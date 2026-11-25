"""From probabilities to an action list: value at risk, expected value, list size, reasons.

V is the revenue the client would generate IF HE STAYED, estimated on his recent rhythm. It is not
a forecast of realized revenue: that would be ~0 for clients who leave and would count the
departure twice (once in P, once in V). EV = P x V, with the same retention rate for
everybody until the pilot says otherwise. (A haircut rho for clients who already declined was set aside
on 25/11; it may come back later.)

The list is the one of the total-churn score alone (no quota for a partial-churn score): the K clients of the
perimeter with the highest EV. `score_batch` is the simulated production step that writes it.

Import note: `churn.snapshots` imports `compute_value` from this module, so everything that needs `churn.models`,
`churn.snapshots`, `churn.data` or `churn.pipeline` is imported inside the functions that use it.
"""

from __future__ import annotations

import math
import pickle
import re
import string
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from churn.config import RUNS_DIR, Config, load_config, override_config, resolve

# ----------------------------------------------------------------------------- value


ANNUALIZATION_FACTOR = 2  # 180 days of revenue -> one year (the team rounds 365 / 180 to 2)


def compute_value(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """V per client according to `ScoringCfg.value_definition` (default: last 180 days, annualized).
    Reads `rev_180d`, the revenue of the 180 days before the snapshot."""
    definition = cfg.scoring.value_definition
    if definition == "ca_180d_annualized":
        return (df["rev_180d"] * ANNUALIZATION_FACTOR).rename("V")
    if definition == "ca_12m":
        # Needs 365 days of history: the first snapshots (6 months of data) do not have it.
        raise ValueError("scoring.value_definition='ca_12m' is not supported (needs 365 days of history); use 'ca_180d_annualized'")
    raise ValueError(f"Unknown value_definition {definition!r}")


# -------------------------------------------------------------------- expected value


def expected_value(p_cal: pd.Series, value: pd.Series) -> pd.Series:
    """EV = p_cal * V."""
    return (p_cal * value).rename("ev")


def apply_min_proba(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Drop the clients whose calibrated probability `p_cal` is below `ScoringCfg.min_proba` (0 keeps everybody)."""
    return df.loc[df["p_cal"] >= cfg.scoring.min_proba]


# ------------------------------------------------------------------------- list size
# Capacity hypothesis: `calls_per_week` company-wide (500), converted to the dataset scale (`dataset_share`
# ~ 30 % of revenue): K ~ 700 for a monthly list, ~ 4 000 for six lists cumulated. The list has no quota for a second
# score: it is the total-churn score alone.


def list_size(cfg: Config) -> int:
    """K of a monthly list at the dataset scale: `ScoringCfg.list_size_k` (700), a positive integer."""
    k = cfg.scoring.list_size_k
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"scoring.list_size_k must be a positive integer, got {k!r}")
    return k


# ---------------------------------------------------------------------------- reasons
# Display texts are French (deliverable) and live in `reasons_fr.yaml`; keys are English feature names. A template
# may use `{value}` (the client's own value of that feature) and the name of another feature in braces.
# SHAP is computed only for the clients of the list, never for all rows.
#
# What a reason is. Only the variables that RAISE the client's risk are candidates (positive SHAP value, or positive
# coefficient x standardized value for the logistic regression); the three largest become `reason_1..3`. A variable
# without a template is never a reason (no reason without a validated sentence), and a sentence is used only if it
# is true for the client (a "baisse" is shown only for a negative trend). Fewer than three candidates: the remaining
# reasons are empty strings. The baseline rules and BG/NBD have no per-variable effect: all their reasons are empty.
# The attributions explain the raw model score, before the monotone calibration, so the direction is the same.


def load_reasons() -> dict[str, str]:
    """Read `reasons_fr.yaml` (feature name -> French sentence template), checking the placeholders."""
    path = Path(__file__).with_name("reasons_fr.yaml")
    templates = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(templates, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in templates.items()):
        raise ValueError(f"{path.name}: expected a mapping feature name -> sentence")
    for name, text in templates.items():
        for _, field, _, _ in string.Formatter().parse(text):
            if field and field != "value" and field not in templates:
                raise ValueError(f"{path.name}: the sentence of {name!r} uses the unknown placeholder {{{field}}}")
    return templates


def _num(x: float, decimals: int = 0) -> str:
    """French number: decimal comma, non-breaking space between thousands."""
    text = f"{x:,.{decimals}f}".replace(",", " ").replace(".", ",")
    return text.rstrip("0").rstrip(",") if decimals and "," in text else text


def _fmt_days(x: float) -> str:
    return _num(x, 0)


def _fmt_gap(x: float) -> str:
    return _num(x, 1 if x < 10 else 0)


def _fmt_money(x: float) -> str:
    return _num(x, 0) + " €"


def _fmt_ratio(x: float) -> str:
    return _num(x, 1)


def _fmt_decline_pct(trend: float) -> str:
    """The sentence says "en baisse de {value} %": `trend` is a log-ratio, the drop is 100 * (1 - exp(trend))."""
    return _num(100.0 * (1.0 - math.exp(trend)), 0)


_FORMATTERS = {
    "recency_days": _fmt_days, "recency_ratio": _fmt_ratio, "freq_90d": _fmt_days, "freq_180d": _fmt_days,
    "freq_prev_90d": _fmt_days, "rev_90d": _fmt_money, "rev_180d": _fmt_money, "rev_prev_90d": _fmt_money,
    "gap_median": _fmt_gap, "gap_p90": _fmt_gap, "freq_trend": _fmt_decline_pct, "rev_trend": _fmt_decline_pct,
}
_DECLINE_ONLY = ("freq_trend", "rev_trend")  # their sentence says "baisse": usable only for a drop of at least 1 %
_SILENCE_ONLY = ("recency_ratio",)  # "silence de x fois son rythme": usable only from 1 (past his usual longest pause)


def _usable(name: str, value: float) -> bool:
    if value is None or not np.isfinite(value):
        return False
    if name in _DECLINE_ONLY:
        return round(100.0 * (1.0 - math.exp(value))) >= 1
    if name in _SILENCE_ONLY:
        return value >= 1.0
    return True


def _render(name: str, row: pd.Series, templates: dict[str, str]) -> str | None:
    """The sentence of feature `name` for one client, or None when it cannot be stated truthfully."""
    if name not in templates or name not in _FORMATTERS:
        return None
    needed = [name] + [f for _, f, _, _ in string.Formatter().parse(templates[name]) if f and f != "value"]
    values = {}
    for feature in needed:
        if feature not in row.index or feature not in _FORMATTERS or not _usable(feature, row[feature]):
            return None
        values[feature] = _FORMATTERS[feature](float(row[feature]))
    fields = {f: values[f] for f in needed if f != name}
    sentence = templates[name].format(value=values[name], **fields)
    return re.sub(r"(?<![\d,\u00a0])1 jours", "1 jour", sentence)  # "1 jours" -> "1 jour"; "11 jours" is left alone


def _contributions(model: Any, X: pd.DataFrame) -> pd.DataFrame | None:
    """Per-client, per-variable contribution to the risk (log-odds scale; > 0 raises the risk), or None when the model
    has no per-variable effect (baseline rules, BG/NBD)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier

    from churn.models import BGNBD, RuleBaseline

    if isinstance(model, (RuleBaseline, BGNBD)):
        return None
    if isinstance(model, (HistGradientBoostingClassifier, XGBClassifier)):
        import shap

        values = shap.TreeExplainer(model).shap_values(X)
        if isinstance(values, list):  # older shap versions: one array per class
            values = values[-1]
        values = np.asarray(values)
        if values.ndim == 3:  # (rows, features, classes): the class "lost"
            values = values[:, :, -1]
        return pd.DataFrame(values, index=X.index, columns=list(X.columns))
    if hasattr(model, "steps") and isinstance(model[-1], LogisticRegression):
        # coefficient x standardized value: how far above the average client the variable pushes the risk. The
        # missing-value indicators the pipeline adds are not variables of the client and are dropped.
        names = list(model[:-1].get_feature_names_out(list(X.columns)))
        standardized = model[:-1].transform(X)
        parts = pd.DataFrame(standardized * model[-1].coef_[0], index=X.index, columns=names)
        return parts[[c for c in X.columns if c in parts.columns]]
    raise ValueError(f"top_reasons: model {type(model).__name__} is not supported (hgb, xgb and logreg are)")


def top_reasons(model: Any, X: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """`n` reasons per client, as French sentences built from `load_reasons()`.

    SHAP values (`TreeExplainer`) for HGB and XGB, coefficients x standardized values for the logistic regression, no
    reason (empty strings) for the baseline rules and BG/NBD. `X` has the model's feature columns, one row per client
    (the clients of the list, not all rows). Returns a frame indexed like `X` with columns `reason_1` ... `reason_n`
    ("" where the client has fewer than `n` usable reasons), the largest contribution first."""
    columns = [f"reason_{i}" for i in range(1, n + 1)]
    out = pd.DataFrame("", index=X.index, columns=columns)
    contributions = _contributions(model, X) if len(X) else None
    if contributions is None:
        return out
    templates = load_reasons()
    candidates = [c for c in contributions.columns if c in templates]
    for i, (_, row) in enumerate(X.iterrows()):
        pushes = contributions[candidates].iloc[i]
        sentences = []
        for name in pushes[pushes > 0].sort_values(ascending=False, kind="stable").index:
            sentence = _render(name, row, templates)
            if sentence is not None:
                sentences.append(sentence)
            if len(sentences) == n:
                break
        out.iloc[i, : len(sentences)] = sentences
    return out


# ------------------------------------------------------------------------ batch list

LIST_COLUMNS = ["rank", "client_id", "p_cal", "V", "ev", "segment", "reason_1", "reason_2", "reason_3"]


def _load_run(run_dir: Path) -> tuple[dict[str, Any], Config]:
    """The `model.pkl` bundle (model, calibrator, features, config) and the run's configuration (trusted local file)."""
    pkl = run_dir / "model.pkl"
    if not pkl.exists():
        raise FileNotFoundError(f"{pkl} not found: is {run_dir.name!r} a complete run?")
    import churn.models  # noqa: F401  (the classes pickled in the bundle live there)

    with pkl.open("rb") as f:
        bundle = pickle.load(f)
    missing = {"model", "calibrator", "features", "config"} - set(bundle)
    if missing:
        raise ValueError(f"{pkl}: missing key(s) {sorted(missing)}")
    # Optional key (`pipeline.run` may record `SNAPSHOT_VERSION`): a model trained on features defined otherwise than
    # the ones computed here would be fed wrong values without any error.
    from churn.snapshots import SNAPSHOT_VERSION

    trained_on = bundle.get("snapshot_version")
    if trained_on is not None and trained_on != SNAPSHOT_VERSION:
        raise ValueError(f"{run_dir.name} was trained on snapshot version {trained_on}, the code computes version {SNAPSHOT_VERSION}: rerun it")
    return bundle, load_config(run_dir / "config.yaml")


def score_batch(
    run_id: str, snapshot_date: date, *, runs_dir: Path | None = None, output_dir: Path | None = None
) -> Path:
    """Simulated 'production' step (no API): load the model of `runs/<run_id>/`, build the snapshot of `snapshot_date`
    from the history before that date only (no label, no `gone`: the future is not looked at), keep the regular
    clients, rank them by EV = p_cal x V, keep the K = `list_size(cfg)` first, add the three reasons, write
    `action_list_<date>.csv` next to the model (or in `output_dir`) and return its path.

    Output columns: rank, client_id, p_cal, V, ev, segment, reason_1, reason_2, reason_3. `p_cal` below
    `ScoringCfg.min_proba` is dropped before the cut. Fewer than K eligible clients: the list is shorter.
    The final test date on the real dataset is refused unless `CHURN_FINAL_TEST=1` (same guard as `pipeline.run`).
    `runs_dir` replaces `runs/`."""
    from churn import models
    from churn.data import build_client_day_table, connect
    from churn.pipeline import _assert_test_allowed
    from churn.snapshots import assign_segments, compute_features, in_perimeter, rhythm_class

    run_dir = Path(runs_dir if runs_dir is not None else RUNS_DIR) / run_id
    bundle, cfg = _load_run(run_dir)
    _assert_test_allowed(override_config(cfg, {"snapshots.test_date": f"{snapshot_date:%Y-%m-%d}"}))
    features: list[str] = list(bundle["features"])

    con = connect(resolve(cfg.dataset.shifted_parquet))
    try:
        build_client_day_table(con)
        # Features only: `build_snapshot` would also compute the label and `gone` from the days after T.
        df = compute_features(
            con, snapshot_date, cfg.snapshots.feature_set,
            lookback_days=cfg.snapshots.lookback_days, min_purchases=cfg.label.min_purchases,
        )
    finally:
        con.close()
    needed = {"freq_180d", "gap_median", "rev_180d", *features}
    if needed - set(df.columns):
        raise ValueError(f"The features of the snapshot lack {sorted(needed - set(df.columns))}")
    df["rhythm"] = rhythm_class(df, cfg)
    df["V"] = compute_value(df, cfg)
    df = assign_segments(df, cfg)  # on the whole snapshot, like `build_snapshot`
    df = df.loc[in_perimeter(df)].sort_values("client_id", kind="stable").reset_index(drop=True)

    p_raw = bundle["model"].predict_proba(df[features])[:, 1]
    df["p_cal"] = models.apply_calibrator(bundle["calibrator"], p_raw, df["segment"])
    df["ev"] = expected_value(df["p_cal"], df["V"])
    ranked = apply_min_proba(df, cfg).sort_values(["ev", "client_id"], ascending=[False, True], kind="stable")
    top = ranked.head(list_size(cfg)).reset_index(drop=True)
    top["rank"] = np.arange(1, len(top) + 1)
    top = pd.concat([top, top_reasons(bundle["model"], top[features], n=3)], axis=1)

    out_dir = Path(output_dir) if output_dir is not None else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"action_list_{snapshot_date:%Y-%m-%d}.csv"
    top[LIST_COLUMNS].to_csv(path, index=False)
    return path
