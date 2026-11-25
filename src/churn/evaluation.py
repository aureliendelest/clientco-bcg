"""Evaluation, always computed from saved predictions (never at training time).

Prediction schema:
    run_id, client_id, snapshot_date, split, p_raw, p_cal, V, ev, label, gone, horizon_days, segment

Core metrics (the long table written by `evaluate_predictions`, one row per metric):

    capture_at_k       revenue captured among the top K, ranked by `ev`:
                       sum(V of true churners in the top K) / sum(V of all true churners).
                       "True churner" means `gone` (no purchase within 180 days) whatever the training
                       label: labels A and B are compared on the same truth. THE metric of decision.
    size_only_capture  same, ranked by V alone ("the K biggest clients", no model): the benchmark.
    oracle_capture     same, ranking the true churners by V: the best any model could do with K calls.
    pr_auc             average precision of `p_raw` against `gone` (ranking quality, comparable across labels).
    brier_raw/_cal     mean squared error of `p_raw` / `p_cal` against `label`, the event the model predicts
                       (NOT `gone`: with label B the two events differ, so Brier is not comparable across labels).

K is the same for every run (`evaluation.k_values`, headline K = 700). With several snapshot dates in a split,
the top K is taken per date and the revenues are pooled. Per-segment rows only carry the probability metrics
(pr_auc, brier): a top K inside a segment would not be the list the reps receive.
Confidence intervals (95 %) resample clients, not rows, with the same draws for every metric.
Precision, recall, F1 and ROC-AUC are not part of the core and not implemented here.

Total-churn metrics (only when the universe of the test date is given, see `load_universe`). The core metrics
above divide by the churned revenue of the regular clients (the perimeter); these three divide by the churned
revenue of ALL clients of the snapshot (regular or not), so they say how much of the whole loss a list of K calls
recovers. Segment "all", same K, same bootstrap (clients of the universe are resampled):

    capture_total_at_k       V of the churners in the model's top K (ranked by `ev`, regular clients only)
                             / V of all the churners of the universe.
    oracle_total_capture     top K of ALL clients ranked by V x gone / same denominator.
    size_only_total_capture  the K biggest clients of the whole universe (by V) / same denominator.

They are not in `evaluation.metrics` (the config hash would change and every run would be rerun): they are
computed whenever a universe is passed to `evaluate_predictions`.
Impact scenarios and the tornado live in `impact.py`, not here.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from churn.config import Config, resolve

CORE_METRICS = ("capture_at_k", "size_only_capture", "oracle_capture", "pr_auc", "brier_raw", "brier_cal")
_METRICS_AT_K = ("capture_at_k", "size_only_capture", "oracle_capture")
# Computed when a universe is given, never listed in `evaluation.metrics` (see the module docstring).
TOTAL_METRICS = ("capture_total_at_k", "oracle_total_capture", "size_only_total_capture")
UNIVERSE_COLUMNS = ["client_id", "snapshot_date", "V", "gone", "rhythm", "rev_90d"]

# ------------------------------------------------------------------------------ metrics


def pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Average precision; NaN when there is no positive."""
    y = np.asarray(y)
    return float(average_precision_score(y, p)) if y.sum() > 0 else float("nan")


def brier(y: np.ndarray, p: np.ndarray) -> float:
    """Mean squared error of a probability; NaN when empty."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2)) if len(y) else float("nan")


def _capture(
    key: np.ndarray, lost: np.ndarray, date: np.ndarray, ks: Sequence[int], den: float | None = None
) -> np.ndarray:
    """Captured share of `lost` in the top K of `key`, for every K at once, top K taken per snapshot date.
    Ties keep the row order (rows are sorted by client_id), so the result is reproducible.
    `den`: total to divide by; by default the total of `lost` over the rows."""
    num, total = np.zeros(len(ks)), 0.0
    for d in np.unique(date):
        m = date == d
        cum = np.cumsum(lost[m][np.argsort(-key[m], kind="stable")])
        num += [cum[min(k, len(cum)) - 1] for k in ks]
        total += cum[-1]
    total = total if den is None else den
    return num / total if total > 0 else np.full(len(ks), np.nan)


def capture_at_k(pred: pd.DataFrame, k: int, rank_by: str = "ev") -> float:
    """Share of the churned value captured in the top `k` clients ranked by `rank_by`
    (`ev` = the model, `V` = the "biggest clients" benchmark). Rows whose truth is unknown are ignored."""
    a = _arrays(_clean(pred), extra=rank_by)
    return float(_capture(a[rank_by], a["V"] * a["gone"], a["date"], [k])[0])


def oracle_capture(pred: pd.DataFrame, k: int) -> float:
    """Best possible capture with `k` places: rank the true churners by V."""
    return capture_at_k(pred.assign(oracle=pred["V"] * pred["gone"]), k, rank_by="oracle")


# --------------------------------------------------------------------------- bootstrap


def _clean(pred: pd.DataFrame) -> pd.DataFrame:
    """Rows with a known truth, sorted by (snapshot_date, client_id): the order that makes ties reproducible."""
    df = pred.loc[pred["gone"].notna()].sort_values(["snapshot_date", "client_id"], kind="stable")
    return df.reset_index(drop=True)


def _arrays(df: pd.DataFrame, extra: str | None = None) -> dict[str, np.ndarray]:
    cols = ["V", "gone", "ev", "p_raw", "p_cal", "label"] + ([extra] if extra else [])
    out = {c: df[c].to_numpy(dtype=float) for c in dict.fromkeys(cols)}
    out["client"] = df["client_id"].to_numpy()
    out["date"] = pd.factorize(df["snapshot_date"])[0]
    return out


def _resample_rows(client: np.ndarray, n: int, seed: int) -> Iterator[np.ndarray]:
    """`n` bootstrap draws as row indices: clients are drawn with replacement and each drawn client brings
    ALL its rows (every snapshot date). Rows come back sorted by client."""
    uniq, codes = np.unique(client, return_inverse=True)
    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes)
    starts = np.cumsum(counts) - counts
    rng = np.random.default_rng(seed)
    for _ in range(n):
        c = np.sort(rng.integers(0, len(uniq), len(uniq)))
        lens = counts[c]
        within = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
        yield order[np.repeat(starts[c], lens) + within]


def bootstrap_ci(pred: pd.DataFrame, fn: Callable[[pd.DataFrame], float], n: int, seed: int) -> tuple[float, float]:
    """95 % confidence interval by resampling clients (not rows). A client drawn twice takes two places
    in the top K (K is not rescaled)."""
    df = pred.sort_values(["snapshot_date", "client_id"], kind="stable").reset_index(drop=True)
    draws = [fn(df.iloc[idx]) for idx in _resample_rows(df["client_id"].to_numpy(), n, seed)]
    if not draws or not np.isfinite(draws).any():
        return float("nan"), float("nan")
    low, high = np.nanpercentile(draws, [2.5, 97.5])
    return float(low), float(high)


# ------------------------------------------------------------------------- long table


def _core(a: dict[str, np.ndarray], ks: Sequence[int], wanted: Sequence[str], prob_only: bool) -> dict:
    """All requested metrics of one set of rows: {(metric, k or None): value}."""
    out: dict[tuple[str, int | None], float] = {}
    lost = a["V"] * a["gone"]
    if not prob_only:
        for name, key in (("capture_at_k", a["ev"]), ("size_only_capture", a["V"]), ("oracle_capture", lost)):
            if name in wanted:
                out |= {(name, k): float(v) for k, v in zip(ks, _capture(key, lost, a["date"], ks))}
    if "pr_auc" in wanted:
        out[("pr_auc", None)] = pr_auc(a["gone"], a["p_raw"])
    known = np.isfinite(a["label"])
    for name, col in (("brier_raw", "p_raw"), ("brier_cal", "p_cal")):
        if name in wanted:
            out[(name, None)] = brier(a["label"][known], a[col][known])
    return out


# --------------------------------------------------------------------------- universe


def load_universe(cfg: Config) -> pd.DataFrame:
    """Every client of the snapshot of `cfg.snapshots.test_date`, read from the snapshot cache only.

    Columns: client_id, snapshot_date, V, gone, rhythm, rev_90d. The perimeter (regular clients) is
    `rhythm in snapshots.PERIMETER_RHYTHMS`. `gone` is NaN where the truth window is not fully observed.
    The cache is never built here (no DuckDB): the file must already exist, because a run on the same
    configuration wrote it.
    """
    from churn import snapshots  # imported here: `data` pulls DuckDB, which evaluation does not need
    from churn.data import fingerprint

    parquet = resolve(cfg.dataset.shifted_parquet)
    if not parquet.exists():
        raise FileNotFoundError(f"{parquet} not found: cannot compute the snapshot cache key of this configuration")
    # `snapshots.PROCESSED_DIR` is read at call time: the tests redirect the cache folder there
    folder = snapshots.PROCESSED_DIR / f"snapshots-{snapshots.cache_key(cfg, fingerprint(parquet))}"
    path = folder / f"{cfg.snapshots.test_date}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No cached snapshot for {cfg.snapshots.test_date} at {path}. load_universe only reads the cache "
            "(it never builds snapshots): run an experiment with this configuration first."
        )
    return pd.read_parquet(path, columns=UNIVERSE_COLUMNS)


def _universe_block(df: pd.DataFrame, universe: pd.DataFrame) -> dict[str, np.ndarray] | None:
    """Arrays of the universe rows of the dates of `df` (the cleaned predictions of one split), with the model's
    score `ev` on the clients that were predicted (-inf elsewhere) and `ranked`, which marks them.
    None when the universe does not hold all the dates of the split."""
    dates = pd.unique(df["snapshot_date"])
    if not set(dates) <= set(pd.unique(universe["snapshot_date"])):
        return None
    uni = universe.loc[universe["snapshot_date"].isin(dates) & universe["gone"].notna()]
    uni = uni.sort_values(["snapshot_date", "client_id"], kind="stable").reset_index(drop=True)
    if uni.duplicated(["snapshot_date", "client_id"]).any() or df.duplicated(["snapshot_date", "client_id"]).any():
        raise ValueError("Duplicate (snapshot_date, client_id) in the predictions or in the universe")
    keys = ["snapshot_date", "client_id"]
    joined = uni.merge(df[[*keys, "ev", "V", "gone"]], on=keys, how="left", suffixes=("", "_pred"), indicator=True)
    if (df.merge(uni[keys], on=keys, how="left", indicator=True)["_merge"] != "both").any():
        raise ValueError("Some predicted clients are not in the universe: wrong cache or wrong configuration")
    seen = joined["_merge"] == "both"
    if not np.allclose(joined.loc[seen, "V"], joined.loc[seen, "V_pred"], rtol=1e-9, equal_nan=True) or not (
        joined.loc[seen, "gone"] == joined.loc[seen, "gone_pred"]
    ).all():
        raise ValueError("V or gone differ between the predictions and the universe: wrong cache or configuration")
    ev = joined["ev"].to_numpy(dtype=float)
    ranked = seen.to_numpy() & np.isfinite(ev)
    return {
        "V": joined["V"].to_numpy(dtype=float),
        "gone": joined["gone"].to_numpy(dtype=float),
        "ev": np.where(ranked, ev, -np.inf),
        "ranked": ranked,
        "client": joined["client_id"].to_numpy(),
        "date": pd.factorize(joined["snapshot_date"])[0],
    }


def _total_core(a: dict[str, np.ndarray], ks: Sequence[int]) -> dict[tuple[str, int], float]:
    """The three total-churn metrics of one set of universe rows: {(metric, k): value}.
    One denominator (all the churned V of the universe, per date then pooled) for the three."""
    lost = a["V"] * a["gone"]
    den = sum(np.cumsum(np.sort(lost[a["date"] == d])[::-1])[-1] for d in np.unique(a["date"]))
    out: dict[tuple[str, int], float] = {}
    for name, key, num in (
        ("capture_total_at_k", a["ev"], lost * a["ranked"]),
        ("oracle_total_capture", lost, lost),
        ("size_only_total_capture", a["V"], lost),
    ):
        out |= {(name, k): float(v) for k, v in zip(ks, _capture(key, num, a["date"], ks, den=den))}
    return out


def _total_rows(run_id: str, split: str, a: dict[str, np.ndarray], ks: Sequence[int], n_boot: int, seed: int) -> list:
    """Rows of the long table for the total-churn metrics, with the bootstrap interval (clients of the universe
    are resampled, so the clients that were never ranked count in the denominator's randomness too)."""
    point = _total_core(a, ks)
    draws: dict = {key: [] for key in point}
    for idx in _resample_rows(a["client"], n_boot, seed):
        for key, value in _total_core({c: v[idx] for c, v in a.items()}, ks).items():
            draws[key].append(value)
    rows = []
    for (metric, k), value in point.items():
        d = np.asarray(draws[(metric, k)], dtype=float)
        low, high = np.nanpercentile(d, [2.5, 97.5]) if np.isfinite(d).any() else (np.nan, np.nan)
        rows.append((run_id, split, "all", metric, k, value, low, high))
    return rows


def evaluate_predictions(pred: pd.DataFrame, cfg: Config, universe: pd.DataFrame | None = None) -> pd.DataFrame:
    """Long table (run_id, split, segment, metric, k, value, ci_low, ci_high) for one run.

    One block per split, with segment "all" (every core metric) and one segment per value of `segment`
    (probability metrics only). `k` is empty for metrics that do not depend on K.

    `universe` (see `load_universe`): without it, nothing changes. With it, the three `TOTAL_METRICS` are added
    (segment "all", one row per K) for every split whose snapshot dates are all in the universe; the other
    splits are left as they are. The universe must cover at least one split.
    """
    wanted = tuple(cfg.evaluation.metrics)
    unknown = sorted(set(wanted) - set(CORE_METRICS))
    if unknown:
        raise ValueError(f"Unknown or not implemented metric(s) {unknown}; available: {list(CORE_METRICS)}")
    if pred["run_id"].nunique() != 1:
        raise ValueError("evaluate_predictions takes the predictions of exactly one run")
    if universe is not None:
        missing = sorted({"client_id", "snapshot_date", "V", "gone"} - set(universe.columns))
        if missing:
            raise ValueError(f"The universe lacks the column(s) {missing}")
    ks, n_boot = list(cfg.evaluation.k_values), cfg.evaluation.bootstrap_n

    rows = []
    covered = False
    for split, part in pred.groupby("split", sort=True):
        df = _clean(part)
        blocks = [("all", df)] + [(str(seg), g) for seg, g in df.groupby("segment", sort=True, observed=True)]
        for seg, block in blocks:
            if block.empty:
                continue
            a, prob_only = _arrays(block), seg != "all"
            point = _core(a, ks, wanted, prob_only)
            draws: dict = {key: [] for key in point}
            for idx in _resample_rows(a["client"], n_boot, cfg.seed):
                for key, value in _core({c: v[idx] for c, v in a.items()}, ks, wanted, prob_only).items():
                    draws[key].append(value)
            for (metric, k), value in point.items():
                d = np.asarray(draws[(metric, k)], dtype=float)
                low, high = np.nanpercentile(d, [2.5, 97.5]) if np.isfinite(d).any() else (np.nan, np.nan)
                rows.append((pred["run_id"].iloc[0], split, seg, metric, k, value, low, high))
        if universe is not None and not df.empty:
            u = _universe_block(df, universe)
            if u is not None:
                covered = True
                rows += _total_rows(pred["run_id"].iloc[0], split, u, ks, n_boot, cfg.seed)
    if universe is not None and not covered:
        raise ValueError("The universe holds none of the snapshot dates of the predictions")

    out = pd.DataFrame(rows, columns=["run_id", "split", "segment", "metric", "k", "value", "ci_low", "ci_high"])
    out["k"] = out["k"].astype("Int64")
    return out
