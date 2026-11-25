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
Precision, recall, F1 and ROC-AUC are not part of the core (see `threshold_metrics`).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from churn.config import Config

CORE_METRICS = ("capture_at_k", "size_only_capture", "oracle_capture", "pr_auc", "brier_raw", "brier_cal")
_METRICS_AT_K = ("capture_at_k", "size_only_capture", "oracle_capture")

# ------------------------------------------------------------------------------ metrics


def threshold_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    """Precision, recall and F1 when predicting 'lost' for p >= threshold. Annex, not in the core."""
    raise NotImplementedError("evaluation.threshold_metrics")


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold maximising F1 (call it on validation data). Annex, not in the core."""
    raise NotImplementedError("evaluation.best_f1_threshold")


def pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Average precision; NaN when there is no positive."""
    y = np.asarray(y)
    return float(average_precision_score(y, p)) if y.sum() > 0 else float("nan")


def brier(y: np.ndarray, p: np.ndarray) -> float:
    """Mean squared error of a probability; NaN when empty."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2)) if len(y) else float("nan")


def _capture(key: np.ndarray, lost: np.ndarray, date: np.ndarray, ks: Sequence[int]) -> np.ndarray:
    """Captured share of `lost` in the top K of `key`, for every K at once, top K taken per snapshot date.
    Ties keep the row order (rows are sorted by client_id), so the result is reproducible."""
    num, den = np.zeros(len(ks)), 0.0
    for d in np.unique(date):
        m = date == d
        cum = np.cumsum(lost[m][np.argsort(-key[m], kind="stable")])
        num += [cum[min(k, len(cum)) - 1] for k in ks]
        den += cum[-1]
    return num / den if den > 0 else np.full(len(ks), np.nan)


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


def evaluate_predictions(pred: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Long table (run_id, split, segment, metric, k, value, ci_low, ci_high) for one run.

    One block per split, with segment "all" (every core metric) and one segment per value of `segment`
    (probability metrics only). `k` is empty for metrics that do not depend on K.
    """
    wanted = tuple(cfg.evaluation.metrics)
    unknown = sorted(set(wanted) - set(CORE_METRICS))
    if unknown:
        raise ValueError(f"Unknown or not implemented metric(s) {unknown}; available: {list(CORE_METRICS)}")
    if pred["run_id"].nunique() != 1:
        raise ValueError("evaluate_predictions takes the predictions of exactly one run")
    ks, n_boot = list(cfg.evaluation.k_values), cfg.evaluation.bootstrap_n

    rows = []
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

    out = pd.DataFrame(rows, columns=["run_id", "split", "segment", "metric", "k", "value", "ci_low", "ci_high"])
    out["k"] = out["k"].astype("Int64")
    return out


# -------------------------------------------------------------------------------- impact
# revenue x lost share x detection x retention x margin. Detection comes from the backtest,
# the lost share is measured on the dataset, retention is a team hypothesis (scenarios) until the
# pilot measures it. The net impact subtracts the opportunity cost of dedicated reps.


def impact_scenarios(metrics: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Low / central / high scenarios (retention 10 / 20 / 30 %), in M EUR of gross margin per year."""
    raise NotImplementedError("evaluation.impact_scenarios")


def tornado_table(cfg: Config, ranges: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """One row per parameter (lost share, detection, retention): impact at its low and high
    value, others at the central case, sorted by amplitude."""
    raise NotImplementedError("evaluation.tornado_table")
