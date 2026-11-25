"""Evaluation, always computed from saved predictions (never at training time).

Prediction schema:
    run_id, client_id, snapshot_date, split, p_raw, p_cal, V, ev, label, gone, horizon_days, segment

Main metric (comparable when H varies by client): revenue captured among the top K =
sum(V of true churners in the top K) / sum(V of all true churners), where "true churner" means `gone`
(no purchase within 180 days) whatever the training label: labels A and B are compared on the same truth.
Probability metrics (precision, recall, F1, PR-AUC, ROC-AUC, Brier) are for tuning and comparison;
thresholds are chosen on validation, never on the test snapshot.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from churn.config import Config

# ------------------------------------------------------------------------------ metrics


def threshold_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    """Precision, recall and F1 when predicting 'lost' for p >= threshold."""
    raise NotImplementedError("evaluation.threshold_metrics")


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold maximising F1 (call it on validation data)."""
    raise NotImplementedError("evaluation.best_f1_threshold")


def capture_at_k(pred: pd.DataFrame, k: int, rank_by: str = "ev") -> float:
    """Share of the churned value captured in the top `k` clients ranked by `rank_by`."""
    raise NotImplementedError("evaluation.capture_at_k")


def calibration_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Brier score, log loss and a reliability table summary."""
    raise NotImplementedError("evaluation.calibration_metrics")


def bootstrap_ci(pred: pd.DataFrame, fn: Callable[[pd.DataFrame], float], n: int, seed: int) -> tuple[float, float]:
    """95 % confidence interval by resampling clients (not rows)."""
    raise NotImplementedError("evaluation.bootstrap_ci")


def evaluate_predictions(pred: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Long table (run_id, split, segment, metric, k, value, ci_low, ci_high) for one run."""
    raise NotImplementedError("evaluation.evaluate_predictions")


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
