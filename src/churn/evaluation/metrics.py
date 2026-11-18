"""Evaluation metrics, computed from the predictions table.

Prediction schema:
    run_id, client_id, snapshot_date, split, p_raw, p_cal, V, rho, ev, label, horizon_days, segment

Main metric (comparable when H varies by client): revenue captured among the top K =
sum(V of true churners in the top K) / sum(V of all true churners), by capacity tier.
Probability metrics (precision, recall, F1, PR-AUC, ROC-AUC, Brier) are for tuning and comparison;
thresholds are chosen on validation, never on the test snapshot.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from churn.config import Config


def threshold_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    """Precision, recall and F1 when predicting 'lost' for p >= threshold."""
    raise NotImplementedError("evaluation.metrics.threshold_metrics")


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold maximising F1 (call it on validation data)."""
    raise NotImplementedError("evaluation.metrics.best_f1_threshold")


def capture_at_k(pred: pd.DataFrame, k: int, rank_by: str = "ev") -> float:
    """Share of the churned value captured in the top `k` clients ranked by `rank_by`."""
    raise NotImplementedError("evaluation.metrics.capture_at_k")


def calibration_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Brier score, log loss and a reliability table summary."""
    raise NotImplementedError("evaluation.metrics.calibration_metrics")


def bootstrap_ci(pred: pd.DataFrame, fn: Callable[[pd.DataFrame], float], n: int, group_by: str, seed: int) -> tuple[float, float]:
    """95 % confidence interval by resampling clients (not rows)."""
    raise NotImplementedError("evaluation.metrics.bootstrap_ci")


def evaluate_predictions(pred: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Long table (run_id, split, segment, metric, k, value, ci_low, ci_high) for one run."""
    raise NotImplementedError("evaluation.metrics.evaluate_predictions")
