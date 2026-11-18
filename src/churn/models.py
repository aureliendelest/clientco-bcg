"""Models: registry, rule baseline, probability calibration and hyperparameter tuning."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from churn.config import Config, ModelCfg

# ---------------------------------------------------------------------------- registry


def build_baseline(params: dict[str, Any], seed: int) -> Any:
    """Rule baseline: relative recency (recency / client's p90 gap). The score to beat."""
    raise NotImplementedError("models.build_baseline")


def build_logreg(params: dict[str, Any], seed: int) -> Any:
    """Logistic regression on imputed, scaled features (readable coefficients)."""
    raise NotImplementedError("models.build_logreg")


def build_hgb(params: dict[str, Any], seed: int) -> Any:
    """scikit-learn HistGradientBoostingClassifier."""
    raise NotImplementedError("models.build_hgb")


def build_xgb(params: dict[str, Any], seed: int) -> Any:
    """XGBoost classifier (tree method `hist`)."""
    raise NotImplementedError("models.build_xgb")


MODEL_REGISTRY: dict[str, Callable[[dict[str, Any], int], Any]] = {
    "baseline": build_baseline,
    "logreg": build_logreg,
    "hgb": build_hgb,
    "xgb": build_xgb,
}
# Optional, time-boxed (add to the registry only if tried): small MLP, 'tested, not retained' if no gain.


def get_model(cfg: ModelCfg, seed: int) -> Any:
    """Build the estimator selected by the configuration."""
    raise NotImplementedError("models.get_model")


class RelativeRecencyBaseline:
    """Score = recency / p90 gap of the client (higher = more at risk).

    Follows the scikit-learn estimator API so that it goes through the same pipeline as the models.
    The raw score is monotone with risk; calibration turns it into a probability.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RelativeRecencyBaseline:
        raise NotImplementedError("models.RelativeRecencyBaseline.fit")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return an (n, 2) array, like scikit-learn classifiers."""
        raise NotImplementedError("models.RelativeRecencyBaseline.predict_proba")


# ------------------------------------------------------------------------- calibration
# Required: the expected value multiplies P by V, so 0.05 must really mean 5 %.


@dataclass
class Calibrator:
    """Fitted calibrators, one per segment (or a single one)."""

    method: str
    by_segment: bool
    models: dict[str, Any]


def fit_calibrator(p_raw: np.ndarray, y: np.ndarray, segment: pd.Series, cfg: ModelCfg) -> Calibrator:
    """Fit on validation data only, never on the test snapshot."""
    raise NotImplementedError("models.fit_calibrator")


def apply_calibrator(calibrator: Calibrator, p_raw: np.ndarray, segment: pd.Series) -> np.ndarray:
    """Return calibrated probabilities `p_cal`."""
    raise NotImplementedError("models.apply_calibrator")


# ------------------------------------------------------------------------------ tuning


def tune(cfg: Config, train: pd.DataFrame) -> dict[str, Any]:
    """Run `cfg.model.tuning_trials` Optuna trials with grouped-by-client K-fold on the training
    snapshots. The test snapshot is never used. Returns the best hyperparameters."""
    raise NotImplementedError("models.tune")
