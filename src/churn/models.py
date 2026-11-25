"""Models: registry, rule baseline, probability calibration and hyperparameter tuning."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from churn.config import Config, ModelCfg

# ---------------------------------------------------------------------------- registry


def build_baseline(params: dict[str, Any], seed: int) -> Any:
    """Deterministic rule baseline, the score to beat. `params["rule"]` selects it (see `RuleBaseline`);
    the default is relative recency (recency / the client's p90 gap). `seed` is unused: no randomness."""
    return RuleBaseline(**params)


def build_logreg(params: dict[str, Any], seed: int) -> Any:
    """Logistic regression on imputed, scaled features (readable coefficients).

    Median imputation with missing-value indicators (rhythm features are NaN below `min_purchases`), then
    standardisation, so the coefficients are comparable. No class weighting: probabilities are calibrated later."""
    options = {"C": 1.0, "max_iter": 1000, "random_state": seed} | params
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), LogisticRegression(**options))


def build_hgb(params: dict[str, Any], seed: int) -> Any:
    """scikit-learn HistGradientBoostingClassifier. Defaults of the notebooks that chose the label (`07`, `08`).

    Handles NaN natively. Reproducible only with one OpenMP thread (`OMP_NUM_THREADS=1`) and rows sorted by client
    (see docs/03_evidence.md)."""
    options = {"max_iter": 200, "learning_rate": 0.08, "early_stopping": False, "random_state": seed} | params
    return HistGradientBoostingClassifier(**options)


def build_xgb(params: dict[str, Any], seed: int) -> Any:
    """XGBoost classifier (tree method `hist`). One thread by default: the multithreaded run is not reproducible."""
    options = {
        "tree_method": "hist", "n_estimators": 200, "learning_rate": 0.08, "max_depth": 6,
        "n_jobs": 1, "random_state": seed, "eval_metric": "logloss",
    } | params
    return XGBClassifier(**options)


MODEL_REGISTRY: dict[str, Callable[[dict[str, Any], int], Any]] = {
    "baseline": build_baseline,
    "logreg": build_logreg,
    "hgb": build_hgb,
    "xgb": build_xgb,
}
# Optional, time-boxed (add to the registry only if tried): small MLP, 'tested, not retained' if no gain.


def get_model(cfg: ModelCfg, seed: int) -> Any:
    """Build the estimator selected by the configuration (`cfg.type`, `cfg.params`).

    Every estimator has `fit(X, y)` and `predict_proba(X)` (an (n, 2) array, column 1 = P(lost))."""
    try:
        build = MODEL_REGISTRY[cfg.type]
    except KeyError:
        raise ValueError(f"Unknown model type {cfg.type!r}; known: {sorted(MODEL_REGISTRY)}") from None
    return build(dict(cfg.params), seed)


# ----------------------------------------------------------------------- rule baselines

_RULES = ("relative_recency", "recency_days", "business_rules")


class RuleBaseline(ClassifierMixin, BaseEstimator):
    """Deterministic risk score from a fixed rule, with no learning (`fit` only records the classes).

    It follows the scikit-learn estimator API so that it goes through the same pipeline as the models. The score
    is monotone with risk and is mapped to [0, 1) by s / (1 + s); calibration then turns it into a probability.

    Rules (`rule`):
      - `relative_recency` (default): s = recency_days / client's p90 gap (the feature `recency_ratio`). A client
        silent for longer than his usual longest pause is at risk.
      - `recency_days`: s = days since the last purchase day. The naive rule, with no notion of the client's rhythm.
      - `business_rules`: s = number of fired rules + the relative recency squashed to [0, 1) as a tie-breaker.
        Rules: (1) `recency_ratio >= ratio_threshold`; (2) `freq_trend <= trend_threshold` (purchase frequency of
        the last 90 days at most that of the 90 days before); (3) `rev_trend <= trend_threshold` (same, revenue).
    A missing input (a client with too short a history has no `recency_ratio`) counts as no risk: such clients
    are outside the perimeter of the model.
    """

    def __init__(self, rule: str = "relative_recency", ratio_threshold: float = 1.0, trend_threshold: float = 0.5):
        self.rule = rule
        self.ratio_threshold = ratio_threshold
        self.trend_threshold = trend_threshold

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> RuleBaseline:
        if self.rule not in _RULES:
            raise ValueError(f"Unknown rule {self.rule!r}; known: {list(_RULES)}")
        self.classes_ = np.array([0, 1])
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Raw risk score s >= 0 (higher = more at risk)."""
        if self.rule not in _RULES:
            raise ValueError(f"Unknown rule {self.rule!r}; known: {list(_RULES)}")
        needed = {
            "relative_recency": ["recency_ratio"], "recency_days": ["recency_days"],
            "business_rules": ["recency_ratio", "freq_trend", "rev_trend"],
        }[self.rule]
        missing = [c for c in needed if c not in X.columns]
        if missing:
            raise KeyError(f"rule {self.rule!r} needs the columns {missing}")
        if self.rule == "recency_days":
            return X["recency_days"].fillna(0.0).to_numpy(dtype=float)
        ratio = X["recency_ratio"].to_numpy(dtype=float)
        relative = np.nan_to_num(ratio, nan=0.0)
        if self.rule == "relative_recency":
            return relative
        fired = (
            (ratio >= self.ratio_threshold).astype(float)  # NaN compares False: rule not fired
            + (X["freq_trend"].to_numpy(dtype=float) <= self.trend_threshold)
            + (X["rev_trend"].to_numpy(dtype=float) <= self.trend_threshold)
        )
        return fired + relative / (1.0 + relative)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return an (n, 2) array, like scikit-learn classifiers. Column 1 is the monotone score s / (1 + s)."""
        s = self.score(X)
        p = s / (1.0 + s)
        return np.column_stack([1.0 - p, p])


# ------------------------------------------------------------------------- calibration
# Required: the expected value multiplies P by V, so 0.05 must really mean 5 %.


ALL_SEGMENTS = "__all__"  # key of the calibrator fitted on every client; fallback for a segment with no calibrator of its own
MIN_SEGMENT_FIT = 200  # below this many clients (or with a single class) a segment is calibrated by the global calibrator


@dataclass
class Calibrator:
    """Fitted calibrators, one per segment (or a single one).

    `models` maps a segment value (as a string) to a fitted `IsotonicRegression`, plus `ALL_SEGMENTS` for the
    calibrator fitted on every client. It is empty when `method == "none"` (identity)."""

    method: str
    by_segment: bool
    models: dict[str, Any]


def _isotonic(p_raw: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    # Bounds in [0, 1] and clipping outside the fitted range: a raw score never seen at fit time gets the
    # probability of the nearest one. Isotonic output is a step function: clients sharing a step tie on P (the
    # ranking by P x V then falls back on V inside the step).
    return IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip").fit(p_raw, y)


def fit_calibrator(p_raw: np.ndarray, y: np.ndarray, segment: pd.Series, cfg: ModelCfg) -> Calibrator:
    """Fit on validation data only, never on the test snapshot.

    `p_raw` are out-of-fold raw probabilities (or scores), `y` the training label (0 / 1, no missing value: a client
    whose label is not observed does not belong to the calibration data). With `cfg.calibration_by_segment`, one
    isotonic regression per segment; a segment with fewer than `MIN_SEGMENT_FIT` clients or a single class uses the
    global one. `cfg.calibration == "none"` returns the identity."""
    by_segment = bool(cfg.calibration_by_segment)
    if cfg.calibration == "none":
        return Calibrator("none", by_segment, {})
    if cfg.calibration != "isotonic":
        raise ValueError(f"Unknown calibration {cfg.calibration!r}; known: ['isotonic', 'none']")
    p_raw, y = np.asarray(p_raw, dtype=float), np.asarray(y, dtype=float)
    if not (len(p_raw) == len(y) == len(segment)):
        raise ValueError(f"p_raw, y and segment must have the same length, got {len(p_raw)}, {len(y)}, {len(segment)}")
    if np.isnan(p_raw).any() or np.isnan(y).any():
        raise ValueError("p_raw and y must not contain missing values: keep only the clients whose label is observed")
    if not set(np.unique(y)) <= {0.0, 1.0}:
        raise ValueError("y must be 0 / 1")
    if len(np.unique(y)) < 2:
        raise ValueError("Calibration needs both classes in y")
    models: dict[str, Any] = {ALL_SEGMENTS: _isotonic(p_raw, y)}
    if by_segment:
        seg = segment.astype(str).to_numpy()
        for value in np.unique(seg):
            mask = seg == value
            if mask.sum() >= MIN_SEGMENT_FIT and len(np.unique(y[mask])) == 2:
                models[value] = _isotonic(p_raw[mask], y[mask])
    return Calibrator("isotonic", by_segment, models)


def apply_calibrator(calibrator: Calibrator, p_raw: np.ndarray, segment: pd.Series) -> np.ndarray:
    """Return calibrated probabilities `p_cal`, in the order of `p_raw`."""
    p_raw = np.asarray(p_raw, dtype=float)
    if calibrator.method == "none":
        return p_raw.copy()
    if len(p_raw) != len(segment):
        raise ValueError(f"p_raw and segment must have the same length, got {len(p_raw)} and {len(segment)}")
    if calibrator.by_segment:
        seg = segment.astype(str).to_numpy()
        out = np.empty(len(p_raw))
        for value in np.unique(seg):
            mask = seg == value
            out[mask] = calibrator.models.get(value, calibrator.models[ALL_SEGMENTS]).predict(p_raw[mask])
        return out
    return calibrator.models[ALL_SEGMENTS].predict(p_raw)


# ------------------------------------------------------------------------------ tuning


def tune(cfg: Config, train: pd.DataFrame) -> dict[str, Any]:
    """Run `cfg.model.tuning_trials` Optuna trials with grouped-by-client K-fold on the training
    snapshots. The test snapshot is never used. Returns the best hyperparameters."""
    raise NotImplementedError("models.tune")
