"""Model registry: `ModelCfg.type` -> scikit-learn-compatible estimator."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from churn.config import ModelCfg


def build_baseline(params: dict[str, Any], seed: int) -> Any:
    """Rule baseline: relative recency (recency / client's p90 gap). The score to beat."""
    raise NotImplementedError("models.registry.build_baseline")


def build_logreg(params: dict[str, Any], seed: int) -> Any:
    """Logistic regression on imputed, scaled features (readable coefficients)."""
    raise NotImplementedError("models.registry.build_logreg")


def build_hgb(params: dict[str, Any], seed: int) -> Any:
    """scikit-learn HistGradientBoostingClassifier."""
    raise NotImplementedError("models.registry.build_hgb")


def build_xgb(params: dict[str, Any], seed: int) -> Any:
    """XGBoost classifier (tree method `hist`)."""
    raise NotImplementedError("models.registry.build_xgb")


def build_mlp(params: dict[str, Any], seed: int) -> Any:
    """Optional, time-boxed: small multilayer perceptron. Documented as 'tested, not retained' if no gain."""
    raise NotImplementedError("models.registry.build_mlp (optional)")


MODEL_REGISTRY: dict[str, Callable[[dict[str, Any], int], Any]] = {
    "baseline": build_baseline,
    "logreg": build_logreg,
    "hgb": build_hgb,
    "xgb": build_xgb,
    "mlp": build_mlp,
}


def get_model(cfg: ModelCfg, seed: int) -> Any:
    """Build the estimator selected by the configuration."""
    raise NotImplementedError("models.registry.get_model")
