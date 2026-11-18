"""Hyperparameter search (Optuna) once the best model family is chosen."""

from __future__ import annotations

from typing import Any

import pandas as pd

from churn.config import Config


def tune(cfg: Config, train: pd.DataFrame) -> dict[str, Any]:
    """Run `cfg.tuning.n_trials` Optuna trials with grouped-by-client K-fold on the training snapshots.

    The test snapshot is never used. Every trial is recorded so it can be collected like a run.
    Returns the best hyperparameters.
    """
    raise NotImplementedError("models.tune.tune")
