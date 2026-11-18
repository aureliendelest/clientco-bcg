"""Value at risk V and haircut rho.

V is the revenue the client would generate IF HE STAYED, estimated on his recent rhythm.
It is not a forecast of realized revenue: that would be ~0 for clients who leave and would
count the departure twice (once in P, once in V).
"""

from __future__ import annotations

import pandas as pd

from churn.config import Config


def compute_value(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """V per client according to `ScoringCfg.value.definition` (default: last 180 days, annualized)."""
    raise NotImplementedError("scoring.value.compute_value")


def fit_rho(train: pd.DataFrame, cfg: Config) -> dict[str, float]:
    """Haircut rho per segment: share of the previous revenue level that clients recover after
    a drop of more than `drop_threshold` without leaving. Observational, label it 'Deduced'.

    Fitted on training data only.
    """
    raise NotImplementedError("scoring.value.fit_rho")
