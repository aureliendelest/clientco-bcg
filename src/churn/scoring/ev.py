"""Expected value EV = P x rho x V (same retention rate for everybody until the pilot says otherwise)."""

from __future__ import annotations

import pandas as pd

from churn.config import Config


def expected_value(p_cal: pd.Series, rho: pd.Series, value: pd.Series) -> pd.Series:
    """EV = p_cal * rho * V."""
    raise NotImplementedError("scoring.ev.expected_value")


def apply_min_proba(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Drop clients whose calibrated probability is below `ScoringCfg.min_proba`."""
    raise NotImplementedError("scoring.ev.apply_min_proba")
