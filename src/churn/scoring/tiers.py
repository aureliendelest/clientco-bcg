"""Capacity tiers: a share of the sales force is dedicated to churn prevention and handles the top
of the ranking; the rest of the list is handled by calls. K is the sum over tiers."""

from __future__ import annotations

import pandas as pd

from churn.config import Config


def capacity_k(cfg: Config, scenario: dict[str, float] | None = None) -> dict[str, int]:
    """Number of clients each tier can handle per week:
    n_reps * share_of_reps * actions_per_week. `scenario` overrides the shares or actions."""
    raise NotImplementedError("scoring.tiers.capacity_k")


def allocate_tiers(ranked: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add the `tier` column to a ranking sorted by EV (top clients go to the dedicated tier)."""
    raise NotImplementedError("scoring.tiers.allocate_tiers")
