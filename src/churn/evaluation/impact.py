"""Impact simulation: revenue x lost share x detection x retention x rho x margin.

Detection comes from the backtest, the lost share is measured on the dataset, retention is a team
hypothesis (scenarios) until the pilot measures it. The net impact subtracts the opportunity cost of
dedicated reps.
"""

from __future__ import annotations

import pandas as pd

from churn.config import Config


def impact_scenarios(metrics: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Low / central / high scenarios (retention 10 / 20 / 30 %), in M EUR of gross margin per year."""
    raise NotImplementedError("evaluation.impact.impact_scenarios")


def tornado_table(cfg: Config, ranges: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """One row per parameter (lost share, detection, retention, rho): impact at its low and high value,
    others at the central case, sorted by amplitude."""
    raise NotImplementedError("evaluation.impact.tornado_table")
