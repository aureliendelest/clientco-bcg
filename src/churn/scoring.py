"""From probabilities to an action list: value at risk, expected value, capacity tiers, reasons.

V is the revenue the client would generate IF HE STAYED, estimated on his recent rhythm. It is not
a forecast of realized revenue: that would be ~0 for clients who leave and would count the
departure twice (once in P, once in V). EV = P x rho x V, with the same retention rate for
everybody until the pilot says otherwise.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from churn.config import Config

# ------------------------------------------------------------------- value and haircut


def compute_value(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """V per client according to `ScoringCfg.value_definition` (default: last 180 days, annualized)."""
    raise NotImplementedError("scoring.compute_value")


def fit_rho(train: pd.DataFrame, cfg: Config) -> dict[str, float]:
    """Haircut rho per segment: share of the previous revenue level that clients recover after a
    drop of more than `rho_drop_threshold` without leaving. Observational: label it 'Deduced'.
    Fitted on training data only."""
    raise NotImplementedError("scoring.fit_rho")


# -------------------------------------------------------------------- expected value


def expected_value(p_cal: pd.Series, rho: pd.Series, value: pd.Series) -> pd.Series:
    """EV = p_cal * rho * V."""
    raise NotImplementedError("scoring.expected_value")


def apply_min_proba(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Drop clients whose calibrated probability is below `ScoringCfg.min_proba`."""
    raise NotImplementedError("scoring.apply_min_proba")


# ------------------------------------------------------------------------------ tiers
# A share of the sales force is dedicated to churn prevention and handles the top of the ranking;
# the rest of the list is handled by calls. K is the sum over tiers.


def capacity_k(cfg: Config, scenario: dict[str, float] | None = None) -> dict[str, int]:
    """Clients each tier can handle per week: n_reps * share_of_reps * actions_per_week.
    `scenario` overrides the shares or the actions."""
    raise NotImplementedError("scoring.capacity_k")


def allocate_tiers(ranked: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add the `tier` column to a ranking sorted by EV (top clients go to the dedicated tier)."""
    raise NotImplementedError("scoring.allocate_tiers")


# ---------------------------------------------------------------------------- reasons
# Display texts are French (deliverable) and live in `reasons_fr.yaml`; keys are English feature names.
# SHAP is computed only for the clients of the list, never for all rows.


def load_reasons() -> dict[str, str]:
    """Read `reasons_fr.yaml` (feature name -> French sentence template)."""
    raise NotImplementedError("scoring.load_reasons")


def top_reasons(model: Any, X: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """SHAP values (tree models) or coefficients (logistic regression) -> `n` reasons per client,
    as French sentences built from `load_reasons()`."""
    raise NotImplementedError("scoring.top_reasons")


# ------------------------------------------------------------------------ batch list


def score_batch(run_id: str, snapshot_date: date) -> Path:
    """Simulated 'production' step (no API): load the model of `runs/<run_id>/`, score the snapshot,
    rank by EV, apply the minimal probability and the tiers, add the three reasons, write a CSV.

    Output columns: rank, client_id, p_cal, V, rho, ev, segment, tier, suggested_channel,
    reason_1, reason_2, reason_3. Returns the CSV path.
    """
    raise NotImplementedError("scoring.score_batch")
