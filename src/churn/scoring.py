"""From probabilities to an action list: value at risk, expected value, list size, reasons.

V is the revenue the client would generate IF HE STAYED, estimated on his recent rhythm. It is not
a forecast of realized revenue: that would be ~0 for clients who leave and would count the
departure twice (once in P, once in V). EV = P x V, with the same retention rate for
everybody until the pilot says otherwise. (A haircut rho for clients who already declined was set aside
on 25/11; it may come back later.)
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from churn.config import Config

# ----------------------------------------------------------------------------- value


def compute_value(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """V per client according to `ScoringCfg.value_definition` (default: last 180 days, annualized)."""
    raise NotImplementedError("scoring.compute_value")


# -------------------------------------------------------------------- expected value


def expected_value(p_cal: pd.Series, value: pd.Series) -> pd.Series:
    """EV = p_cal * V."""
    raise NotImplementedError("scoring.expected_value")


def apply_min_proba(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Drop clients whose calibrated probability is below `ScoringCfg.min_proba`."""
    raise NotImplementedError("scoring.apply_min_proba")


# ------------------------------------------------------------------------- list size
# Capacity hypothesis: `calls_per_week` company-wide (500), converted to the dataset scale
# (`dataset_share` ~ 30 % of revenue): K ~ 700 for a monthly list, ~ 4 000 for six lists cumulated.
# The list is shared between the two scores: `total_share_of_list` (25 %) of the places go to the total-churn
# score, the rest to the partial score.


def list_size(cfg: Config) -> int:
    """K of a monthly list at the dataset scale (`ScoringCfg.list_size_k`, ~ calls_per_week x 4.3 x dataset_share)."""
    raise NotImplementedError("scoring.list_size")


def mixed_list(total_ranked: pd.DataFrame, partial_ranked: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Merge the two rankings (both sorted by EV): `total_share_of_list` x K places to the total-churn
    score, the rest to the partial score, without duplicate clients. Adds the `score_kind` column."""
    raise NotImplementedError("scoring.mixed_list")


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
    rank by EV, apply the minimal probability and the mixed list, add the three reasons, write a CSV.

    Output columns: rank, client_id, p_cal, V, ev, segment, score_kind, suggested_channel,
    reason_1, reason_2, reason_3. Returns the CSV path.
    """
    raise NotImplementedError("scoring.score_batch")
