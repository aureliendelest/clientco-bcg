"""Batch scoring: the simulated 'production' step. Writes the action list as CSV (no API)."""

from __future__ import annotations

from datetime import date
from pathlib import Path


def score_batch(run_id: str, snapshot_date: date) -> Path:
    """Load the model of `runs/<run_id>/`, score the snapshot at `snapshot_date`, rank by EV,
    apply the minimal probability and the capacity tiers, add the three reasons.

    Output columns: rank, client_id, p_cal, V, rho, ev, segment, tier, suggested_channel,
    reason_1, reason_2, reason_3. Returns the CSV path.
    """
    raise NotImplementedError("scoring.batch.score_batch")
