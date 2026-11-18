"""Feature sets. Features at date T use only data strictly before T."""

from __future__ import annotations

from datetime import date

import duckdb
import pandas as pd

FEATURE_SETS: dict[str, list[str]] = {
    "basic": [
        "recency_days",
        "freq_180d",
        "freq_90d",
        "freq_prev_90d",
        "rev_180d",
        "rev_90d",
        "rev_prev_90d",
        "gap_median",
        "gap_p90",
        "recency_ratio",  # recency / gap_p90, only with enough purchase history
        "freq_trend",  # freq_90d / (freq_prev_90d + 1)
        "rev_trend",
    ],
    # Planned extensions (not implemented): returns share, channel mix, number of branches,
    # basket diversity, month indicator. Never a day-of-week feature (dates are shifted).
    "extended": [],
}


def compute_features(con: duckdb.DuckDBPyConnection, snapshot_date: date, feature_set: str) -> pd.DataFrame:
    """One row per client active before `snapshot_date`, with the columns of `FEATURE_SETS[feature_set]`.

    Rhythm features (gap_*, recency_ratio) are NaN below `label.min_purchases` purchases.
    """
    raise NotImplementedError("snapshots.features.compute_features")
