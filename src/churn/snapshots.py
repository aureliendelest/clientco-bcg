"""Snapshots: one row per active client at a date T (features before T, label after T).

Contract (columns of a snapshot table):
    client_id, snapshot_date, <features>, label, horizon_days, V, segment
Features only use history strictly before `snapshot_date`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import date

import duckdb
import numpy as np
import pandas as pd

from churn.config import Config, LabelCfg

# ------------------------------------------------------------------------------ build


def snapshot_dates(cfg: Config) -> list[date]:
    """Snapshot dates between `train_start` and `train_end` (monthly or weekly), plus the test date."""
    raise NotImplementedError("snapshots.snapshot_dates")


def build_snapshot(con: duckdb.DuckDBPyConnection, snapshot_date: date, cfg: Config) -> pd.DataFrame:
    """Snapshot at one date: active clients over the last `lookback_days`, their features,
    their label and horizon, their value V and their segment."""
    raise NotImplementedError("snapshots.build_snapshot")


def cache_key(cfg: Config, data_fingerprint: str) -> str:
    """Key of the Parquet cache: data fingerprint + snapshot, label and feature configuration."""
    raise NotImplementedError("snapshots.cache_key")


def load_or_build(cfg: Config) -> pd.DataFrame:
    """All snapshots for this configuration, read from `data/processed/` when the cache key matches,
    otherwise built once and written."""
    raise NotImplementedError("snapshots.load_or_build")


# ---------------------------------------------------------------------------- features

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
    Rhythm features (gap_*, recency_ratio) are NaN below `label.min_purchases` purchases."""
    raise NotImplementedError("snapshots.compute_features")


# -------------------------------------------------------------------------------- labels
# `LabelCfg.kind`: class_horizon (step 1) | relative_horizon (step 2) | durable_drop (variant).
# Each function adds `label` (1 = lost) and `horizon_days` to a snapshot.

LabelFn = Callable[[pd.DataFrame, duckdb.DuckDBPyConnection, date, LabelCfg], pd.DataFrame]


def label_class_horizon(df: pd.DataFrame, con: duckdb.DuckDBPyConnection, snapshot_date: date, cfg: LabelCfg) -> pd.DataFrame:
    """Lost = no purchase within the horizon of the client's rhythm class."""
    raise NotImplementedError("snapshots.label_class_horizon")


def label_relative_horizon(df: pd.DataFrame, con: duckdb.DuckDBPyConnection, snapshot_date: date, cfg: LabelCfg) -> pd.DataFrame:
    """Lost = no purchase within H_i = clip(k * p90_gap_i, h_min, h_max)."""
    raise NotImplementedError("snapshots.label_relative_horizon")


def label_durable_drop(df: pd.DataFrame, con: duckdb.DuckDBPyConnection, snapshot_date: date, cfg: LabelCfg) -> pd.DataFrame:
    """Lost = revenue after T stays more than `drop_threshold` below the revenue before T."""
    raise NotImplementedError("snapshots.label_durable_drop")


LABELS: dict[str, LabelFn] = {
    "class_horizon": label_class_horizon,
    "relative_horizon": label_relative_horizon,
    "durable_drop": label_durable_drop,
}

# ------------------------------------------------------------------------------ segments


def rhythm_class(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """`fast` / `medium` / `slow` from the client's median gap, `short_history` below `min_purchases`."""
    raise NotImplementedError("snapshots.rhythm_class")


def frequency_segment(df: pd.DataFrame) -> pd.Series:
    """Segment by number of purchase days over the lookback window."""
    raise NotImplementedError("snapshots.frequency_segment")


def value_segment(df: pd.DataFrame) -> pd.Series:
    """Segment by revenue decile (key accounts live in the top deciles)."""
    raise NotImplementedError("snapshots.value_segment")


def assign_segments(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add the `segment` column according to `ModelCfg.segment_by`."""
    raise NotImplementedError("snapshots.assign_segments")


# --------------------------------------------------------------------------------- splits


def temporal_split(snapshots: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, test): train = snapshots from `train_start` to `train_end`, test = the `test_date`
    snapshot. Calls `assert_embargo` before returning."""
    raise NotImplementedError("snapshots.temporal_split")


def assert_embargo(train_dates: list[date], test_date: date, embargo_days: int) -> None:
    """Raise if the label window of any training snapshot ends after `test_date`
    (that would let the model see the future)."""
    raise NotImplementedError("snapshots.assert_embargo")


def client_group_folds(df: pd.DataFrame, n_splits: int, group_by: str = "client_id") -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Grouped K-fold: all snapshots of a client stay in the same fold."""
    raise NotImplementedError("snapshots.client_group_folds")
