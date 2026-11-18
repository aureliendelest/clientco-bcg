"""Build and cache snapshot tables.

Contract (columns of a snapshot table):
    client_id, snapshot_date, <features>, label, horizon_days, V, segment
Features only use history strictly before `snapshot_date`.
"""

from __future__ import annotations

from datetime import date

import duckdb
import pandas as pd

from churn.config import Config


def snapshot_dates(cfg: Config) -> list[date]:
    """Snapshot dates between `train_start` and `train_end` (monthly or weekly), plus the test date."""
    raise NotImplementedError("snapshots.build.snapshot_dates")


def build_snapshot(con: duckdb.DuckDBPyConnection, snapshot_date: date, cfg: Config) -> pd.DataFrame:
    """Snapshot at one date: active clients over the last `lookback_days`, their features,
    their label and horizon, their value V and their segment."""
    raise NotImplementedError("snapshots.build.build_snapshot")


def cache_key(cfg: Config, data_fingerprint: str) -> str:
    """Key of the Parquet cache: data fingerprint + snapshot, label and feature configuration."""
    raise NotImplementedError("snapshots.build.cache_key")


def load_or_build(cfg: Config) -> pd.DataFrame:
    """All snapshots for this configuration, read from `data/processed/` when the cache key matches,
    otherwise built once and written."""
    raise NotImplementedError("snapshots.build.load_or_build")
