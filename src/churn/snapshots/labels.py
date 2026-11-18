"""Label definitions. Each function adds `label` (1 = lost) and `horizon_days` to a snapshot.

The definition of "lost" is a business choice, selected by `LabelCfg.kind`:
    class_horizon     step 1: horizon per rhythm class
    relative_horizon  step 2: H_i = clip(k * p90_gap_i, h_min, h_max)
    durable_drop      variant: durable revenue drop of more than `drop_threshold`
    survival          step 3 (optional): discrete-time survival model, not planned yet
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

import duckdb
import pandas as pd

from churn.config import LabelCfg

LabelFn = Callable[[pd.DataFrame, duckdb.DuckDBPyConnection, date, LabelCfg], pd.DataFrame]


def label_class_horizon(df: pd.DataFrame, con: duckdb.DuckDBPyConnection, snapshot_date: date, cfg: LabelCfg) -> pd.DataFrame:
    """Lost = no purchase within the horizon of the client's rhythm class."""
    raise NotImplementedError("snapshots.labels.label_class_horizon")


def label_relative_horizon(df: pd.DataFrame, con: duckdb.DuckDBPyConnection, snapshot_date: date, cfg: LabelCfg) -> pd.DataFrame:
    """Lost = no purchase within H_i = clip(k * p90_gap_i, h_min, h_max)."""
    raise NotImplementedError("snapshots.labels.label_relative_horizon")


def label_durable_drop(df: pd.DataFrame, con: duckdb.DuckDBPyConnection, snapshot_date: date, cfg: LabelCfg) -> pd.DataFrame:
    """Lost = revenue after T stays more than `drop_threshold` below the revenue before T."""
    raise NotImplementedError("snapshots.labels.label_durable_drop")


def label_survival(df: pd.DataFrame, con: duckdb.DuckDBPyConnection, snapshot_date: date, cfg: LabelCfg) -> pd.DataFrame:
    """Optional step 3: kept in the registry on purpose, not planned."""
    raise NotImplementedError("snapshots.labels.label_survival (optional, not planned)")


LABELS: dict[str, LabelFn] = {
    "class_horizon": label_class_horizon,
    "relative_horizon": label_relative_horizon,
    "durable_drop": label_durable_drop,
    "survival": label_survival,
}
