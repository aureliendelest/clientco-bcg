"""Temporal split with embargo, and grouped folds for tuning."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date

import numpy as np
import pandas as pd

from churn.config import Config


def temporal_split(snapshots: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, test): train = snapshots from `train_start` to `train_end`, test = the `test_date` snapshot.

    Calls `assert_embargo` before returning.
    """
    raise NotImplementedError("snapshots.splits.temporal_split")


def assert_embargo(train_dates: list[date], test_date: date, embargo_days: int) -> None:
    """Raise if the label window of any training snapshot ends after `test_date`
    (that would let the model see the future)."""
    raise NotImplementedError("snapshots.splits.assert_embargo")


def client_group_folds(df: pd.DataFrame, n_splits: int, group_by: str = "client_id") -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Grouped K-fold: all snapshots of a client stay in the same fold."""
    raise NotImplementedError("snapshots.splits.client_group_folds")
