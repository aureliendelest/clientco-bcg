"""Client segments used for evaluation, calibration and (optionally) per-segment models."""

from __future__ import annotations

import pandas as pd

from churn.config import Config


def rhythm_class(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """`fast` / `medium` / `slow` from the client's median gap, `short_history` below `min_purchases`."""
    raise NotImplementedError("snapshots.segments.rhythm_class")


def frequency_segment(df: pd.DataFrame) -> pd.Series:
    """Segment by number of purchase days over the lookback window."""
    raise NotImplementedError("snapshots.segments.frequency_segment")


def value_segment(df: pd.DataFrame) -> pd.Series:
    """Segment by revenue decile (key accounts live in the top deciles)."""
    raise NotImplementedError("snapshots.segments.value_segment")


def assign_segments(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add the `segment` column according to `SegmentationCfg.by`."""
    raise NotImplementedError("snapshots.segments.assign_segments")
