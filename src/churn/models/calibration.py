"""Probability calibration. Required: the expected value multiplies P by V,
so 0.05 must really mean 5 %."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from churn.config import CalibrationCfg


@dataclass
class Calibrator:
    """Fitted calibrators, one per segment (or a single one)."""

    method: str
    by_segment: bool
    models: dict[str, Any]


def fit_calibrator(p_raw: np.ndarray, y: np.ndarray, segment: pd.Series, cfg: CalibrationCfg) -> Calibrator:
    """Fit on validation data only, never on the test snapshot."""
    raise NotImplementedError("models.calibration.fit_calibrator")


def apply_calibrator(calibrator: Calibrator, p_raw: np.ndarray, segment: pd.Series) -> np.ndarray:
    """Return calibrated probabilities `p_cal`."""
    raise NotImplementedError("models.calibration.apply_calibrator")
