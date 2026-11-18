"""Rule baseline used as the score to beat."""

from __future__ import annotations

import numpy as np
import pandas as pd


class RelativeRecencyBaseline:
    """Score = recency / p90 gap of the client (higher = more at risk).

    Follows the scikit-learn estimator API so that it goes through the same pipeline as the models.
    The raw score is monotone with risk; calibration turns it into a probability.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RelativeRecencyBaseline:
        raise NotImplementedError("models.baseline.RelativeRecencyBaseline.fit")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return an (n, 2) array, like scikit-learn classifiers."""
        raise NotImplementedError("models.baseline.RelativeRecencyBaseline.predict_proba")
