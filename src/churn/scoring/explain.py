"""Per-client reasons (top 3) shown next to each client in the action list.

SHAP is computed only for the clients of the list, never for all rows.
Display texts are French (deliverable) and live in `reasons_fr.yaml`; keys are English feature names.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def load_reasons() -> dict[str, str]:
    """Read `reasons_fr.yaml` (feature name -> French sentence template)."""
    raise NotImplementedError("scoring.explain.load_reasons")


def top_reasons(model: Any, X: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """SHAP values (tree models) or coefficients (logistic regression) -> `n` reasons per client,
    as French sentences built from `load_reasons()`."""
    raise NotImplementedError("scoring.explain.top_reasons")
