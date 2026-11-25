"""Hand-made sales for the snapshot tests: named clients with a purchase every N days."""

from __future__ import annotations

import pandas as pd

T = pd.Timestamp("2025-04-15").date()  # the snapshot date of most tests


def every(start: str, end: str, days: int) -> list[str]:
    """Purchase dates from `start` to `end` (included), one every `days` days."""
    return [f"{d:%Y-%m-%d}" for d in pd.date_range(start, end, freq=f"{days}D")]


def before(n: int, step: int) -> list[str]:
    """`n` purchase dates, one every `step` days, the last one the day before the snapshot date T."""
    return [f"{T - pd.Timedelta(days=1 + step * i):%Y-%m-%d}" for i in reversed(range(n))]


class _Sales:
    """`sales_for({"name": [dates]})` -> {client_id: [dates]}; `sales_for.ids["name"]` gives the numeric id back."""

    ids: dict[str, int] = {}

    def __call__(self, named: dict[str, list[str]]) -> dict[int, list[str]]:
        self.ids = {name: 100 + i for i, name in enumerate(named)}
        return {self.ids[name]: dates for name, dates in named.items()}


sales_for = _Sales()
