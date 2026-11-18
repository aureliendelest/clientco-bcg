"""DuckDB access to the shifted Parquet (heavy computation stays in SQL)."""

from __future__ import annotations

from pathlib import Path

import duckdb


def connect(parquet: str | Path | None = None) -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB connection and expose the sales Parquet as the view `sales`.

    Defaults to `DatasetCfg.shifted_parquet`.
    """
    raise NotImplementedError("data.store.connect")


def build_client_day_table(con: duckdb.DuckDBPyConnection) -> None:
    """Create the tables `cd` (one row per client and order date) and `bd` (purchase days only,
    i.e. positive sales). A purchase day is a (client_id, date_order) with sales_net > 0.

    Reference implementation: cell 16 of notebooks/churn_exploration_viz.ipynb.
    """
    raise NotImplementedError("data.store.build_client_day_table")
