"""Raw data ingestion and DuckDB access.

The raw CSV is never modified. The shifted Parquet is the only data source used by the rest of
the project, so no other module applies any date shift. Heavy computation stays in DuckDB.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from churn.config import Config


def ingest(cfg: Config, *, overwrite: bool = False) -> Path:
    """Read `cfg.dataset.raw_csv`, add `shift_years` years to `date_order` and `date_invoice`,
    and write `cfg.dataset.shifted_parquet` (zstd) plus a small note recording the shift.

    Fails if the output already exists and `overwrite` is False (prevents a double shift).
    Returns the path of the written Parquet file.
    """
    raise NotImplementedError("data.ingest")


def fingerprint(path: str | Path) -> str:
    """Cheap, stable fingerprint of a data file (size + hash of a sample of its content).
    Recorded in `meta.json` of every run."""
    raise NotImplementedError("data.fingerprint")


def connect(parquet: str | Path | None = None) -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB connection and expose the sales Parquet as the view `sales`.
    Defaults to `DatasetCfg.shifted_parquet`."""
    raise NotImplementedError("data.connect")


def build_client_day_table(con: duckdb.DuckDBPyConnection) -> None:
    """Create the tables `cd` (one row per client and order date, with the net revenue of the day) and `bn`
    (purchase days only). A purchase day is a (client_id, date_order) whose net revenue (sales minus the
    returns of the day) is > 0.01 EUR: a purchase returned the same day is not a purchase.

    Reference implementation: the "Table (client, jour d'achat)" cells of
    notebooks/01_data_quality_and_activity_overview.ipynb and notebooks/04_churn_scope_total_vs_partial.ipynb.
    """
    raise NotImplementedError("data.build_client_day_table")
