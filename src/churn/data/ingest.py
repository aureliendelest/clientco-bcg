"""One-shot ingestion: raw CSV -> shifted Parquet.

The raw CSV is never modified. The shifted file is the only data source used by
the rest of the project, so no other module applies any date shift.
"""

from __future__ import annotations

from pathlib import Path

from churn.config import Config


def ingest(cfg: Config, *, overwrite: bool = False) -> Path:
    """Read `cfg.dataset.raw_csv`, add `shift_years` years to `date_order` and `date_invoice`,
    and write `cfg.dataset.shifted_parquet` (zstd) plus a small note recording the shift.

    Fails if the output already exists and `overwrite` is False (prevents a double shift).
    Returns the path of the written Parquet file.
    """
    raise NotImplementedError("data.ingest.ingest")


def fingerprint(path: str | Path) -> str:
    """Cheap, stable fingerprint of a data file (size + hash of a sample of its content).

    Stored in every run so that two runs are only comparable on the same data.
    """
    raise NotImplementedError("data.ingest.fingerprint")
