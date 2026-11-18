"""Merge all runs into `results/`, the only folder read by the report notebook.

Tables written to `results/`:
    runs.parquet         one row per run (flattened config, git sha, data fingerprint, timings)  [versioned]
    metrics.parquet      long table from `evaluation.metrics.evaluate_predictions`               [versioned]
    predictions.parquet  all predictions                                                          [git-ignored]
    importances.parquet  feature importances                                                      [git-ignored]
"""

from __future__ import annotations

from pathlib import Path


def collect() -> dict[str, Path]:
    """Scan `runs/`, recompute metrics from each run's predictions, write the four tables.
    Returns table name -> path."""
    raise NotImplementedError("pipeline.collect.collect")
