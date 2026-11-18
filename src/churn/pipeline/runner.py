"""One experiment = one run.

Stages (called in order, snapshots cached in data/processed/):
    load snapshots -> split (embargo) -> [tune] -> train -> calibrate -> value, rho, EV -> predictions

A run writes `runs/<run_id>/`:
    config.yaml, meta.json (git sha, data fingerprint, seed, timings), predictions.parquet,
    importances.parquet, model.pkl (kept local, never committed).
Metrics are NOT computed here: `collect` computes them from predictions with a single module.
"""

from __future__ import annotations

from pathlib import Path

from churn.config import Config


def run(cfg: Config) -> Path:
    """Execute the experiment and return `runs/<run_id>/`.

    If a run with the same id (config + git sha + data fingerprint) already exists, it is not rerun.
    """
    raise NotImplementedError("pipeline.runner.run")
