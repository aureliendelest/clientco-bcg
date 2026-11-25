"""Run orchestration: one run, a series of runs, and the merge of all runs into results/.

A run executes: load snapshots (cached in data/processed/) -> split (training labels known before the test) -> [tune] -> train ->
calibrate -> value, EV -> predictions. It writes `runs/<run_id>/`:
    config.yaml, meta.json (git sha, data fingerprint, seed, timings), predictions.parquet,
    importances.parquet, model.pkl (kept local, never committed).
Metrics are NOT computed in a run: `collect` computes them from predictions with one module.
"""

from __future__ import annotations

from pathlib import Path

from churn.config import Config

# --------------------------------------------------------------------------------- run


def run(cfg: Config, *, force: bool = False) -> Path:
    """Execute the experiment and return `runs/<run_id>/`.

    If the run folder already exists (same name and configuration hash), it is skipped unless `force`.
    """
    raise NotImplementedError("pipeline.run")


# ------------------------------------------------------------------------------- sweep
# A sweep file lists a base experiment and a grid of overrides (dot paths), e.g.
#     base: configs/experiments/hgb.yaml
#     grid:
#       model.type: [logreg, hgb, xgb]
#       label.kind: [fixed_horizon, relative_horizon]


def expand_sweep(sweep_path: str | Path) -> list[Config]:
    """Expand the grid into one validated `Config` per combination, each with a distinct `name`."""
    raise NotImplementedError("pipeline.expand_sweep")


def run_sweep(sweep_path: str | Path, *, force: bool = False) -> list[Path]:
    """Run every configuration of the sweep and return the run folders."""
    raise NotImplementedError("pipeline.run_sweep")


# ----------------------------------------------------------------------------- collect
# Tables written to `results/`, the only folder read by the report notebook:
#     runs.parquet         one row per run (flattened config, git sha, data fingerprint, timings)  [versioned]
#     metrics.parquet      long table from `evaluation.evaluate_predictions`                       [versioned]
#     predictions.parquet  all predictions                                                          [git-ignored]
#     importances.parquet  feature importances                                                      [git-ignored]


def collect() -> dict[str, Path]:
    """Scan `runs/`, recompute metrics from each run's predictions, write the four tables.
    Returns table name -> path."""
    raise NotImplementedError("pipeline.collect")
