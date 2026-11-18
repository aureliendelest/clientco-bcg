"""A sweep file lists a base experiment and a grid of overrides (dot paths), e.g.

    base: configs/experiments/hgb.yaml
    grid:
      model.type: [logreg, hgb, xgb]
      label.kind: [class_horizon, relative_horizon]
"""

from __future__ import annotations

from pathlib import Path

from churn.config import Config


def expand_sweep(sweep_path: str | Path) -> list[Config]:
    """Expand the grid into one validated `Config` per combination, each with a distinct `name`."""
    raise NotImplementedError("pipeline.sweep.expand_sweep")


def run_sweep(sweep_path: str | Path) -> list[Path]:
    """Run every configuration of the sweep and return the run folders."""
    raise NotImplementedError("pipeline.sweep.run_sweep")
