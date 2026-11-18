"""Project paths. Everything is resolved from the repository root."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"  # cached snapshots (git-ignored)
RUNS_DIR = ROOT / "runs"  # one folder per run (git-ignored)
RESULTS_DIR = ROOT / "results"  # merged tables read by the report notebook
CONFIGS_DIR = ROOT / "configs"


def resolve(path: str | Path) -> Path:
    """Return an absolute path; relative paths are taken from the repository root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
