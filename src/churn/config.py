"""Typed experiment configuration and project paths.

One YAML file describes one experiment. A file may start from another one with
`extends: <path>` and only override what changes. Every option that we want to
compare between runs MUST live here, otherwise two runs are not comparable.

Values marked "team hypothesis" are placeholders to confirm (tutoring, pilot).
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

# ------------------------------------------------------------------------------ paths

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"  # cached snapshots (git-ignored)
RUNS_DIR = ROOT / "runs"  # one folder per run (git-ignored)
RESULTS_DIR = ROOT / "results"  # merged tables read by the report notebook


def resolve(path: str | Path) -> Path:
    """Return an absolute path; relative paths are taken from the repository root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


# ------------------------------------------------------------------------ dataclasses


@dataclass
class DatasetCfg:
    raw_csv: str = "data/dataset_sales_2026.csv"
    shifted_parquet: str = "data/sales_shifted.parquet"
    shift_years: int = 2  # fictitious dates: +2 years, applied once by `churn ingest`


@dataclass
class SnapshotsCfg:
    lookback_days: int = 180  # features use history strictly before the snapshot date
    step: Literal["monthly", "weekly"] = "monthly"
    train_start: str = "2025-03-22"  # shifted dates
    train_end: str = "2025-09-22"
    test_date: str = "2026-03-22"  # never touched during development
    embargo_days: int = 180  # train label windows must end before the test date
    feature_set: str = "basic"


@dataclass
class LabelCfg:
    kind: Literal["class_horizon", "relative_horizon", "durable_drop"] = "class_horizon"
    # Step 1: horizon per rhythm class, in days (starting values, to calibrate on data)
    class_horizons: dict[str, int] = field(
        default_factory=lambda: {"fast": 30, "medium": 60, "slow": 120, "short_history": 180}
    )
    rhythm_bounds_days: dict[str, int] = field(default_factory=lambda: {"fast_max": 7, "medium_max": 30})
    # Step 2: relative horizon H_i = clip(k * p90_gap_i, h_min, h_max)
    k: float = 2.0
    h_min: int = 30
    h_max: int = 180
    min_purchases: int = 6  # rhythm features need at least this many purchases
    drop_threshold: float = 0.5  # durable_drop variant


@dataclass
class ModelCfg:
    type: Literal["baseline", "logreg", "hgb", "xgb"] = "hgb"
    params: dict[str, Any] = field(default_factory=dict)
    calibration: Literal["isotonic", "none"] = "isotonic"
    calibration_by_segment: bool = True
    segmentation: Literal["single", "per_segment"] = "single"
    segment_by: Literal["frequency", "value"] = "frequency"
    tuning: bool = False  # Optuna, once the best model family is chosen
    tuning_trials: int = 50
    cv_folds: int = 5  # grouped by client (an internal embargo is impossible on 6 months)


@dataclass
class TierCfg:
    name: str
    share_of_reps: float
    actions_per_week: int
    channel: str


@dataclass
class ScoringCfg:
    # V = revenue the client would generate if he stayed (NOT a forecast of realized revenue)
    value_definition: Literal["ca_180d_annualized", "ca_12m"] = "ca_180d_annualized"
    # Haircut on V: a saved at-risk client buys less than before
    rho_method: Literal["rebound_calibrated", "fixed"] = "rebound_calibrated"
    rho_fixed_value: float = 1.0
    rho_drop_threshold: float = 0.5  # clients that dropped by more than this without leaving
    min_proba: float = 0.0  # minimal calibrated probability to be listed
    n_reps: int = 200
    # Team hypothesis, to confirm at tutoring: a share of reps is dedicated to churn
    tiers: list[TierCfg] = field(
        default_factory=lambda: [
            TierCfg(name="dedicated", share_of_reps=0.2, actions_per_week=10, channel="visit_or_call"),
            TierCfg(name="general", share_of_reps=0.8, actions_per_week=5, channel="call"),
        ]
    )
    opportunity_cost_per_action: float | None = None  # team hypothesis, unknown


@dataclass
class EvaluationCfg:
    k_values: list[int] = field(default_factory=lambda: [2000, 5000, 10000])
    metrics: list[str] = field(
        default_factory=lambda: ["capture_at_k", "precision", "recall", "f1", "pr_auc", "roc_auc", "brier"]
    )
    bootstrap_n: int = 200  # clients are resampled, not rows


@dataclass
class ImpactCfg:
    retention_scenarios: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])  # team hypothesis
    lost_revenue_share: tuple[float, float] = (0.034, 0.048)  # measured on the dataset
    gross_margin: float = 0.2465
    annual_revenue_meur: float = 15504.212


@dataclass
class Config:
    name: str = "experiment"
    seed: int = 0
    dataset: DatasetCfg = field(default_factory=DatasetCfg)
    snapshots: SnapshotsCfg = field(default_factory=SnapshotsCfg)
    label: LabelCfg = field(default_factory=LabelCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    scoring: ScoringCfg = field(default_factory=ScoringCfg)
    evaluation: EvaluationCfg = field(default_factory=EvaluationCfg)
    impact: ImpactCfg = field(default_factory=ImpactCfg)


# --------------------------------------------------------------------------- loading


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(path: Path) -> dict:
    """Read a YAML file, following `extends:` (path relative to the repo root)."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = data.pop("extends", None)
    if parent:
        return _deep_merge(_read_yaml(resolve(parent)), data)
    return data


def _build(cls: type, data: dict[str, Any]) -> Any:
    """Build a dataclass from a dict, recursively; unknown keys are an error."""
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown option(s) for {cls.__name__}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        hint = hints[name]
        origin = typing.get_origin(hint)
        if dataclasses.is_dataclass(hint) and isinstance(value, dict):
            kwargs[name] = _build(hint, value)
        elif origin is list and value and dataclasses.is_dataclass(typing.get_args(hint)[0]):
            kwargs[name] = [_build(typing.get_args(hint)[0], v) for v in value]
        elif origin is tuple and isinstance(value, list):
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    """Load and validate an experiment YAML file (with `extends:` support)."""
    return _build(Config, _read_yaml(resolve(path)))


def config_to_dict(cfg: Config) -> dict[str, Any]:
    """Plain-dict view of a configuration (hashing and `runs/<id>/config.yaml`)."""
    return dataclasses.asdict(cfg)


def config_hash(cfg: Config) -> str:
    """Stable hash of the resolved configuration."""
    payload = json.dumps(config_to_dict(cfg), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_id(cfg: Config) -> str:
    """Run identity = name + short hash of the configuration. The git commit and the data
    fingerprint are recorded in `meta.json`, not in the id. A run whose folder already exists
    is skipped unless forced."""
    return f"{cfg.name}-{config_hash(cfg)[:10]}"
