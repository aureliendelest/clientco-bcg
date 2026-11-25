"""Run orchestration: one run, a series of runs, and the merge of all runs into results/.

A run executes: load snapshots (cached in data/processed/) -> split (training labels known before the test) -> [tune] -> train ->
calibrate -> value, EV -> predictions. It writes `runs/<run_id>/`:
    config.yaml, meta.json (git sha, data fingerprint, seed, timings), predictions.parquet,
    importances.parquet, model.pkl (kept local, never committed).
Metrics are NOT computed in a run: `collect` computes them from predictions with one module.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import os
import pickle
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import xgboost
import yaml
from collections.abc import Callable
from threadpoolctl import threadpool_limits

from churn import models
from churn.config import (
    RESULTS_DIR, ROOT, RUNS_DIR, Config, DatasetCfg, config_hash, config_to_dict, load_config, override_config, resolve,
)
from churn.config import run_id as make_run_id
from churn.data import fingerprint
from churn.evaluation import evaluate_predictions, load_universe
from churn.scoring import expected_value
from churn.snapshots import (
    FEATURE_SETS, SNAPSHOT_VERSION, cache_key, client_group_folds, in_perimeter, load_or_build, temporal_split,
)

FINAL_TEST_DATE = "2026-03-22"  # the test snapshot: run once, after the configuration is frozen
FINAL_TEST_ENV = "CHURN_FINAL_TEST"


def _assert_test_allowed(cfg: Config) -> None:
    """Refuse a run on the real dataset whose test date is the final test date, unless `CHURN_FINAL_TEST=1`.

    Docs/02_modeling_decisions.md, section 10: the test is never touched during development. A synthetic
    dataset (another Parquet file) is not concerned: the tests of the package run on it."""
    real = resolve(DatasetCfg().shifted_parquet)
    if cfg.snapshots.test_date == FINAL_TEST_DATE and resolve(cfg.dataset.shifted_parquet) == real:
        if os.environ.get(FINAL_TEST_ENV) != "1":
            raise PermissionError(
                f"Refusing to run on the final test date {FINAL_TEST_DATE} with the real dataset: "
                f"set {FINAL_TEST_ENV}=1 once the configuration is frozen"
            )


PREDICTION_COLUMNS = [
    "run_id", "client_id", "snapshot_date", "split", "p_raw", "p_cal", "V", "ev", "label", "gone", "horizon_days", "segment",
]

# ------------------------------------------------------------------------------ training


@dataclass
class Fitted:
    """What `fit_predict` returns: the fitted estimator and calibrator, and the predictions of every target."""

    model: Any
    calibrator: models.Calibrator
    features: list[str]
    params: dict[str, Any]
    predictions: pd.DataFrame  # PREDICTION_COLUMNS without `run_id`
    importances: pd.DataFrame
    timings: dict[str, float]


def _perimeter(df: pd.DataFrame) -> pd.DataFrame:
    """The regular clients, sorted by (client, date): the row order makes the training reproducible."""
    out = df.loc[in_perimeter(df)].sort_values(["client_id", "snapshot_date"], kind="stable")
    return out.reset_index(drop=True)


def fit_predict(cfg: Config, train: pd.DataFrame, targets: dict[str, pd.DataFrame]) -> Fitted:
    """THE training function (docs/02_modeling_decisions.md, section 10): tune if asked, grouped CV on the training
    snapshots, calibrator fitted on the out-of-fold probabilities, refit on all the training rows, then prediction
    of every target. The rolling backtest and the final model both go through it, so the backtest evaluates exactly
    the deployed procedure. `targets` maps a split name to a snapshot table; the target rows never take part in
    tuning, CV, calibration or training."""
    if cfg.model.segmentation != "single":
        raise NotImplementedError(f"pipeline: model.segmentation={cfg.model.segmentation!r} (only 'single' is implemented)")
    features = FEATURE_SETS[cfg.snapshots.feature_set]
    train = _perimeter(train)
    if train.empty or train["label"].isna().any():
        raise ValueError("The training rows are empty or have no label: the label window is not observed")
    X, y = train[features], train["label"].astype(int)
    timings: dict[str, float] = {}

    with threadpool_limits(limits=1):  # one thread: the same run twice gives the same predictions
        t = time.time()
        params = dict(cfg.model.params)
        if cfg.model.tuning:
            params |= models.tune(cfg, train)
        model_cfg = dataclasses.replace(cfg.model, params=params)
        timings["tune_s"] = time.time() - t

        t = time.time()
        if model_cfg.calibration == "none":
            calibrator = models.fit_calibrator(np.empty(0), np.empty(0), pd.Series(dtype=str), model_cfg)
        else:
            oof = np.full(len(train), np.nan)
            for fit_rows, val_rows in client_group_folds(train, model_cfg.cv_folds, seed=cfg.seed):
                fold = models.get_model(model_cfg, cfg.seed).fit(X.iloc[fit_rows], y.iloc[fit_rows])
                oof[val_rows] = fold.predict_proba(X.iloc[val_rows])[:, 1]
            calibrator = models.fit_calibrator(oof, y.to_numpy(), train["segment"], model_cfg)
        timings["cv_calibration_s"] = time.time() - t

        t = time.time()
        model = models.get_model(model_cfg, cfg.seed).fit(X, y)
        timings["fit_s"] = time.time() - t

        t = time.time()
        frames = []
        for split, snap in targets.items():
            df = _perimeter(snap)
            p_raw = model.predict_proba(df[features])[:, 1]
            p_cal = models.apply_calibrator(calibrator, p_raw, df["segment"])
            out = df[["client_id", "snapshot_date", "label", "gone", "horizon_days", "V", "segment"]].copy()
            out["split"], out["p_raw"], out["p_cal"] = split, p_raw, p_cal
            out["ev"] = expected_value(out["p_cal"], out["V"]).to_numpy()
            frames.append(out)
        timings["predict_s"] = time.time() - t
        importances = models.feature_importances(model, X, y, cfg.seed)

    predictions = pd.concat(frames, ignore_index=True)[[c for c in PREDICTION_COLUMNS if c != "run_id"]]
    return Fitted(model, calibrator, features, params, predictions, importances, timings)


# --------------------------------------------------------------------------------- run


def _git_state() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git("status", "--porcelain", "--untracked-files=no")
    return {"git_sha": git("rev-parse", "HEAD"), "git_dirty": None if status is None else bool(status)}


def run(cfg: Config, *, force: bool = False, runs_dir: Path | None = None) -> Path:
    """Execute the experiment and return `runs/<run_id>/`.

    If the run folder already exists (same name and configuration hash), it is skipped unless `force`.
    The run scores the snapshot of `snapshots.test_date` (`split = "test"`) after training on the snapshots up to
    `Config.last_train_date()`. A rolling-backtest date is a run whose `test_date` is that date. The folder is
    written under a temporary name and moved in place at the end, so a failed run leaves nothing behind.
    `runs_dir` replaces the default `runs/` (used by the tests).
    """
    _assert_test_allowed(cfg)
    started = time.time()
    rid = make_run_id(cfg)
    root = Path(runs_dir) if runs_dir is not None else RUNS_DIR
    out = root / rid
    if out.exists() and not force:
        return out

    snapshots = load_or_build(cfg)
    train, test = temporal_split(snapshots, cfg)
    snapshots_s = time.time() - started
    fitted = fit_predict(cfg, train, {"test": test})
    predictions = fitted.predictions.assign(run_id=rid)[PREDICTION_COLUMNS]

    tmp = root / f".{rid}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        (tmp / "config.yaml").write_text(yaml.safe_dump(config_to_dict(cfg), sort_keys=False), encoding="utf-8")
        predictions.to_parquet(tmp / "predictions.parquet", index=False)
        fitted.importances.assign(run_id=rid).to_parquet(tmp / "importances.parquet", index=False)
        with (tmp / "model.pkl").open("wb") as f:
            pickle.dump({"model": fitted.model, "calibrator": fitted.calibrator, "features": fitted.features,
                         "config": config_to_dict(cfg), "snapshot_version": SNAPSHOT_VERSION}, f)
        in_train, in_test = _perimeter(train), _perimeter(test)
        meta = {
            "run_id": rid, "name": cfg.name, "config_hash": config_hash(cfg), "seed": cfg.seed,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **_git_state(),
            "data_fingerprint": fingerprint(resolve(cfg.dataset.shifted_parquet)),
            "train_dates": sorted({f"{d:%Y-%m-%d}" for d in pd.to_datetime(train["snapshot_date"])}),
            "test_date": cfg.snapshots.test_date,
            "n_train_rows": len(in_train), "n_train_clients": int(in_train["client_id"].nunique()),
            "train_positive_rate": float(in_train["label"].mean()), "n_test_rows": len(in_test),
            "test_positive_rate": float(in_test["gone"].mean()) if in_test["gone"].notna().any() else None,
            "params": fitted.params,
            "timings_s": {"snapshots_s": snapshots_s, **fitted.timings, "total_s": time.time() - started},
            "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                         "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__},
        }
        (tmp / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")
        if out.exists():
            shutil.rmtree(out)  # only reached with force=True
        tmp.rename(out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return out


# ------------------------------------------------------------------------------- sweep
# A sweep file lists a base experiment and a grid of overrides (dot paths), e.g.
#     base: configs/experiments/hgb.yaml
#     grid:
#       model.type: [logreg, hgb, xgb]
#       label.kind: [fixed_horizon, relative_horizon]


def expand_sweep(sweep_path: str | Path) -> list[Config]:
    """Expand the grid into one validated `Config` per combination, each with a distinct `name`.

    The name is `<sweep file name>__<path>-<value>__...`, in the order of the grid (the first key varies slowest),
    so a run folder says what changed. A grid key is a dot path in the configuration; its value is a non-empty
    list of distinct values. `name` cannot be swept. An empty grid gives the base experiment alone."""
    path = resolve(sweep_path)
    spec = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    unknown = set(spec) - {"base", "grid"}
    if unknown:
        raise ValueError(f"{path.name}: unknown key(s) {sorted(unknown)}; a sweep has `base` and `grid`")
    if "base" not in spec:
        raise ValueError(f"{path.name}: missing `base` (path of the base experiment)")
    grid = spec.get("grid") or {}
    for key, values in grid.items():
        if key == "name":
            raise ValueError(f"{path.name}: `name` is set by the sweep and cannot be in the grid")
        if not isinstance(values, list) or not values:
            raise ValueError(f"{path.name}: grid[{key!r}] must be a non-empty list")
        if len({json.dumps(v, sort_keys=True, default=str) for v in values}) != len(values):
            raise ValueError(f"{path.name}: grid[{key!r}] has duplicate values")

    base = load_config(spec["base"])
    configs = []
    for combo in itertools.product(*grid.values()):
        # YAML reads an unquoted 2025-09-22 as a date object; the configuration holds ISO strings
        overrides = {k: v.isoformat() if isinstance(v, date) else v for k, v in zip(grid, combo)}
        label = "__".join(f"{key}-{value}" for key, value in overrides.items())
        name = "__".join([path.stem, label]) if label else path.stem
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
        configs.append(override_config(base, overrides | {"name": safe}))
    return configs


def run_sweep(sweep_path: str | Path, *, force: bool = False, runs_dir: Path | None = None) -> list[Path]:
    """Run every configuration of the sweep and return the run folders (in the order of `expand_sweep`).

    A run whose folder already exists is skipped unless `force`. The first failure stops the sweep: the runs
    already written are kept, so a second call resumes where the first stopped."""
    configs = expand_sweep(sweep_path)
    root = Path(runs_dir) if runs_dir is not None else RUNS_DIR
    folders = []
    for i, cfg in enumerate(configs, 1):
        skipped = (root / make_run_id(cfg)).exists() and not force
        print(f"[{i}/{len(configs)}] {cfg.name}: {'skipped (exists)' if skipped else 'running'}", flush=True)
        t = time.time()
        folders.append(run(cfg, force=force, runs_dir=runs_dir))
        if not skipped:
            print(f"[{i}/{len(configs)}] {cfg.name}: done in {time.time() - t:.0f}s", flush=True)
    return folders


# ----------------------------------------------------------------------------- collect
# Tables written to `results/`, the only folder read by the report notebook:
#     runs.parquet         one row per run (flattened config, git sha, data fingerprint, timings)  [versioned]
#     metrics.parquet      long table from `evaluation.evaluate_predictions`                       [versioned]
#     predictions.parquet  all predictions                                                          [git-ignored]
#     importances.parquet  feature importances                                                      [git-ignored]


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    """Nested dict -> flat `a.b.c` columns; lists and free-form dicts become JSON strings (one Parquet column each)."""
    if isinstance(value, dict) and prefix and prefix.split(".")[-1] not in ("params", "rhythm_bounds_days"):
        for key, v in value.items():
            _flatten(f"{prefix}.{key}", v, out)
    elif isinstance(value, dict) and not prefix:
        for key, v in value.items():
            _flatten(key, v, out)
    elif isinstance(value, (dict, list, tuple)):
        out[prefix] = json.dumps(value, sort_keys=True, default=str)
    else:
        out[prefix] = value


def _run_row(meta: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """One row of `runs.parquet`: the flattened configuration (`model.type`, `label.kind`, ...) and the run's metadata."""
    row: dict[str, Any] = {"run_id": meta["run_id"]}
    _flatten("", {k: v for k, v in meta.items() if k != "run_id"}, row)
    for key, value in config.items():
        if isinstance(value, dict):
            _flatten(key, value, row)
        else:
            row.setdefault(key, value)  # `name` and `seed` are already there, from the metadata
    return row


def _universe(cfg: Config, cache: dict) -> pd.DataFrame | None:
    """The cached snapshot of the run's test date with every client (denominator of the `*_total_*` metrics), or
    None with a warning when it is not in the cache. Runs that share a cache key and a date share one read."""
    key = (cache_key(cfg, fingerprint(resolve(cfg.dataset.shifted_parquet))), cfg.snapshots.test_date)
    if key not in cache:
        try:
            cache[key] = load_universe(cfg)
        except FileNotFoundError as exc:
            print(f"collect: no universe for {cfg.name} ({exc}); total-churn metrics skipped", flush=True)
            cache[key] = None
    return cache[key]


def collect(
    *, runs_dir: Path | None = None, results_dir: Path | None = None, universe_for: Callable[[Config], bool] | None = None
) -> dict[str, Path]:
    """Scan `runs/`, recompute metrics from each run's predictions, write the four tables.
    Returns table name -> path.

    The tables are rebuilt from the run folders at every call (a deleted run disappears from them) and written
    atomically. Folders that are not complete runs (hidden temporary folders, a missing file) are skipped with a
    warning. Metrics are always recomputed, never read back from an older table.

    `universe_for(cfg)`: for the runs where it is true, the metrics with the whole client base as denominator
    (`capture_total_at_k`, ...) are added; it costs 15 to 25 s per run, so it is off by default."""
    root = Path(runs_dir) if runs_dir is not None else RUNS_DIR
    out = Path(results_dir) if results_dir is not None else RESULTS_DIR
    needed = ("config.yaml", "meta.json", "predictions.parquet", "importances.parquet")
    folders = []
    for folder in sorted(p for p in root.glob("*") if p.is_dir() and not p.name.startswith(".")) if root.exists() else []:
        missing = [f for f in needed if not (folder / f).exists()]
        if missing:
            print(f"collect: skipping {folder.name} (missing {', '.join(missing)})", flush=True)
        else:
            folders.append(folder)
    if not folders:
        raise ValueError(f"No complete run in {root}: run `churn run <config>` or `churn sweep <sweep>` first")

    runs, metrics, predictions, importances = [], [], [], []
    universes: dict = {}
    for folder in folders:
        cfg = load_config(folder / "config.yaml")
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        pred = pd.read_parquet(folder / "predictions.parquet")
        runs.append(_run_row(meta, config_to_dict(cfg)))
        uni = _universe(cfg, universes) if universe_for is not None and universe_for(cfg) else None
        metrics.append(evaluate_predictions(pred, cfg, universe=uni))
        predictions.append(pred)
        importances.append(pd.read_parquet(folder / "importances.parquet"))

    tables = {
        "runs": pd.DataFrame(runs),
        "metrics": pd.concat(metrics, ignore_index=True),
        "predictions": pd.concat(predictions, ignore_index=True),
        "importances": pd.concat(importances, ignore_index=True),
    }
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, table in tables.items():
        paths[name] = out / f"{name}.parquet"
        tmp = paths[name].with_name(paths[name].name + ".tmp")
        table.to_parquet(tmp, index=False)
        tmp.replace(paths[name])  # atomic: the report notebook never reads a half-written table
    return paths
