"""pipeline.run and sweeps: files written, schema, skip / force, reproducibility, no use of the test snapshot."""

import dataclasses
import json
import pickle

import numpy as np
import pandas as pd
import pytest
import yaml

from churn.config import ModelCfg, config_hash, config_to_dict, load_config, override_config, run_id
from churn.pipeline import PREDICTION_COLUMNS, expand_sweep, fit_predict, run, run_sweep
from churn.snapshots import in_perimeter, load_or_build, temporal_split


def test_run_writes_the_run_folder(pipeline_cfg, tmp_path):
    out = run(pipeline_cfg, runs_dir=tmp_path)
    assert out == tmp_path / run_id(pipeline_cfg)
    assert {p.name for p in out.iterdir()} == {"config.yaml", "meta.json", "predictions.parquet", "importances.parquet", "model.pkl"}
    assert not list(tmp_path.glob(".*.tmp"))

    pred = pd.read_parquet(out / "predictions.parquet")
    assert list(pred.columns) == PREDICTION_COLUMNS
    assert set(pred["split"]) == {"test"} and set(pred["run_id"]) == {out.name}
    assert (pd.to_datetime(pred["snapshot_date"]) == pd.Timestamp(pipeline_cfg.snapshots.test_date)).all()
    assert pred["client_id"].is_unique and pred["client_id"].is_monotonic_increasing
    assert pred[["p_raw", "p_cal"]].apply(lambda c: c.between(0, 1).all()).all()
    np.testing.assert_allclose(pred["ev"], pred["p_cal"] * pred["V"])
    assert set(pred["segment"]) <= {"fast", "medium"}  # the perimeter only

    _, test = temporal_split(load_or_build(pipeline_cfg), pipeline_cfg)
    assert len(pred) == in_perimeter(test).sum()

    meta = json.loads((out / "meta.json").read_text())
    assert meta["run_id"] == out.name and meta["config_hash"] == config_hash(pipeline_cfg)
    assert meta["train_dates"][0] == pipeline_cfg.snapshots.train_start
    assert max(meta["train_dates"]) == pipeline_cfg.last_train_date() < pipeline_cfg.snapshots.test_date
    assert meta["data_fingerprint"] and meta["n_train_rows"] > 0 and 0 < meta["train_positive_rate"] < 1
    assert set(pd.read_parquet(out / "importances.parquet")["kind"]) == {"coefficient"}
    with (out / "model.pkl").open("rb") as f:
        bundle = pickle.load(f)
    assert bundle["model"].predict_proba(pd.DataFrame(np.ones((1, len(bundle["features"]))), columns=bundle["features"])).shape == (1, 2)


def test_the_saved_config_gives_the_same_run_id(pipeline_cfg, tmp_path):
    out = run(pipeline_cfg, runs_dir=tmp_path)
    assert run_id(load_config(out / "config.yaml")) == out.name


def test_existing_run_is_skipped_unless_forced(pipeline_cfg, tmp_path):
    out = run(pipeline_cfg, runs_dir=tmp_path)
    (out / "marker").write_text("x")
    assert run(pipeline_cfg, runs_dir=tmp_path) == out and (out / "marker").exists()
    assert run(pipeline_cfg, force=True, runs_dir=tmp_path) == out and not (out / "marker").exists()


def test_a_failed_run_leaves_nothing_behind(pipeline_cfg, tmp_path):
    bad = dataclasses.replace(pipeline_cfg, model=dataclasses.replace(pipeline_cfg.model, segmentation="per_segment"))
    with pytest.raises(NotImplementedError, match="per_segment"):
        run(bad, runs_dir=tmp_path)
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("kind", ["logreg", "hgb", "xgb", "baseline"])
def test_same_configuration_same_predictions(pipeline_cfg, tmp_path, kind):
    cfg = dataclasses.replace(pipeline_cfg, name=f"repro-{kind}", model=ModelCfg(type=kind, cv_folds=3))
    a = pd.read_parquet(run(cfg, runs_dir=tmp_path / "a") / "predictions.parquet")
    b = pd.read_parquet(run(cfg, runs_dir=tmp_path / "b") / "predictions.parquet")
    pd.testing.assert_frame_equal(a, b)


def test_calibration_none_skips_the_cv_and_keeps_raw_probabilities(pipeline_cfg):
    cfg = dataclasses.replace(pipeline_cfg, model=ModelCfg(type="logreg", calibration="none"))
    train, test = temporal_split(load_or_build(cfg), cfg)
    pred = fit_predict(cfg, train, {"test": test}).predictions
    assert pred["p_cal"].equals(pred["p_raw"])


def test_rule_baseline_runs_through_the_same_pipeline(pipeline_cfg):
    cfg = dataclasses.replace(pipeline_cfg, model=ModelCfg(type="baseline", params={"rule": "business_rules"}, cv_folds=3))
    train, test = temporal_split(load_or_build(cfg), cfg)
    fitted = fit_predict(cfg, train, {"test": test})
    assert fitted.importances.empty
    assert fitted.predictions["p_cal"].between(0, 1).all()
    # calibrated, the score keeps its ranking inside each segment (the isotonic regression is monotone)
    for _, part in fitted.predictions.groupby("segment"):
        assert (np.diff(part.sort_values("p_raw", kind="stable")["p_cal"].to_numpy()) >= -1e-12).all()


# ------------------------------------------------------------------------------ sweeps


def _write_sweep(tmp_path, base_cfg, grid, name="grid"):
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump(config_to_dict(base_cfg), sort_keys=False))
    sweep = tmp_path / f"{name}.yaml"
    sweep.write_text(yaml.safe_dump({"base": str(base), "grid": grid}, sort_keys=False))
    return sweep


def test_the_project_sweep_expands_to_the_four_models_on_the_two_labels():
    configs = expand_sweep("configs/sweeps/models.yaml")
    assert len(configs) == 8 and len({c.name for c in configs}) == 8 and len({run_id(c) for c in configs}) == 8
    assert {(c.model.type, c.label.kind) for c in configs} == {
        (m, k) for m in ("baseline", "logreg", "hgb", "xgb") for k in ("fixed_horizon", "relative_horizon")
    }
    assert configs[0].name == "models__model.type-baseline__label.kind-fixed_horizon"
    assert configs[1].label.kind == "relative_horizon"  # the last key varies fastest
    rest = lambda c: config_to_dict(override_config(c, {"name": "x", "model.type": "hgb", "label.kind": "fixed_horizon"}))
    assert all(rest(c) == rest(configs[0]) for c in configs)  # nothing else differs from the base


def test_a_grid_can_reach_into_model_params(tmp_path, pipeline_cfg):
    sweep = _write_sweep(tmp_path, pipeline_cfg, {"model.params.C": [0.1, 1.0], "seed": [0, 1]})
    configs = expand_sweep(sweep)
    assert [(c.model.params["C"], c.seed) for c in configs] == [(0.1, 0), (0.1, 1), (1.0, 0), (1.0, 1)]
    assert all(c.name.startswith("grid__model.params.C-") for c in configs)


def test_an_empty_grid_is_the_base_experiment(tmp_path, pipeline_cfg):
    (cfg,) = expand_sweep(_write_sweep(tmp_path, pipeline_cfg, {}))
    assert cfg.name == "grid" and cfg.model == pipeline_cfg.model


@pytest.mark.parametrize("grid, message", [
    ({"model.typo": ["hgb"]}, "Unknown option"),
    ({"modell.type": ["hgb"]}, "Unknown option"),
    ({"model.type": "hgb"}, "non-empty list"),
    ({"model.type": []}, "non-empty list"),
    ({"model.type": ["hgb", "hgb"]}, "duplicate"),
    ({"name": ["a", "b"]}, "cannot be in the grid"),
    ({"model.type.x": ["a"]}, "not a section"),
])
def test_a_bad_grid_is_refused(tmp_path, pipeline_cfg, grid, message):
    with pytest.raises(ValueError, match=message):
        expand_sweep(_write_sweep(tmp_path, pipeline_cfg, grid))


def test_a_bad_sweep_file_is_refused(tmp_path):
    for content, message in (("grid: {}\n", "missing `base`"), ("base: x\nbases: y\n", "unknown key")):
        (tmp_path / "s.yaml").write_text(content)
        with pytest.raises(ValueError, match=message):
            expand_sweep(tmp_path / "s.yaml")


def test_run_sweep_runs_every_combination_then_resumes(pipeline_cfg, tmp_path, capsys):
    sweep = _write_sweep(tmp_path, pipeline_cfg, {"model.type": ["baseline", "logreg"]})
    runs = tmp_path / "runs"
    folders = run_sweep(sweep, runs_dir=runs)
    assert [f.name for f in folders] == [run_id(c) for c in expand_sweep(sweep)]
    assert all((f / "predictions.parquet").exists() for f in folders)
    (folders[0] / "marker").write_text("x")
    assert run_sweep(sweep, runs_dir=runs) == folders and (folders[0] / "marker").exists()  # skipped, not rerun
    assert "skipped" in capsys.readouterr().out
    run_sweep(sweep, force=True, runs_dir=runs)
    assert not (folders[0] / "marker").exists()


def test_run_sweep_stops_at_the_first_failure_and_keeps_finished_runs(pipeline_cfg, tmp_path):
    sweep = _write_sweep(tmp_path, pipeline_cfg, {"model.segmentation": ["single", "per_segment"]})
    with pytest.raises(NotImplementedError):
        run_sweep(sweep, runs_dir=tmp_path / "runs")
    assert [p.name for p in (tmp_path / "runs").iterdir()] == [run_id(expand_sweep(sweep)[0])]


def test_dates_in_a_grid_become_iso_strings(tmp_path, pipeline_cfg):
    sweep = tmp_path / "dates.yaml"  # unquoted dates are parsed by YAML as `date` objects
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump(config_to_dict(pipeline_cfg), sort_keys=False))
    sweep.write_text(f"base: {base}\ngrid:\n  snapshots.test_date: [2025-09-22, 2025-10-22]\n")
    assert [c.snapshots.test_date for c in expand_sweep(sweep)] == ["2025-09-22", "2025-10-22"]
