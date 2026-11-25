"""pipeline.run: files written, schema, skip / force, reproducibility, and no use of the test snapshot."""

import dataclasses
import json
import pickle

import numpy as np
import pandas as pd
import pytest

from churn.config import ModelCfg, config_hash, load_config, run_id
from churn.pipeline import PREDICTION_COLUMNS, fit_predict, run
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
