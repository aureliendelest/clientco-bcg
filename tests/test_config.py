"""Configuration loading is real infrastructure (not a stub), so these tests run."""

import glob

import pytest

from churn.config import Config, config_hash, load_config, run_id


def test_base_config_loads_completely():
    cfg = load_config("configs/base.yaml")
    assert isinstance(cfg, Config)
    assert cfg.label.kind == "fixed_horizon" and cfg.label.horizon_days == cfg.label.truth_days == 180
    assert cfg.scoring.calls_per_week == 500 and cfg.scoring.list_size_k == 700
    assert cfg.label_wait_days() == 180
    assert cfg.last_train_date() == "2025-09-22"   # 7 monthly photos, label A


@pytest.mark.parametrize("path", sorted(glob.glob("configs/experiments/*.yaml")))
def test_experiments_extend_base(path):
    cfg = load_config(path)
    if path.endswith("final.yaml"):
        # The frozen configuration of the final test (docs/plan_gel.md) is named after its role, not its model.
        assert cfg.name == "final"
    else:
        assert cfg.name.startswith(cfg.model.type)
    assert cfg.label.kind in ("fixed_horizon", "relative_horizon")


def test_label_b_waits_90_days_and_trains_until_december():
    cfg = load_config("configs/experiments/hgb_label_b.yaml")
    assert (cfg.label.k, cfg.label.h_min, cfg.label.h_max) == (3.0, 45, 90)
    assert cfg.label_wait_days() == 90
    assert cfg.last_train_date() == "2025-12-22"   # 10 monthly photos, label B


def test_unknown_option_is_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("extends: configs/base.yaml\nmodel:\n  typo_option: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(bad)


def test_run_id_is_stable_and_depends_on_the_config():
    a, b = load_config("configs/experiments/hgb.yaml"), load_config("configs/experiments/xgb.yaml")
    assert config_hash(a) != config_hash(b)
    assert run_id(a) == run_id(a)
    assert run_id(a) != run_id(b)
    assert run_id(a).startswith("hgb-")
