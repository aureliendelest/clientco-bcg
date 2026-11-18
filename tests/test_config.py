"""Configuration loading is real infrastructure (not a stub), so these tests run."""

import glob

import pytest

from churn.config import Config, config_hash, load_config, run_id


def test_base_config_loads_completely():
    cfg = load_config("configs/base.yaml")
    assert isinstance(cfg, Config)
    assert cfg.snapshots.embargo_days == 180
    assert [t.name for t in cfg.scoring.tiers] == ["dedicated", "general"]


@pytest.mark.parametrize("path", sorted(glob.glob("configs/experiments/*.yaml")))
def test_experiments_extend_base(path):
    cfg = load_config(path)
    assert cfg.model.type == cfg.name
    assert cfg.label.kind == "class_horizon"  # inherited from base


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
