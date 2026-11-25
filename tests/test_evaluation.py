"""Evaluation on toy predictions: total-churn metrics against a universe, the universe loader, the paired comparison.

Toy universe (one date, 12 clients). Regular clients (fast / medium) are the only ones predicted:

    client  V    gone  rhythm         ev
    1       100  1     fast           9
    2        80  1     fast           8
    3        60  0     medium         7
    4        50  1     medium         6
    5        40  1     fast           5
    6        30  0     fast           4
    7       500  1     slow           -     (out of perimeter)
    8       300  0     short_history  -
    9       200  0     short_history  -
    10       10  1     slow           -
    11     1000  NaN   fast           10    (truth unknown: ignored everywhere)

Churned V: perimeter 270 (clients 1, 2, 4, 5), whole universe 780 (+ 500 + 10).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn import evaluation
from churn.comparison import accept, paired_by_date
from churn.config import Config
from churn.evaluation import TOTAL_METRICS, evaluate_predictions, load_universe

DATE = pd.Timestamp("2025-10-22")
TOTAL = 780.0
KS = [2, 3, 100]  # 100 is larger than the number of clients


def _universe(dates=(DATE,)) -> pd.DataFrame:
    rows = [
        (1, 100, 1, "fast"), (2, 80, 1, "fast"), (3, 60, 0, "medium"), (4, 50, 1, "medium"), (5, 40, 1, "fast"),
        (6, 30, 0, "fast"), (7, 500, 1, "slow"), (8, 300, 0, "short_history"), (9, 200, 0, "short_history"),
        (10, 10, 1, "slow"), (11, 1000, np.nan, "fast"),
    ]
    uni = pd.DataFrame(rows, columns=["client_id", "V", "gone", "rhythm"])
    uni["rev_90d"] = uni["V"] / 4
    return pd.concat([uni.assign(snapshot_date=d) for d in dates], ignore_index=True)


def _pred(universe: pd.DataFrame, run_id: str = "toy") -> pd.DataFrame:
    ev = {1: 9, 2: 8, 3: 7, 4: 6, 5: 5, 6: 4, 11: 10}
    p = universe.loc[universe["client_id"].isin(ev)].copy()
    p["ev"] = p["client_id"].map(ev).astype(float)
    p["p_raw"] = p["p_cal"] = p["ev"] / 10
    p["label"] = p["gone"]
    p["horizon_days"] = 180.0
    p["split"] = "test"
    p["run_id"] = run_id
    p["segment"] = p["rhythm"]
    return p[["run_id", "client_id", "snapshot_date", "split", "p_raw", "p_cal", "V", "ev", "label", "gone", "horizon_days", "segment"]]


def _cfg() -> Config:
    cfg = Config()
    cfg.evaluation.k_values, cfg.evaluation.bootstrap_n = KS, 30
    cfg.snapshots.test_date = "2025-10-22"
    return cfg


def _value(table: pd.DataFrame, metric: str, k: int, split: str = "test") -> float:
    row = table[(table.metric == metric) & (table.k == k) & (table.split == split) & (table.segment == "all")]
    assert len(row) == 1
    return float(row["value"].iloc[0])


# ------------------------------------------------------------------------------ total-churn metrics


def test_total_metrics_by_hand_with_out_of_perimeter_clients():
    uni = _universe()
    out = evaluate_predictions(_pred(uni), _cfg(), universe=uni)
    # K = 2: the model lists clients 1 and 2 (180), the oracle takes 7 and 1 (600), the biggest are 7 and 8 (500)
    assert _value(out, "capture_total_at_k", 2) == pytest.approx(180 / TOTAL)
    assert _value(out, "oracle_total_capture", 2) == pytest.approx(600 / TOTAL)
    assert _value(out, "size_only_total_capture", 2) == pytest.approx(500 / TOTAL)
    # K = 3: client 3 stays (not gone); oracle 7, 1, 2; biggest 7, 8, 9
    assert _value(out, "capture_total_at_k", 3) == pytest.approx(180 / TOTAL)
    assert _value(out, "oracle_total_capture", 3) == pytest.approx(680 / TOTAL)
    assert _value(out, "size_only_total_capture", 3) == pytest.approx(500 / TOTAL)


def test_k_larger_than_the_churners_and_than_the_predicted_clients():
    uni = _universe()
    out = evaluate_predictions(_pred(uni), _cfg(), universe=uni)
    # every regular churner is listed, but the 510 of the other clients are never reachable by the model
    assert _value(out, "capture_total_at_k", 100) == pytest.approx(270 / TOTAL)
    assert _value(out, "oracle_total_capture", 100) == pytest.approx(1.0)
    assert _value(out, "size_only_total_capture", 100) == pytest.approx(1.0)
    # the regular denominator is untouched: same K, the old metrics
    assert _value(out, "capture_at_k", 100) == pytest.approx(1.0)


def test_total_capture_is_below_regular_capture_and_below_the_oracle():
    uni = _universe()
    out = evaluate_predictions(_pred(uni), _cfg(), universe=uni)
    for k in KS:
        assert _value(out, "capture_total_at_k", k) <= _value(out, "capture_at_k", k) + 1e-12
        assert _value(out, "capture_total_at_k", k) <= _value(out, "oracle_total_capture", k) + 1e-12
        assert _value(out, "size_only_total_capture", k) <= _value(out, "oracle_total_capture", k) + 1e-12


def test_unknown_truth_is_ignored_in_numerator_and_denominator():
    uni = _universe()
    with_nan = evaluate_predictions(_pred(uni), _cfg(), universe=uni)
    # the same toy without client 11 (V = 1000, gone unknown, best ev): identical numbers
    uni2 = uni[uni.client_id != 11]
    without = evaluate_predictions(_pred(uni2), _cfg(), universe=uni2)
    pd.testing.assert_frame_equal(with_nan, without)


def test_existing_metrics_are_unchanged_by_the_universe_and_new_ones_are_added():
    uni = _universe()
    cfg = _cfg()
    plain = evaluate_predictions(_pred(uni), cfg)
    with_uni = evaluate_predictions(_pred(uni), cfg, universe=uni)
    old = with_uni[~with_uni.metric.isin(TOTAL_METRICS)].reset_index(drop=True)
    pd.testing.assert_frame_equal(plain, old)
    assert not plain.metric.isin(TOTAL_METRICS).any()
    new = with_uni[with_uni.metric.isin(TOTAL_METRICS)]
    assert set(new.metric) == set(TOTAL_METRICS) and set(new.segment) == {"all"}
    assert sorted(new.k.unique()) == KS and len(new) == len(KS) * len(TOTAL_METRICS)
    assert list(with_uni.columns) == ["run_id", "split", "segment", "metric", "k", "value", "ci_low", "ci_high"]
    assert evaluate_predictions(_pred(uni), cfg, universe=uni).equals(with_uni)  # deterministic given the seed
    # the new metrics are not part of the configured metrics
    assert not set(TOTAL_METRICS) & set(cfg.evaluation.metrics)


def test_confidence_intervals_of_the_total_metrics():
    uni = _universe()
    out = evaluate_predictions(_pred(uni), _cfg(), universe=uni)
    new = out[out.metric.isin(TOTAL_METRICS)]
    assert new[["ci_low", "ci_high"]].notna().all().all()
    assert (new.ci_low <= new.ci_high).all()
    assert ((new.value >= 0) & (new.value <= 1)).all()


def test_top_k_is_taken_per_date_and_pooled():
    uni = _universe(dates=(DATE, DATE + pd.Timedelta(days=30)))
    out = evaluate_predictions(_pred(uni), _cfg(), universe=uni)
    # two identical dates: same ratios as a single date
    assert _value(out, "capture_total_at_k", 2) == pytest.approx(180 / TOTAL)
    assert _value(out, "oracle_total_capture", 2) == pytest.approx(600 / TOTAL)


def test_no_churner_at_all_gives_nan_not_an_error():
    uni = _universe().assign(gone=0.0)
    out = evaluate_predictions(_pred(uni), _cfg(), universe=uni)
    assert np.isnan(out[out.metric.isin(TOTAL_METRICS)]["value"]).all()


def test_universe_that_does_not_match_the_predictions_is_refused():
    uni = _universe()
    cfg = _cfg()
    with pytest.raises(ValueError, match="none of the snapshot dates"):
        evaluate_predictions(_pred(uni), cfg, universe=_universe(dates=(DATE + pd.Timedelta(days=30),)))
    with pytest.raises(ValueError, match="not in the universe"):
        evaluate_predictions(_pred(uni), cfg, universe=uni[uni.client_id != 1])
    with pytest.raises(ValueError, match="differ"):
        evaluate_predictions(_pred(uni), cfg, universe=uni.assign(V=uni["V"] * 2))
    with pytest.raises(ValueError, match="lacks"):
        evaluate_predictions(_pred(uni), cfg, universe=uni.drop(columns="V"))


def test_a_split_outside_the_universe_is_left_without_total_metrics():
    uni = _universe()
    other = _pred(uni).assign(split="valid", snapshot_date=DATE - pd.Timedelta(days=30))
    out = evaluate_predictions(pd.concat([_pred(uni), other]), _cfg(), universe=uni)
    assert set(out[out.metric.isin(TOTAL_METRICS)].split) == {"test"}
    assert "valid" in set(out.split)


def test_unknown_metric_in_the_config_is_still_refused():
    cfg = _cfg()
    cfg.evaluation.metrics = ["capture_total_at_k"]  # computed with a universe, never requested through the config
    with pytest.raises(ValueError, match="capture_total_at_k"):
        evaluate_predictions(_pred(_universe()), cfg)


def test_threshold_stubs_are_gone():
    for name in ("threshold_metrics", "best_f1_threshold", "impact_scenarios", "tornado_table"):
        assert not hasattr(evaluation, name)


# ------------------------------------------------------------------------------ load_universe


def _cache_config(tmp_path, monkeypatch) -> Config:
    from churn.data import fingerprint
    from churn.snapshots import cache_key

    parquet = tmp_path / "sales.parquet"
    parquet.write_bytes(b"not a real parquet: only its fingerprint is read" * 10)
    cfg = _cfg()
    cfg.dataset.shifted_parquet = str(parquet)
    monkeypatch.setattr("churn.snapshots.PROCESSED_DIR", tmp_path / "processed")
    cfg._key = cache_key(cfg, fingerprint(parquet))  # type: ignore[attr-defined]
    return cfg


def test_load_universe_reads_the_cached_snapshot_of_the_test_date(tmp_path, monkeypatch):
    cfg = _cache_config(tmp_path, monkeypatch)
    folder = tmp_path / "processed" / f"snapshots-{cfg._key}"  # type: ignore[attr-defined]
    folder.mkdir(parents=True)
    snap = _universe().assign(freq_90d=3, label=1.0)  # a snapshot has more columns than the universe
    snap.to_parquet(folder / "2025-10-22.parquet", index=False)
    snap.assign(V=-1.0).to_parquet(folder / "2025-11-22.parquet", index=False)  # another date: not read
    out = load_universe(cfg)
    assert list(out.columns) == ["client_id", "snapshot_date", "V", "gone", "rhythm", "rev_90d"]
    assert len(out) == len(snap) and (out["V"] > 0).all()  # all the clients, whatever the rhythm
    assert set(out["rhythm"]) == {"fast", "medium", "slow", "short_history"}


def test_load_universe_never_builds_and_says_what_is_missing(tmp_path, monkeypatch):
    cfg = _cache_config(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError, match="2025-10-22"):
        load_universe(cfg)
    assert not (tmp_path / "processed").exists()  # nothing was created: no build, not even the folder
    cfg.dataset.shifted_parquet = str(tmp_path / "absent.parquet")
    with pytest.raises(FileNotFoundError, match="absent.parquet"):
        load_universe(cfg)


# ------------------------------------------------------------------------------ paired comparison

DATES = ["2025-09-22", "2025-10-22", "2025-11-22", "2025-12-22", "2026-01-22", "2026-02-22"]


def _runs_and_metrics(gain700, gain2000, model="hgb", base=0.40):
    """Runs of A and B (one per date) and their metrics; the gaps are given in points of capture."""
    runs_a = pd.DataFrame({"run_id": [f"{model}-a-{d}" for d in DATES], "snapshots.test_date": DATES})
    runs_b = pd.DataFrame({"run_id": [f"{model}-b-{d}" for d in DATES], "snapshots.test_date": DATES})
    rows = []
    for i, d in enumerate(DATES):
        for k, gain in ((700, gain700[i]), (2000, gain2000[i])):
            rows.append((f"{model}-a-{d}", "test", "all", "capture_at_k", k, base))
            rows.append((f"{model}-b-{d}", "test", "all", "capture_at_k", k, base + gain / 100))
    rows.append((f"{model}-a-{DATES[0]}", "test", "all", "pr_auc", None, 0.5))  # other metrics are ignored
    metrics = pd.DataFrame(rows, columns=["run_id", "split", "segment", "metric", "k", "value"])
    metrics["k"] = metrics["k"].astype("Int64")
    return metrics, runs_a, runs_b


def test_paired_by_date_one_row_per_date_with_gap_in_points_and_winner():
    metrics, a, b = _runs_and_metrics([2, -1, 3, 0, 1, 2], [0] * 6)
    t = paired_by_date(metrics, a, b, 700)
    assert list(t.columns) == ["date", "k", "run_a", "run_b", "value_a", "value_b", "diff_pts", "winner"]
    assert len(t) == 6 and list(t["date"]) == [pd.Timestamp(d) for d in DATES]
    assert t["diff_pts"].tolist() == pytest.approx([2, -1, 3, 0, 1, 2])
    assert t["winner"].tolist() == ["B", "A", "B", "tie", "B", "B"]
    assert (t["value_a"] == 0.40).all()
    both = paired_by_date(metrics, a, b, [700, 2000], model="hgb")
    assert len(both) == 12 and set(both["model"]) == {"hgb"} and both.columns[0] == "model"
    # a mapping {date: run_id} works like a table
    by_map = paired_by_date(metrics, dict(zip(a["snapshots.test_date"], a["run_id"])), b, 700)
    pd.testing.assert_frame_equal(by_map, t)


def test_paired_by_date_refuses_ambiguous_or_missing_input():
    metrics, a, b = _runs_and_metrics([1] * 6, [0] * 6)
    with pytest.raises(ValueError, match="same dates"):
        paired_by_date(metrics, a, b.iloc[1:], 700)
    with pytest.raises(ValueError, match="several runs"):
        paired_by_date(metrics, pd.concat([a, a.iloc[:1]]), b, 700)
    with pytest.raises(ValueError, match="K=4000"):
        paired_by_date(metrics, a, b, 4000)
    with pytest.raises(ValueError, match="lacks"):
        paired_by_date(metrics, a.drop(columns="snapshots.test_date"), b, 700)


def _table(model, g700, g2000):
    metrics, a, b = _runs_and_metrics(g700, g2000, model=model)
    return paired_by_date(metrics, a, b, [700, 2000], model=model)


def test_accept_default_rule_passes_and_excludes_the_first_date():
    # the 22/09 date is a huge loss: it would fail the rule if it were counted
    t = _table("hgb", [-30, 2, 2, 1, 2, 1.5], [-30, 0, 1, -1, 0, 0])
    v = accept(t)
    m = v["models"]["hgb"]
    assert v["accepted"] and m["accepted"] and m["n_dates"] == 5
    assert m["mean_gain_pts"] == pytest.approx(1.7) and m["wins"] == 5
    assert m["mean_gain_k_check_pts"] == pytest.approx(0.0)
    assert v["thresholds"] == {"min_gain": 1.5, "min_wins": 4, "k_main": 700, "k_check": 2000, "excluded_dates": ["2025-09-22"]}
    # counted when the exclusion is lifted
    assert not accept(t, excluded_dates=())["accepted"]


def test_accept_each_condition_can_fail_alone():
    ok2000 = [0] * 6
    assert not accept(_table("m", [0, 1, 1, 1, 1, 1], ok2000))["accepted"]  # mean gain 1.0 < 1.5
    low_wins = accept(_table("m", [0, 4, 4, 4, -1, 0.5], ok2000))  # mean 2.3, 4 wins of the 5 counted dates
    assert low_wins["models"]["m"]["wins"] == 4 and low_wins["accepted"]
    three = accept(_table("m", [0, 5, 5, 5, -1, -1], ok2000))  # mean 2.6 but 3 wins
    assert not three["models"]["m"]["wins_ok"] and three["models"]["m"]["gain_ok"] and not three["accepted"]
    k_fail = accept(_table("m", [0, 2, 2, 2, 2, 2], [0, -3, 0, 0, 0, 0]))  # K = 2000 mean -0.6
    assert not k_fail["models"]["m"]["k_check_ok"] and not k_fail["accepted"]


def test_accept_thresholds_are_parameters_and_boundary_is_inclusive():
    t = _table("m", [0, 1.5, 1.5, 1.5, 1.5, 1.5], [0] * 6)
    assert accept(t)["accepted"]  # exactly 1.5 points (floating point slack)
    assert not accept(t, min_gain=1.6)["accepted"]
    assert not accept(t, min_wins=6)["accepted"]
    assert accept(t, k_check=700, min_gain=0)["accepted"]


def test_accept_gives_a_verdict_per_model_and_a_global_one():
    good = [0, 2, 2, 2, 2, 2]
    t = pd.concat([_table("hgb", good, [0] * 6), _table("xgb", [0, 0, 0, 0, 0, 1], [0] * 6)], ignore_index=True)
    v = accept(t)
    assert v["models"]["hgb"]["accepted"] and not v["models"]["xgb"]["accepted"]
    assert not v["accepted"]  # both models must pass
    # without a model column, the table is one model
    assert list(accept(_table("hgb", good, [0] * 6).drop(columns="model"))["models"]) == ["all"]


def test_accept_needs_both_ks():
    metrics, a, b = _runs_and_metrics([2] * 6, [0] * 6)
    with pytest.raises(ValueError, match="K=700 and K=2000"):
        accept(paired_by_date(metrics, a, b, 700))
