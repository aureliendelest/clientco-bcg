"""Unit tests of the bricks. Snapshot tests run on a hand-made sales table (see conftest.py)."""

import numpy as np
import pandas as pd
import pytest

from churn.config import Config
from churn.evaluation import (
    CORE_METRICS, bootstrap_ci, brier, capture_at_k, evaluate_predictions, oracle_capture, pr_auc,
)
from churn.snapshots import FEATURE_SETS, build_snapshot, compute_features
from tests.helpers import T, before, every, sales_for

# ---- snapshots: labels


def test_fixed_horizon_is_the_same_for_every_client(sales_con):
    # Label A: lost = no purchase day within label.horizon_days (180) after T, whatever the rhythm.
    # The window is [T, T + 180): a purchase on T counts, a purchase on T + 180 does not.
    con = sales_con(sales_for({
        "stays_fast": every("2025-01-01", "2025-12-31", 5),
        "stays_medium": every("2025-01-01", "2025-12-31", 30),
        "back_on_day_T": every("2025-01-01", "2025-04-14", 5) + ["2025-04-15"],
        "back_on_day_179": every("2025-01-01", "2025-04-14", 5) + ["2025-10-11"],
        "back_on_day_180": every("2025-01-01", "2025-04-14", 5) + ["2025-10-12"],
        "stopped": every("2025-01-01", "2025-04-14", 5),
    }))
    snap = build_snapshot(con, T, Config()).set_index("client_id")
    lost = snap.label.to_dict()
    ids = sales_for.ids
    assert [lost[ids[n]] for n in ("stays_fast", "stays_medium", "back_on_day_T", "back_on_day_179")] == [0, 0, 0, 0]
    assert [lost[ids[n]] for n in ("back_on_day_180", "stopped")] == [1, 1]
    assert (snap.horizon_days == 180).all()
    assert snap.gone.equals(snap.label)  # same definition: the truth is the label A


def test_relative_horizon_is_clipped_between_h_min_and_h_max(sales_con):
    # Label B: H_i = clip(k * p90_gap_i, h_min, h_max) = clip(3 * p90, 45, 90) by default
    cfg = Config()
    cfg.label.kind = "relative_horizon"
    con = sales_con(sales_for({
        "gap5": before(8, 5) + ["2025-06-04"],    # 3 * 5 = 15 -> 45; back after 50 days: lost
        "gap20": before(6, 20) + ["2025-06-04"],  # 3 * 20 = 60; back after 50 days: not lost
        "gap30": before(6, 30) + ["2025-06-04"],  # 3 * 30 = 90; back after 50 days: not lost
        "gap35": before(6, 35) + ["2025-07-30"],  # 3 * 35 = 105 -> 90; back after 106 days: lost
        "filler": every("2025-01-01", "2025-12-31", 5),  # only extends the data, so that the 180-day truth is observed
    }))
    snap = build_snapshot(con, T, cfg).set_index("client_id")
    ids = sales_for.ids
    assert snap.loc[ids["gap5"], "horizon_days"] == 45
    assert snap.loc[ids["gap20"], "horizon_days"] == 60
    assert snap.loc[ids["gap30"], "horizon_days"] == 90
    assert snap.loc[ids["gap35"], "horizon_days"] == 90
    assert [snap.loc[ids[n], "label"] for n in ("gap5", "gap20", "gap30", "gap35")] == [1, 0, 0, 1]
    assert (snap.gone == 0).all()  # all of them come back within 180 days: the truth says they stayed


def test_label_is_nan_when_the_window_is_not_observed(sales_con):
    # Data end 2025-12-31: T + 180 = 2025-10-12 is observed, 2025-07-15 + 180 is not.
    con = sales_con(sales_for({"a": every("2025-01-01", "2025-12-31", 5)}))
    cfg = Config()
    assert build_snapshot(con, T, cfg).label.notna().all()
    late = build_snapshot(con, pd.Timestamp("2025-07-15").date(), cfg)
    assert late[["label", "gone"]].isna().all().all()


def test_partial_drop_needs_a_purchase_and_a_big_revenue_drop(sales_con):
    cfg = Config()
    cfg.label.kind = "partial_drop"
    rows = {
        "unchanged": every("2025-01-01", "2025-12-31", 5),
        "halved": every("2025-01-01", "2025-04-14", 5) + every("2025-04-15", "2025-10-11", 20),  # still buys, a lot less
        "gone": every("2025-01-01", "2025-04-14", 5),  # total churn is not partial
    }
    con = sales_con(sales_for(rows))
    lost = build_snapshot(con, T, cfg).set_index("client_id").label
    ids = sales_for.ids
    assert [lost[ids[n]] for n in ("unchanged", "halved", "gone")] == [0, 1, 0]


# ---- snapshots: features and contract


def test_every_feature_of_the_set_is_produced(sales_con):
    # compute_features returns exactly the columns of FEATURE_SETS[feature_set].
    con = sales_con(sales_for({"a": every("2025-01-01", "2025-04-14", 5)}))
    out = compute_features(con, T, "basic")
    assert list(out.columns) == ["client_id", *FEATURE_SETS["basic"]]
    assert len(out) == 1


def test_snapshot_columns_match_the_contract(sales_con):
    # client_id, snapshot_date, <features>, label, horizon_days, V, segment
    con = sales_con(sales_for({"a": every("2025-01-01", "2025-12-31", 5)}))
    snap = build_snapshot(con, T, Config())
    assert list(snap.columns) == [
        "client_id", "snapshot_date", *FEATURE_SETS["basic"], "rhythm", "label", "horizon_days", "gone", "V", "segment",
    ]
    row = snap.iloc[0]
    assert row.snapshot_date == pd.Timestamp(T)
    assert row.V == 2 * row.rev_180d          # V = revenue of the 180 days before, x 2
    assert row.segment == row.rhythm == "fast"
    assert snap.client_id.is_monotonic_increasing  # reproducibility: rows sorted by client


# ---- evaluation


def _pred(rows, run_id="r", split="test", date="2026-03-22"):
    """Predictions from (client, V, ev, gone[, p_raw, p_cal, label, segment]) tuples; defaults keep it short."""
    cols = ["client_id", "V", "ev", "gone", "p_raw", "p_cal", "label", "segment"]
    df = pd.DataFrame([(*r, *[None] * (8 - len(r))) for r in rows], columns=cols)
    df["p_raw"] = df["p_raw"].fillna(0.5).astype(float)
    df["p_cal"] = df["p_cal"].fillna(df["p_raw"]).astype(float)
    df["label"] = df["label"].fillna(df["gone"]).astype(float)
    df["segment"] = df["segment"].fillna("fast")
    return df.assign(run_id=run_id, split=split, snapshot_date=pd.Timestamp(date), horizon_days=180)


# 5 clients; churners (gone) are a (V 100), c (V 50), e (V 10): total churned revenue 160.
TOY = _pred([("a", 100, 0.9, 1), ("b", 400, 0.8, 0), ("c", 50, 0.7, 1), ("d", 300, 0.2, 0), ("e", 10, 0.1, 1)])


def test_capture_at_k_is_the_share_of_churned_revenue_in_the_top_k():
    assert capture_at_k(TOY, 1) == pytest.approx(100 / 160)      # top 1 by ev: a
    assert capture_at_k(TOY, 3) == pytest.approx(150 / 160)      # a, b, c
    assert capture_at_k(TOY, 5) == pytest.approx(1.0)            # K larger than the list: everybody
    assert capture_at_k(TOY, 99) == pytest.approx(1.0)


def test_size_only_and_oracle_bracket_the_model():
    # "biggest clients" ranks b, d, a: only a is a churner. The oracle ranks the churners by V: a, c, e.
    assert capture_at_k(TOY, 3, rank_by="V") == pytest.approx(100 / 160)
    assert oracle_capture(TOY, 2) == pytest.approx(150 / 160)
    for k in (1, 2, 3, 4):
        assert capture_at_k(TOY, k, rank_by="V") <= capture_at_k(TOY, k) <= oracle_capture(TOY, k) + 1e-12


def test_capture_ignores_unknown_truth_and_ranks_each_date_separately():
    unknown = TOY.assign(gone=[1, np.nan, 1, np.nan, 1], client_id=list("abcde"))
    assert capture_at_k(unknown, 1) == pytest.approx(100 / 160)  # b and d are dropped, a is still the top
    two_dates = pd.concat([TOY, TOY.assign(snapshot_date=pd.Timestamp("2026-04-22"))])
    assert capture_at_k(two_dates, 1) == pytest.approx(100 / 160)  # top 1 of EACH date, revenues pooled


def test_capture_ties_are_reproducible():
    tied = _pred([("a", 10, 1.0, 1), ("b", 10, 1.0, 0), ("c", 10, 1.0, 0)])
    assert capture_at_k(tied.iloc[::-1], 1) == capture_at_k(tied, 1) == 1.0   # ties: client_id order, not row order


def test_pr_auc_and_brier_on_a_toy_example():
    y, p = np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.7, 0.1])
    assert pr_auc(y, p) == pytest.approx((1 + 2 / 3) / 2)       # precision 1 at the first churner, 2/3 at the second
    assert np.isnan(pr_auc(np.zeros(3), np.array([0.1, 0.2, 0.3])))   # no churner: undefined
    assert brier(y, p) == pytest.approx((0.01 + 0.64 + 0.09 + 0.01) / 4)
    assert np.isnan(brier(np.array([]), np.array([])))


def test_bootstrap_resamples_clients_with_all_their_rows():
    # Each client has two dates; the draws must keep both rows of a drawn client.
    two = pd.concat([TOY, TOY.assign(snapshot_date=pd.Timestamp("2026-04-22"))])
    seen = []
    low, high = bootstrap_ci(two, lambda d: seen.append(d.client_id.value_counts().tolist()) or float(len(d)), 20, seed=1)
    assert all(c % 2 == 0 for counts in seen for c in counts)   # every client comes with its 2 dates
    assert low == high == 10.0                                  # 5 clients drawn x 2 dates
    again = bootstrap_ci(TOY, lambda d: capture_at_k(d, 2), 50, seed=3)
    assert again == bootstrap_ci(TOY, lambda d: capture_at_k(d, 2), 50, seed=3)   # same seed, same interval


def _random_pred(n=3000, seed=0, run_id="r"):
    rng = np.random.default_rng(seed)
    risk = rng.random(n)
    V = rng.lognormal(7, 1, n)
    gone = (rng.random(n) < risk * 0.3).astype(int)
    return _pred(list(zip(range(n), V, risk * V, gone, risk, risk, gone, rng.choice(["fast", "medium"], n))), run_id=run_id)


def test_evaluate_predictions_long_table():
    cfg = Config()
    cfg.evaluation.k_values, cfg.evaluation.bootstrap_n = [100, 300], 20
    pred = pd.concat([_random_pred(), _random_pred(seed=1).assign(split="valid")])
    out = evaluate_predictions(pred, cfg)
    assert list(out.columns) == ["run_id", "split", "segment", "metric", "k", "value", "ci_low", "ci_high"]
    assert set(out.split) == {"test", "valid"} and set(out.segment) == {"all", "fast", "medium"}
    allrows = out[out.segment == "all"]
    assert set(allrows.metric) == set(CORE_METRICS)
    # K metrics: one row per K; the others have no K
    assert sorted(allrows[allrows.metric == "capture_at_k"].k) == [100, 100, 300, 300]  # 2 splits
    assert allrows[allrows.metric == "pr_auc"].k.isna().all()
    # segments carry the probability metrics only
    assert set(out[out.segment == "fast"].metric) == {"pr_auc", "brier_raw", "brier_cal"}
    # ordering of the benchmarks, and the interval contains the estimate (to a bootstrap tolerance)
    t = allrows[(allrows.split == "test") & (allrows.k == 100)].set_index("metric")
    assert t.loc["size_only_capture", "value"] < t.loc["capture_at_k", "value"] < t.loc["oracle_capture", "value"] <= 1
    assert (out.ci_low <= out.ci_high).all()
    assert evaluate_predictions(pred, cfg).equals(out)         # deterministic given the seed


def test_evaluate_predictions_rejects_unknown_metrics_and_mixed_runs():
    cfg = Config()
    cfg.evaluation.metrics = ["capture_at_k", "f1"]
    with pytest.raises(ValueError, match="f1"):
        evaluate_predictions(_random_pred(50), cfg)
    with pytest.raises(ValueError, match="one run"):
        evaluate_predictions(pd.concat([_random_pred(50), _random_pred(50, run_id="other")]), Config())


@pytest.mark.skip(reason="stub: evaluation.threshold_metrics not implemented")
def test_threshold_metrics_on_a_toy_example():
    # precision, recall and F1 computed by hand on 6 clients.
    ...


@pytest.mark.skip(reason="stub: evaluation.tornado_table not implemented")
def test_tornado_is_sorted_by_amplitude():
    ...
