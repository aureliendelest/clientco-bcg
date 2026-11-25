"""Segments, dates, folds and cache of the snapshots (labels and leakage rules: test_logic.py, test_no_leakage.py)."""

import duckdb
import numpy as np
import pandas as pd
import pytest

import churn.snapshots as snapshots
from churn.config import Config, load_config
from churn.data import PURCHASE_DAY_MIN_NET, has_feature_tables
from churn.snapshots import (
    EXTENDED_FEATURES,
    FEATURE_SETS,
    MIN_BRANCH_DAYS,
    assign_segments,
    build_snapshot,
    cache_key,
    client_group_folds,
    in_perimeter,
    load_or_build,
    rhythm_class,
    snapshot_dates,
    value_segment,
)
from tests.helpers import T, before, every, sales_for
from tests.helpers_features import features, line, lines, make_con


def test_rhythm_classes_and_perimeter(sales_con):
    con = sales_con(sales_for({
        "fast": before(10, 5),         # median gap 5
        "medium": before(6, 20),       # median gap 20
        "slow": before(6, 35),         # 6 purchases but median gap 35 > 30
        "short": before(4, 10),        # fewer than 6 purchases
        "limit_fast": before(6, 7),    # 7 = fast_max: still fast
        "limit_medium": before(6, 30), # 30 = max_median_gap_days: still in the perimeter
    }))
    snap = build_snapshot(con, T, Config()).set_index("client_id")
    ids = sales_for.ids
    got = {n: snap.loc[i, "rhythm"] for n, i in ids.items()}
    assert got == {"fast": "fast", "medium": "medium", "slow": "slow", "short": "short_history",
                   "limit_fast": "fast", "limit_medium": "medium"}
    assert in_perimeter(snap.reset_index()).sum() == 4
    assert dict(zip(snap.index, rhythm_class(snap.reset_index().assign(freq_180d=snap.freq_180d.values), Config()))) \
        == {i: got[n] for n, i in ids.items()}  # rhythm_class alone gives the same classes as the snapshot


def test_segment_by_frequency_is_the_rhythm_class_and_by_value_is_a_decile():
    df = pd.DataFrame({"freq_180d": [20, 8, 3], "gap_median": [5.0, 20.0, np.nan], "rev_180d": [10.0, 20.0, 30.0]})
    assert list(assign_segments(df, Config()).segment) == ["fast", "medium", "short_history"]
    cfg = Config()
    cfg.model.segment_by = "value"
    big = pd.DataFrame({"rev_180d": np.arange(1.0, 101.0), "snapshot_date": pd.Timestamp("2025-04-15")})
    seg = assign_segments(big, cfg).segment
    assert seg.iloc[0] == "d01" and seg.iloc[-1] == "d10" and (seg.value_counts() == 10).all()
    two_dates = pd.concat([big, big.assign(snapshot_date=pd.Timestamp("2025-05-15"), rev_180d=big.rev_180d * 1000)])
    assert (value_segment(two_dates).value_counts() == 20).all()  # deciles are computed within each date


def test_snapshot_dates_are_monthly_on_the_22nd_then_the_test_date():
    cfg = Config()
    dates = snapshot_dates(cfg)
    assert len(dates) == 8 and dates[0].isoformat() == "2025-03-22" and dates[-2].isoformat() == "2025-09-22"
    assert dates[-1].isoformat() == "2026-03-22"
    assert len(snapshot_dates(cfg, last="2026-03-22")) == 13  # backtest grid: the test date is not repeated
    cfg.snapshots.step = "weekly"
    weekly = snapshot_dates(cfg)
    assert weekly[1] - weekly[0] == pd.Timedelta(weeks=1).to_pytimedelta() and weekly[-1].isoformat() == "2026-03-22"


def test_group_folds_keep_a_client_in_one_fold_and_do_not_move_it_when_rows_are_added():
    df = pd.DataFrame({"client_id": np.repeat(np.arange(200), 3)})
    folds = list(client_group_folds(df, 5))
    assert len(folds) == 5
    seen = set()
    for train, valid in folds:
        assert set(df.client_id.iloc[train]).isdisjoint(df.client_id.iloc[valid])
        seen |= set(valid)
    assert seen == set(range(len(df)))  # every row is validated exactly once
    bigger = pd.DataFrame({"client_id": np.concatenate([df.client_id.to_numpy(), np.arange(150, 400)])})
    fold_of = lambda d: {c: k for k, (_, v) in enumerate(client_group_folds(d, 5)) for c in d.client_id.iloc[v]}  # noqa: E731
    small, large = fold_of(df), fold_of(bigger)
    assert all(small[c] == large[c] for c in small)  # the fold of a client does not depend on the other rows
    assert list(client_group_folds(df, 5, seed=1))[0][1].tolist() != folds[0][1].tolist()
    with pytest.raises(ValueError):
        list(client_group_folds(df, 1))


def test_cache_key_depends_on_what_changes_a_snapshot_only():
    base = load_config("configs/experiments/hgb.yaml")
    key = cache_key(base, "fp")
    other_model = load_config("configs/experiments/xgb.yaml")
    assert cache_key(other_model, "fp") == key                    # the model does not change the snapshots
    shorter = load_config("configs/experiments/hgb.yaml")
    shorter.snapshots.train_end = "2025-06-22"
    assert cache_key(shorter, "fp") == key                        # nor does the training window
    assert cache_key(load_config("configs/experiments/hgb_label_b.yaml"), "fp") != key
    assert cache_key(base, "other-data") != key
    changed = load_config("configs/experiments/hgb.yaml")
    changed.label.horizon_days = 120
    assert cache_key(changed, "fp") != key
    unused = load_config("configs/experiments/hgb.yaml")
    unused.label.k = 5.0                                          # label A does not read k
    assert cache_key(unused, "fp") == key


def test_load_or_build_builds_each_date_once(tmp_path, monkeypatch):
    purchases = [(c, d) for c in (1, 2) for d in every("2024-09-22", "2026-09-22", 5)]
    df = pd.DataFrame(purchases, columns=["client_id", "date_order"])
    df["date_order"] = pd.to_datetime(df.date_order).dt.date
    df["date_invoice"], df["product_id"], df["quantity"], df["order_channel"], df["branch_id"], df["sales_net"] = (
        df.date_order, 1, 1.0, "online", 1, 50.0)
    parquet = tmp_path / "sales_shifted.parquet"
    duckdb.connect().register("df", df).execute(f"COPY df TO '{parquet}' (FORMAT parquet)")
    monkeypatch.setattr(snapshots, "PROCESSED_DIR", tmp_path / "processed")
    cfg = Config()
    cfg.dataset.shifted_parquet = str(parquet)
    calls = []
    real = snapshots.build_snapshot
    monkeypatch.setattr(snapshots, "build_snapshot", lambda con, d, c: calls.append(d) or real(con, d, c))
    first = load_or_build(cfg)
    assert len(calls) == 8 and len(first) == 16 and first.snapshot_date.nunique() == 8
    again = load_or_build(cfg)
    assert len(calls) == 8 and again.equals(first)                # second call reads the cache only
    load_or_build(cfg, last="2025-10-22")
    assert len(calls) == 9                                        # only the new date is built



# ------------------------------------------------------------------------------ extended features
# Toy lines with a product, a channel and a branch each (`tests/helpers_features.py`); dates are offsets in days
# before T: the last half is offsets 1..90, the previous half 91..180, offset 0 and below is not history.


def test_feature_sets_contract():
    basic, market, extended = FEATURE_SETS["basic"], FEATURE_SETS["market_only"], FEATURE_SETS["extended"]
    assert basic == [
        "recency_days", "freq_180d", "freq_90d", "freq_prev_90d", "rev_180d", "rev_90d", "rev_prev_90d",
        "gap_median", "gap_p90", "recency_ratio", "freq_trend", "rev_trend",
    ]  # unchanged: the cache of `basic` stays valid
    assert market == basic[:10] + ["mkt_freq_trend", "mkt_rev_trend"]  # same positions as the raw trends
    assert len(extended) == 17 and len(set(extended)) == 17
    assert extended[:10] == basic[:10]
    assert extended[10:] == [
        "mkt_freq_trend", "mkt_rev_trend", "negative_lines_share_180d", "product_div_trend", "product_loss_share",
        "channel_shift", "branch_freq_trend",
    ]
    for fs in (market, extended):  # the raw and the market-adjusted trend never go together
        assert not {"freq_trend", "rev_trend"} & set(fs)
    assert set(EXTENDED_FEATURES) == set(extended[10:])


def test_snapshot_version_and_cache_keys_keep_basic_valid():
    assert snapshots.SNAPSHOT_VERSION == 2  # a bump would invalidate the cache of `basic`
    cfg = load_config("configs/experiments/hgb.yaml")
    keys = {}
    for fs in ("basic", "market_only", "extended"):
        cfg.snapshots.feature_set = fs
        keys[fs] = cache_key(cfg, "fp")
    assert len(set(keys.values())) == 3
    assert keys["basic"] == cache_key(load_config("configs/experiments/hgb.yaml"), "fp")


def test_basic_does_not_touch_the_extended_tables():
    # `basic` only needs `cd` / `bn`: it works, and stays cheap, on a connection without the feature tables.
    con = make_con(lines(1, range(1, 60, 5)))
    snap = build_snapshot(con, T, Config())
    assert list(snap.columns[:14]) == ["client_id", "snapshot_date", *FEATURE_SETS["basic"]]
    assert not has_feature_tables(con)
    cfg = Config()
    cfg.snapshots.feature_set = "extended"
    ext = build_snapshot(con, T, cfg)
    assert has_feature_tables(con)  # built on demand
    assert list(ext.columns[:19]) == ["client_id", "snapshot_date", *FEATURE_SETS["extended"]]
    shared = [c for c in snap.columns if c in ext.columns]
    pd.testing.assert_frame_equal(snap[shared], ext[shared])  # the columns they share hold the same values


def test_feature_tables_follow_the_purchase_day_definition():
    rows = [
        line(1, 10, 100.0, product=1), line(1, 10, -30.0, product=2, branch=2),  # purchase day, two branches
        line(1, 20, 50.0, product=3), line(1, 20, -50.0, product=3),              # returned the same day: no purchase
        line(1, 30, 5.0, product=4, channel="by phone"),
    ]
    con = make_con(rows, with_feature_tables=True)
    assert PURCHASE_DAY_MIN_NET == 0.01
    q = lambda sql: con.execute(sql).fetchall()  # noqa: E731
    assert q("SELECT n_days, rev FROM act_day ORDER BY d") == [(1, 5.0), (1, 70.0)]
    assert sorted(q("SELECT pos, neg FROM cd_lines"), key=lambda r: r[0]) == [(5.0, None), (50.0, -50.0), (100.0, -30.0)]
    assert sorted(q("SELECT product_id, rev FROM cpd")) == [(1, 100.0), (2, -30.0), (4, 5.0)]  # purchase days only
    assert sorted(q("SELECT channel, rev FROM ccd")) == [("by phone", 5.0), ("online", 70.0)]
    # the branch table applies the rule at branch level: branch 2 netted -30 that day, so it is no branch purchase day
    assert sorted(q("SELECT branch_id, rev FROM cbd")) == [(1, 5.0), (1, 100.0)]


def test_market_adjusted_trends_replace_the_raw_trends():
    rows = (
        lines(1, [5, 10, 20, 100]) + lines(1, [181, 0, -5])  # outside the history: before the lookback, day T, the future
        + [line(1, 30, 100.0), line(1, 30, -100.0, product=2)]  # bought and returned the same day: no purchase day
        + lines(2, [7, 90]) + lines(2, [91, 120, 180]) + lines(2, [181])  # offsets 90 and 180 are inside
    )
    ext = features(rows)
    raw = features(rows, "basic")
    assert not {"freq_trend", "rev_trend"} & set(ext.columns)
    # market, ALL clients: 5 purchase days (500 EUR) in the last half, 4 (400 EUR) in the previous half
    mkt_f, mkt_r = np.log(5) - np.log(4), np.log(500) - np.log(400)
    assert raw.loc[1, "freq_trend"] == pytest.approx(np.log1p(3) - np.log1p(1))
    assert ext.loc[1, "mkt_freq_trend"] == pytest.approx(np.log1p(3) - np.log1p(1) - mkt_f)
    assert ext.loc[2, "mkt_freq_trend"] == pytest.approx(np.log1p(2) - np.log1p(3) - mkt_f)
    assert ext.loc[1, "mkt_rev_trend"] == pytest.approx(np.log1p(300) - np.log1p(100) - mkt_r)
    assert ext.loc[2, "mkt_rev_trend"] == pytest.approx(np.log1p(200) - np.log1p(300) - mkt_r)
    # the adjustment is one constant for the whole date
    assert np.ptp(raw["freq_trend"] - ext["mkt_freq_trend"]) < 1e-12
    assert np.ptp(raw["rev_trend"] - ext["mkt_rev_trend"]) < 1e-12


def test_negative_lines_share():
    rows = (
        [line(1, 10, 100.0), line(1, 10, -20.0, product=2)]  # purchase day with a line of -20
        + [line(1, 30, 100.0), line(1, 40, -50.0)]           # a credit note on a day without purchase still counts
        + [line(1, 200, -1000.0), line(1, 0, -999.0), line(1, -3, -999.0)]  # outside the history: ignored
        + lines(2, [10, 20])                                  # no negative line at all
    )
    ext = features(rows)
    assert ext.loc[1, "negative_lines_share_180d"] == pytest.approx(70 / 200)  # |negatives| / positives
    assert ext.loc[2, "negative_lines_share_180d"] == 0.0


def test_product_diversity_trend():
    rows = (
        [line(1, 5, 100.0, product=1), line(1, 5, 100.0, product=2), line(1, 10, 100.0, product=3)]
        + [line(1, 15, 100.0, product=4)]                                   # product 4 is bought...
        + [line(1, 20, -100.0, product=4), line(1, 20, 200.0, product=1)]   # ...and returned: no revenue left, not counted
        + [line(1, 100, 100.0, product=1), line(1, 120, 100.0, product=1), line(1, 120, 50.0, product=8)]
        + [line(1, 3, 100.0, product=77), line(1, -2, 100.0, product=78)]   # 77 counts (offset 3), 78 is the future
    )
    ext = features(rows)
    assert ext.loc[1, "product_div_trend"] == pytest.approx(np.log1p(4) - np.log1p(2))  # {1,2,3,77} against {1,8}


def test_product_loss_share_and_its_nan_rules():
    rows = (
        [line(1, 100, 100.0, product=1), line(1, 110, 300.0, product=2), line(1, 120, 100.0, product=3)]  # previous: 500
        + [line(1, 10, 100.0, product=1), line(1, 20, 50.0, product=9)]    # last half: only product 1 is bought again
        + lines(2, [100, 110, 120])                                        # no purchase in the last half
        + lines(3, [10, 20])                                               # no purchase in the previous half
        + lines(4, [10, 100], product=5)                                   # everything bought again
    )
    ext = features(rows)
    assert ext.loc[1, "product_loss_share"] == pytest.approx(400 / 500)  # revenue of products 2 and 3
    assert np.isnan(ext.loc[2, "product_loss_share"]) and np.isnan(ext.loc[3, "product_loss_share"])
    assert ext.loc[4, "product_loss_share"] == 0.0
    # nothing bought in the last half is not a NaN rule for the diversity trend
    assert ext.loc[2, "product_div_trend"] == pytest.approx(np.log1p(0) - np.log1p(1))


def test_channel_shift_and_its_nan_rules():
    store, phone, online = "at the store", "by phone", "online"
    rows = (
        [line(1, 100, 300.0, channel=store), line(1, 110, 100.0, channel=phone)]   # previous: dominant channel = store
        + [line(1, 10, 100.0, channel=store), line(1, 20, 300.0, channel=phone)]   # last half: 25 % in store
        + [line(2, 100, 100.0, channel=store), line(2, 10, 100.0, channel=store)]  # same channel: 0
        + [line(3, 100, 100.0, channel=store), line(3, 10, 100.0, channel=online)]  # all moved: 1
        + [line(4, 100, 100.0, channel=phone), line(4, 110, 100.0, channel=store), line(4, 10, 100.0, channel=store)]  # tie
        + lines(5, [10, 20], channel=store)                                          # no previous purchase
        + lines(6, [100, 110], channel=store)                                        # no purchase in the last half
    )
    ext = features(rows)
    assert ext.loc[1, "channel_shift"] == pytest.approx(1 - 100 / 400)
    assert ext.loc[2, "channel_shift"] == 0.0
    assert ext.loc[3, "channel_shift"] == 1.0
    assert ext.loc[4, "channel_shift"] == 0.0  # tie in the previous half: the first channel by name ("at the store")
    assert np.isnan(ext.loc[5, "channel_shift"])
    assert np.isnan(ext.loc[6, "channel_shift"])


def _crowd(client: int, branch: int, recent: int, previous: int) -> list[tuple]:
    """A client buying every day in the last half (`recent` days) and in the previous half (`previous` days) at one branch."""
    return lines(client, range(1, recent + 1), branch=branch) + lines(client, range(91, 91 + previous), branch=branch)


def test_branch_frequency_trend_is_leave_one_out_with_a_minimum_of_purchase_days():
    rows = (
        _crowd(90, 7, 40, 35)                                                   # branch 7: 40 and 35 days
        + lines(20, [5, 10, 15, 20, 25], branch=7) + lines(20, [100, 105], branch=7) + [line(20, 30, branch=8)]
        + lines(21, [3, 6, 9], 10.0, branch=7) + lines(21, [95, 99], 10.0, branch=7)  # many small days at branch 7...
        + [line(21, 12, 500.0, branch=8), line(21, 130, 500.0, branch=8)]             # ...but the revenue is at branch 8
        + _crowd(91, 8, 30, 30)
        + _crowd(92, 9, 29, 30) + lines(22, [50, 150], branch=9)                # 29 days in the last half: below 30
        + _crowd(93, 10, 40, 29) + lines(23, [50, 150], branch=10)              # 29 days in the previous half: below 30
    )
    ext = features(rows)
    assert MIN_BRANCH_DAYS == 30
    # client 20: main branch 7, with 40 + 5 + 3 = 48 days in the last half and 35 + 2 + 2 = 39 before; without his own 43 and 37
    assert ext.loc[20, "branch_freq_trend"] == pytest.approx(np.log(43) - np.log(37))
    assert ext.loc[20, "branch_freq_trend"] != pytest.approx(np.log(48) - np.log(39))  # he is not in his own branch
    # client 21: main branch = highest revenue (8), not the most frequent (7). Branch 8 has 30 + 1 + 1 days in the
    # last half and 30 + 0 + 1 before; without his own 31 and 30, the minimum is reached exactly
    assert ext.loc[21, "branch_freq_trend"] == pytest.approx(np.log(31) - np.log(30))
    assert np.isnan(ext.loc[90, "branch_freq_trend"])  # without his own days his branch has 8 and 4 days
    assert np.isnan(ext.loc[22, "branch_freq_trend"])  # 29 days in the last half
    assert np.isnan(ext.loc[23, "branch_freq_trend"])  # 29 days in the previous half
