"""Segments, dates, folds and cache of the snapshots (labels and leakage rules: test_logic.py, test_no_leakage.py)."""

import duckdb
import numpy as np
import pandas as pd
import pytest

import churn.snapshots as snapshots
from churn.config import Config, load_config
from churn.snapshots import (
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
