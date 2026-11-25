"""Simulations on toy predictions computed by hand: value definitions (#7) and lists with rest (#9).

#7 toy (date 2025-10-22, 6 clients; V_recent = rev_90d x 4, ev = p_cal x V):

    id   V     V_recent  gone  p_cal  declining  ev     p_cal x V_recent
    1    1000  200       1     0.5    yes        500    100
    2     400  400       1     0.6    no         240    240
    3     300  300       0     0.9    no         270    270
    4     100    0       1     0.2    yes         20      0
    5     800  800       0     0.1    no          80     80
    6      50  100       1     0.3    no          15     30

Churned V: 1000 + 400 + 100 + 50 = 1550. Churned V_recent: 200 + 400 + 0 + 100 = 700.
Ranking by ev: 1, 3, 2, 5, 4, 6. Ranking by p_cal x V_recent: 3, 2, 1, 5, 6, 4.

#9 toy (K = 2, clients 1 to 4, ev always 1 > 2 > 3 > 4), V and gone by date:

    date   V (1,2,3,4)        gone (1,2,3,4)
    d1     100, 80, 60, 40    1, 0, 1, 1
    d2     110, 90, 60, 50    1, 1, 0, 1
    d3     120, 100, 70, 60   0, 1, 1, 0
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn import evaluation, simulation
from churn.simulation import cooldown_lists, load_backtest, value_alternatives

D1, D2, D3 = "2025-09-22", "2025-10-22", "2025-11-22"


# ------------------------------------------------------------------------------ #7 toy data


def _toy(date="2025-10-22", p=(0.5, 0.6, 0.9, 0.2, 0.1, 0.3)):
    v = np.array([1000, 400, 300, 100, 800, 50], float)
    rev90 = np.array([50, 100, 75, 0, 200, 25], float)
    gone = np.array([1, 1, 0, 1, 0, 1], float)
    f90 = np.array([1, 3, 3, 0, 2, 4])
    fprev = np.array([3, 3, 2, 2, 2, 2])
    p = np.array(p)
    ids = np.arange(1, 7)
    pred = pd.DataFrame(
        {"client_id": ids, "snapshot_date": pd.Timestamp(date), "p_cal": p, "V": v, "ev": p * v, "gone": gone}
    )
    snap = pd.DataFrame(
        {"client_id": ids, "snapshot_date": pd.Timestamp(date), "V": v, "rev_90d": rev90, "freq_90d": f90,
         "freq_prev_90d": fprev}
    )
    return pred, snap


def _cap(table, date, k, ranking, loss, col="capture"):
    row = table[(table["date"] == date) & (table["k"] == k) & (table["ranking"] == ranking) & (table["loss"] == loss)]
    assert len(row) == 1
    return float(row[col].iloc[0])


def test_value_alternatives_by_hand_k2():
    cap, _ = value_alternatives(*_toy(), k_values=(2, 3))
    d = "2025-10-22"
    # current ranking {1, 3}: V churned 1000 / 1550, V_recent churned 200 / 700
    assert _cap(cap, d, 2, "V", "V") == pytest.approx(1000 / 1550, abs=1e-12)
    assert _cap(cap, d, 2, "V", "V_recent") == pytest.approx(200 / 700, abs=1e-12)
    # recent ranking {3, 2}: client 2 only
    assert _cap(cap, d, 2, "V_recent", "V") == pytest.approx(400 / 1550, abs=1e-12)
    assert _cap(cap, d, 2, "V_recent", "V_recent") == pytest.approx(400 / 700, abs=1e-12)
    # numerators and denominators are exposed
    assert _cap(cap, d, 2, "V", "V", "captured") == 1000
    assert _cap(cap, d, 2, "V", "V", "total") == 1550
    assert _cap(cap, d, 2, "V", "V_recent", "total") == 700
    # K = 3: both rankings take {1, 2, 3}
    for rank in ("V", "V_recent"):
        assert _cap(cap, d, 3, rank, "V") == pytest.approx(1400 / 1550, abs=1e-12)
        assert _cap(cap, d, 3, rank, "V_recent") == pytest.approx(600 / 700, abs=1e-12)


def test_value_alternatives_k_larger_than_perimeter_captures_everything():
    cap, _ = value_alternatives(*_toy(), k_values=(100,))
    d = cap[cap["date"] == "2025-10-22"]
    assert np.allclose(d["capture"], 1.0)


def test_current_ranking_uses_the_ev_column():
    pred, snap = _toy()
    pred["ev"] = [1, 2, 3, 9, 8, 7]  # ranking 4, 5, 6 first: only client 4 and 6 churn, V 100 + 50
    cap, _ = value_alternatives(pred, snap, k_values=(3,))
    assert _cap(cap, "2025-10-22", 3, "V", "V") == pytest.approx(150 / 1550)


def test_overestimation_by_hand():
    _, over = value_alternatives(*_toy(), k_values=(2,))
    o = over[over["date"] == "2025-10-22"].set_index("group")
    # churners 1, 2, 4, 6: V / V_recent = 5, 1, undefined (V_recent = 0), 0.5
    a = o.loc["all"]
    assert a["n_churners"] == 4 and a["share_of_churners"] == 1 and a["share_of_lost_v"] == pytest.approx(1)
    assert a["median_ratio"] == pytest.approx(1.0) and a["mean_ratio"] == pytest.approx(6.5 / 3)
    assert a["ratio_of_sums"] == pytest.approx(1550 / 700) and a["n_v_recent_zero"] == 1
    # declining churners: 1 (ratio 5) and 4 (V_recent = 0)
    dcl = o.loc["declining"]
    assert dcl["n_churners"] == 2 and dcl["share_of_churners"] == pytest.approx(0.5)
    assert dcl["share_of_lost_v"] == pytest.approx(1100 / 1550)
    assert dcl["median_ratio"] == pytest.approx(5.0) and dcl["mean_ratio"] == pytest.approx(5.0)
    assert dcl["ratio_of_sums"] == pytest.approx(1100 / 200) and dcl["n_v_recent_zero"] == 1
    nd = o.loc["not_declining"]
    assert nd["median_ratio"] == pytest.approx(0.75) and nd["share_of_lost_v"] == pytest.approx(450 / 1550)


def test_v_recent_zero_everywhere_gives_nan_not_an_error():
    pred, snap = _toy()
    snap["rev_90d"] = 0.0
    cap, over = value_alternatives(pred, snap, k_values=(2,))
    assert np.isnan(_cap(cap, "2025-10-22", 2, "V", "V_recent"))  # nothing lost under V_recent
    assert np.isnan(_cap(cap, "2025-10-22", 2, "V_recent", "V_recent"))
    assert _cap(cap, "2025-10-22", 2, "V", "V") == pytest.approx(1000 / 1550)  # the V loss is unaffected
    a = over[(over["date"] == "2025-10-22") & (over["group"] == "all")].iloc[0]
    assert np.isnan(a["median_ratio"]) and np.isnan(a["mean_ratio"]) and np.isnan(a["ratio_of_sums"])
    assert a["n_v_recent_zero"] == 4


def test_no_churner_gives_nan():
    pred, snap = _toy()
    pred["gone"] = 0.0
    cap, over = value_alternatives(pred, snap, k_values=(2,))
    assert cap["capture"].isna().all()
    a = over[(over["date"] == "2025-10-22") & (over["group"] == "all")].iloc[0]
    assert a["n_churners"] == 0 and np.isnan(a["share_of_churners"]) and np.isnan(a["share_of_lost_v"])


def test_unknown_truth_is_ignored():
    pred, snap = _toy()
    extra_p = pd.DataFrame(
        {"client_id": [7], "snapshot_date": pd.Timestamp("2025-10-22"), "p_cal": [0.99], "V": [5000.0],
         "ev": [4950.0], "gone": [np.nan]}
    )
    extra_s = pd.DataFrame(
        {"client_id": [7], "snapshot_date": pd.Timestamp("2025-10-22"), "V": [5000.0], "rev_90d": [10.0],
         "freq_90d": [1], "freq_prev_90d": [2]}
    )
    cap0, over0 = value_alternatives(pred, snap, k_values=(2,))
    cap1, over1 = value_alternatives(pd.concat([pred, extra_p]), pd.concat([snap, extra_s]), k_values=(2,))
    pd.testing.assert_frame_equal(cap0, cap1)
    pd.testing.assert_frame_equal(over0, over1)


def test_mean_excludes_the_first_date():
    p_other = (0.01, 0.02, 0.3, 0.4, 0.5, 0.6)  # a different ranking on 2025-09-22
    parts = [_toy(D1, p_other), _toy("2025-10-22"), _toy(D3)]
    pred = pd.concat([x[0] for x in parts])
    snap = pd.concat([x[1] for x in parts])
    cap, over = value_alternatives(pred, snap, k_values=(2,))
    first = _cap(cap, D1, 2, "V", "V")
    assert first != pytest.approx(1000 / 1550)  # the first date does differ
    m = cap[(cap["date"] == "mean") & (cap["ranking"] == "V") & (cap["loss"] == "V")]
    assert len(m) == 1 and m["capture"].iloc[0] == pytest.approx(1000 / 1550) and m["n_dates"].iloc[0] == 2
    assert np.isnan(m["captured"].iloc[0])
    # the dates stay in the table, the mean is a summary row
    assert set(cap["date"]) == {D1, "2025-10-22", D3, "mean"}
    mo = over[(over["date"] == "mean") & (over["group"] == "declining")].iloc[0]
    assert mo["median_ratio"] == pytest.approx(5.0) and mo["n_dates"] == 2
    # nothing excluded: the mean moves
    cap_all, _ = value_alternatives(pred, snap, k_values=(2,), excluded_dates=())
    m_all = cap_all[(cap_all["date"] == "mean") & (cap_all["ranking"] == "V") & (cap_all["loss"] == "V")]
    assert m_all["capture"].iloc[0] == pytest.approx((first + 2 * 1000 / 1550) / 3)


def test_inconsistent_inputs_raise():
    pred, snap = _toy()
    with pytest.raises(ValueError, match="V differs"):
        value_alternatives(pred, snap.assign(V=snap["V"] + 1))
    with pytest.raises(ValueError, match="missing from the snapshot"):
        value_alternatives(pred, snap.iloc[:-1])
    with pytest.raises(ValueError, match="miss columns"):
        value_alternatives(pred.drop(columns="p_cal"), snap)
    with pytest.raises(ValueError, match="Duplicate"):
        value_alternatives(pd.concat([pred, pred]), snap)


def test_final_test_date_is_refused():
    pred, snap = _toy("2026-03-22")
    with pytest.raises(ValueError, match="final test date"):
        value_alternatives(pred, snap)
    with pytest.raises(ValueError, match="final test date"):
        cooldown_lists({"2026-03-22": _lists_frames()[D1]})
    with pytest.raises(ValueError, match="final test date"):
        load_backtest(("2025-12-22", "2026-03-22"))


# ------------------------------------------------------------------------------ #9 toy data


def _frame(v, gone):
    return pd.DataFrame({"client_id": [1, 2, 3, 4], "V": v, "gone": gone, "ev": [4.0, 3.0, 2.0, 1.0]})


def _lists_frames():
    return {
        D1: _frame([100.0, 80, 60, 40], [1.0, 0, 1, 1]),
        D2: _frame([110.0, 90, 60, 50], [1.0, 1, 0, 1]),
        D3: _frame([120.0, 100, 70, 60], [0.0, 1, 1, 0]),
    }


def _rows(table, rest):
    return table[table["rest_lists"] == rest].reset_index(drop=True)


def test_rest_excludes_the_clients_of_the_previous_list_by_hand():
    t = cooldown_lists(_lists_frames(), k=2, rest_lists=1)
    assert set(t["rest_lists"]) == {0, 1}
    r = _rows(t, 1)
    assert list(r["date"]) == [D1, D2, D3] and list(r["list"]) == [1, 2, 3]
    assert list(r["n_listed"]) == [2, 2, 2] and list(r["n_banned_present"]) == [0, 2, 2]
    # lists: {1, 2}, then {3, 4} (1 and 2 rest), then {1, 2} again (3 and 4 rest); rest of 1 list only
    assert list(r["n_already_listed_before"]) == [0, 0, 2]
    assert list(r["captured_v"]) == [100, 50, 100]
    assert list(r["den_date"]) == [200, 250, 170]
    assert r["capture_list"].to_numpy() == pytest.approx([100 / 200, 50 / 250, 100 / 170])
    # unique counting: client 1 at 100, client 4 at 50, client 2 at 100 (V at its first capture)
    assert list(r["captured_unique_cum"]) == [100, 150, 250]
    # unique churners of the horizon, V at the first gone: 1 -> 100, 3 -> 60, 4 -> 40, 2 -> 90 (d2): 290
    assert list(r["den_unique_all"]) == [290, 290, 290]
    assert r["capture_cum_unique"].to_numpy() == pytest.approx([100 / 290, 150 / 290, 250 / 290])
    assert list(r["den_unique_to_date"]) == [200, 290, 290]
    assert r["capture_cum_unique_to_date"].to_numpy() == pytest.approx([100 / 200, 150 / 290, 250 / 290])
    assert r["capture_cum_pooled"].to_numpy() == pytest.approx([100 / 200, 150 / 450, 250 / 620])


def test_no_rest_variant_counts_each_churner_once():
    r = _rows(cooldown_lists(_lists_frames(), k=2, rest_lists=3), 0)
    # {1, 2} every month; client 1 captured at d1 (V = 100), client 2 at d2 (V = 90)
    assert list(r["n_already_listed_before"]) == [0, 2, 2] and list(r["n_banned_present"]) == [0, 0, 0]
    assert list(r["captured_v"]) == [100, 200, 100]
    assert list(r["captured_unique_cum"]) == [100, 190, 190]
    assert r["capture_cum_unique"].to_numpy() == pytest.approx([100 / 290, 190 / 290, 190 / 290])
    # counting the repetitions would give more: 300 / 450 then 400 / 620
    assert r["capture_cum_pooled"].to_numpy() == pytest.approx([100 / 200, 300 / 450, 400 / 620])
    # with rest_lists = 0 asked, a single variant is returned
    assert set(cooldown_lists(_lists_frames(), k=2, rest_lists=0)["rest_lists"]) == {0}


def test_rest_lists_longer_than_the_clients_available_gives_a_short_list():
    r = _rows(cooldown_lists(_lists_frames(), k=2, rest_lists=2), 2)
    # lists {1, 2}, {3, 4}, then every client rests: empty list
    assert list(r["n_listed"]) == [2, 2, 0]
    assert r["captured_v"].iloc[2] == 0 and r["capture_list"].iloc[2] == 0
    assert list(r["captured_unique_cum"]) == [100, 150, 150]


def test_stacked_frame_input_gives_the_same_result():
    frames = _lists_frames()
    stacked = pd.concat([f.assign(snapshot_date=pd.Timestamp(d)) for d, f in frames.items()])
    pd.testing.assert_frame_equal(cooldown_lists(frames, 2, 1), cooldown_lists(stacked, 2, 1))
    reordered = dict(reversed(list(frames.items())))
    pd.testing.assert_frame_equal(cooldown_lists(frames, 2, 1), cooldown_lists(reordered, 2, 1))


def test_cooldown_without_any_churner_is_nan_not_an_error():
    frames = {d: f.assign(gone=0.0) for d, f in _lists_frames().items()}
    r = cooldown_lists(frames, k=2, rest_lists=1)
    for c in ("capture_list", "capture_cum_unique", "capture_cum_unique_to_date", "capture_cum_pooled"):
        assert r[c].isna().all()
    assert (r["captured_v"] == 0).all()


def test_cooldown_unknown_truth_ignored_and_ties_by_client_id():
    frames = _lists_frames()
    frames[D1] = frames[D1].assign(ev=1.0)  # all tied: the lowest client_id come first
    extra = pd.DataFrame({"client_id": [9], "V": [1e6], "gone": [np.nan], "ev": [99.0]})
    frames[D1] = pd.concat([frames[D1], extra])
    r = _rows(cooldown_lists(frames, k=2, rest_lists=1), 1)
    assert r["n_listed"].iloc[0] == 2 and r["captured_v"].iloc[0] == 100  # clients 1 and 2, client 9 ignored


def test_cooldown_bad_arguments():
    with pytest.raises(ValueError):
        cooldown_lists(_lists_frames(), k=0)
    with pytest.raises(ValueError):
        cooldown_lists(_lists_frames(), k=2, rest_lists=-1)
    with pytest.raises(ValueError, match="miss columns"):
        cooldown_lists({D1: _lists_frames()[D1].drop(columns="ev")}, k=2)


# ------------------------------------------------------------------------------ real data


def test_capture_matches_capture_at_k_of_the_run_on_real_data():
    try:
        preds, snaps = load_backtest(("2025-12-22",))
    except FileNotFoundError as e:  # runs or cache not available on this machine
        pytest.skip(str(e))
    pred = preds["2025-12-22"]
    cap, over = value_alternatives(pred, snaps["2025-12-22"], k_values=(700, 2000))
    for k in (700, 2000):
        expected = evaluation.capture_at_k(pred, k)
        got = _cap(cap, "2025-12-22", k, "V", "V")
        assert abs(got - expected) < 1e-9
    assert (cap["capture"].dropna().between(0, 1)).all()
    assert len(cap[cap["date"] == "2025-12-22"]) == 2 * 2 * 2  # 2 K x 2 rankings x 2 losses
    assert set(over["group"]) == {"all", "declining", "not_declining"}


def test_cooldown_on_real_data_is_consistent():
    try:
        preds, _ = load_backtest(("2025-11-22", "2025-12-22", "2026-01-22"))
    except FileNotFoundError as e:
        pytest.skip(str(e))
    r = cooldown_lists(preds, k=700, rest_lists=1)
    with_rest, no_rest = _rows(r, 1), _rows(r, 0)
    assert (with_rest["n_listed"] == 700).all()
    # a client listed on the previous date is never listed again with a rest of 1
    assert with_rest["n_banned_present"].iloc[1] > 0
    assert with_rest["capture_cum_unique"].is_monotonic_increasing and no_rest["capture_cum_unique"].is_monotonic_increasing
    # the first list is the same with and without rest
    assert with_rest["capture_list"].iloc[0] == no_rest["capture_list"].iloc[0]
    # the model's own list on the first date equals the metric of the run
    assert abs(with_rest["capture_list"].iloc[0] - evaluation.capture_at_k(preds["2025-11-22"], 700)) < 1e-9
    assert simulation.FORBIDDEN_DATE not in set(r["date"])
