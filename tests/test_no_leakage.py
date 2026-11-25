"""Anti-leakage suite, to run before every sweep (`make test`).

These are the five rules from docs/06_engineering.md (section 5). Rules 3 and 5 run the pipeline on the small
synthetic dataset of `conftest.pipeline_cfg`.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from churn.config import Config
from churn.snapshots import (
    EXTENDED_FEATURES,
    FEATURE_SETS,
    assert_labels_known_before_test,
    build_snapshot,
    compute_features,
    load_or_build,
    temporal_split,
)
from tests.helpers import T, before, every, sales_for
from tests.helpers_features import features, line, lines, rich_rows


def test_features_on_truncated_data_are_identical(sales_con):
    # Rule 1: features of snapshot T computed on data truncated at T must be identical to the
    # features computed on the full data (no feature may look at dates >= T).
    named = {
        "regular": every("2025-01-01", "2025-12-31", 5),
        "returns_later": before(7, 9) + ["2025-04-15", "2025-04-20", "2025-09-01"],  # buys ON day T and after
        "stops": before(6, 12),
        "starts_at_T": ["2025-04-15", "2025-04-16"],  # no history yet: absent from the snapshot
    }
    full = compute_features(sales_con(sales_for(named)), T, "basic")
    truncated = {name: [d for d in dates if d < f"{T}"] for name, dates in named.items()}
    cut = compute_features(sales_con(sales_for(truncated)), T, "basic")
    pd.testing.assert_frame_equal(full, cut)
    assert set(full.client_id) == {sales_for.ids[n] for n in ("regular", "returns_later", "stops")}


def test_train_label_windows_end_before_test_date():
    # Rule 2: for every training snapshot, snapshot_date + Config.label_wait_days() <= test_date (180 for label A, h_max for label B).
    test = date(2026, 3, 22)
    assert_labels_known_before_test([date(2025, 3, 22), date(2025, 9, 22)], test, 180)
    assert_labels_known_before_test([date(2025, 9, 23)], test, 180)  # window [T, T+180) ends on 21/03: fine
    with pytest.raises(ValueError, match="see the future"):
        assert_labels_known_before_test([date(2025, 9, 24)], test, 180)
    assert_labels_known_before_test([date(2025, 12, 22)], test, 90)
    with pytest.raises(ValueError):
        assert_labels_known_before_test([date(2025, 12, 23)], test, 90)


def _fake_snapshots(dates: list[str]) -> pd.DataFrame:
    n = len(dates)
    return pd.DataFrame({
        "client_id": np.arange(n), "snapshot_date": pd.to_datetime(dates),
        "rhythm": "fast", "label": 0.0, "gone": 0.0,
    })


def test_temporal_split_refuses_a_training_window_that_reaches_the_test():
    cfg = Config()
    dates = ["2025-03-22", "2025-09-22", "2025-10-22", "2026-03-22"]
    train, test = temporal_split(_fake_snapshots(dates), cfg)  # derived train_end: 22/09/2025 for label A
    assert list(train.snapshot_date.dt.strftime("%Y-%m-%d")) == dates[:2] and len(test) == 1
    cfg.snapshots.train_end = "2025-10-22"  # forced: label A of 22/10 ends in April, after the test
    with pytest.raises(ValueError, match="see the future"):
        temporal_split(_fake_snapshots(dates), cfg)
    cfg.label.kind = "relative_horizon"  # label B waits only 90 days: 22/12 is the last allowed date
    cfg.snapshots.train_end = None
    assert cfg.last_train_date() == "2025-12-22"


def test_shuffled_labels_give_auc_near_half(pipeline_cfg):
    # Rule 3: a model trained on shuffled labels has nothing to say about the test snapshot. One shuffled model is a
    # random direction in a space where recency carries the whole signal, so its AUC is near 0.1 or 0.9 depending on
    # the sign of the noise: the expectation is 0.5 over many shuffles. A leak of the labels into the features or
    # the split would keep every shuffled AUC far above 0.5.
    import dataclasses

    from sklearn.metrics import roc_auc_score

    from churn.config import ModelCfg
    from churn.pipeline import fit_predict

    cfg = dataclasses.replace(pipeline_cfg, model=ModelCfg(type="logreg", calibration="none"))
    train, test = temporal_split(load_or_build(cfg), cfg)
    real = fit_predict(cfg, train, {"test": test}).predictions
    assert roc_auc_score(real["gone"], real["p_raw"]) > 0.8  # the synthetic signal is there...
    aucs = []
    for seed in range(30):
        shuffled = train.assign(label=np.random.default_rng(seed).permutation(train["label"].to_numpy()))
        fake = fit_predict(cfg, shuffled, {"test": test}).predictions
        aucs.append(roc_auc_score(fake["gone"], fake["p_raw"]))
    assert 0.35 < np.mean(aucs) < 0.65  # ...and it needs the true labels
    assert min(aucs) < 0.5 < max(aucs)


def test_rhythm_features_use_only_prior_purchases_and_need_min_purchases(sales_con):
    # Rule 4: gap_* and recency_ratio use purchases strictly before T, and are NaN below
    # label.min_purchases purchases (the class value is used instead).
    con = sales_con(sales_for({
        "five": before(5, 10), "six": before(6, 10),
        "on_day_T": before(6, 10) + ["2025-04-15"],  # the purchase of day T must not enter the features
    }))
    feats = compute_features(con, T, "basic").set_index("client_id")
    ids = sales_for.ids
    rhythm = ["gap_median", "gap_p90", "recency_ratio"]
    assert feats.loc[ids["five"], rhythm].isna().all()                 # 5 < min_purchases = 6
    assert feats.loc[ids["five"], ["recency_days", "freq_180d"]].notna().all()  # volume features stay
    assert feats.loc[ids["six"], rhythm].notna().all()
    assert feats.loc[ids["six"], "gap_median"] == 10
    pd.testing.assert_series_equal(feats.loc[ids["on_day_T"]], feats.loc[ids["six"]], check_names=False)


def test_calibration_is_fitted_without_test_data(pipeline_cfg):
    # Rule 5: the calibrator (like the model) never sees the test snapshot: changing the test labels, or adding other
    # targets to score, leaves the test predictions untouched.
    from churn.pipeline import fit_predict

    train, test = temporal_split(load_or_build(pipeline_cfg), pipeline_cfg)
    base = fit_predict(pipeline_cfg, train, {"test": test}).predictions
    rng = np.random.default_rng(1)
    altered = test.assign(label=rng.permutation(test["label"].to_numpy()), gone=rng.permutation(test["gone"].to_numpy()))
    other = fit_predict(pipeline_cfg, train, {"test": altered, "extra": test.head(50)}).predictions
    pd.testing.assert_frame_equal(
        base[["client_id", "p_raw", "p_cal"]].reset_index(drop=True),
        other.loc[other["split"] == "test", ["client_id", "p_raw", "p_cal"]].reset_index(drop=True),
    )


def _future_lines(scale: float) -> list[tuple]:
    """Lines dated T and later, wild on purpose: new products, channels and branches, returns, new clients, and
    amounts or places that change with `scale`."""
    rows = []
    for client in (1, 2, 3, 4, 50):  # 50 only exists from T on
        for offset in (0, -1, -10, -60):
            rows.append(line(client, offset, 1000.0 * scale, product=900 + int(scale), channel="other", branch=int(scale) + 20))
            rows.append(line(client, offset, -400.0 * scale, product=7, channel="by phone", branch=7))
    rows += lines(51, range(0, -40, -1), 10.0 * scale, branch=7)  # a crowd at the main branch of the others, from T on
    return rows


@pytest.mark.parametrize("feature_set", sorted(FEATURE_SETS))
def test_every_feature_ignores_lines_dated_from_T(feature_set):
    # Rule 1 for all the feature sets: the features of T computed on the history only equal the ones computed when
    # lines dated T or later are added, whatever those lines are (amounts, products, channels, branches, returns,
    # new clients). Two different futures give the same table.
    history = rich_rows()
    base = features(history, feature_set)
    for scale in (1.0, 5.0):
        pd.testing.assert_frame_equal(base, features(history + _future_lines(scale), feature_set))
    assert 50 not in base.index and 51 not in base.index
    assert list(base.columns) == FEATURE_SETS[feature_set]


def test_the_new_features_react_to_the_history_before_T():
    # The invariance test above would pass on constant features: every new variable must change when the history
    # changes (extra lines the day before T, a return, a new product and channel, a new client at the main branch).
    history = rich_rows()
    changed = history + [
        line(1, 2, 900.0, product=999, channel="at the store", branch=8),
        line(1, 3, -300.0, product=998, channel="by phone", branch=7),
        line(1, 4, 50.0, product=996),
        line(1, 120, 300.0, product=997),  # a product of the previous half that is never bought again
        *lines(5, range(1, 60), 10.0, branch=7),
    ]
    a, b = features(history), features(changed)
    assert a.loc[[1, 2], "branch_freq_trend"].notna().all()  # the test is not vacuous: the minimum of 30 days is met
    for col in EXTENDED_FEATURES:
        both_nan = a[col].isna() & b.reindex(a.index)[col].isna()
        assert not np.allclose(a[col][~both_nan], b.reindex(a.index)[col][~both_nan], equal_nan=True), col
