"""Anti-leakage suite, to run before every sweep (`make test`).

These are the five rules from docs/churn_enjeux_decisions_vigilances.md. They are skipped until
the corresponding bricks are implemented.
"""

import pytest


@pytest.mark.skip(reason="stub: features.compute_features not implemented")
def test_features_on_truncated_data_are_identical():
    # Rule 1: features of snapshot T computed on data truncated at T must be identical to the
    # features computed on the full data (no feature may look at dates >= T).
    ...


@pytest.mark.skip(reason="stub: splits.temporal_split not implemented")
def test_train_label_windows_end_before_test_date():
    # Rule 2: for every training snapshot, snapshot_date + embargo_days <= test_date.
    ...


@pytest.mark.skip(reason="stub: pipeline.runner.run not implemented")
def test_shuffled_labels_give_auc_near_half():
    # Rule 3: after shuffling the training labels, the test ROC-AUC must stay close to 0.5.
    ...


@pytest.mark.skip(reason="stub: features.compute_features not implemented")
def test_rhythm_features_use_only_prior_purchases_and_need_min_purchases():
    # Rule 4: gap_* and recency_ratio use purchases strictly before T, and are NaN below
    # label.min_purchases purchases (the class value is used instead).
    ...


@pytest.mark.skip(reason="stub: calibration and rho not implemented")
def test_rho_and_calibration_are_fitted_without_test_data():
    # Rule 5: fit_calibrator and fit_rho never see the test snapshot.
    ...
