"""Placeholders for the unit tests of the stub bricks, skipped until implemented."""

import pytest

# ---- snapshots: labels


@pytest.mark.skip(reason="stub: snapshots.label_fixed_horizon not implemented")
def test_fixed_horizon_is_the_same_for_every_client():
    # Label A: lost = no purchase day within label.horizon_days (180) after T, whatever the rhythm.
    ...


@pytest.mark.skip(reason="stub: snapshots.label_relative_horizon not implemented")
def test_relative_horizon_is_clipped_between_h_min_and_h_max():
    # Label B: H_i = clip(k * p90_gap_i, h_min, h_max) = clip(3 * p90, 45, 90) by default
    ...


# ---- snapshots: features and contract


@pytest.mark.skip(reason="stub: snapshots.compute_features not implemented")
def test_every_feature_of_the_set_is_produced():
    # compute_features returns exactly the columns of FEATURE_SETS[feature_set].
    ...


@pytest.mark.skip(reason="stub: snapshots.build_snapshot not implemented")
def test_snapshot_columns_match_the_contract():
    # client_id, snapshot_date, <features>, label, horizon_days, V, segment
    ...


# ---- evaluation


@pytest.mark.skip(reason="stub: evaluation.capture_at_k not implemented")
def test_capture_at_k_is_one_when_top_k_contains_all_churners():
    ...


@pytest.mark.skip(reason="stub: evaluation.threshold_metrics not implemented")
def test_threshold_metrics_on_a_toy_example():
    # precision, recall and F1 computed by hand on 6 clients.
    ...


@pytest.mark.skip(reason="stub: evaluation.tornado_table not implemented")
def test_tornado_is_sorted_by_amplitude():
    ...
