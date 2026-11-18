"""Placeholders for the unit tests of the stub bricks, skipped until implemented."""

import pytest

# ---- snapshots: labels


@pytest.mark.skip(reason="stub: snapshots.label_class_horizon not implemented")
def test_class_horizon_uses_rhythm_class_horizon():
    # A 'fast' client is lost after label.class_horizons['fast'] days without purchase, not 180.
    ...


@pytest.mark.skip(reason="stub: snapshots.label_relative_horizon not implemented")
def test_relative_horizon_is_clipped_between_h_min_and_h_max():
    # H_i = clip(k * p90_gap_i, h_min, h_max)
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
