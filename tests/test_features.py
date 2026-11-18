import pytest


@pytest.mark.skip(reason="stub: snapshots.features not implemented")
def test_every_feature_of_the_set_is_produced():
    # compute_features returns exactly the columns of FEATURE_SETS[feature_set].
    ...


@pytest.mark.skip(reason="stub: snapshots.build not implemented")
def test_snapshot_columns_match_the_contract():
    # client_id, snapshot_date, <features>, label, horizon_days, V, segment
    ...
