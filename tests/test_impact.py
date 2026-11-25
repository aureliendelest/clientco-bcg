"""Impact scenarios and tornado: the central case of docs/05_impact_and_pilot.md, the bounds, the sorting."""

import dataclasses

import pytest

from churn.config import Config, ImpactCfg, ScoringCfg
from churn.impact import impact_meur, impact_scenarios, tornado_table

CENTRAL = 15504.212 * 0.041 * 0.5 * 0.2 * 0.2465  # the doc: ~ 15.7 M EUR of margin a year


def test_central_case_of_the_doc():
    assert CENTRAL == pytest.approx(15.67, abs=0.01)
    assert impact_meur(Config(), detection=0.5, retention=0.2, lost_share=0.041) == pytest.approx(CENTRAL)


def test_scenarios_grid_and_bounds():
    table = impact_scenarios(0.5, Config())
    assert len(table) == 6  # 3 retentions x (low, high) lost share
    assert set(table["retention"]) == {0.1, 0.2, 0.3} and set(table["lost_share_case"]) == {"low", "high"}
    low = table.query("retention == 0.1 and lost_share_case == 'low'").iloc[0]
    high = table.query("retention == 0.3 and lost_share_case == 'high'").iloc[0]
    assert low["impact_gross_meur"] == pytest.approx(15504.212 * 0.034 * 0.5 * 0.1 * 0.2465)  # ~ 6.5
    assert high["impact_gross_meur"] == pytest.approx(15504.212 * 0.048 * 0.5 * 0.3 * 0.2465)  # ~ 27.5
    assert table["impact_gross_meur"].min() == low["impact_gross_meur"]
    assert table["impact_gross_meur"].max() == high["impact_gross_meur"]
    # proportional to the detection and to the retention
    twice = impact_scenarios(1.0, Config())
    assert (twice["impact_gross_meur"] / table["impact_gross_meur"]).round(9).eq(2.0).all()


def test_gross_when_no_opportunity_cost():
    table = impact_scenarios(0.5, Config())
    assert not table["net_of_opportunity_cost"].any()
    assert (table["opportunity_cost_meur"] == 0).all()
    assert table["impact_net_meur"].equals(table["impact_gross_meur"])


def test_opportunity_cost_is_subtracted_and_flagged_only_when_given():
    cfg = Config(scoring=ScoringCfg(opportunity_cost_per_action=100.0, list_size_k=700))
    table = impact_scenarios(0.5, cfg)
    assert table["net_of_opportunity_cost"].all()
    cost = 100.0 * 700 * 12 / 1e6  # 0.84 M EUR a year
    assert table["opportunity_cost_meur"].tolist() == pytest.approx([cost] * 6)
    assert table["impact_net_meur"].tolist() == pytest.approx((table["impact_gross_meur"] - cost).tolist())


def test_lost_share_override_and_validation():
    assert len(impact_scenarios(0.5, Config(), lost_share=0.041)) == 3  # one case
    pair = impact_scenarios(0.5, Config(), lost_share=(0.02, 0.06))
    assert sorted(pair["lost_share"].unique()) == [0.02, 0.06]
    for bad in (1.5, (0.06, 0.02), (0.1, 0.2, 0.3)):
        with pytest.raises(ValueError):
            impact_scenarios(0.5, Config(), lost_share=bad)
    with pytest.raises(ValueError, match="detection"):
        impact_scenarios(1.2, Config())


def test_scenarios_follow_the_configuration():
    cfg = Config(impact=ImpactCfg(retention_scenarios=[0.05], lost_revenue_share=(0.03, 0.05), gross_margin=0.3,
                                  annual_revenue_meur=1000.0))
    table = impact_scenarios(0.4, cfg)
    assert list(table["impact_gross_meur"]) == pytest.approx([1000 * 0.03 * 0.4 * 0.05 * 0.3, 1000 * 0.05 * 0.4 * 0.05 * 0.3])


def test_tornado_matches_the_doc_table_and_is_sorted():
    t = tornado_table(Config(), {"detection": (0.4, 0.6)})  # retention and lost share from the configuration
    assert list(t["parameter"]) == ["retention", "detection", "lost_share"]
    assert t["central_meur"].eq(t["central_meur"].iloc[0]).all() and t["central_meur"].iloc[0] == pytest.approx(CENTRAL)
    rows = t.set_index("parameter")
    # docs/05_impact_and_pilot.md, section 2
    assert rows.loc["retention", ["impact_low_meur", "impact_high_meur"]].round(1).tolist() == [7.8, 23.5]
    assert rows.loc["detection", ["impact_low_meur", "impact_high_meur"]].round(1).tolist() == [12.5, 18.8]
    # the doc prints 18.4 for the high lost share; the formula gives 18.34
    assert rows.loc["lost_share", ["impact_low_meur", "impact_high_meur"]].tolist() == pytest.approx([13.0, 18.4], abs=0.1)
    assert rows.loc["retention", "amplitude_meur"] == pytest.approx(15.7, abs=0.05)
    assert t["amplitude_meur"].is_monotonic_decreasing
    assert (rows["impact_low_meur"] <= rows["central_meur"]).all() and (rows["central_meur"] <= rows["impact_high_meur"]).all()


def test_tornado_order_changes_with_the_ranges():
    t = tornado_table(Config(), {"detection": (0.1, 0.9), "retention": (0.18, 0.22)})
    assert t["parameter"].iloc[0] == "detection"  # a wide range comes first
    assert t["amplitude_meur"].is_monotonic_decreasing
    explicit = tornado_table(Config(), {"detection": (0.4, 0.5, 0.6)})
    assert explicit.set_index("parameter").loc["detection", "centre"] == 0.5


def test_tornado_errors():
    with pytest.raises(ValueError, match="detection"):
        tornado_table(Config(), {})
    with pytest.raises(ValueError, match="Unknown"):
        tornado_table(Config(), {"detection": (0.4, 0.6), "margin": (0.2, 0.3)})
    with pytest.raises(ValueError):
        tornado_table(Config(), {"detection": (0.6, 0.4)})
    with pytest.raises(ValueError):
        tornado_table(Config(), {"detection": (0.4, 1.6)})


def test_the_configuration_is_not_modified():
    cfg = Config()
    before = dataclasses.asdict(cfg)
    impact_scenarios(0.5, cfg)
    tornado_table(cfg, {"detection": (0.4, 0.6)})
    assert dataclasses.asdict(cfg) == before
