"""Business impact in M EUR of gross margin per year, and its sensitivity (tornado).

    impact = annual revenue x lost share x detection x retention x gross margin

- annual revenue, lost share (low / high) and margin: `ImpactCfg`, measured on the dataset;
- detection: the share of the lost revenue captured by the list of K clients, measured by the backtest and passed
  in by the caller (the reference is `capture_total_at_k` at K = 700, denominator "all the churn");
- retention: a team hypothesis (scenarios 10 / 20 / 30 %) until a randomized pilot measures it.

The numbers are gross of the opportunity cost of the reps, which is subtracted only when
`ScoringCfg.opportunity_cost_per_action` is given (and the table then says so).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from churn.config import Config
from churn.scoring import list_size

LISTS_PER_YEAR = 12  # monthly lists

TORNADO_PARAMETERS = ("retention", "detection", "lost_share")  # in the order of the doc; the table is sorted by amplitude


def _check_share(name: str, value: float) -> float:
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be a share between 0 and 1, got {value!r}")
    return float(value)


def impact_meur(cfg: Config, *, detection: float, retention: float, lost_share: float) -> float:
    """Gross margin saved per year (M EUR) for one combination of the three uncertain parameters."""
    return (
        cfg.impact.annual_revenue_meur
        * _check_share("lost_share", lost_share)
        * _check_share("detection", detection)
        * _check_share("retention", retention)
        * cfg.impact.gross_margin
    )


def _lost_share_cases(lost_share: float | Sequence[float] | None, cfg: Config) -> list[tuple[str, float]]:
    """`None`: the low and high shares of the configuration; a number: that single share; a pair: (low, high)."""
    if lost_share is None:
        lost_share = cfg.impact.lost_revenue_share
    if isinstance(lost_share, (int, float)):
        return [("given", _check_share("lost_share", lost_share))]
    values = [_check_share("lost_share", v) for v in lost_share]
    if len(values) != 2 or values[0] > values[1]:
        raise ValueError(f"lost_share must be a number or a (low, high) pair, got {lost_share!r}")
    return [("low", values[0]), ("high", values[1])]


def impact_scenarios(detection: float, cfg: Config, *, lost_share: float | Sequence[float] | None = None) -> pd.DataFrame:
    """Retention scenarios (`ImpactCfg.retention_scenarios`, 10 / 20 / 30 %) x lost share (low and high), in M EUR of
    gross margin per year. One row per combination.

    `detection` is the share of the lost revenue captured by the list (0 to 1). `lost_share` replaces
    `ImpactCfg.lost_revenue_share`: a single number gives one case, a (low, high) pair two.

    The opportunity cost is subtracted only if `ScoringCfg.opportunity_cost_per_action` is set: it is then
    `cost x list_size x 12` (one monthly list of K actions, twelve lists a year) expressed in M EUR; otherwise
    `opportunity_cost_meur` is 0 and `net_of_opportunity_cost` is False, so that a gross figure is never read as a net one.

    Columns: retention, lost_share_case, lost_share, detection, annual_revenue_meur, gross_margin,
    impact_gross_meur, opportunity_cost_meur, impact_net_meur, net_of_opportunity_cost.
    """
    _check_share("detection", detection)
    cost_per_action = cfg.scoring.opportunity_cost_per_action
    cost_meur = 0.0 if cost_per_action is None else cost_per_action * list_size(cfg) * LISTS_PER_YEAR / 1e6
    rows = []
    for retention in cfg.impact.retention_scenarios:
        for case, share in _lost_share_cases(lost_share, cfg):
            gross = impact_meur(cfg, detection=detection, retention=retention, lost_share=share)
            rows.append({
                "retention": retention,
                "lost_share_case": case,
                "lost_share": share,
                "detection": float(detection),
                "annual_revenue_meur": cfg.impact.annual_revenue_meur,
                "gross_margin": cfg.impact.gross_margin,
                "impact_gross_meur": gross,
                "opportunity_cost_meur": cost_meur,
                "impact_net_meur": gross - cost_meur,
                "net_of_opportunity_cost": cost_per_action is not None,
            })
    return pd.DataFrame(rows)


def _range(name: str, bounds: Sequence[float]) -> tuple[float, float, float]:
    """(low, centre, high) of a tornado range: a pair (the centre is the middle) or a triple (explicit centre)."""
    values = [float(v) for v in bounds]
    if len(values) == 2:
        low, high = values
        centre = (low + high) / 2
    elif len(values) == 3:
        low, centre, high = values
    else:
        raise ValueError(f"ranges[{name!r}] must be (low, high) or (low, centre, high), got {bounds!r}")
    if not low <= centre <= high:
        raise ValueError(f"ranges[{name!r}] must satisfy low <= centre <= high, got {bounds!r}")
    for v in (low, centre, high):
        _check_share(name, v)
    return low, centre, high


def tornado_table(cfg: Config, ranges: Mapping[str, Sequence[float]]) -> pd.DataFrame:
    """One row per parameter (retention, detection, lost share): the impact (M EUR of gross margin per year) at its low
    and at its high value, the two others at their centre, sorted by decreasing amplitude.

    `ranges` maps `retention`, `detection` and `lost_share` to `(low, high)` (the centre is the middle) or
    `(low, centre, high)`. `detection` is required: it comes from the backtest. `retention` defaults to the span of
    `ImpactCfg.retention_scenarios` and `lost_share` to `ImpactCfg.lost_revenue_share`. An unknown key is an error.

    Columns: parameter, low, centre, high, impact_low_meur, impact_high_meur, central_meur, amplitude_meur."""
    unknown = set(ranges) - set(TORNADO_PARAMETERS)
    if unknown:
        raise ValueError(f"Unknown tornado parameter(s) {sorted(unknown)}; known: {list(TORNADO_PARAMETERS)}")
    if "detection" not in ranges:
        raise ValueError("ranges must contain 'detection' (measured by the backtest)")
    bounds = {
        "retention": (min(cfg.impact.retention_scenarios), max(cfg.impact.retention_scenarios)),
        "lost_share": tuple(cfg.impact.lost_revenue_share),
    } | dict(ranges)
    spans = {name: _range(name, bounds[name]) for name in TORNADO_PARAMETERS}
    centre = {name: span[1] for name, span in spans.items()}
    central = impact_meur(cfg, **centre)
    rows = []
    for name in TORNADO_PARAMETERS:
        low, mid, high = spans[name]
        at_low = impact_meur(cfg, **(centre | {name: low}))
        at_high = impact_meur(cfg, **(centre | {name: high}))
        rows.append({
            "parameter": name, "low": low, "centre": mid, "high": high,
            "impact_low_meur": at_low, "impact_high_meur": at_high, "central_meur": central,
            "amplitude_meur": abs(at_high - at_low),
        })
    return pd.DataFrame(rows).sort_values("amplitude_meur", ascending=False, kind="stable").reset_index(drop=True)
