"""Paired comparison of two sets of runs, date by date, and the acceptance rule written before the results.

`paired_by_date` turns the long metrics table (`results/metrics.parquet`) and two sets of backtest runs (one run per
snapshot date, same model and label, only the feature set or the model differs) into one row per date.
`accept` applies the rule of docs/plan_gel.md, section 2 (defaults = that rule):

    mean paired gain at the main K (700)  >=  min_gain points      (5 dates of the count)
    dates won at the main K               >=  min_wins
    mean gap at k_check (2 000)           >=  0

The excluded dates (22/09/2025: reported, not counted) stay in the table and are left out of the three tests.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence

import numpy as np
import pandas as pd

DATE_COLUMN = "snapshots.test_date"  # column of `results/runs.parquet` that holds the date of a run
_TOL = 1e-9  # floating point slack on the thresholds (gain in points)


def _by_date(runs: pd.DataFrame | Mapping | Sequence, name: str) -> pd.Series:
    """Run ids indexed by snapshot date (a Timestamp), whatever the way the runs are given."""
    if isinstance(runs, pd.DataFrame):
        missing = {"run_id", DATE_COLUMN} - set(runs.columns)
        if missing:
            raise ValueError(f"{name}: the runs table lacks the column(s) {sorted(missing)}")
        ids = pd.Series(runs["run_id"].to_numpy(), index=pd.to_datetime(runs[DATE_COLUMN]))
    elif isinstance(runs, Mapping):
        ids = pd.Series(list(runs.values()), index=pd.to_datetime(list(runs.keys())))
    else:
        raise TypeError(f"{name}: expected a runs DataFrame (run_id, {DATE_COLUMN}) or a {{date: run_id}} mapping")
    if ids.index.has_duplicates:
        dup = sorted({f"{d:%Y-%m-%d}" for d in ids.index[ids.index.duplicated()]})
        raise ValueError(f"{name}: several runs for the same date {dup}: select one run per date")
    return ids.sort_index()


def paired_by_date(
    metrics: pd.DataFrame,
    runs_a: pd.DataFrame | Mapping,
    runs_b: pd.DataFrame | Mapping,
    k: int | Sequence[int],
    *,
    metric: str = "capture_at_k",
    split: str = "test",
    segment: str = "all",
    model: str | None = None,
) -> pd.DataFrame:
    """One row per (date, k): the value of A, the value of B, the gap in points and the winner.

    `metrics`: the long table of `evaluation.evaluate_predictions` (several runs). `runs_a` / `runs_b`: the runs to
    compare, as a table with `run_id` and `snapshots.test_date` (a selection of `results/runs.parquet`) or as a
    {date: run_id} mapping. The dates of A and B must be the same. `k`: one K or a list of Ks.
    `diff_pts` = (B - A) x 100: positive when B (the challenger) is better; `winner` is "A", "B" or "tie".
    `model`, when given, is written in a `model` column so that several tables can be stacked for `accept`.
    Columns: [model], date, k, run_a, run_b, value_a, value_b, diff_pts, winner.
    """
    a, b = _by_date(runs_a, "runs_a"), _by_date(runs_b, "runs_b")
    if not a.index.equals(b.index):
        only_a = sorted(f"{d:%Y-%m-%d}" for d in a.index.difference(b.index))
        only_b = sorted(f"{d:%Y-%m-%d}" for d in b.index.difference(a.index))
        raise ValueError(f"A and B do not cover the same dates (only in A: {only_a}, only in B: {only_b})")
    ks = [int(k)] if np.isscalar(k) else [int(x) for x in k]
    sel = metrics.loc[(metrics["metric"] == metric) & (metrics["split"] == split) & (metrics["segment"] == segment)]
    sel = sel.loc[sel["k"].isin(ks)]
    if sel.duplicated(["run_id", "k"]).any():
        raise ValueError(f"Several {metric} rows for the same run and K (split={split!r}, segment={segment!r})")
    value = sel.set_index(["run_id", "k"])["value"]

    def get(run_id: str, kk: int) -> float:
        if (run_id, kk) not in value.index:
            raise ValueError(f"No {metric} value for run {run_id} at K={kk} (split={split!r}, segment={segment!r})")
        return float(value[(run_id, kk)])

    rows = []
    for date in a.index:
        for kk in ks:
            va, vb = get(a[date], kk), get(b[date], kk)
            diff = (vb - va) * 100
            winner = "B" if diff > 0 else "A" if diff < 0 else "tie"
            rows.append((date, kk, a[date], b[date], va, vb, diff, winner))
    out = pd.DataFrame(rows, columns=["date", "k", "run_a", "run_b", "value_a", "value_b", "diff_pts", "winner"])
    if model is not None:
        out.insert(0, "model", model)
    return out


def accept(
    table: pd.DataFrame,
    min_gain: float = 1.5,
    min_wins: int = 4,
    excluded_dates: Collection[str] = ("2025-09-22",),
    k_check: int = 2000,
    *,
    k_main: int = 700,
) -> dict:
    """Apply the acceptance rule to the stacked output of `paired_by_date` (it needs the K `k_main` and `k_check`).

    A model passes when, over the dates that are not excluded: the mean gap at `k_main` is at least `min_gain`
    points, B wins (strictly) at `k_main` on at least `min_wins` dates, and the mean gap at `k_check` is at least 0.
    The global verdict is True when every model passes. Without a `model` column, the table is one model, "all".

    Returns {"accepted": bool, "models": {model: {...}}, "thresholds": {...}}; each model holds `accepted`,
    `n_dates`, `mean_gain_pts`, `wins`, `mean_gain_k_check_pts` and the three booleans `gain_ok`, `wins_ok`,
    `k_check_ok`. Nothing here depends on the data: the thresholds are arguments.
    """
    need = {"date", "k", "diff_pts"} - set(table.columns)
    if need:
        raise ValueError(f"The table lacks the column(s) {sorted(need)}")
    skip = {pd.Timestamp(d) for d in excluded_dates}
    t = table.assign(date=pd.to_datetime(table["date"]))
    t = t.loc[~t["date"].isin(skip)]
    if "model" not in t.columns:
        t = t.assign(model="all")
    if t.empty:
        raise ValueError("No date left after the exclusions")
    models: dict[str, dict] = {}
    for name, g in t.groupby("model", sort=True):
        main, check = g.loc[g["k"] == k_main], g.loc[g["k"] == k_check]
        if main.empty or check.empty:
            raise ValueError(f"Model {name!r}: the table needs rows at K={k_main} and K={k_check}")
        if set(main["date"]) != set(check["date"]) or main["date"].duplicated().any() or check["date"].duplicated().any():
            raise ValueError(f"Model {name!r}: the dates of K={k_main} and K={k_check} differ or repeat")
        mean_gain, wins = float(main["diff_pts"].mean()), int((main["diff_pts"] > 0).sum())
        mean_check = float(check["diff_pts"].mean())
        verdict = {
            "n_dates": int(len(main)),
            "mean_gain_pts": mean_gain,
            "wins": wins,
            "mean_gain_k_check_pts": mean_check,
            "gain_ok": mean_gain >= min_gain - _TOL,
            "wins_ok": wins >= min_wins,
            "k_check_ok": mean_check >= -_TOL,
        }
        verdict["accepted"] = bool(verdict["gain_ok"] and verdict["wins_ok"] and verdict["k_check_ok"])
        models[str(name)] = verdict
    return {
        "accepted": all(v["accepted"] for v in models.values()),
        "models": models,
        "thresholds": {
            "min_gain": min_gain, "min_wins": min_wins, "k_main": k_main, "k_check": k_check,
            "excluded_dates": sorted(f"{d:%Y-%m-%d}" for d in skip),
        },
    }
