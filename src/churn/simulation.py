"""Post-hoc simulations on the backtest predictions: value definitions (#7) and monthly lists with rest (#9).

Pure functions returning DataFrames (no plot, no I/O except `load_backtest`). Everything works on the saved
predictions and on the cached snapshots of the SAME run configuration; the final test date is never read.

Perimeter, ranking, capture
---------------------------
"Perimeter" = the clients that were predicted (regular clients: rhythm fast / medium) whose truth `gone` is known.
`ev` (= p_cal x V) is the ranking of the current pipeline. The capture of a list is, as in
`evaluation.capture_at_k`, the lost value of the churners in the list divided by the lost value of ALL the
churners of the perimeter. Ties keep the order of `client_id`, so the results are reproducible.

#7 `value_alternatives`: is V (= 2 x revenue of the last 180 days, annualized) biased by clients in decline?
Two definitions of the value of a client are compared:

    V         = rev_180d x 2    (current definition, in the predictions)
    V_recent  = rev_90d x 4     (recent run rate)

Each of the two rankings (p_cal x V, p_cal x V_recent) is scored under each of the two losses (V and V_recent):
capture = sum(loss x gone) over the top K / sum(loss x gone) over all the churners of the perimeter, top K taken
per date. A "declining" client has freq_90d < freq_prev_90d (strict). The over-estimation table compares V with
V_recent for the churners (gone = 1) of the perimeter: ratio V / V_recent per client (undefined where
V_recent = 0: counted in `n_v_recent_zero`, left out of the median and the mean of the ratio, but kept in
`ratio_of_sums` = sum V / sum V_recent), and the share of the total lost V that each group carries.
Summary rows ("mean") average the per-date numbers over the dates not in `EXCLUDED_FROM_MEAN`
(2025-09-22: a single training snapshot), simple mean of the dates, not a pooled ratio.

#9 `cooldown_lists`: one list of K clients per date, in date order, ranked by `ev`. A client listed in one of
the `rest_lists` previous lists (the lists as built, so after exclusions) cannot be listed again; the list is
completed with the next best clients. The denominators are ambiguous, so they are fixed here:

  * capture of ONE list (`capture_list`): sum of V x gone of the list / sum of V x gone of ALL churners of the
    perimeter of THIS date (a churner that was already listed or already lost counts again: each date is
    its own snapshot, as in the backtest metric).
  * cumulative, each churner counted ONCE (`capture_cum_unique`): numerator = sum of V of the distinct churners
    that appeared with gone = 1 in at least one list so far, V taken at the FIRST date where the client was listed
    with gone = 1; denominator = sum of V of the distinct churners of ALL the dates of the simulation, each client
    counted once, V taken at the first date where its gone = 1 (perimeter of that date). The denominator is the
    same for every list (the whole horizon), so the curve rises from 0 to the final share; the numerator can use
    a different date's V than the denominator for the same client (V changes between dates), so this is not a
    strict share of the same quantity.
  * `capture_cum_unique_to_date`: same numerator, denominator restricted to the distinct churners of the dates up
    to this list (what was known at that time).
  * `capture_cum_pooled`: for contrast, repetitions counted: sum of the lists' captured V / sum of the dates'
    perimeter churned V (pooled over dates, a churner seen at several dates counts at each).
The table holds both `rest_lists = 0` (no rest, unique counting: the effect of the repetition) and the requested
`rest_lists`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from churn.evaluation import _capture

BACKTEST_DATES = ("2025-09-22", "2025-10-22", "2025-11-22", "2025-12-22", "2026-01-22", "2026-02-22")
EXCLUDED_FROM_MEAN = ("2025-09-22",)
FORBIDDEN_DATE = "2026-03-22"  # final test date: never read here
MEAN_LABEL = "mean"

RANKINGS = ("V", "V_recent")  # "V" = current ranking p_cal x V (the `ev` column), "V_recent" = p_cal x V_recent
LOSSES = ("V", "V_recent")
SNAPSHOT_COLUMNS = ["client_id", "snapshot_date", "V", "gone", "rev_90d", "freq_90d", "freq_prev_90d"]


# ------------------------------------------------------------------------------ helpers


def _date_str(d) -> str:
    return pd.Timestamp(d).strftime("%Y-%m-%d")


def _check_dates(dates) -> None:
    if any(_date_str(d) == FORBIDDEN_DATE for d in dates):
        raise ValueError(f"{FORBIDDEN_DATE} is the final test date: it must not be used in simulations")


def _prepare(pred: pd.DataFrame, snapshot: pd.DataFrame) -> pd.DataFrame:
    """Predictions joined to the snapshot columns they need, rows with an unknown truth dropped, sorted by
    (snapshot_date, client_id). Adds `V_recent` = rev_90d x 4 and `declining`. Raises on an inconsistent pair."""
    need_p = {"client_id", "snapshot_date", "p_cal", "V", "ev", "gone"}
    need_s = {"client_id", "snapshot_date", "V", "rev_90d", "freq_90d", "freq_prev_90d"}
    if need_p - set(pred.columns):
        raise ValueError(f"predictions miss columns {sorted(need_p - set(pred.columns))}")
    if need_s - set(snapshot.columns):
        raise ValueError(f"snapshot misses columns {sorted(need_s - set(snapshot.columns))}")
    _check_dates(pd.unique(pred["snapshot_date"]))
    keys = ["snapshot_date", "client_id"]
    if pred.duplicated(keys).any() or snapshot.duplicated(keys).any():
        raise ValueError("Duplicate (snapshot_date, client_id) in the predictions or in the snapshot")
    snap = snapshot[[*keys, "V", "rev_90d", "freq_90d", "freq_prev_90d"]].rename(columns={"V": "V_snapshot"})
    df = pred[[*keys, "p_cal", "V", "ev", "gone"]].merge(snap, on=keys, how="left", validate="one_to_one")
    if df["V_snapshot"].isna().any():
        raise ValueError("Some predicted clients are missing from the snapshot")
    if not np.allclose(df["V"], df["V_snapshot"], rtol=1e-9, atol=1e-9):
        raise ValueError("V differs between the predictions and the snapshot (wrong cache or configuration)")
    df = df.drop(columns="V_snapshot")
    df = df.loc[df["gone"].notna()].sort_values(keys, kind="stable").reset_index(drop=True)
    df["V_recent"] = df["rev_90d"] * 4.0
    df["declining"] = df["freq_90d"] < df["freq_prev_90d"]
    return df


def _mean_rows(table: pd.DataFrame, by: list[str], value_cols: list[str], excluded: Sequence[str]) -> pd.DataFrame:
    """Summary rows: mean over the dates not excluded, for every combination of `by`."""
    kept = table.loc[~table["date"].isin(list(excluded))]
    if kept.empty:
        return table.iloc[0:0]
    grouped = kept.groupby(by, sort=False)
    out = grouped[value_cols].mean().reset_index()
    out.insert(0, "date", MEAN_LABEL)
    out["n_dates"] = grouped["date"].nunique().to_numpy()
    return out


# ------------------------------------------------------------------------------ #7 value definitions


def capture_by_ranking(df: pd.DataFrame, k_values: Sequence[int]) -> pd.DataFrame:
    """Per date, K, ranking and loss: captured and total lost value, and their ratio. `df` comes from `_prepare`."""
    rows = []
    for date, g in df.groupby("snapshot_date", sort=True):
        keys = {"V": g["ev"].to_numpy(float), "V_recent": (g["p_cal"] * g["V_recent"]).to_numpy(float)}
        dates = np.zeros(len(g), dtype=int)
        for loss in LOSSES:
            lost = (g[loss] * g["gone"]).to_numpy(float)
            total = float(lost.sum())
            for rank, key in keys.items():
                num = _capture(key, lost, dates, list(k_values), den=1.0)  # raw captured value
                for k, n in zip(k_values, num):
                    rows.append((_date_str(date), int(k), rank, loss, n, total, n / total if total > 0 else np.nan))
    cols = ["date", "k", "ranking", "loss", "captured", "total", "capture"]
    return pd.DataFrame(rows, columns=cols)


def overestimation(df: pd.DataFrame) -> pd.DataFrame:
    """Over-estimation of V against V_recent among the churners of the perimeter, per date and group
    (`all`, `declining`, `not_declining`). `df` comes from `_prepare`."""
    rows = []
    for date, g in df.groupby("snapshot_date", sort=True):
        churn = g.loc[g["gone"] == 1]
        total_v = float(churn["V"].sum())
        for name, sub in (
            ("all", churn),
            ("declining", churn[churn["declining"]]),
            ("not_declining", churn[~churn["declining"]]),
        ):
            defined = sub[sub["V_recent"] > 0]
            ratio = defined["V"] / defined["V_recent"]
            sum_recent = float(sub["V_recent"].sum())
            rows.append(
                (
                    _date_str(date), name, len(sub), len(sub) / len(churn) if len(churn) else np.nan,
                    float(sub["V"].sum()) / total_v if total_v > 0 else np.nan,
                    float(ratio.median()) if len(ratio) else np.nan,
                    float(ratio.mean()) if len(ratio) else np.nan,
                    float(sub["V"].sum()) / sum_recent if sum_recent > 0 else np.nan,
                    int((sub["V_recent"] <= 0).sum()),
                )
            )
    cols = ["date", "group", "n_churners", "share_of_churners", "share_of_lost_v", "median_ratio", "mean_ratio",
            "ratio_of_sums", "n_v_recent_zero"]
    return pd.DataFrame(rows, columns=cols)


def value_alternatives(
    pred: pd.DataFrame,
    snapshot: pd.DataFrame,
    k_values: Sequence[int] = (700, 2000),
    excluded_dates: Sequence[str] = EXCLUDED_FROM_MEAN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """#7. `pred`: predictions of the backtest runs stacked (any number of dates); `snapshot`: the cached snapshots
    of the same dates (columns of SNAPSHOT_COLUMNS at least). Returns `(capture, overest)`.

    `capture`: one row per (date, k, ranking, loss) with `captured`, `total`, `capture`, then, for each (k,
    ranking, loss), a summary row (date = "mean", `n_dates`) averaging `capture` over the dates not in
    `excluded_dates` (`captured` and `total` are not averaged: NaN). `ranking` and `loss` are "V" (current) or
    "V_recent". `overest`: see `overestimation`, with mean rows too. See the module docstring for the definitions."""
    df = _prepare(pred, snapshot)
    cap = capture_by_ranking(df, k_values)
    cap_mean = _mean_rows(cap, ["k", "ranking", "loss"], ["capture"], excluded_dates)
    cap = pd.concat([cap, cap_mean], ignore_index=True)
    over = overestimation(df)
    over_mean = _mean_rows(
        over, ["group"],
        ["n_churners", "share_of_churners", "share_of_lost_v", "median_ratio", "mean_ratio", "ratio_of_sums",
         "n_v_recent_zero"],
        excluded_dates,
    )
    over = pd.concat([over, over_mean], ignore_index=True)
    return cap, over


# ------------------------------------------------------------------------------ #9 lists with rest


def _frames(pred_by_date: Mapping | pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """The per-date frames in date order, rows with a known truth, sorted by client_id."""
    if isinstance(pred_by_date, pd.DataFrame):
        items = [(d, g) for d, g in pred_by_date.groupby("snapshot_date", sort=True)]
    else:
        items = list(pred_by_date.items())
    _check_dates([d for d, _ in items])
    out = []
    for d, g in sorted(items, key=lambda it: pd.Timestamp(it[0])):
        missing = {"client_id", "V", "gone", "ev"} - set(g.columns)
        if missing:
            raise ValueError(f"predictions of {d} miss columns {sorted(missing)}")
        g = g.loc[g["gone"].notna(), ["client_id", "V", "gone", "ev"]].sort_values("client_id", kind="stable")
        if g["client_id"].duplicated().any():
            raise ValueError(f"Duplicate client_id on {d}")
        out.append((_date_str(d), g.reset_index(drop=True)))
    return out


def _simulate(frames: list[tuple[str, pd.DataFrame]], k: int, rest_lists: int) -> list[dict]:
    # unique churners of the whole horizon: V at the first date where gone = 1
    first_gone: dict = {}
    den_to_date: list[float] = []
    for _, g in frames:
        for cid, v in zip(g.loc[g["gone"] == 1, "client_id"], g.loc[g["gone"] == 1, "V"]):
            first_gone.setdefault(cid, float(v))
        den_to_date.append(sum(first_gone.values()))
    den_unique = den_to_date[-1] if den_to_date else 0.0

    lists: list[set] = []
    captured_unique: dict = {}
    listed_ever: set = set()
    cum_list, cum_den = 0.0, 0.0
    rows = []
    for i, (date, g) in enumerate(frames):
        banned = set().union(*lists[max(0, len(lists) - rest_lists):]) if rest_lists > 0 else set()
        order = np.argsort(-g["ev"].to_numpy(float), kind="stable")
        eligible = order[~g["client_id"].iloc[order].isin(banned).to_numpy()]
        top = g.iloc[eligible[:k]]
        ids = set(top["client_id"])
        lists.append(ids)
        lost = float((top["V"] * top["gone"]).sum())
        den_date = float((g["V"] * g["gone"]).sum())
        for cid, v in zip(top.loc[top["gone"] == 1, "client_id"], top.loc[top["gone"] == 1, "V"]):
            captured_unique.setdefault(cid, float(v))
        num_unique = sum(captured_unique.values())
        cum_list += lost
        cum_den += den_date
        rows.append(
            {
                "list": i + 1, "date": date, "k": k, "n_listed": len(top),
                "n_already_listed_before": len(ids & listed_ever),
                "n_banned_present": int(g["client_id"].isin(banned).sum()),
                "n_churners_in_list": int((top["gone"] == 1).sum()),
                "captured_v": lost, "den_date": den_date,
                "capture_list": lost / den_date if den_date > 0 else np.nan,
                "captured_unique_cum": num_unique, "den_unique_all": den_unique,
                "capture_cum_unique": num_unique / den_unique if den_unique > 0 else np.nan,
                "den_unique_to_date": den_to_date[i],
                "capture_cum_unique_to_date": num_unique / den_to_date[i] if den_to_date[i] > 0 else np.nan,
                "capture_cum_pooled": cum_list / cum_den if cum_den > 0 else np.nan,
            }
        )
        listed_ever |= ids
    return rows


def cooldown_lists(pred_by_date: Mapping | pd.DataFrame, k: int = 700, rest_lists: int = 3) -> pd.DataFrame:
    """#9. `pred_by_date`: {date: predictions} (or one stacked frame with `snapshot_date`); columns client_id, V,
    gone, ev. One list of K per date in date order; a client in one of the `rest_lists` previous lists is excluded.
    One row per list, for `rest_lists = 0` (no rest) and for the requested value (same table if it is 0), with the
    column `rest_lists`. Definitions of the captures: module docstring."""
    if k <= 0 or rest_lists < 0:
        raise ValueError("k must be positive and rest_lists >= 0")
    frames = _frames(pred_by_date)
    out = []
    for r in dict.fromkeys((0, int(rest_lists))):
        t = pd.DataFrame(_simulate(frames, int(k), r))
        t.insert(0, "rest_lists", r)
        out.append(t)
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------------------------ loading (cache only)


def load_backtest(
    dates: Sequence[str] = BACKTEST_DATES,
    *,
    model: str = "xgb",
    label: str = "fixed_horizon",
    feature_set: str = "basic",
    runs_dir: str | Path | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Predictions of the backtest runs and cached snapshots of the same dates, as `({date: pred}, {date: snapshot})`.
    Reads `runs/<run>/predictions.parquet` and `data/processed/snapshots-<key>/<date>.parquet` only (no DuckDB query,
    no build). One run per date is required. The final test date is refused before any file is touched."""
    _check_dates(dates)
    from churn import snapshots  # lazy: the module imports the data layer
    from churn.config import RUNS_DIR, load_config

    runs_dir = Path(runs_dir) if runs_dir is not None else RUNS_DIR
    preds, snaps = {}, {}
    for d in dates:
        pattern = f"*model.type-{model}__label.kind-{label}__snapshots.feature_set-{feature_set}__snapshots.test_date-{d}-*"
        found = sorted(p for p in runs_dir.glob(pattern) if p.is_dir())
        if len(found) != 1:
            raise FileNotFoundError(f"Expected one run for {d} matching {pattern} in {runs_dir}, found {len(found)}")
        run = found[0]
        cfg = load_config(run / "config.yaml")
        fp = json.loads((run / "meta.json").read_text())["data_fingerprint"]
        path = snapshots.PROCESSED_DIR / f"snapshots-{snapshots.cache_key(cfg, fp)}" / f"{d}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"No cached snapshot for {d} at {path}")
        preds[d] = pd.read_parquet(run / "predictions.parquet")
        snaps[d] = pd.read_parquet(path, columns=SNAPSHOT_COLUMNS)
    return preds, snaps
