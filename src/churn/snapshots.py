"""Snapshots: one row per active client at a date T (features before T, label after T).

Contract (columns of a snapshot table):
    client_id, snapshot_date, <features>, rhythm, label, horizon_days, gone, V, segment
Features only use history strictly before `snapshot_date` (days < T); labels look at days >= T, in the half-open
window [T, T + H). A snapshot holds every client active over the lookback window, whatever the rhythm: the model
perimeter (regular clients) is `in_perimeter(df)`. `label`, `horizon_days` and `gone` are NaN when the window is not
fully observed (the last dates of the data, and the dates scored in production).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator
from datetime import date, timedelta

import duckdb
import numpy as np
import pandas as pd

from churn.config import PROCESSED_DIR, Config, LabelCfg, config_to_dict, resolve
from churn.data import build_client_day_table, connect, fingerprint
from churn.scoring import compute_value

SNAPSHOT_VERSION = 2  # v2: freq_trend and rev_trend became log-ratios. Bump when the content of a snapshot changes for a given config: invalidates the cache
PERIMETER_RHYTHMS = ("fast", "medium")  # regular clients: the perimeter of the model

# ------------------------------------------------------------------------------ build


def snapshot_dates(cfg: Config, *, last: str | date | None = None) -> list[date]:
    """Snapshot dates from `train_start` (monthly or weekly) up to `last`, plus the test date.

    `last` defaults to the last training date (`Config.last_train_date()`). The rolling backtest passes a later
    date to get the dates between the end of training and the test.
    """
    start = pd.Timestamp(cfg.snapshots.train_start)
    end = pd.Timestamp(last) if last is not None else pd.Timestamp(cfg.last_train_date())
    dates: list[date] = []
    j = 0
    while True:
        # offsets from the start (not cumulated) so that a month end never drifts
        d = start + (pd.DateOffset(months=j) if cfg.snapshots.step == "monthly" else pd.Timedelta(weeks=j))
        if d > end:
            break
        dates.append(d.date())
        j += 1
    dates.append(date.fromisoformat(cfg.snapshots.test_date))
    return sorted(set(dates))


def build_snapshot(con: duckdb.DuckDBPyConnection, snapshot_date: date, cfg: Config) -> pd.DataFrame:
    """Snapshot at one date: active clients over the last `lookback_days`, their features,
    their label and horizon, their value V and their segment.

    `con` must hold the tables of `data.build_client_day_table`."""
    base = _aggregates(con, snapshot_date, cfg.snapshots.lookback_days, cfg.label.min_purchases)
    base["rhythm"] = rhythm_class(base, cfg)
    base["V"] = compute_value(base, cfg)
    base = LABELS[cfg.label.kind](base, con, snapshot_date, cfg.label)
    base = assign_segments(base, cfg)
    base["snapshot_date"] = pd.Timestamp(snapshot_date)
    features = _feature_columns(cfg.snapshots.feature_set)
    cols = ["client_id", "snapshot_date", *features, "rhythm", "label", "horizon_days", "gone", "V", "segment"]
    return base[cols].sort_values("client_id").reset_index(drop=True)


_LABEL_KEYS = {
    "fixed_horizon": ("horizon_days",),
    "relative_horizon": ("k", "h_min", "h_max"),
    "partial_drop": ("horizon_days", "drop_threshold"),
}
_PERIMETER_KEYS = ("truth_days", "min_purchases", "max_median_gap_days", "rhythm_bounds_days")


def cache_key(cfg: Config, data_fingerprint: str) -> str:
    """Key of the Parquet cache: data fingerprint + snapshot, label and feature configuration.

    Only what changes the content of a snapshot is in the key (not the model, the train/test dates or the
    step), so every experiment on the same data, labels and features shares the same cached snapshots."""
    label = config_to_dict(cfg).get("label", {})
    parts = {
        "version": SNAPSHOT_VERSION,
        "data": data_fingerprint,
        "lookback_days": cfg.snapshots.lookback_days,
        "feature_set": cfg.snapshots.feature_set,
        "segment_by": cfg.model.segment_by,
        "value_definition": cfg.scoring.value_definition,
        "label_kind": cfg.label.kind,
        "label": {k: label[k] for k in (*_LABEL_KEYS[cfg.label.kind], *_PERIMETER_KEYS)},
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def load_or_build(cfg: Config, *, last: str | date | None = None) -> pd.DataFrame:
    """All snapshots for this configuration, read from `data/processed/` when the cache key matches,
    otherwise built once and written (one Parquet file per date). `last`: see `snapshot_dates`."""
    parquet = resolve(cfg.dataset.shifted_parquet)
    key = cache_key(cfg, fingerprint(parquet))
    folder = PROCESSED_DIR / f"snapshots-{key}"
    dates = snapshot_dates(cfg, last=last)
    missing = [d for d in dates if not (folder / f"{d:%Y-%m-%d}.parquet").exists()]
    if missing:
        folder.mkdir(parents=True, exist_ok=True)
        con = connect(parquet)
        build_client_day_table(con)
        for d in missing:
            tmp = folder / f"{d:%Y-%m-%d}.parquet.tmp"
            build_snapshot(con, d, cfg).to_parquet(tmp, index=False)
            tmp.replace(folder / f"{d:%Y-%m-%d}.parquet")  # atomic: an interrupted build leaves no partial file
        con.close()
    return pd.concat([pd.read_parquet(folder / f"{d:%Y-%m-%d}.parquet") for d in dates], ignore_index=True)


# ---------------------------------------------------------------------------- features

FEATURE_SETS: dict[str, list[str]] = {
    "basic": [
        "recency_days",
        "freq_180d",
        "freq_90d",
        "freq_prev_90d",
        "rev_180d",
        "rev_90d",
        "rev_prev_90d",
        "gap_median",
        "gap_p90",
        "recency_ratio",  # recency / gap_p90, only with enough purchase history
        "freq_trend",  # log(1 + freq_90d) - log(1 + freq_prev_90d): 0 = stable, < 0 = decline
        "rev_trend",  # same on revenue
    ],
    # Planned extensions (not implemented): returns share, channel mix, number of branches,
    # basket diversity, month indicator. Never a day-of-week feature (dates are shifted).
    "extended": [],
}
_RHYTHM_FEATURES = ("gap_median", "gap_p90", "recency_ratio")  # NaN below `min_purchases` purchase days


def _feature_columns(feature_set: str) -> list[str]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set {feature_set!r}; known: {sorted(FEATURE_SETS)}")
    if not FEATURE_SETS[feature_set]:
        raise NotImplementedError(f"feature set {feature_set!r} is not implemented yet")
    return FEATURE_SETS[feature_set]


def _aggregates(con: duckdb.DuckDBPyConnection, snapshot_date: date, lookback_days: int, min_purchases: int) -> pd.DataFrame:
    """Every aggregate computed from the history before `snapshot_date`, one row per active client.

    The columns are named for the default 180 / 90 day windows; the windows follow `lookback_days` (and its half)."""
    t = f"{snapshot_date:%Y-%m-%d}"
    half = lookback_days // 2
    df = con.execute(f"""
        WITH hist AS (
            SELECT client_id, d, rev, d - lag(d) OVER (PARTITION BY client_id ORDER BY d) AS gap
            FROM bn WHERE d < DATE '{t}' AND d >= DATE '{t}' - {lookback_days})
        SELECT client_id,
               count(*) AS freq_180d,
               median(gap) AS gap_median,
               quantile_cont(gap, 0.9) AS gap_p90,
               date_diff('day', max(d), DATE '{t}') AS recency_days,
               sum(rev) AS rev_180d,
               count(*) FILTER (WHERE d >= DATE '{t}' - {half}) AS freq_90d,
               count(*) FILTER (WHERE d <  DATE '{t}' - {half}) AS freq_prev_90d,
               coalesce(sum(rev) FILTER (WHERE d >= DATE '{t}' - {half}), 0) AS rev_90d,
               coalesce(sum(rev) FILTER (WHERE d <  DATE '{t}' - {half}), 0) AS rev_prev_90d
        FROM hist GROUP BY client_id ORDER BY client_id""").fetchdf()
    df["recency_ratio"] = df["recency_days"] / df["gap_p90"].clip(lower=1)
    # Log-ratios: 0 = stable, < 0 = decline, symmetric, and bounded by the log of the largest value (a plain ratio
    # exploded when the previous revenue was close to zero: max 2 million). `log1p` also handles the zeros.
    df["freq_trend"] = np.log1p(df["freq_90d"]) - np.log1p(df["freq_prev_90d"])
    df["rev_trend"] = np.log1p(df["rev_90d"]) - np.log1p(df["rev_prev_90d"])
    too_short = df["freq_180d"] < min_purchases  # gaps are not reliable below `min_purchases` purchase days
    df.loc[too_short, list(_RHYTHM_FEATURES)] = np.nan
    return df


def compute_features(
    con: duckdb.DuckDBPyConnection,
    snapshot_date: date,
    feature_set: str,
    *,
    lookback_days: int = 180,
    min_purchases: int = 6,
) -> pd.DataFrame:
    """One row per client active before `snapshot_date`, with the columns of `FEATURE_SETS[feature_set]`.
    Rhythm features (gap_*, recency_ratio) are NaN below `label.min_purchases` purchases."""
    columns = _feature_columns(feature_set)
    return _aggregates(con, snapshot_date, lookback_days, min_purchases)[["client_id", *columns]]


# -------------------------------------------------------------------------------- labels
# `LabelCfg.kind`: fixed_horizon (label A) | relative_horizon (label B) | partial_drop (second score).
# Each function adds `label` (1 = lost) and `horizon_days` to a snapshot. Whatever the kind, the snapshot also
# carries `gone` (no purchase within `truth_days`): the evaluation truth.

LabelFn = Callable[[pd.DataFrame, duckdb.DuckDBPyConnection, date, LabelCfg], pd.DataFrame]


def _days_observed(con: duckdb.DuckDBPyConnection, snapshot_date: date) -> int:
    """Days of data from `snapshot_date` (included) to the end of the data: a window of H days is observed when H <= this."""
    last = con.execute("SELECT max(d) FROM bn").fetchone()[0]
    return (pd.Timestamp(last) - pd.Timestamp(snapshot_date)).days


def _keep_if(values: pd.Series, observed: bool | pd.Series) -> pd.Series:
    """`values` where `observed`, NaN elsewhere (`observed` may be one boolean for the whole snapshot)."""
    if isinstance(observed, (bool, np.bool_)):
        return values if observed else pd.Series(np.nan, index=values.index)
    return values.where(observed)


def _add_truth(df: pd.DataFrame, con: duckdb.DuckDBPyConnection, snapshot_date: date, cfg: LabelCfg) -> pd.DataFrame:
    """Add `next_gap` (days from T to the first purchase day in [T, T + window), NaN if none) and `gone`
    (no purchase within `truth_days`; NaN when that window is not observed)."""
    window = max(cfg.truth_days, cfg.horizon_days, cfg.h_max)
    t = f"{snapshot_date:%Y-%m-%d}"
    nxt = con.execute(f"""
        SELECT client_id, min(d) - DATE '{t}' AS next_gap FROM bn
        WHERE d >= DATE '{t}' AND d < DATE '{t}' + {window} GROUP BY client_id""").fetchdf()
    df = df.drop(columns=["next_gap"], errors="ignore").merge(nxt, on="client_id", how="left")
    df["next_gap"] = df["next_gap"].astype(float)
    gone = (df["next_gap"].isna() | (df["next_gap"] >= cfg.truth_days)).astype(float)
    df["gone"] = _keep_if(gone, cfg.truth_days <= _days_observed(con, snapshot_date))
    return df


def _lost_within(df: pd.DataFrame, horizon: pd.Series | float, available: int) -> pd.Series:
    """1.0 when the first purchase comes at or after `horizon` days (or never), NaN when the horizon is not observed."""
    lost = (df["next_gap"].isna() | (df["next_gap"] >= horizon)).astype(float)
    return _keep_if(lost, horizon <= available)


def label_fixed_horizon(df: pd.DataFrame, con: duckdb.DuckDBPyConnection, snapshot_date: date, cfg: LabelCfg) -> pd.DataFrame:
    """Label A. Lost = no purchase day within `horizon_days` (180) after the snapshot."""
    df = _add_truth(df, con, snapshot_date, cfg)
    df["horizon_days"] = float(cfg.horizon_days)
    df["label"] = _lost_within(df, cfg.horizon_days, _days_observed(con, snapshot_date))
    return df


def label_relative_horizon(df: pd.DataFrame, con: duckdb.DuckDBPyConnection, snapshot_date: date, cfg: LabelCfg) -> pd.DataFrame:
    """Label B. Lost = no purchase within H_i = clip(k * p90_gap_i, h_min, h_max) (3, 45, 90 by default).
    Needs `gap_p90`: NaN (so no label) for the clients with too short a history."""
    df = _add_truth(df, con, snapshot_date, cfg)
    df["horizon_days"] = (cfg.k * df["gap_p90"]).clip(cfg.h_min, cfg.h_max)
    df["label"] = _lost_within(df, df["horizon_days"], _days_observed(con, snapshot_date)).where(df["horizon_days"].notna())
    return df


def label_partial_drop(df: pd.DataFrame, con: duckdb.DuckDBPyConnection, snapshot_date: date, cfg: LabelCfg) -> pd.DataFrame:
    """Second score. Lost = at least one purchase after T, but revenue over the next 180 days below
    `drop_threshold` x the revenue of the 180 days before T."""
    df = _add_truth(df, con, snapshot_date, cfg)
    t = f"{snapshot_date:%Y-%m-%d}"
    after = con.execute(f"""
        SELECT client_id, count(*) AS n_after, sum(rev) AS rev_after FROM bn
        WHERE d >= DATE '{t}' AND d < DATE '{t}' + {cfg.horizon_days} GROUP BY client_id""").fetchdf()
    df = df.merge(after, on="client_id", how="left")
    partial = (df["n_after"].fillna(0) > 0) & (df["rev_after"].fillna(0.0) < cfg.drop_threshold * df["rev_180d"])
    df["horizon_days"] = float(cfg.horizon_days)
    df["label"] = _keep_if(partial.astype(float), cfg.horizon_days <= _days_observed(con, snapshot_date))
    return df.drop(columns=["n_after", "rev_after"])


LABELS: dict[str, LabelFn] = {
    "fixed_horizon": label_fixed_horizon,
    "relative_horizon": label_relative_horizon,
    "partial_drop": label_partial_drop,
}

# ------------------------------------------------------------------------------ segments


def rhythm_class(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """`fast` / `medium` / `slow` from the client's median gap, `short_history` below `min_purchases`.
    The model perimeter keeps `fast` and `medium` (regular clients).

    Bounds: `fast` up to `rhythm_bounds_days.fast_max`, `medium` up to `max_median_gap_days` (the upper bound of
    the perimeter), `slow` above."""
    fast_max = cfg.label.rhythm_bounds_days["fast_max"]
    regular_max = cfg.label.max_median_gap_days
    cls = np.select(
        [df["freq_180d"] < cfg.label.min_purchases, df["gap_median"] <= fast_max, df["gap_median"] <= regular_max],
        ["short_history", "fast", "medium"],
        default="slow",
    )
    return pd.Series(cls, index=df.index, name="rhythm")


def in_perimeter(df: pd.DataFrame) -> pd.Series:
    """True for the regular clients (fast or medium rhythm): the clients the model is trained and evaluated on."""
    return df["rhythm"].isin(PERIMETER_RHYTHMS)


def frequency_segment(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Segment by purchase rhythm: the rhythm class itself (fast / medium inside the perimeter)."""
    return rhythm_class(df, cfg)


def value_segment(df: pd.DataFrame) -> pd.Series:
    """Segment by revenue decile (key accounts live in the top deciles): `d01` (smallest) to `d10`, on `rev_180d`,
    separately for each `snapshot_date` when the column exists."""

    def deciles(rev: pd.Series) -> pd.Series:
        rank = rev.rank(method="first")  # ties broken by order: deciles of equal size
        return pd.Series(["d%02d" % (b + 1) for b in np.minimum((rank - 1) * 10 // len(rev), 9).astype(int)], index=rev.index)

    if "snapshot_date" in df.columns:
        return df.groupby("snapshot_date", group_keys=False)["rev_180d"].apply(deciles).rename("segment")
    return deciles(df["rev_180d"]).rename("segment")


def assign_segments(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add the `segment` column according to `ModelCfg.segment_by`."""
    df = df.copy()
    if cfg.model.segment_by == "frequency":
        df["segment"] = frequency_segment(df, cfg)
    elif cfg.model.segment_by == "value":
        df["segment"] = value_segment(df)
    else:
        raise ValueError(f"Unknown segment_by {cfg.model.segment_by!r}")
    return df


# --------------------------------------------------------------------------------- splits


def temporal_split(snapshots: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, test): train = snapshots from `train_start` to `train_end`, test = the `test_date`
    snapshot. The last training date is `Config.last_train_date()`. Calls `assert_labels_known_before_test`
    before returning. All clients are kept: the perimeter (`in_perimeter`) is applied by the caller."""
    dates = pd.to_datetime(snapshots["snapshot_date"])
    start, end = pd.Timestamp(cfg.snapshots.train_start), pd.Timestamp(cfg.last_train_date())
    test_date = pd.Timestamp(cfg.snapshots.test_date)
    train = snapshots[(dates >= start) & (dates <= end)].reset_index(drop=True)
    test = snapshots[dates == test_date].reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError(f"No snapshot to train on ({len(train)} rows) or test on ({len(test)} rows)")
    assert_labels_known_before_test(
        sorted({d.date() for d in pd.to_datetime(train["snapshot_date"])}), test_date.date(), cfg.label_wait_days()
    )
    if train.loc[in_perimeter(train), "label"].isna().any():
        raise ValueError("Training rows of the perimeter have no label: the data ends before their label window")
    return train, test


def assert_labels_known_before_test(train_dates: list[date], test_date: date, wait_days: int) -> None:
    """Raise if the label window of any training snapshot ends after `test_date`
    (that would let the model see the future). `wait_days` = `Config.label_wait_days()`.

    A label window is [T, T + wait_days), so its last day is T + wait_days - 1: it may end the day before the test date."""
    late = [d for d in train_dates if d + timedelta(days=wait_days) > test_date]
    if late:
        raise ValueError(
            f"Label windows of {len(late)} training snapshot(s) (from {late[0]}) end after the test date {test_date} "
            f"(wait {wait_days} days): the model would see the future"
        )


def client_group_folds(
    df: pd.DataFrame, n_splits: int, group_by: str = "client_id", seed: int = 0
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Grouped K-fold: all snapshots of a client stay in the same fold.

    The fold of a client is a hash of its id (and the seed), not a function of the other rows: a client keeps
    its fold when snapshots are added, which keeps the inner folds consistent from one backtest date to the next.
    Yields (train positions, validation positions)."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    with np.errstate(over="ignore"):  # uint64 arithmetic wraps around on purpose (splitmix64 mixing of the seed)
        x = pd.util.hash_array(df[group_by].to_numpy()) + np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    fold = (x % np.uint64(n_splits)).astype(np.int64)
    for k in range(n_splits):
        valid = np.flatnonzero(fold == k)
        if valid.size == 0:
            raise ValueError(f"Fold {k} is empty: too few clients for {n_splits} splits")
        yield np.flatnonzero(fold != k), valid
