"""Build the local replay dashboard: `data/front/index.html`, one self-contained file.

    .venv/bin/python front/build.py

What it does: on a weekly grid, scores the regular clients with the saved monthly `final_backtest` models (the latest
one dated on or before the snapshot; no retraining), keeps the clients any list could contain, adds the reasons, and
embeds the result in `front/template.html` together with `front/engine.js` (which builds the lists in the browser).

The output holds client-level data: it stays local (`data/` is not versioned) and must not be published or shared
outside the team. The final test date is never read: the grid stops on 2026-03-02.
"""

from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from churn import models
from churn.config import RUNS_DIR, load_config, resolve
from churn.data import build_client_day_table, connect
from churn.scoring import _load_run, top_reasons
from churn.snapshots import build_snapshot, in_perimeter

FRONT = Path(__file__).parent
OUT_DIR = resolve("data/front")
MODEL_DATES = pd.to_datetime(["2025-09-22", "2025-10-22", "2025-11-22", "2025-12-22", "2026-01-22", "2026-02-22"])
# 24 weekly dates: cadences of 1, 2 or 4 weeks nest in it and cover the same 24 weeks, hence the same call budget.
GRID = pd.date_range("2025-09-22", periods=24, freq="7D")
FINAL_TEST = pd.Timestamp("2026-03-22")
# Clients embedded per date, from the top of each ranking (payload key -> depth). A list never goes below that depth: the
# page warns when it runs out. The size ranking is stable from one date to the next, so the rest period digs deeper.
DEPTH = {"ev": 5000, "evr": 5000, "v": 8000}


def score_grid() -> tuple[list[pd.DataFrame], dict]:
    """One scored frame per grid date (all the regular clients), and the model bundle used for each date."""
    bundles = {}
    for d in MODEL_DATES:
        found = sorted(RUNS_DIR.glob(f"final_backtest__snapshots.test_date-{d:%Y-%m-%d}-*"))
        if len(found) != 1:
            raise FileNotFoundError(f"Expected one final_backtest run for {d:%F}, found {len(found)} in {RUNS_DIR}")
        bundles[d] = _load_run(found[0])

    con = connect(resolve(bundles[MODEL_DATES[0]][1].dataset.shifted_parquet))
    build_client_day_table(con)
    frames, used = [], {}
    for d in GRID:
        model_date = MODEL_DATES[MODEL_DATES <= d].max()
        bundle, cfg = bundles[model_date]
        df = build_snapshot(con, d.date(), cfg)
        df = df.loc[in_perimeter(df)].sort_values("client_id").reset_index(drop=True)
        raw = bundle["model"].predict_proba(df[list(bundle["features"])])[:, 1]
        df["p_cal"] = models.apply_calibrator(bundle["calibrator"], raw, df["segment"])
        df["ev"] = df["p_cal"] * df["V"]
        df["ev_recent"] = df["p_cal"] * df["rev_90d"] * 4.0
        frames.append(df)
        used[d] = (bundle, model_date)
        print(f"{d:%F}  model {model_date:%F}  {len(df)} regular clients", flush=True)
    con.close()
    return frames, used


def _top_ids(df: pd.DataFrame, column: str, n: int) -> set:
    return set(df.sort_values([column, "client_id"], ascending=[False, True]).head(n)["client_id"])


def encode_date(d: pd.Timestamp, df: pd.DataFrame, bundle: dict, model_date, loss: pd.Series, reasons: dict) -> dict:
    """Columnar payload of one date. `reasons` maps a sentence to its index (shared by all dates)."""
    by_model = _top_ids(df, "ev", DEPTH["ev"]) | _top_ids(df, "ev_recent", DEPTH["evr"])
    keep = by_model | _top_ids(df, "V", DEPTH["v"]) | set(df.loc[df["gone"] == 1, "client_id"])
    sub = df.loc[df["client_id"].isin(keep)].reset_index(drop=True)

    codes = np.zeros((len(sub), 3), dtype=int)  # 0 = no reason
    listed = sub["client_id"].isin(by_model).to_numpy()
    text = top_reasons(bundle["model"], sub.loc[listed, list(bundle["features"])], n=3)
    codes[listed] = [[reasons.setdefault(s, len(reasons)) if s else 0 for s in row] for row in text.to_numpy()]

    def ints(col):
        return sub[col].round().astype("int64").tolist()

    return {
        "date": f"{d:%Y-%m-%d}",
        "model": f"{model_date:%Y-%m-%d}",
        "n_regular": len(df),
        "churned_v": float((df["V"] * df["gone"]).sum()),
        "id": ints("client_id"),
        "seg": sub["segment"].astype(str).tolist(),
        "p": sub["p_cal"].round(5).tolist(),
        "ev": sub["ev"].round(4).tolist(),
        "evr": sub["ev_recent"].round(4).tolist(),
        "v": sub["V"].round(2).tolist(),
        "rev180": ints("rev_180d"),
        "rev90": ints("rev_90d"),
        "revprev": ints("rev_prev_90d"),
        "freq180": ints("freq_180d"),
        "freq90": ints("freq_90d"),
        "recency": ints("recency_days"),
        "gap": sub["gap_median"].round(1).tolist(),
        "gone": ints("gone"),
        "loss": sub["client_id"].map(loss).fillna(0).round(2).tolist(),
        "r": codes.tolist(),
    }


def main() -> None:
    assert GRID.max() < FINAL_TEST, "the grid must stop before the final test date"
    frames, used = score_grid()

    full = pd.concat(frames, ignore_index=True)
    if full["gone"].isna().any():
        raise ValueError("Unknown truth on a grid date")
    # One fixed loss per churner, whatever the cadence: V at the first grid date where it is a churner.
    loss = full.loc[full["gone"] == 1].sort_values("snapshot_date").groupby("client_id")["V"].first()

    reasons: dict[str, int] = {"": 0}
    dates = []
    for d, df in zip(GRID, frames):
        bundle, model_date = used[d]
        dates.append(encode_date(d, df, bundle, model_date, loss, reasons))
        print(f"{d:%F}  {len(dates[-1]['id'])} candidates", flush=True)

    cfg = load_config(resolve("configs/experiments/final.yaml"))
    payload = {
        "dates": dates,
        "reasons": list(reasons),
        "depth": DEPTH,
        "total_loss": float(loss.sum()),
        "n_churners": len(loss),
        "defaults": {
            "k_per_week": 160,
            "gross_margin": cfg.impact.gross_margin,
            "retention": cfg.impact.retention_scenarios,
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    columns = ["client_id", "snapshot_date", "p_cal", "V", "ev", "ev_recent", "gone"]
    full[columns].to_parquet(OUT_DIR / "scores.parquet", index=False)  # full perimeter, to check the page against
    blob = base64.b64encode(gzip.compress(json.dumps(payload, allow_nan=False).encode(), 9)).decode()
    html = (FRONT / "template.html").read_text(encoding="utf-8")
    html = html.replace("/*__ENGINE__*/", (FRONT / "engine.js").read_text(encoding="utf-8")).replace("__DATA__", blob)
    path = OUT_DIR / "index.html"
    path.write_text(html, encoding="utf-8")
    print(f"written {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
