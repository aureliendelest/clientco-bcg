"""Shared fixtures: a tiny, hand-made sales table exposed to DuckDB like the real `sales` view."""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from churn.data import build_client_day_table


def make_sales_con(purchases: dict[int, list[str]], *, amount: float = 100.0, extra: list[tuple] | None = None):
    """DuckDB connection with a `sales` view and the `cd` / `bn` tables.

    `purchases` maps a client to its purchase dates (one line of `amount` EUR each). `extra` lists more raw lines
    (client_id, date, sales_net), for returns for example.
    """
    rows = [(c, pd.Timestamp(d), amount) for c, dates in purchases.items() for d in dates]
    rows += [(c, pd.Timestamp(d), v) for c, d, v in (extra or [])]
    df = pd.DataFrame(rows, columns=["client_id", "date_order", "sales_net"])
    df["date_order"] = df["date_order"].dt.date
    df["date_invoice"] = df["date_order"]
    df["product_id"], df["quantity"], df["order_channel"], df["branch_id"] = 1, 1.0, "online", 1
    con = duckdb.connect()
    con.register("sales_df", df)
    con.execute("CREATE VIEW sales AS SELECT * FROM sales_df")
    build_client_day_table(con)
    return con


@pytest.fixture()
def sales_con():
    """Factory fixture: `sales_con({client: [dates]}, ...)`."""
    return make_sales_con


# --------------------------------------------------------------------------- a small dataset for pipeline tests


def _write_synthetic_sales(path) -> None:
    """800 regular clients buying every 3 to 15 days from 2024-09-22 to 2026-09-22 (the shifted dates of the
    real dataset); 45 % of them stop at a random date between 2025-04-01 and 2026-06-30."""
    rng = np.random.default_rng(0)
    start, end = pd.Timestamp("2024-09-22"), pd.Timestamp("2026-09-22")
    rows = []
    for client in range(1, 801):
        step = int(rng.integers(3, 16))
        stop = pd.Timestamp("2025-04-01") + pd.Timedelta(days=int(rng.integers(0, 456))) if rng.uniform() < 0.45 else end
        day = start + pd.Timedelta(days=int(rng.integers(0, step)))
        while day <= stop:
            rows.append((client, day, float(rng.uniform(50, 500))))
            day += pd.Timedelta(days=int(max(1, step + rng.integers(-1, 2))))
    df = pd.DataFrame(rows, columns=["client_id", "date_order", "sales_net"])
    df["date_order"] = df["date_order"].dt.date
    df["date_invoice"] = df["date_order"]
    df["product_id"], df["quantity"], df["order_channel"], df["branch_id"] = 1, 1.0, "online", 1
    duckdb.connect().register("df", df).execute(f"COPY df TO '{path}' (FORMAT parquet)")


@pytest.fixture(scope="session")
def pipeline_cfg(tmp_path_factory):
    """A `Config` on the synthetic dataset (label A, 3 CV folds, logistic regression). The snapshot cache and the
    shifted Parquet live in a temporary folder."""
    from churn.config import Config, DatasetCfg, ModelCfg
    from churn.snapshots import load_or_build

    folder = tmp_path_factory.mktemp("pipeline")
    parquet = folder / "sales_shifted.parquet"
    _write_synthetic_sales(parquet)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("churn.snapshots.PROCESSED_DIR", folder / "processed")
        cfg = Config(name="synthetic", dataset=DatasetCfg(shifted_parquet=str(parquet)),
                     model=ModelCfg(type="logreg", cv_folds=3))
        load_or_build(cfg)  # build the snapshots once for the whole session
        yield cfg
