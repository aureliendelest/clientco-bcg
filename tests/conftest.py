"""Shared fixtures: a tiny, hand-made sales table exposed to DuckDB like the real `sales` view."""

from __future__ import annotations

import duckdb
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
