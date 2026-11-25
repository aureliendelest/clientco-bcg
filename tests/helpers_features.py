"""Toy sales lines for the `extended` features: product, channel and branch on every line.

`tests/conftest.py` only knows one product, one channel and one branch, which is enough for the `basic` features
but not for these ones. Dates are given as an offset in days before the snapshot date `T` (`day(10)` = ten days
before T, `day(-3)` = three days after). With the default 180-day lookback the windows, in offsets, are:

    last half      1 .. 90      (offset 90 = T - 90 is in)
    previous half  91 .. 180    (offset 180 = T - 180 is in, offset 181 is out)
    not history    0 and below  (day T itself and the future)
"""

from __future__ import annotations

import duckdb
import pandas as pd

from churn.data import build_client_day_table
from tests.helpers import T

LINE_COLUMNS = ["client_id", "date_order", "product_id", "order_channel", "branch_id", "sales_net"]


def day(offset: int) -> str:
    """ISO date `offset` days before T."""
    return f"{T - pd.Timedelta(days=offset):%Y-%m-%d}"


def line(client: int, offset: int, net: float = 100.0, *, product: int = 1, channel: str = "online", branch: int = 1) -> tuple:
    """One raw sales line dated `offset` days before T."""
    return (client, day(offset), product, channel, branch, net)


def lines(client: int, offsets, net: float = 100.0, **kw) -> list[tuple]:
    """One line per offset, same product, channel and branch."""
    return [line(client, o, net, **kw) for o in offsets]


def make_con(rows: list[tuple], *, with_feature_tables: bool = False) -> duckdb.DuckDBPyConnection:
    """DuckDB connection with a `sales` view (columns of the real Parquet) and the `cd` / `bn` tables. The tables of
    the extended features are NOT built (the snapshot code builds them on demand) unless asked."""
    df = pd.DataFrame(rows, columns=LINE_COLUMNS)
    df["date_order"] = pd.to_datetime(df["date_order"]).dt.date
    df["date_invoice"] = df["date_order"]
    df["quantity"] = 1.0
    con = duckdb.connect()
    con.register("sales_df", df)
    con.execute("CREATE VIEW sales AS SELECT * FROM sales_df")
    build_client_day_table(con)
    if with_feature_tables:
        from churn.data import build_feature_tables

        build_feature_tables(con)
    return con


def features(rows: list[tuple], feature_set: str = "extended", **kw) -> pd.DataFrame:
    """`compute_features` at T on toy lines, indexed by client_id."""
    from churn.snapshots import compute_features

    return compute_features(make_con(rows), T, feature_set, **kw).set_index("client_id")


def rich_rows() -> list[tuple]:
    """A dataset that exercises every variable at once (used by the no-leakage test). Clients:

    1 regular buyer with several products, channels and branches, returns and a credit note
    2 a second regular client at the same branches (so the branch has history)
    3 a client that stopped buying 100 days before T
    4 a crowd client making branch 7 big enough for `branch_freq_trend`
    """
    rows: list[tuple] = []
    for o in range(5, 180, 7):
        rows.append(line(1, o, 120.0, product=o % 5, channel="online" if o % 2 else "by phone", branch=7 if o % 3 else 8))
    rows += [line(1, 33, -40.0, product=2), line(1, 61, -15.0, product=3, channel="by phone")]
    for o in range(2, 175, 6):
        rows.append(line(2, o, 80.0, product=10 + o % 3, channel="at the store", branch=7))
    rows += lines(3, range(100, 175, 9), 60.0, product=5, branch=8)
    rows += lines(4, range(1, 181, 2), 30.0, product=7, channel="by phone", branch=7)
    return rows
