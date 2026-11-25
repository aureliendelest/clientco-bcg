"""Raw data ingestion and DuckDB access.

The raw CSV is never modified. The shifted Parquet is the only data source used by the rest of
the project, so no other module applies any date shift. Heavy computation stays in DuckDB.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from churn.config import ROOT, Config, DatasetCfg, resolve

# A purchase day is a (client, date) whose net revenue is above one cent. The threshold is not cosmetic: a
# purchase returned the same day nets to +-1e-13, and DuckDB gives no guarantee on the order of a floating point sum.
PURCHASE_DAY_MIN_NET = 0.01

_CSV_COLUMNS = (
    "{'date_order':'DATE','date_invoice':'DATE','product_id':'BIGINT','client_id':'BIGINT',"
    "'sales_net':'DOUBLE','quantity':'DOUBLE','order_channel':'VARCHAR','branch_id':'BIGINT'}"
)
_TEMP_DIR = ROOT / "data" / ".duckdb_tmp"  # DuckDB spills here when a query exceeds the memory limit
_FINGERPRINT_CHUNK = 1 << 20  # 1 MiB read at the start, the middle and the end of the file


def _new_connection(memory_limit: str = "4GB") -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    _TEMP_DIR.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET temp_directory='{_TEMP_DIR}'")
    return con


def _note_path(parquet: Path) -> Path:
    return parquet.with_name(parquet.name + ".shift.json")


def ingest(cfg: Config, *, overwrite: bool = False) -> Path:
    """Read `cfg.dataset.raw_csv`, add `shift_years` years to `date_order` and `date_invoice`,
    and write `cfg.dataset.shifted_parquet` (zstd) plus a small note recording the shift.

    Fails if the output already exists and `overwrite` is False (prevents a double shift).
    Returns the path of the written Parquet file.
    """
    raw = resolve(cfg.dataset.raw_csv)
    out = resolve(cfg.dataset.shifted_parquet)
    years = int(cfg.dataset.shift_years)
    if not raw.exists():
        raise FileNotFoundError(f"Raw CSV not found: {raw}")
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} already exists (use overwrite=True / --overwrite to rebuild it from the raw CSV)")

    # Calendar shift: 29/02/2024 becomes 28/02/2026, so these two days merge into one (the only day of the
    # dataset concerned, one purchase day fewer for the clients who bought on both).
    shifted = "CAST({col} + to_years(" + str(years) + ") AS DATE)"
    tmp = out.with_name(out.name + ".tmp")
    out.parent.mkdir(parents=True, exist_ok=True)
    con = _new_connection()
    try:
        con.execute(f"""
            COPY (
                SELECT {shifted.format(col="date_order")} AS date_order,
                       {shifted.format(col="date_invoice")} AS date_invoice,
                       product_id, client_id, sales_net, quantity, order_channel, branch_id
                FROM read_csv('{raw}', delim=';', header=true, columns={_CSV_COLUMNS})
            ) TO '{tmp}' (FORMAT parquet, COMPRESSION zstd)""")
        rows, first, last = con.execute(
            f"SELECT count(*), min(date_order), max(date_order) FROM read_parquet('{tmp}')"
        ).fetchone()
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    finally:
        con.close()
    tmp.replace(out)  # atomic: a half-written file is never taken for the dataset

    note = {
        "shift_years": years,
        "raw_csv": str(raw),
        "raw_csv_bytes": raw.stat().st_size,
        "rows": rows,
        "first_date_order": str(first),
        "last_date_order": str(last),
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "warning": "Dates are fictitious (shifted). Never shift this file again; rebuild it from the raw CSV.",
    }
    _note_path(out).write_text(json.dumps(note, indent=2) + "\n", encoding="utf-8")
    return out


def fingerprint(path: str | Path) -> str:
    """Cheap, stable fingerprint of a data file (size + hash of a sample of its content).
    Recorded in `meta.json` of every run."""
    p = Path(path)
    size = p.stat().st_size
    digest = hashlib.sha256(str(size).encode())
    with p.open("rb") as f:
        for offset in (0, max(size // 2 - _FINGERPRINT_CHUNK // 2, 0), max(size - _FINGERPRINT_CHUNK, 0)):
            f.seek(offset)
            digest.update(f.read(_FINGERPRINT_CHUNK))
    return f"{size}-{digest.hexdigest()[:16]}"


def connect(parquet: str | Path | None = None) -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB connection and expose the sales Parquet as the view `sales`.
    Defaults to `DatasetCfg.shifted_parquet`."""
    path = resolve(parquet if parquet is not None else DatasetCfg().shifted_parquet)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Build it once from the raw CSV with `python -m churn ingest`.")
    con = _new_connection()
    con.execute(f"CREATE OR REPLACE VIEW sales AS SELECT * FROM read_parquet('{path}')")
    return con


def build_client_day_table(con: duckdb.DuckDBPyConnection) -> None:
    """Create the tables `cd` (one row per client and order date, with the net revenue of the day) and `bn`
    (purchase days only). A purchase day is a (client_id, date_order) whose net revenue (sales minus the
    returns of the day) is > 0.01 EUR: a purchase returned the same day is not a purchase.

    Reference implementation: the "Table (client, jour d'achat)" cells of
    notebooks/01_data_quality_and_activity_overview.ipynb and notebooks/04_churn_scope_total_vs_partial.ipynb.
    """
    con.execute("""
        CREATE OR REPLACE TABLE cd AS
        SELECT client_id, date_order AS d, sum(sales_net) AS rev_net
        FROM sales
        GROUP BY client_id, date_order""")
    con.execute(f"""
        CREATE OR REPLACE TABLE bn AS
        SELECT client_id, d, rev_net AS rev
        FROM cd
        WHERE rev_net > {PURCHASE_DAY_MIN_NET}""")


# Tables behind the `extended` features, all built from the view `sales` and `bn`. They are cheap (a few seconds on
# the 63 M rows) and only needed by feature sets that use them, so they are separate from `cd` / `bn`.
FEATURE_TABLES = ("act_day", "cd_lines", "cpd", "ccd", "cbd")


def has_feature_tables(con: duckdb.DuckDBPyConnection) -> bool:
    """True when `build_feature_tables` has already run on this connection."""
    names = {row[0] for row in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    return all(t in names for t in FEATURE_TABLES)


def build_feature_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Create the tables of the `extended` features (needs `build_client_day_table` first).

    Every table is dated by `date_order` (column `d`) so the snapshot queries cut the history strictly before T.
    The notion of a purchase day is the one of `bn` (net revenue of the client-day > 0.01 EUR):

    - `act_day(d, n_days, rev)`: the whole market, one row per day: purchase days of all clients and their revenue
      (the global activity behind `mkt_freq_trend` and `mkt_rev_trend`).
    - `cd_lines(client_id, d, pos, neg)`: per client-day, the sum of the lines with `sales_net` > 0 (`pos`) and the sum
      of the lines with `sales_net` < 0 (`neg`, a negative number). ALL lines, purchase day or not: a credit note on a
      day without purchase is still a negative line (`negative_lines_share_180d`).
    - `cpd(client_id, d, product_id, rev)`, `ccd(client_id, d, channel, rev)`: net revenue per product / per channel and
      client-day, **on the purchase days of `bn` only**, lines of the day netted (so the revenue of a window sums to the
      `rev_*` columns of the snapshot). `product_div_trend`, `product_loss_share`, `channel_shift`.
    - `cbd(client_id, d, branch_id, rev)`: per client, branch and day, only when the net revenue of the client at that
      branch that day is > 0.01: a purchase day of the branch (`branch_freq_trend`). The rule is applied at branch level
      (not through `bn`) so that a client buying at two branches on one day counts one day at each.
    """
    con.execute("""
        CREATE OR REPLACE TABLE act_day AS
        SELECT d, count(*) AS n_days, sum(rev) AS rev FROM bn GROUP BY d""")
    con.execute("""
        CREATE OR REPLACE TABLE cd_lines AS
        SELECT client_id, date_order AS d,
               sum(sales_net) FILTER (WHERE sales_net > 0) AS pos,
               sum(sales_net) FILTER (WHERE sales_net < 0) AS neg
        FROM sales GROUP BY client_id, date_order""")
    # Group the lines first and keep the purchase days afterwards: joining the 63 M raw lines to `bn` first was
    # 10 times slower for the channel table (string keys).
    con.execute("""
        CREATE OR REPLACE TABLE cpd AS
        SELECT g.client_id, g.d, g.product_id, g.rev
        FROM (SELECT client_id, date_order AS d, product_id, sum(sales_net) AS rev
              FROM sales GROUP BY client_id, date_order, product_id) g
        SEMI JOIN bn ON g.client_id = bn.client_id AND g.d = bn.d""")
    con.execute("""
        CREATE OR REPLACE TABLE ccd AS
        SELECT g.client_id, g.d, g.channel, g.rev
        FROM (SELECT client_id, date_order AS d, order_channel AS channel, sum(sales_net) AS rev
              FROM sales GROUP BY client_id, date_order, order_channel) g
        SEMI JOIN bn ON g.client_id = bn.client_id AND g.d = bn.d""")
    con.execute(f"""
        CREATE OR REPLACE TABLE cbd AS
        SELECT client_id, d, branch_id, rev
        FROM (SELECT client_id, date_order AS d, branch_id, sum(sales_net) AS rev
              FROM sales GROUP BY client_id, date_order, branch_id)
        WHERE rev > {PURCHASE_DAY_MIN_NET}""")
