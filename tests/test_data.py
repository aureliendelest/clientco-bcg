"""Ingestion and client-day tables, on a tiny CSV (the real one is 4 GB)."""

import json

import duckdb
import pytest

from churn.config import Config
from churn.data import build_client_day_table, connect, fingerprint, ingest

CSV = """date_order;date_invoice;product_id;client_id;sales_net;quantity;order_channel;branch_id
2022-09-22;2022-09-23;1;10;100.0;1;online;5
2024-02-29;2024-03-01;2;10;50.0;1;online;5
2024-09-22;2024-09-22;3;11;75.5;2;at the store;6
"""


def _cfg(tmp_path, years=2) -> Config:
    raw = tmp_path / "raw.csv"
    raw.write_text(CSV, encoding="utf-8")
    cfg = Config()
    cfg.dataset.raw_csv = str(raw)
    cfg.dataset.shifted_parquet = str(tmp_path / "out" / "sales_shifted.parquet")
    cfg.dataset.shift_years = years
    return cfg


def test_ingest_shifts_both_dates_once_and_keeps_the_raw_csv(tmp_path):
    cfg = _cfg(tmp_path)
    before = (tmp_path / "raw.csv").read_bytes()
    out = ingest(cfg)
    assert (tmp_path / "raw.csv").read_bytes() == before
    rows = duckdb.connect().execute(
        f"SELECT date_order::VARCHAR, date_invoice::VARCHAR FROM read_parquet('{out}') ORDER BY client_id, date_order"
    ).fetchall()
    # 29/02/2024 + 2 years has no equivalent day: it falls on 28/02/2026
    assert rows == [("2024-09-22", "2024-09-23"), ("2026-02-28", "2026-03-01"), ("2026-09-22", "2026-09-22")]
    note = json.loads((out.with_name(out.name + ".shift.json")).read_text())
    assert note["shift_years"] == 2 and note["rows"] == 3


def test_ingest_refuses_a_second_shift_unless_overwrite(tmp_path):
    cfg = _cfg(tmp_path)
    out = ingest(cfg)
    with pytest.raises(FileExistsError):
        ingest(cfg)
    assert ingest(cfg, overwrite=True) == out  # rebuilt from the raw CSV, not shifted twice
    assert duckdb.connect().execute(f"SELECT min(date_order)::VARCHAR FROM read_parquet('{out}')").fetchone()[0] == "2024-09-22"


def test_connect_exposes_the_sales_view_and_fails_without_ingest(tmp_path):
    out = ingest(_cfg(tmp_path))
    assert connect(out).execute("SELECT count(*) FROM sales").fetchone()[0] == 3
    with pytest.raises(FileNotFoundError, match="churn ingest"):
        connect(tmp_path / "missing.parquet")


def test_fingerprint_is_stable_and_sees_changes(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"a" * 100)
    first = fingerprint(f)
    assert fingerprint(f) == first
    f.write_bytes(b"a" * 99 + b"b")
    assert fingerprint(f) != first


def test_purchase_day_is_net_of_same_day_returns(sales_con):
    con = sales_con(
        {1: ["2025-01-10", "2025-01-12"]},
        extra=[(1, "2025-01-10", -100.0),    # fully returned the same day: not a purchase day
               (2, "2025-01-10", 40.0), (2, "2025-01-10", -39.995)],  # net of 0.005 EUR: below the cent
    )
    days = con.execute("SELECT client_id, d::VARCHAR FROM bn ORDER BY 1, 2").fetchall()
    assert days == [(1, "2025-01-12")]
    assert con.execute("SELECT count(*) FROM cd").fetchone()[0] == 3  # cd keeps every (client, day)
