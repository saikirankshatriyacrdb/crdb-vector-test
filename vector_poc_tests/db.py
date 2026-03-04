"""
Database connection helpers and common query utilities.
"""
from __future__ import annotations
import psycopg
from psycopg.rows import dict_row
from contextlib import contextmanager
import config


@contextmanager
def connect(autocommit: bool = True):
    """Yield a psycopg connection using the configured DSN."""
    conn = psycopg.connect(config.get_dsn(), autocommit=autocommit, row_factory=dict_row)
    try:
        yield conn
    finally:
        conn.close()


def get_conn(autocommit: bool = True):
    """Return a plain connection (caller manages lifecycle)."""
    return psycopg.connect(config.get_dsn(), autocommit=autocommit, row_factory=dict_row)


def table_row_count(conn, table: str) -> int:
    """Return approximate row count for a table."""
    row = conn.execute(f"SELECT count(*) AS cnt FROM {table}").fetchone()
    return row["cnt"]


def table_size_bytes(conn, table: str) -> int:
    """Return total relation size in bytes (data + indexes + toast).
    Falls back to SHOW RANGES if pg_total_relation_size is unavailable
    (CockroachDB Cloud virtual clusters)."""
    try:
        row = conn.execute(
            "SELECT pg_total_relation_size(%s) AS sz", [table]
        ).fetchone()
        return row["sz"]
    except Exception:
        conn.rollback()
        # Fallback for CockroachDB Cloud virtual clusters:
        # SHOW RANGES WITH DETAILS exposes range_size in bytes
        rows = conn.execute(
            f"SELECT range_size FROM [SHOW RANGES FROM TABLE {table} WITH DETAILS]"
        ).fetchall()
        return sum(int(r["range_size"]) for r in rows)


def explain_query(conn, sql: str, params=None) -> str:
    """Run EXPLAIN on a query and return the plan text."""
    explain_sql = f"EXPLAIN {sql}"
    rows = conn.execute(explain_sql, params or []).fetchall()
    # CockroachDB returns 'info' column; PostgreSQL returns 'QUERY PLAN'
    key = "info" if "info" in rows[0] else list(rows[0].keys())[0]
    return "\n".join(r[key] for r in rows)
