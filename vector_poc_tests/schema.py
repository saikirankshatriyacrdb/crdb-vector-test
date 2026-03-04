"""
Schema management for the Vector PoC.

Creates / drops tables for both the INLINE and SIDECAR models,
plus C-SPANN vector indexes (CockroachDB's native vector indexing).

Updated to support beam_size tuning for C-SPANN index configuration.
"""
from __future__ import annotations
import psycopg
from rich.console import Console
import config

console = Console()


# -- SQL Templates ------------------------------------------------------------

INLINE_TABLE = """
CREATE TABLE IF NOT EXISTS {table} (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    payload    JSONB,
    embedding  VECTOR({dim})
);
"""

SIDECAR_RECORD_TABLE = """
CREATE TABLE IF NOT EXISTS {records_table} (
    id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    payload  JSONB
);
"""

SIDECAR_VECTOR_TABLE = """
CREATE TABLE IF NOT EXISTS {vectors_table} (
    record_id  UUID PRIMARY KEY REFERENCES {records_table}(id),
    embedding  VECTOR({dim})
);
"""

# CockroachDB C-SPANN vector index with tunable parameters
CSPANN_INDEX = """
CREATE INDEX IF NOT EXISTS {idx_name}
ON {table} USING CSPANN (embedding);
"""

# C-SPANN index with explicit partition/beam parameters
CSPANN_INDEX_WITH_OPTS = """
CREATE INDEX IF NOT EXISTS {idx_name}
ON {table} USING CSPANN (embedding)
WITH (
    min_partition_size = {min_partition_size},
    max_partition_size = {max_partition_size},
    build_beam_size = {build_beam_size}
);
"""

# Alternative: CREATE VECTOR INDEX syntax
VECTOR_INDEX = """
CREATE VECTOR INDEX IF NOT EXISTS {idx_name}
ON {table} (embedding);
"""

DROP_TABLE = "DROP TABLE IF EXISTS {table} CASCADE;"


def _run(conn, sql: str, label: str):
    """Execute DDL and log outcome."""
    try:
        conn.execute(sql)
        conn.commit()
        console.print(f"  [green]\u2713[/green] {label}")
    except Exception as e:
        conn.rollback()
        console.print(f"  [red]\u2717[/red] {label}: {e}")
        raise


def enable_vector_extension(conn):
    """Enable pgvector if available."""
    for stmt in [
        "CREATE EXTENSION IF NOT EXISTS vector;",
        "SET CLUSTER SETTING sql.defaults.vectorize = 'on';",
    ]:
        try:
            conn.execute(stmt)
            conn.commit()
        except Exception:
            conn.rollback()


def setup_inline(conn, table: str = "embeddings_inline", dim: int = config.VECTOR_DIM):
    """Create inline-model table."""
    console.print(f"\n[bold]Setting up inline table:[/bold] {table}")
    _run(conn, INLINE_TABLE.format(table=table, dim=dim), f"CREATE TABLE {table}")
    return table


def setup_sidecar(conn, prefix: str = "sidecar", dim: int = config.VECTOR_DIM):
    """Create sidecar-model tables."""
    rtbl = f"{prefix}_records"
    vtbl = f"{prefix}_vectors"
    console.print(f"\n[bold]Setting up sidecar tables:[/bold] {rtbl}, {vtbl}")
    _run(conn, SIDECAR_RECORD_TABLE.format(records_table=rtbl), f"CREATE TABLE {rtbl}")
    _run(conn, SIDECAR_VECTOR_TABLE.format(vectors_table=vtbl, records_table=rtbl, dim=dim), f"CREATE TABLE {vtbl}")
    return rtbl, vtbl


def create_vector_index(conn, table: str, op: str = "cosine", method: str = "cspann",
                        use_tuned_params: bool = True):
    """
    Create a C-SPANN vector index on a CockroachDB table.

    When use_tuned_params=True, applies the partition and build beam size
    settings from config for optimal recall/latency trade-off.
    """
    idx_name = f"idx_{table}_{op}"
    console.print(f"\n[bold]Creating C-SPANN vector index:[/bold] {idx_name}")

    # Try with tuned parameters first
    if use_tuned_params:
        try:
            sql = CSPANN_INDEX_WITH_OPTS.format(
                idx_name=idx_name,
                table=table,
                min_partition_size=config.MIN_PARTITION_SIZE,
                max_partition_size=config.MAX_PARTITION_SIZE,
                build_beam_size=config.BUILD_BEAM_SIZE,
            )
            _run(conn, sql, f"C-SPANN index {idx_name} (tuned: partition={config.MIN_PARTITION_SIZE}-{config.MAX_PARTITION_SIZE}, build_beam={config.BUILD_BEAM_SIZE})")
            return idx_name
        except Exception:
            console.print("  [yellow]Tuned C-SPANN syntax failed, trying default...[/yellow]")

    # Try default USING CSPANN
    try:
        sql = CSPANN_INDEX.format(idx_name=idx_name, table=table)
        _run(conn, sql, f"C-SPANN index {idx_name}")
        return idx_name
    except Exception:
        console.print("  [yellow]USING CSPANN syntax failed, trying CREATE VECTOR INDEX...[/yellow]")

    sql = VECTOR_INDEX.format(idx_name=idx_name, table=table)
    _run(conn, sql, f"VECTOR INDEX {idx_name}")
    return idx_name


def set_search_beam_size(conn, beam_size: int = config.VECTOR_SEARCH_BEAM_SIZE):
    """
    Set the vector_search_beam_size session variable.

    Controls the number of partitions explored during vector search queries.
    Higher values increase recall but reduce QPS:
      - 32 (default): low recall on random data, moderate on real data
      - 128: ~90% recall on real-world data (recommended production setting)
      - 2048: ~95% recall on random data (too slow for production)
    """
    try:
        conn.execute(f"SET vector_search_beam_size = {beam_size}")
        console.print(f"  [green]\u2713[/green] vector_search_beam_size = {beam_size}")
    except Exception as e:
        # Try alternative syntax for different CRDB versions
        try:
            conn.execute(f"SET SESSION vector_search_beam_size = {beam_size}")
            console.print(f"  [green]\u2713[/green] vector_search_beam_size = {beam_size} (session)")
        except Exception:
            console.print(f"  [yellow]Could not set beam_size: {e}[/yellow]")


def drop_all(conn, tables=None):
    """Drop specified tables or the default set (includes all per-test tables)."""
    if tables is None:
        tables = [
            # Legacy shared tables
            "sidecar_vectors", "sidecar_records",
            "embeddings_inline",
            "embeddings_test",
            # Per-test-case tables
            "tc_qp01_inline",
            "tc_qp02_inline",
            "tc_qp03_inline",
            "tc_iu01_inline",
            "tc_iu02_inline",
            "tc_iu03_inline",
            "tc_fc01_vectors",
            "tc_fc02_vectors",
            "tc_as01_sidecar_vectors", "tc_as01_sidecar_records",
            "tc_as01_inline",
            "tc_as02_sidecar_vectors", "tc_as02_sidecar_records",
            "tc_as02_inline",
        ]
    for t in tables:
        try:
            conn.execute(DROP_TABLE.format(table=t))
            conn.commit()
            console.print(f"  [dim]Dropped {t}[/dim]")
        except Exception:
            conn.rollback()


def reset(conn):
    """Full reset: drop everything and recreate."""
    drop_all(conn)
    enable_vector_extension(conn)
    setup_inline(conn)
    setup_sidecar(conn)
