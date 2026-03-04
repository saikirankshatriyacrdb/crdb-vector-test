"""
TC-AS-01: Inline vs. Sidecar Model Comparison
TC-AS-02: Storage Footprint at Scale       -> 3-4 GB for 500K x 1536-dim

Updated to use VectorDBBench real-world dataset (Performance1536D500K).
Each test case uses its own dedicated tables to avoid conflicts.
"""
from __future__ import annotations
import time
import numpy as np
from rich.console import Console
import config
import db
import schema
import ingest
from datagen import (
    load_train_vectors, load_query_vectors, load_dataset,
    vec_to_pgvector, has_real_dataset, generate_vectors,
)

console = Console()

# -- Table names (unique per test case) ----------------------------------------
TABLE_AS01_INLINE = "tc_as01_inline"
TABLE_AS01_SIDECAR_RECORDS = "tc_as01_sidecar_records"
TABLE_AS01_SIDECAR_VECTORS = "tc_as01_sidecar_vectors"

TABLE_AS02_INLINE = "tc_as02_inline"
TABLE_AS02_SIDECAR_RECORDS = "tc_as02_sidecar_records"
TABLE_AS02_SIDECAR_VECTORS = "tc_as02_sidecar_vectors"


# -- TC-AS-01: Inline vs. Sidecar ---------------------------------------------

def _measure_query_latency(conn, table: str, query_vecs: np.ndarray, k: int = 10,
                           id_col: str = "id") -> dict:
    """Measure query latency stats against a table."""
    latencies = []
    for qv in query_vecs:
        vec_str = vec_to_pgvector(qv)
        t0 = time.perf_counter()
        conn.execute(
            f"SELECT {id_col}, embedding <-> %s::vector AS score FROM {table} ORDER BY embedding <-> %s::vector LIMIT %s",
            [vec_str, vec_str, k],
        ).fetchall()
        latencies.append((time.perf_counter() - t0) * 1000)
    return {
        "avg_ms": round(np.mean(latencies), 2),
        "p50_ms": round(np.percentile(latencies, 50), 2),
        "p95_ms": round(np.percentile(latencies, 95), 2),
        "p99_ms": round(float(np.percentile(latencies, 99)), 2),
    }


def _measure_insert_latency(conn, model: str, vecs: np.ndarray,
                              inline_table: str = None,
                              records_table: str = None,
                              vectors_table: str = None) -> dict:
    """Measure single-insert latency for a model."""
    latencies = []
    for v in vecs:
        if model == "inline":
            _, elapsed = ingest.insert_single_inline(conn, v, table=inline_table)
        else:
            _, elapsed = ingest.insert_single_sidecar(
                conn, v, records_table=records_table, vectors_table=vectors_table
            )
        latencies.append(elapsed)
    return {
        "avg_ms": round(np.mean(latencies), 2),
        "p95_ms": round(np.percentile(latencies, 95), 2),
        "p99_ms": round(float(np.percentile(latencies, 99)), 2),
    }


def run_tc_as_01() -> dict:
    """
    TC-AS-01: Inline vs. Sidecar Model Comparison.
    Load same dataset into both models, compare query and insert latency.
    Pass criteria: Both models meet query and ingestion thresholds.
    """
    console.rule("[bold blue]TC-AS-01: Inline vs. Sidecar Comparison")
    console.print(f"  Dataset: {'VectorDBBench real-world' if has_real_dataset() else 'Synthetic'}")
    test_size = min(50000, config.SMALL_DATASETS[0])
    n_queries = 200
    n_inserts = 200

    query_vecs = load_query_vectors(n=n_queries) if has_real_dataset() else generate_vectors(n_queries, seed=4444)
    # Use vectors from later in the dataset for insert testing
    if has_real_dataset():
        insert_vecs = load_train_vectors(n=test_size + n_inserts)[test_size:]
    else:
        insert_vecs = generate_vectors(n_inserts, seed=5556)
    dataset = load_dataset(f"ds_{test_size}")

    results = {}

    # -- Inline Model --
    console.print("\n[bold]Testing INLINE model[/bold]")
    with db.connect() as conn:
        conn.execute(f"DROP TABLE IF EXISTS {TABLE_AS01_INLINE} CASCADE")
        conn.commit()
        schema.enable_vector_extension(conn)
        schema.setup_inline(conn, table=TABLE_AS01_INLINE)

        console.print(f"  Loading {test_size:,} vectors...")
        ingest.batch_insert_inline(conn, dataset, show_progress=True, table=TABLE_AS01_INLINE)
        schema.create_vector_index(conn, TABLE_AS01_INLINE, op="l2")
        schema.set_search_beam_size(conn, config.VECTOR_SEARCH_BEAM_SIZE)

        console.print("  Measuring query latency...")
        q_stats = _measure_query_latency(conn, TABLE_AS01_INLINE, query_vecs)
        console.print(f"    Query — avg: {q_stats['avg_ms']}ms  p95: {q_stats['p95_ms']}ms  p99: {q_stats['p99_ms']}ms")

        console.print("  Measuring insert latency...")
        i_stats = _measure_insert_latency(conn, "inline", insert_vecs, inline_table=TABLE_AS01_INLINE)
        console.print(f"    Insert — avg: {i_stats['avg_ms']}ms  p95: {i_stats['p95_ms']}ms  p99: {i_stats['p99_ms']}ms")

    results["inline"] = {
        "query": q_stats,
        "insert": i_stats,
        "query_pass": q_stats["p95_ms"] < config.QUERY_P95_SMALL_MS,
        "insert_pass": i_stats["avg_ms"] < config.INSERT_AVG_MS,
    }

    # -- Sidecar Model --
    console.print("\n[bold]Testing SIDECAR model[/bold]")
    with db.connect() as conn:
        conn.execute(f"DROP TABLE IF EXISTS {TABLE_AS01_SIDECAR_VECTORS} CASCADE")
        conn.execute(f"DROP TABLE IF EXISTS {TABLE_AS01_SIDECAR_RECORDS} CASCADE")
        conn.commit()
        schema.enable_vector_extension(conn)
        schema.setup_sidecar(conn, prefix="tc_as01_sidecar")

        console.print(f"  Loading {test_size:,} vectors...")
        ingest.batch_insert_sidecar(conn, dataset, show_progress=True,
                                     records_table=TABLE_AS01_SIDECAR_RECORDS,
                                     vectors_table=TABLE_AS01_SIDECAR_VECTORS)
        schema.create_vector_index(conn, TABLE_AS01_SIDECAR_VECTORS, op="l2")
        schema.set_search_beam_size(conn, config.VECTOR_SEARCH_BEAM_SIZE)

        console.print("  Measuring query latency...")
        q_stats = _measure_query_latency(conn, TABLE_AS01_SIDECAR_VECTORS, query_vecs, id_col="record_id")
        console.print(f"    Query — avg: {q_stats['avg_ms']}ms  p95: {q_stats['p95_ms']}ms  p99: {q_stats['p99_ms']}ms")

        console.print("  Measuring insert latency...")
        i_stats = _measure_insert_latency(conn, "sidecar", insert_vecs,
                                            records_table=TABLE_AS01_SIDECAR_RECORDS,
                                            vectors_table=TABLE_AS01_SIDECAR_VECTORS)
        console.print(f"    Insert — avg: {i_stats['avg_ms']}ms  p95: {i_stats['p95_ms']}ms  p99: {i_stats['p99_ms']}ms")

    results["sidecar"] = {
        "query": q_stats,
        "insert": i_stats,
        "query_pass": q_stats["p95_ms"] < config.QUERY_P95_SMALL_MS,
        "insert_pass": i_stats["avg_ms"] < config.INSERT_AVG_MS,
    }

    overall_pass = (
        results["inline"]["query_pass"] and results["inline"]["insert_pass"]
        and results["sidecar"]["query_pass"] and results["sidecar"]["insert_pass"]
    )

    console.print(f"\n  Inline  — query p95: {results['inline']['query']['p95_ms']}ms, insert avg: {results['inline']['insert']['avg_ms']}ms")
    console.print(f"  Sidecar — query p95: {results['sidecar']['query']['p95_ms']}ms, insert avg: {results['sidecar']['insert']['avg_ms']}ms")
    console.print(f"\n[bold]TC-AS-01 Overall: {'[green]PASS[/green]' if overall_pass else '[red]FAIL[/red]'}[/bold]")

    return {
        "test_id": "TC-AS-01", "pass": overall_pass,
        "threshold": "Both models meet query + ingestion thresholds",
        "beam_size": config.VECTOR_SEARCH_BEAM_SIZE,
        "dataset_type": "real-world" if has_real_dataset() else "synthetic",
        "results": results,
    }


# -- TC-AS-02: Storage Footprint ----------------------------------------------

def run_tc_as_02() -> dict:
    """
    TC-AS-02: Storage Footprint at Scale.
    Load 500K vectors (or largest available) and measure storage.
    Pass criteria: logical storage within 3-4 GB; no bloat > 5 GB.
    """
    console.rule("[bold blue]TC-AS-02: Storage Footprint at Scale")
    ds_size = config.LARGE_DATASET

    storage_results = {}

    for model in ["inline", "sidecar"]:
        console.print(f"\n[bold]Model: {model.upper()}[/bold]")

        with db.connect() as conn:
            if model == "inline":
                conn.execute(f"DROP TABLE IF EXISTS {TABLE_AS02_INLINE} CASCADE")
                conn.commit()
                schema.enable_vector_extension(conn)
                schema.setup_inline(conn, table=TABLE_AS02_INLINE)

                vecs = load_dataset(f"ds_{ds_size}")
                console.print(f"  Loading {ds_size:,} vectors...")
                ingest.batch_insert_inline(conn, vecs, show_progress=True, table=TABLE_AS02_INLINE)

                row_count = db.table_row_count(conn, TABLE_AS02_INLINE)
                size_bytes = db.table_size_bytes(conn, TABLE_AS02_INLINE)
            else:
                conn.execute(f"DROP TABLE IF EXISTS {TABLE_AS02_SIDECAR_VECTORS} CASCADE")
                conn.execute(f"DROP TABLE IF EXISTS {TABLE_AS02_SIDECAR_RECORDS} CASCADE")
                conn.commit()
                schema.enable_vector_extension(conn)
                schema.setup_sidecar(conn, prefix="tc_as02_sidecar")

                vecs = load_dataset(f"ds_{ds_size}")
                console.print(f"  Loading {ds_size:,} vectors...")
                ingest.batch_insert_sidecar(conn, vecs, show_progress=True,
                                             records_table=TABLE_AS02_SIDECAR_RECORDS,
                                             vectors_table=TABLE_AS02_SIDECAR_VECTORS)

                row_count = db.table_row_count(conn, TABLE_AS02_SIDECAR_VECTORS)
                sz_recs = db.table_size_bytes(conn, TABLE_AS02_SIDECAR_RECORDS)
                sz_vecs = db.table_size_bytes(conn, TABLE_AS02_SIDECAR_VECTORS)
                size_bytes = sz_recs + sz_vecs

        size_gb = size_bytes / (1024 ** 3)
        within_range = config.STORAGE_EXPECTED_MIN_GB <= size_gb <= config.STORAGE_EXPECTED_MAX_GB
        no_bloat = size_gb < config.STORAGE_MAX_GB

        console.print(f"  Row count:    {row_count:,}")
        console.print(f"  Storage:      {size_gb:.2f} GB ({size_bytes:,} bytes)")
        console.print(f"  In range:     {within_range} (expected {config.STORAGE_EXPECTED_MIN_GB}-{config.STORAGE_EXPECTED_MAX_GB} GB)")
        console.print(f"  No bloat:     {no_bloat} (< {config.STORAGE_MAX_GB} GB)")

        storage_results[model] = {
            "row_count": row_count,
            "size_bytes": size_bytes,
            "size_gb": round(size_gb, 3),
            "within_expected_range": within_range,
            "no_bloat": no_bloat,
            "pass": no_bloat,
        }

    overall_pass = all(r["pass"] for r in storage_results.values())
    console.print(f"\n[bold]TC-AS-02 Overall: {'[green]PASS[/green]' if overall_pass else '[red]FAIL[/red]'}[/bold]")

    return {
        "test_id": "TC-AS-02", "pass": overall_pass,
        "threshold": f"< {config.STORAGE_MAX_GB} GB for {ds_size:,} x {config.VECTOR_DIM}-dim",
        "dataset_type": "real-world" if has_real_dataset() else "synthetic",
        "results": storage_results,
    }
