"""
TC-QP-01: Top-K Query Latency (50K-250K)   -> p95 < 100 ms, p99 < 150 ms
TC-QP-02: Top-K Query Latency (500K)       -> p95 < 200 ms, p99 < 300 ms
TC-QP-03: Result Accuracy vs Ground Truth  -> recall@10 >= 90%

Updated to use VectorDBBench real-world dataset (Performance1536D500K).
Real-world embeddings have natural cluster structure, so C-SPANN performs
significantly better than with random vectors.

Key changes from synthetic version:
  - Uses OpenAI embeddings from C4 corpus (via VectorDBBench)
  - Sets vector_search_beam_size for realistic recall (default: 128)
  - Reports both p95 and p99 latency (p99 aligns with VectorDBBench metrics)
  - Uses ground truth neighbors from dataset for recall (TC-QP-03)
  - K=10 per recommendation
"""
from __future__ import annotations
import time
import numpy as np
from rich.console import Console
from rich.table import Table as RichTable
import config
import db
import schema
import ingest
from datagen import (
    load_train_vectors, load_query_vectors, load_ground_truth,
    load_dataset, vec_to_pgvector, has_real_dataset,
    generate_vectors, generate_query_vectors,
)

console = Console()

# -- Table names (unique per test case) ----------------------------------------
TABLE_QP01 = "tc_qp01_inline"
TABLE_QP02 = "tc_qp02_inline"
TABLE_QP03 = "tc_qp03_inline"

# -- Operator mapping for SQL --------------------------------------------------
OPS_SQL = {
    "l2":     "<->",
    "cosine": "<=>",
    "ip":     "<#>",
}


def _run_top_k_queries(conn, query_vecs: np.ndarray, k: int, operator: str,
                        table: str = TABLE_QP01) -> list[float]:
    """Execute top-K similarity queries and return list of latencies in ms."""
    op = OPS_SQL.get(operator, "<->")
    latencies = []
    sql = f"SELECT id, embedding {op} %s::vector AS score FROM {table} ORDER BY embedding {op} %s::vector LIMIT %s"

    for vec in query_vecs:
        vec_str = vec_to_pgvector(vec)
        t0 = time.perf_counter()
        conn.execute(sql, [vec_str, vec_str, k]).fetchall()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

    return latencies


def _percentile(data: list[float], p: float) -> float:
    """Compute percentile from a list of values."""
    arr = np.array(data)
    return float(np.percentile(arr, p))


# -- TC-QP-01: Small datasets (50K-250K) --------------------------------------

def run_tc_qp_01() -> dict:
    """
    TC-QP-01: Top-K Query Latency for datasets <= 250K vectors.
    Uses real-world VectorDBBench dataset with beam_size tuning.
    Pass criteria: p95 < 100 ms for L2 indexed queries.
    """
    console.rule("[bold blue]TC-QP-01: Top-K Query Latency (50K-250K)")
    console.print(f"  Dataset: {'VectorDBBench real-world' if has_real_dataset() else 'Synthetic (fallback)'}")
    console.print(f"  Beam size: {config.VECTOR_SEARCH_BEAM_SIZE}")

    results = {}
    query_vecs = load_query_vectors(n=config.QUERY_COUNT)

    for ds_size in config.SMALL_DATASETS:
        console.print(f"\n[bold]Dataset: {ds_size:,} vectors[/bold]")

        with db.connect() as conn:
            conn.execute(f"DROP TABLE IF EXISTS {TABLE_QP01} CASCADE")
            conn.commit()
            schema.enable_vector_extension(conn)
            schema.setup_inline(conn, table=TABLE_QP01)

            vecs = load_dataset(f"ds_{ds_size}")
            console.print(f"  Loading {ds_size:,} vectors...")
            ingest.batch_insert_inline(conn, vecs, show_progress=True, table=TABLE_QP01)

            # Build C-SPANN index
            schema.create_vector_index(conn, TABLE_QP01, op="l2")

            # Set beam size for search
            schema.set_search_beam_size(conn, config.VECTOR_SEARCH_BEAM_SIZE)

            # Warm up
            console.print("  Warming up (50 queries)...")
            _run_top_k_queries(conn, query_vecs[:50], k=10, operator="l2", table=TABLE_QP01)

            ds_results = {"iterations": [], "pass": True}
            for it in range(config.ITERATIONS):
                for k_val in config.K_VALUES:
                    # Only benchmark L2 — the only operator with C-SPANN index support.
                    # Cosine/IP would do brute-force scans (~660ms each) and are not meaningful.
                    latencies = _run_top_k_queries(conn, query_vecs, k=k_val, operator="l2", table=TABLE_QP01)
                    p50 = _percentile(latencies, 50)
                    p95 = _percentile(latencies, 95)
                    p99 = _percentile(latencies, 99)
                    avg = float(np.mean(latencies))

                    passed = p95 < config.QUERY_P95_SMALL_MS and p99 < config.QUERY_P99_SMALL_MS

                    run_result = {
                        "dataset": ds_size,
                        "iteration": it + 1,
                        "k": k_val,
                        "operator": "l2",
                        "indexed": True,
                        "beam_size": config.VECTOR_SEARCH_BEAM_SIZE,
                        "p50_ms": round(p50, 2),
                        "p95_ms": round(p95, 2),
                        "p99_ms": round(p99, 2),
                        "avg_ms": round(avg, 2),
                        "query_count": len(latencies),
                        "pass": passed,
                    }
                    ds_results["iterations"].append(run_result)
                    if not passed:
                        ds_results["pass"] = False

                    status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
                    console.print(
                        f"    iter={it+1} K={k_val}: "
                        f"p50={p50:.1f}ms  p95={p95:.1f}ms  p99={p99:.1f}ms  {status}"
                    )

            results[ds_size] = ds_results

    overall_pass = all(r["pass"] for r in results.values())
    console.print(f"\n[bold]TC-QP-01 Overall: {'[green]PASS[/green]' if overall_pass else '[red]FAIL[/red]'}[/bold]")
    return {
        "test_id": "TC-QP-01", "pass": overall_pass,
        "threshold": f"p95 < {config.QUERY_P95_SMALL_MS} ms, p99 < {config.QUERY_P99_SMALL_MS} ms",
        "beam_size": config.VECTOR_SEARCH_BEAM_SIZE,
        "dataset_type": "real-world" if has_real_dataset() else "synthetic",
        "datasets": results,
    }


# -- TC-QP-02: Large dataset (500K) -------------------------------------------

def run_tc_qp_02() -> dict:
    """
    TC-QP-02: Top-K Query Latency for 500K vectors.
    Pass criteria: p95 < 200 ms, p99 < 300 ms.
    """
    console.rule("[bold blue]TC-QP-02: Top-K Query Latency (500K)")
    ds_size = config.LARGE_DATASET
    query_vecs = load_query_vectors(n=config.QUERY_COUNT)

    with db.connect() as conn:
        conn.execute(f"DROP TABLE IF EXISTS {TABLE_QP02} CASCADE")
        conn.commit()
        schema.enable_vector_extension(conn)
        schema.setup_inline(conn, table=TABLE_QP02)

        vecs = load_dataset(f"ds_{ds_size}")
        console.print(f"  Loading {ds_size:,} vectors...")
        ingest.batch_insert_inline(conn, vecs, show_progress=True, table=TABLE_QP02)
        schema.create_vector_index(conn, TABLE_QP02, op="l2")

        # Set beam size
        schema.set_search_beam_size(conn, config.VECTOR_SEARCH_BEAM_SIZE)

        # Warm up
        _run_top_k_queries(conn, query_vecs[:50], k=10, operator="l2", table=TABLE_QP02)

        iterations = []
        overall_pass = True
        for it in range(config.ITERATIONS):
            for k_val in config.K_VALUES:
                latencies = _run_top_k_queries(conn, query_vecs, k=k_val, operator="l2", table=TABLE_QP02)
                p95 = _percentile(latencies, 95)
                p99 = _percentile(latencies, 99)
                passed = p95 < config.QUERY_P95_LARGE_MS and p99 < config.QUERY_P99_LARGE_MS
                if not passed:
                    overall_pass = False
                run_result = {
                    "iteration": it + 1, "k": k_val,
                    "beam_size": config.VECTOR_SEARCH_BEAM_SIZE,
                    "p50_ms": round(_percentile(latencies, 50), 2),
                    "p95_ms": round(p95, 2),
                    "p99_ms": round(p99, 2),
                    "avg_ms": round(np.mean(latencies), 2),
                    "pass": passed,
                }
                iterations.append(run_result)
                status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
                console.print(f"    iter={it+1} K={k_val}: p95={p95:.1f}ms  p99={p99:.1f}ms  {status}")

    console.print(f"\n[bold]TC-QP-02 Overall: {'[green]PASS[/green]' if overall_pass else '[red]FAIL[/red]'}[/bold]")
    return {
        "test_id": "TC-QP-02", "pass": overall_pass,
        "threshold": f"p95 < {config.QUERY_P95_LARGE_MS} ms, p99 < {config.QUERY_P99_LARGE_MS} ms",
        "beam_size": config.VECTOR_SEARCH_BEAM_SIZE,
        "dataset_type": "real-world" if has_real_dataset() else "synthetic",
        "dataset": ds_size, "iterations": iterations,
    }


# -- TC-QP-03: Accuracy comparison --------------------------------------------

def _brute_force_top_k(dataset: np.ndarray, query: np.ndarray, k: int) -> list[int]:
    """Compute exact top-K by L2 distance (brute force)."""
    diffs = dataset - query
    dists = np.linalg.norm(diffs, axis=1)
    top_idx = np.argsort(dists)[:k]
    return top_idx.tolist()


def run_tc_qp_03() -> dict:
    """
    TC-QP-03: Result Accuracy vs. Brute-Force Baseline.
    Pass criteria: recall@10 >= 90%.

    Compares CockroachDB ANN results against brute-force exact nearest
    neighbours as the ground truth. Uses real-world VectorDBBench dataset
    with default beam_size to measure production-representative recall.
    """
    console.rule("[bold blue]TC-QP-03: Result Accuracy vs. Ground Truth")
    k = config.TOP_K  # K=10
    n_queries = 500

    # Load dataset and queries
    ds_size = 50000  # Use 50K subset for accuracy test
    dataset = load_dataset(f"ds_{ds_size}")
    query_vecs = load_query_vectors(n=n_queries)

    console.print(f"  Dataset: {'VectorDBBench real-world' if has_real_dataset() else 'Synthetic'}")
    console.print(f"  Beam size: {config.VECTOR_SEARCH_BEAM_SIZE}")

    with db.connect() as conn:
        conn.execute(f"DROP TABLE IF EXISTS {TABLE_QP03} CASCADE")
        conn.commit()
        schema.enable_vector_extension(conn)
        schema.setup_inline(conn, table=TABLE_QP03)

        console.print(f"  Loading {ds_size:,} vectors for accuracy test...")
        ingest.batch_insert_inline(conn, dataset, show_progress=True, table=TABLE_QP03)
        schema.create_vector_index(conn, TABLE_QP03, op="l2")

        # Set beam size
        schema.set_search_beam_size(conn, config.VECTOR_SEARCH_BEAM_SIZE)

        # Get CockroachDB ANN results
        console.print("  Running CockroachDB queries (L2 distance)...")
        crdb_id_results = []
        for qv in query_vecs:
            vec_str = vec_to_pgvector(qv)
            rows = conn.execute(
                f"SELECT id FROM {TABLE_QP03} ORDER BY embedding <-> %s::vector LIMIT %s",
                [vec_str, k],
            ).fetchall()
            crdb_id_results.append(set(r["id"] for r in rows))

        # Fetch IDs in insertion order using payload idx for correct mapping
        # (UUID ordering != insertion order, so ORDER BY id gives wrong mapping)
        console.print("  Fetching ID mapping (by insertion order)...")
        all_ids = conn.execute(
            f"SELECT id FROM {TABLE_QP03} ORDER BY (payload->>'idx')::int"
        ).fetchall()
        id_list = [r["id"] for r in all_ids]

    # Compute brute-force ground truth
    console.print("  Computing brute-force ground truth...")
    recalls = []
    for i, qv in enumerate(query_vecs):
        bf_indices = _brute_force_top_k(dataset, qv, k)
        bf_ids = set(id_list[idx] for idx in bf_indices)
        overlap = len(bf_ids & crdb_id_results[i])
        recalls.append(overlap / k)

    avg_recall = float(np.mean(recalls))
    min_recall = float(min(recalls))
    max_recall = float(max(recalls))
    p10_recall = float(np.percentile(recalls, 10))
    passed = avg_recall >= config.RECALL_AT_K

    console.print(f"\n  Average Recall@{k}: {avg_recall:.4f}")
    console.print(f"  Min Recall@{k}:     {min_recall:.4f}")
    console.print(f"  Max Recall@{k}:     {max_recall:.4f}")
    console.print(f"  p10 Recall@{k}:     {p10_recall:.4f}")
    console.print(f"  Beam size:         {config.VECTOR_SEARCH_BEAM_SIZE}")
    console.print(f"\n[bold]TC-QP-03 Overall: {'[green]PASS[/green]' if passed else '[red]FAIL[/red]'}[/bold]")

    return {
        "test_id": "TC-QP-03", "pass": passed,
        "threshold": f"recall@{k} >= {config.RECALL_AT_K}",
        "avg_recall": round(avg_recall, 4),
        "min_recall": round(min_recall, 4),
        "max_recall": round(max_recall, 4),
        "p10_recall": round(p10_recall, 4),
        "n_queries": n_queries,
        "beam_size": config.VECTOR_SEARCH_BEAM_SIZE,
        "dataset_type": "real-world" if has_real_dataset() else "synthetic",
    }
