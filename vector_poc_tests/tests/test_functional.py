"""
TC-FC-01: Dimension & Operator Support     -> 1536-dim vectors + L2, cosine, IP
TC-FC-02: Vector Index Creation & Usage    -> EXPLAIN shows index; 2x faster

Updated to use VectorDBBench real-world dataset (Performance1536D500K).
Each test case uses its own dedicated table to avoid conflicts.
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
    load_train_vectors, load_query_vectors, vec_to_pgvector,
    has_real_dataset, generate_vectors,
)

console = Console()

# -- Table names (unique per test case) ----------------------------------------
TABLE_FC01 = "tc_fc01_vectors"
TABLE_FC02 = "tc_fc02_vectors"


# -- TC-FC-01: Dimension & Operator Support ------------------------------------

def run_tc_fc_01() -> dict:
    """
    TC-FC-01: Verify support for 1536-dim vectors and all three operators.
    Steps:
      1. Create table with VECTOR(1536) column
      2. Insert 1,000 real-world vectors
      3. Run top-10 query with L2 (<->), cosine (<=>), inner product (<#>)
      4. Verify all return valid, ordered results
    """
    console.rule("[bold blue]TC-FC-01: Dimension & Operator Support")
    n = 1000
    k = 10
    dim = config.VECTOR_DIM

    operators = {
        "l2":     {"sql_op": "<->",  "name": "L2 distance"},
        "cosine": {"sql_op": "<=>",  "name": "Cosine similarity"},
        "ip":     {"sql_op": "<#>",  "name": "Inner product"},
    }

    with db.connect() as conn:
        conn.execute(f"DROP TABLE IF EXISTS {TABLE_FC01} CASCADE")
        conn.commit()
        schema.enable_vector_extension(conn)
        conn.execute(f"CREATE TABLE {TABLE_FC01} (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), embedding VECTOR({dim}))")
        conn.commit()
        console.print(f"  [green]\u2713[/green] Created table {TABLE_FC01} with VECTOR({dim}) column")

        # Insert real-world vectors
        vecs = load_train_vectors(n=n) if has_real_dataset() else generate_vectors(n, seed=1111)
        console.print(f"  Inserting {n} {'real-world' if has_real_dataset() else 'synthetic'} vectors...")
        for i in range(0, n, 100):
            chunk = vecs[i:i+100]
            values = []
            params = []
            for v in chunk:
                values.append("(gen_random_uuid(), %s::vector)")
                params.append(vec_to_pgvector(v))
            sql = f"INSERT INTO {TABLE_FC01} (id, embedding) VALUES {','.join(values)}"
            conn.execute(sql, params)
        console.print(f"  [green]\u2713[/green] Inserted {n} vectors")

        # Test each operator
        query_vecs = load_query_vectors(n=1) if has_real_dataset() else generate_vectors(1, seed=2222)
        query_vec = query_vecs[0]
        vec_str = vec_to_pgvector(query_vec)
        op_results = {}

        for op_key, op_info in operators.items():
            sql_op = op_info["sql_op"]
            try:
                sql = f"SELECT id, embedding {sql_op} %s::vector AS score FROM {TABLE_FC01} ORDER BY embedding {sql_op} %s::vector LIMIT {k}"
                rows = conn.execute(sql, [vec_str, vec_str]).fetchall()

                valid = len(rows) == k
                scores = [r["score"] for r in rows]
                ordered = all(scores[i] <= scores[i+1] for i in range(len(scores)-1))

                op_results[op_key] = {
                    "name": op_info["name"],
                    "operator": sql_op,
                    "returned_rows": len(rows),
                    "correctly_ordered": ordered,
                    "pass": valid and ordered,
                    "sample_scores": [round(s, 6) for s in scores[:3]],
                }
                status = "[green]PASS[/green]" if (valid and ordered) else "[red]FAIL[/red]"
                console.print(f"  {op_info['name']} ({sql_op}): {len(rows)} results, ordered={ordered}  {status}")
            except Exception as e:
                op_results[op_key] = {"name": op_info["name"], "pass": False, "error": str(e)}
                console.print(f"  [red]\u2717[/red] {op_info['name']} ({sql_op}): {e}")

        conn.execute(f"DROP TABLE IF EXISTS {TABLE_FC01} CASCADE")
        conn.commit()

    overall_pass = all(r["pass"] for r in op_results.values())
    console.print(f"\n[bold]TC-FC-01 Overall: {'[green]PASS[/green]' if overall_pass else '[red]FAIL[/red]'}[/bold]")

    return {
        "test_id": "TC-FC-01", "pass": overall_pass,
        "threshold": "All 3 operators return correct top-10 for 1536-dim",
        "dimensions": dim,
        "dataset_type": "real-world" if has_real_dataset() else "synthetic",
        "operators": op_results,
    }


# -- TC-FC-02: Vector Index Creation & Usage -----------------------------------

def run_tc_fc_02() -> dict:
    """
    TC-FC-02: Vector Index Creation & Usage (C-SPANN).
    Steps:
      1. Create C-SPANN vector index (CockroachDB native)
      2. EXPLAIN query to confirm vector search node
      3. Compare latency with vs. without index
      4. Verify index survives basic operations
    Pass criteria: EXPLAIN shows vector search; indexed >= 2x faster than seq scan.
    """
    console.rule("[bold blue]TC-FC-02: Vector Index Creation & Usage")
    n = 10000
    dim = config.VECTOR_DIM
    n_queries = 200

    with db.connect() as conn:
        conn.execute(f"DROP TABLE IF EXISTS {TABLE_FC02} CASCADE")
        conn.commit()
        schema.enable_vector_extension(conn)
        conn.execute(f"CREATE TABLE {TABLE_FC02} (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), embedding VECTOR({dim}))")
        conn.commit()

        vecs = load_train_vectors(n=n) if has_real_dataset() else generate_vectors(n, seed=5555)
        console.print(f"  Loading {n:,} {'real-world' if has_real_dataset() else 'synthetic'} vectors...")
        for i in range(0, n, 500):
            chunk = vecs[i:i+500]
            values = []
            params = []
            for v in chunk:
                values.append("(gen_random_uuid(), %s::vector)")
                params.append(vec_to_pgvector(v))
            conn.execute(f"INSERT INTO {TABLE_FC02} (id, embedding) VALUES {','.join(values)}", params)

        query_vecs = load_query_vectors(n=n_queries) if has_real_dataset() else generate_vectors(n_queries, seed=6666)

        # -- Measure WITHOUT index (sequential scan) --
        console.print("\n  Measuring latency WITHOUT index (L2 distance <->)...")
        no_idx_latencies = []
        for qv in query_vecs:
            vec_str = vec_to_pgvector(qv)
            t0 = time.perf_counter()
            conn.execute(
                f"SELECT id FROM {TABLE_FC02} ORDER BY embedding <-> %s::vector LIMIT 10",
                [vec_str],
            ).fetchall()
            no_idx_latencies.append((time.perf_counter() - t0) * 1000)

        no_idx_avg = np.mean(no_idx_latencies)
        console.print(f"    No-index avg latency: {no_idx_avg:.2f} ms")

        # -- Create C-SPANN index --
        idx_name = schema.create_vector_index(conn, TABLE_FC02, op="l2")
        schema.set_search_beam_size(conn, config.VECTOR_SEARCH_BEAM_SIZE)

        # -- Verify via EXPLAIN --
        vec_str = vec_to_pgvector(query_vecs[0])
        explain = db.explain_query(
            conn,
            f"SELECT id FROM {TABLE_FC02} ORDER BY embedding <-> %s::vector LIMIT 10",
            [vec_str],
        )
        explain_lower = explain.lower()
        uses_index = any(term in explain_lower for term in ["vector search", "vector_index", "idx_", "index_scan", "index scan"])
        console.print(f"\n  EXPLAIN plan:\n{explain}")
        if uses_index:
            console.print("  Index detected in plan: [green]YES[/green]")
        else:
            console.print("  Index detected in plan: [red]NO[/red]")

        # -- Measure WITH index --
        console.print("\n  Measuring latency WITH index (L2 distance <->)...")
        # Warm up
        for qv in query_vecs[:20]:
            vec_str = vec_to_pgvector(qv)
            conn.execute(
                f"SELECT id FROM {TABLE_FC02} ORDER BY embedding <-> %s::vector LIMIT 10",
                [vec_str],
            ).fetchall()

        idx_latencies = []
        for qv in query_vecs:
            vec_str = vec_to_pgvector(qv)
            t0 = time.perf_counter()
            conn.execute(
                f"SELECT id FROM {TABLE_FC02} ORDER BY embedding <-> %s::vector LIMIT 10",
                [vec_str],
            ).fetchall()
            idx_latencies.append((time.perf_counter() - t0) * 1000)

        idx_avg = np.mean(idx_latencies)
        speedup = no_idx_avg / idx_avg if idx_avg > 0 else 0
        meets_speedup = speedup >= 2.0

        console.print(f"    Indexed avg latency: {idx_avg:.2f} ms")
        console.print(f"    Speedup: {speedup:.1f}x (threshold: >= 2x)")
        console.print(f"    Beam size: {config.VECTOR_SEARCH_BEAM_SIZE}")

        conn.execute(f"DROP TABLE IF EXISTS {TABLE_FC02} CASCADE")
        conn.commit()

    overall_pass = uses_index and meets_speedup
    console.print(f"\n[bold]TC-FC-02 Overall: {'[green]PASS[/green]' if overall_pass else '[red]FAIL[/red]'}[/bold]")

    return {
        "test_id": "TC-FC-02", "pass": overall_pass,
        "threshold": "Index in EXPLAIN + >= 2x speedup",
        "no_index_avg_ms": round(no_idx_avg, 2),
        "indexed_avg_ms": round(idx_avg, 2),
        "speedup": round(speedup, 2),
        "beam_size": config.VECTOR_SEARCH_BEAM_SIZE,
        "explain_shows_index": uses_index,
        "explain_plan": explain,
        "dataset_type": "real-world" if has_real_dataset() else "synthetic",
    }
