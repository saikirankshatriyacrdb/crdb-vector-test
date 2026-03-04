"""
Vector ingestion — single-row and batch modes for both inline and sidecar models.
All functions accept a `table` parameter so each test case can use its own table.
"""
from __future__ import annotations
import time
import uuid
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import config
import db
from datagen import vec_to_pgvector


# ── Single-row insert ────────────────────────────────────────────────────

def insert_single_inline(conn, vec: np.ndarray, payload=None,
                          table: str = "embeddings_inline") -> tuple:
    """Insert one vector into an inline table. Returns (id, elapsed_ms)."""
    vid = str(uuid.uuid4())
    vec_str = vec_to_pgvector(vec)
    t0 = time.perf_counter()
    conn.execute(
        f"INSERT INTO {table} (id, payload, embedding) VALUES (%s, %s, %s::vector)",
        [vid, "{}" if payload is None else str(payload), vec_str],
    )
    elapsed = (time.perf_counter() - t0) * 1000
    return vid, elapsed


def insert_single_sidecar(conn, vec: np.ndarray, payload=None,
                            records_table: str = "sidecar_records",
                            vectors_table: str = "sidecar_vectors") -> tuple:
    """Insert one vector into sidecar tables. Returns (record_id, elapsed_ms)."""
    rid = str(uuid.uuid4())
    vec_str = vec_to_pgvector(vec)
    t0 = time.perf_counter()
    conn.execute(
        f"INSERT INTO {records_table} (id, payload) VALUES (%s, %s)",
        [rid, "{}" if payload is None else str(payload)],
    )
    conn.execute(
        f"INSERT INTO {vectors_table} (record_id, embedding) VALUES (%s, %s::vector)",
        [rid, vec_str],
    )
    elapsed = (time.perf_counter() - t0) * 1000
    return rid, elapsed


# ── Pipelined insert / update ────────────────────────────────────────────

def insert_pipelined_inline(conn, vecs: np.ndarray, pipeline_size: int = 100,
                              table: str = "embeddings_inline") -> tuple[list[str], list[float]]:
    """
    Insert vectors using pipeline mode.
    Returns (list_of_ids, list_of_per_operation_latency_ms).
    """
    insert_sql = f"INSERT INTO {table} (id, payload, embedding) VALUES (%s, %s, %s::vector)"
    all_ids: list[str] = []
    latencies: list[float] = []

    for start in range(0, len(vecs), pipeline_size):
        chunk = vecs[start:start + pipeline_size]
        chunk_ids = [str(uuid.uuid4()) for _ in range(len(chunk))]
        chunk_strs = [vec_to_pgvector(v) for v in chunk]

        t0 = time.perf_counter()
        with conn.pipeline():
            for vid, vec_str in zip(chunk_ids, chunk_strs):
                conn.execute(insert_sql, [vid, "{}", vec_str], prepare=True)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        per_op_ms = elapsed_ms / len(chunk)

        all_ids.extend(chunk_ids)
        latencies.extend([per_op_ms] * len(chunk))

    return all_ids, latencies


def update_pipelined_inline(conn, row_ids: list[str], new_vecs: np.ndarray,
                              pipeline_size: int = 100,
                              table: str = "embeddings_inline") -> list[float]:
    """
    Update vectors using pipeline mode.
    Returns list of per-operation latencies in ms.
    """
    update_sql = f"UPDATE {table} SET embedding = %s::vector WHERE id = %s"
    latencies: list[float] = []

    for start in range(0, len(row_ids), pipeline_size):
        chunk_ids = row_ids[start:start + pipeline_size]
        chunk_vecs = new_vecs[start:start + pipeline_size]
        chunk_strs = [vec_to_pgvector(v) for v in chunk_vecs]

        t0 = time.perf_counter()
        with conn.pipeline():
            for vid, vec_str in zip(chunk_ids, chunk_strs):
                conn.execute(update_sql, [vec_str, vid], prepare=True)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        per_op_ms = elapsed_ms / len(chunk_ids)

        latencies.extend([per_op_ms] * len(chunk_ids))

    return latencies


# ── Batch insert ─────────────────────────────────────────────────────────

def _batch_sql_inline(vecs: np.ndarray, start_idx: int,
                       table: str = "embeddings_inline") -> tuple[str, list]:
    """Build a multi-value INSERT statement for an inline table."""
    values = []
    params = []
    for i, vec in enumerate(vecs):
        vid = str(uuid.uuid4())
        values.append(f"(%s, %s, %s::vector)")
        params.extend([vid, f'{{"idx": {start_idx + i}}}', vec_to_pgvector(vec)])
    sql = f"INSERT INTO {table} (id, payload, embedding) VALUES {','.join(values)}"
    return sql, params


def batch_insert_inline(conn, vecs: np.ndarray, batch_size: int = config.BATCH_SIZE,
                         show_progress: bool = True,
                         table: str = "embeddings_inline") -> float:
    """
    Batch-insert vectors into an inline table.
    Returns total elapsed seconds.
    """
    n = len(vecs)
    t0 = time.perf_counter()
    ctx = Progress(
        SpinnerColumn(), TextColumn("[bold blue]{task.description}"),
        BarColumn(), TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(),
    ) if show_progress else None

    if ctx:
        ctx.start()
        task = ctx.add_task(f"Inserting {n} vectors ({table})", total=n)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = vecs[start:end]
        sql, params = _batch_sql_inline(chunk, start, table=table)
        conn.execute(sql, params)
        if ctx:
            ctx.update(task, completed=end)

    if ctx:
        ctx.stop()
    return time.perf_counter() - t0


def _batch_sql_sidecar(vecs: np.ndarray, start_idx: int,
                         records_table: str = "sidecar_records",
                         vectors_table: str = "sidecar_vectors"):
    """Build multi-value INSERT for sidecar tables."""
    rec_values, rec_params = [], []
    vec_values, vec_params = [], []
    for i, vec in enumerate(vecs):
        rid = str(uuid.uuid4())
        rec_values.append("(%s, %s)")
        rec_params.extend([rid, f'{{"idx": {start_idx + i}}}'])
        vec_values.append("(%s, %s::vector)")
        vec_params.extend([rid, vec_to_pgvector(vec)])
    rec_sql = f"INSERT INTO {records_table} (id, payload) VALUES {','.join(rec_values)}"
    vec_sql = f"INSERT INTO {vectors_table} (record_id, embedding) VALUES {','.join(vec_values)}"
    return rec_sql, rec_params, vec_sql, vec_params


def batch_insert_sidecar(conn, vecs: np.ndarray, batch_size: int = config.BATCH_SIZE,
                          show_progress: bool = True,
                          records_table: str = "sidecar_records",
                          vectors_table: str = "sidecar_vectors") -> float:
    """Batch-insert into sidecar model. Returns total elapsed seconds."""
    n = len(vecs)
    t0 = time.perf_counter()
    ctx = Progress(
        SpinnerColumn(), TextColumn("[bold blue]{task.description}"),
        BarColumn(), TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(),
    ) if show_progress else None

    if ctx:
        ctx.start()
        task = ctx.add_task(f"Inserting {n} vectors (sidecar)", total=n)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = vecs[start:end]
        rsql, rp, vsql, vp = _batch_sql_sidecar(chunk, start,
                                                   records_table=records_table,
                                                   vectors_table=vectors_table)
        conn.execute(rsql, rp)
        conn.execute(vsql, vp)
        if ctx:
            ctx.update(task, completed=end)

    if ctx:
        ctx.stop()
    return time.perf_counter() - t0


# ── Concurrent batch insert ─────────────────────────────────────────────

def concurrent_batch_insert_inline(vecs: np.ndarray, concurrency: int = 5,
                                    batch_size: int = config.BATCH_SIZE,
                                    table: str = "embeddings_inline") -> tuple[float, float]:
    """
    Insert vectors using multiple threads.
    Returns (elapsed_seconds, embeddings_per_minute).
    """
    n = len(vecs)
    chunks = []
    for start in range(0, n, batch_size):
        chunks.append((start, min(start + batch_size, n)))

    t0 = time.perf_counter()

    def _worker(span):
        s, e = span
        conn = db.get_conn()
        try:
            chunk = vecs[s:e]
            sql, params = _batch_sql_inline(chunk, s, table=table)
            conn.execute(sql, params)
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_worker, c) for c in chunks]
        for f in as_completed(futures):
            f.result()  # raise on error

    elapsed = time.perf_counter() - t0
    emb_per_min = (n / elapsed) * 60
    return elapsed, emb_per_min


# ── Update ───────────────────────────────────────────────────────────────

def update_single_inline(conn, row_id: str, new_vec: np.ndarray,
                           table: str = "embeddings_inline") -> float:
    """Update one vector in an inline table. Returns elapsed_ms."""
    vec_str = vec_to_pgvector(new_vec)
    t0 = time.perf_counter()
    conn.execute(
        f"UPDATE {table} SET embedding = %s::vector WHERE id = %s",
        [vec_str, row_id],
    )
    elapsed = (time.perf_counter() - t0) * 1000
    return elapsed
