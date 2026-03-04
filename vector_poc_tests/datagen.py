"""
Dataset loader for VectorDBBench Performance1536D500K.

Downloads and caches the real-world OpenAI embeddings dataset (1536-dim,
500K vectors generated from the C4 corpus) used by VectorDBBench.

This replaces synthetic random vectors. Real-world embeddings have natural
cluster structure, so C-SPANN performs much better (higher recall at lower
beam sizes) compared to random vectors where all points are equidistant.

Dataset format (parquet):
  - train.parquet:     id (int), emb (float32 array) — 500K training vectors
  - test.parquet:      id (int), emb (float32 array) — query vectors
  - neighbors.parquet: id (int), neighbors_id (int array) — ground truth top-K

Fallback: If the real dataset is unavailable, generates synthetic vectors
with a warning that results won't be production-representative.
"""
from __future__ import annotations
import os
import sys
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

import config

console = Console()

# -- Dataset paths ------------------------------------------------------------

def _dataset_dir() -> Path:
    """Resolve the dataset directory."""
    if config.DATASET_DIR:
        return Path(config.DATASET_DIR)
    return config.DATA_DIR / "vectordbbench_1536d_500k"


def _check_pyarrow():
    """Check that pyarrow is available for parquet reading."""
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


# -- Download / Load ----------------------------------------------------------

def download_dataset(force: bool = False):
    """
    Download the VectorDBBench Performance1536D500K dataset.

    Tries multiple strategies:
      1. Use vectordb-bench Python package if installed
      2. Use huggingface_hub if available
      3. Provide manual download instructions

    The dataset contains real OpenAI embeddings from the C4 corpus,
    giving production-representative benchmark results.
    """
    ds_dir = _dataset_dir()
    train_path = ds_dir / "train.parquet"

    if train_path.exists() and not force:
        console.print(f"[green]Dataset already cached at {ds_dir}[/green]")
        return ds_dir

    ds_dir.mkdir(parents=True, exist_ok=True)

    # Strategy 1: Try vectordb-bench package
    try:
        console.print("[bold]Attempting download via vectordb-bench package...[/bold]")
        from vectordb_bench.backend.dataset_source.read_s3_source import ReadS3DatasetSource
        source = ReadS3DatasetSource()
        source.download(
            dataset="openai",
            size=500_000,
            dim=1536,
            dest=str(ds_dir),
        )
        console.print("[green]Downloaded via vectordb-bench[/green]")
        return ds_dir
    except (ImportError, Exception) as e:
        console.print(f"  [dim]vectordb-bench not available: {e}[/dim]")

    # Strategy 2: Try huggingface_hub
    try:
        console.print("[bold]Attempting download via huggingface_hub...[/bold]")
        from huggingface_hub import hf_hub_download
        for fname in ["train.parquet", "test.parquet", "neighbors.parquet"]:
            console.print(f"  Downloading {fname}...")
            hf_hub_download(
                repo_id="zilliztech/VectorDBBenchOpenAI1536D500K",
                filename=fname,
                local_dir=str(ds_dir),
                repo_type="dataset",
            )
        console.print("[green]Downloaded via huggingface_hub[/green]")
        return ds_dir
    except (ImportError, Exception) as e:
        console.print(f"  [dim]huggingface_hub not available: {e}[/dim]")

    # Strategy 3: Manual instructions
    console.print("\n[yellow bold]Could not auto-download the dataset.[/yellow bold]")
    console.print(
        f"\nPlease download the VectorDBBench Performance1536D500K dataset manually:\n"
        f"\n  Option A: pip install vectordb-bench && python -c \""
        f"from vectordb_bench.interface import benchMarkRunner; "
        f"benchMarkRunner.run(case_type='Performance1536D500K')\"\n"
        f"\n  Option B: pip install huggingface_hub && huggingface-cli download "
        f"zilliztech/VectorDBBenchOpenAI1536D500K --local-dir {ds_dir} --repo-type dataset\n"
        f"\n  Option C: Download parquet files from "
        f"https://github.com/zilliztech/VectorDBBench and place in:\n"
        f"    {ds_dir}/train.parquet\n"
        f"    {ds_dir}/test.parquet\n"
        f"    {ds_dir}/neighbors.parquet\n"
    )
    console.print("[yellow]Falling back to synthetic vectors (results won't be production-representative)[/yellow]")
    return None


def _load_parquet_vectors(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Load vectors from a parquet file.
    Returns (embeddings_array, ids_array_or_None).
    """
    import pyarrow.parquet as pq
    table = pq.read_table(str(path))
    df = table.to_pandas()

    # VectorDBBench format: 'emb' column contains lists of floats
    if "emb" in df.columns:
        emb_col = "emb"
    elif "embedding" in df.columns:
        emb_col = "embedding"
    else:
        raise ValueError(f"No embedding column found in {path}. Columns: {list(df.columns)}")

    embeddings = np.array(df[emb_col].tolist(), dtype=np.float32)

    ids = None
    if "id" in df.columns:
        ids = np.array(df["id"].tolist())

    return embeddings, ids


def _load_parquet_neighbors(path: Path) -> np.ndarray:
    """Load ground truth neighbors from a parquet file."""
    import pyarrow.parquet as pq
    table = pq.read_table(str(path))
    df = table.to_pandas()

    if "neighbors_id" in df.columns:
        col = "neighbors_id"
    elif "neighbors" in df.columns:
        col = "neighbors"
    else:
        raise ValueError(f"No neighbors column found in {path}. Columns: {list(df.columns)}")

    return np.array(df[col].tolist())


# -- Public API ---------------------------------------------------------------

def load_train_vectors(n: int | None = None) -> np.ndarray:
    """
    Load training vectors from the VectorDBBench dataset.
    If n is specified, return the first n vectors (for subset tests).
    Falls back to synthetic vectors if dataset not available.
    """
    ds_dir = _dataset_dir()
    train_path = ds_dir / "train.parquet"

    if train_path.exists() and _check_pyarrow():
        console.print(f"  Loading real-world vectors from {train_path.name}...")
        vecs, _ = _load_parquet_vectors(train_path)
        if n is not None and n < len(vecs):
            vecs = vecs[:n]
        console.print(f"  [green]Loaded {len(vecs):,} real-world vectors ({config.VECTOR_DIM}D)[/green]")
        return vecs

    # Fallback to synthetic
    console.print("  [yellow]Using synthetic vectors (dataset not available)[/yellow]")
    size = n or 500_000
    return _generate_synthetic(size, seed=size)


def load_query_vectors(n: int | None = None) -> np.ndarray:
    """
    Load query vectors from the VectorDBBench test set.
    Falls back to synthetic vectors if dataset not available.
    """
    ds_dir = _dataset_dir()
    test_path = ds_dir / "test.parquet"

    if test_path.exists() and _check_pyarrow():
        vecs, _ = _load_parquet_vectors(test_path)
        if n is not None and n < len(vecs):
            vecs = vecs[:n]
        return vecs

    # Fallback
    size = n or config.QUERY_COUNT
    return _generate_synthetic(size, seed=99)


def load_ground_truth(n_queries: int | None = None) -> np.ndarray | None:
    """
    Load ground truth neighbor IDs from the VectorDBBench dataset.
    Returns None if not available (tests will compute brute-force baseline).

    Each row contains the indices of the true nearest neighbors in the
    training set, ordered by distance.
    """
    ds_dir = _dataset_dir()
    neighbors_path = ds_dir / "neighbors.parquet"

    if neighbors_path.exists() and _check_pyarrow():
        neighbors = _load_parquet_neighbors(neighbors_path)
        if n_queries is not None and n_queries < len(neighbors):
            neighbors = neighbors[:n_queries]
        return neighbors

    return None


def has_real_dataset() -> bool:
    """Check if the real-world dataset is available."""
    ds_dir = _dataset_dir()
    return (ds_dir / "train.parquet").exists() and _check_pyarrow()


# -- Synthetic fallback (kept for backwards compatibility) --------------------

def _generate_synthetic(n: int, dim: int = config.VECTOR_DIM, seed: int = 42) -> np.ndarray:
    """Return (n, dim) float32 array of L2-normalised random vectors."""
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs /= norms
    return vecs


# Legacy aliases for backwards compatibility
generate_vectors = _generate_synthetic
generate_query_vectors = lambda n=config.QUERY_COUNT, dim=config.VECTOR_DIM, seed=99: _generate_synthetic(n, dim, seed)


def vec_to_pgvector(vec: np.ndarray) -> str:
    """Convert a 1-D numpy vector to pgvector literal string '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def save_dataset(name: str, vecs: np.ndarray):
    """Persist vectors to disk for reproducibility."""
    path = config.DATA_DIR / f"{name}.npy"
    np.save(path, vecs)
    return path


def load_dataset(name: str) -> np.ndarray:
    """
    Load a dataset by name.
    For 'ds_NNNNN' names, loads from the real dataset (subset of train vectors).
    Falls back to .npy files for backwards compatibility.
    """
    # Try real dataset first
    if name.startswith("ds_") and has_real_dataset():
        n = int(name.split("_")[1])
        return load_train_vectors(n=n)

    # Fallback to .npy file
    path = config.DATA_DIR / f"{name}.npy"
    if path.exists():
        return np.load(path)

    # Generate synthetic
    n = int(name.split("_")[1]) if name.startswith("ds_") else 50000
    console.print(f"  [yellow]Generating synthetic dataset '{name}' ({n:,} vectors)[/yellow]")
    vecs = _generate_synthetic(n, seed=n)
    save_dataset(name, vecs)
    return vecs


def ensure_datasets():
    """
    Ensure the VectorDBBench dataset is available.
    Downloads if needed, or falls back to synthetic generation.
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Preparing datasets", total=3)

        # Step 1: Check/download real dataset
        ds_dir = _dataset_dir()
        if not (ds_dir / "train.parquet").exists():
            progress.update(task, description="Downloading VectorDBBench dataset")
            result = download_dataset()
            if result is None:
                # Generate synthetic fallback datasets
                progress.update(task, description="Generating synthetic fallback datasets")
                for size in config.DATASET_SIZES:
                    name = f"ds_{size}"
                    path = config.DATA_DIR / f"{name}.npy"
                    if not path.exists():
                        vecs = _generate_synthetic(size, seed=size)
                        save_dataset(name, vecs)
        progress.advance(task)

        # Step 2: Verify dataset
        if has_real_dataset():
            vecs = load_train_vectors(n=10)
            assert vecs.shape == (10, config.VECTOR_DIM), \
                f"Dataset dimension mismatch: expected {config.VECTOR_DIM}, got {vecs.shape[1]}"
            progress.update(task, description="Dataset verified")
        progress.advance(task)

        # Step 3: Check query vectors
        if has_real_dataset():
            qvecs = load_query_vectors(n=10)
            assert qvecs.shape[1] == config.VECTOR_DIM
            progress.update(task, description="Query vectors verified")
        else:
            qpath = config.DATA_DIR / "query_vectors.npy"
            if not qpath.exists():
                qvecs = _generate_synthetic(config.QUERY_COUNT, seed=99)
                np.save(qpath, qvecs)
        progress.advance(task)


if __name__ == "__main__":
    ensure_datasets()
    if has_real_dataset():
        print(f"Real-world dataset ready at {_dataset_dir()}")
        vecs = load_train_vectors(n=5)
        print(f"  Sample shape: {vecs.shape}, dtype: {vecs.dtype}")
        print(f"  Norm of first vector: {np.linalg.norm(vecs[0]):.4f}")
        gt = load_ground_truth(n_queries=5)
        if gt is not None:
            print(f"  Ground truth shape: {gt.shape}")
    else:
        print("Using synthetic fallback datasets.")
