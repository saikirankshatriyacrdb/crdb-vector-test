#!/bin/bash
# =============================================================================
# Vector PoC Test Suite — EC2 Setup & Run Script
# =============================================================================
# This script sets up and runs the Vector PoC benchmark on a fresh EC2 instance.
#
# Usage:
#   1. Launch a c6a.8xlarge (or similar) in us-west-2
#   2. Connect via Session Manager or EC2 Instance Connect
#   3. Paste this entire script and press Enter
#
# The script will:
#   - Install Python 3.11+ and dependencies
#   - Download the VectorDBBench Performance1536D500K dataset (~4.5 GB)
#   - Run all 10 test cases against CockroachDB
#   - Generate HTML + JSON reports
# =============================================================================

set -e

echo "=========================================="
echo "Vector PoC Test Suite — EC2 Setup"
echo "=========================================="

# -- System packages ----------------------------------------------------------
echo "[1/6] Installing system packages..."
sudo yum install -y python3.11 python3.11-pip git 2>/dev/null || \
sudo dnf install -y python3.11 python3.11-pip git 2>/dev/null || \
sudo apt-get update && sudo apt-get install -y python3.11 python3.11-pip git 2>/dev/null || \
echo "Using existing Python: $(python3 --version)"

# Use python3.11 if available, else python3
PYTHON=$(which python3.11 2>/dev/null || which python3)
PIP="$PYTHON -m pip"
echo "Using Python: $($PYTHON --version)"

# -- Python dependencies ------------------------------------------------------
echo "[2/6] Installing Python dependencies..."
$PIP install --quiet psycopg[binary] numpy rich click python-dotenv pyarrow vectordb-bench

# -- Create project directory -------------------------------------------------
echo "[3/6] Setting up project directory..."
WORKDIR=~/vector_poc_tests
mkdir -p $WORKDIR/tests $WORKDIR/reports $WORKDIR/data

# -- Download CA cert for CockroachDB Cloud -----------------------------------
echo "[4/6] Downloading CockroachDB CA certificate..."
curl -s --create-dirs -o ~/your-cluster-ca.crt \
  'https://cockroachlabs.cloud/clusters/YOUR_CLUSTER_ID/cert' 2>/dev/null || \
echo "Note: Could not auto-download cert. Using existing cert if available."

# If cert download failed, try the CockroachDB Cloud standard cert
if [ ! -f ~/your-cluster-ca.crt ]; then
  curl -s -o ~/your-cluster-ca.crt \
    'https://cockroachlabs.cloud/clusters/cert' 2>/dev/null || true
fi

# -- Write .env file ----------------------------------------------------------
cat > $WORKDIR/.env << 'ENVEOF'
# CockroachDB Connection
CRDB_HOST=your-cluster.aws-us-west-2.cockroachlabs.cloud
CRDB_PORT=26257
CRDB_USER=your_username
CRDB_PASSWORD=your_password
CRDB_DATABASE=vectortest
CRDB_SSLMODE=verify-full
CRDB_SSLROOTCERT=/home/ec2-user/your-cluster-ca.crt

# Test Configuration
VECTOR_DIM=1536
DATASET_SIZES=50000,100000,250000,500000
QUERY_COUNT=1000
TOP_K=10
ITERATIONS=3
BATCH_SIZE=500
CONCURRENCY_LEVELS=1,5,10
FRESHNESS_TRIALS=100

# C-SPANN beam size (default=32, real-world data doesn't need higher)
VECTOR_SEARCH_BEAM_SIZE=32
ENVEOF

echo "[5/6] Downloading VectorDBBench dataset (this takes ~3-5 min)..."
$PYTHON << 'PYEOF'
from vectordb_bench.backend.cases import CaseType
import os, shutil

case_cls = CaseType.Performance1536D500K.case_cls()
ds = case_cls.dataset
print("  Downloading from S3...")
ds.prepare()

# Symlink to expected location
src_dir = "/tmp/vectordb_bench/dataset/openai/openai_medium_500k"
dst_dir = os.path.expanduser("~/vector_poc_tests/data/vectordbbench_1536d_500k")
os.makedirs(dst_dir, exist_ok=True)

for src_name, dst_name in [
    ("shuffle_train.parquet", "train.parquet"),
    ("test.parquet", "test.parquet"),
    ("neighbors.parquet", "neighbors.parquet"),
]:
    src = os.path.join(src_dir, src_name)
    dst = os.path.join(dst_dir, dst_name)
    if os.path.exists(src) and not os.path.exists(dst):
        os.symlink(src, dst)
        print(f"  Linked {dst_name}")

print("  Dataset ready!")
PYEOF

echo "[6/6] All setup complete!"
echo ""
echo "To run the tests:"
echo "  cd $WORKDIR"
echo "  $PYTHON run.py"
echo ""
echo "To run a specific test suite:"
echo "  $PYTHON run.py --suite functional"
echo "  $PYTHON run.py --suite query"
echo "  $PYTHON run.py --suite ingest"
echo "  $PYTHON run.py --suite arch"
echo ""
echo "To run a single test:"
echo "  $PYTHON run.py --test TC-FC-01"
echo ""
echo "=========================================="
echo "Setup complete. Ready to run tests."
echo "=========================================="
