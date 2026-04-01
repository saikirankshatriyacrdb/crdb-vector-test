# Vector PoC Test Suite

Automated test cases for validating CockroachDB (v25.4+) vector search capabilities against the Channel Program success criteria.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure connection (edit .env with your CockroachDB details)
#    The .env file is pre-configured — update if needed.

# 3. Run all tests
python run.py

# Or run specific suites
python run.py --suite functional    # TC-FC-01, TC-FC-02
python run.py --suite ingest        # TC-IU-01, TC-IU-02, TC-IU-03
python run.py --suite query         # TC-QP-01, TC-QP-02, TC-QP-03
python run.py --suite arch          # TC-AS-01, TC-AS-02

# Or run a single test
python run.py --test TC-FC-01
```

## Test Cases

| ID | Name | Success Criteria |
|----|------|-----------------|
| TC-QP-01 | Top-K Query Latency (50K–250K) | p95 < 50 ms |
| TC-QP-02 | Top-K Query Latency (500K) | p95 < 100 ms |
| TC-QP-03 | Result Accuracy vs. Baseline | recall@10 ≥ 95% |
| TC-IU-01 | Single Insert/Update Latency | avg < 50 ms |
| TC-IU-02 | Batch Ingestion Throughput | ≥ 5,000 emb/min |
| TC-IU-03 | Data Freshness | p95 < 5 sec |
| TC-FC-01 | Dimension & Operator Support | 1536-dim + L2/cosine/IP |
| TC-FC-02 | Vector Index Creation | EXPLAIN index + 2x speedup |
| TC-AS-01 | Inline vs. Sidecar Model | Both models meet thresholds |
| TC-AS-02 | Storage Footprint | < 5 GB for 500K vectors |

## Project Structure

```
vector_poc_tests/
├── .env                  # Connection & test config
├── config.py             # Typed settings + thresholds
├── db.py                 # Connection helpers
├── schema.py             # DDL (inline + sidecar)
├── datagen.py            # Synthetic vector generator
├── ingest.py             # Insert/update/batch operations
├── report.py             # HTML report generator
├── run.py                # CLI test runner
├── tests/
│   ├── test_query_performance.py
│   ├── test_ingestion.py
│   ├── test_functional.py
│   └── test_architecture.py
├── reports/              # Generated reports
└── data/                 # Cached datasets (.npy)
```

## Recommended Run Order

For first-time execution, start with functional tests to validate basic connectivity and feature support, then progress to performance tests:

1. `python run.py --test TC-FC-01` — Verify dimensions & operators
2. `python run.py --test TC-FC-02` — Verify index support
3. `python run.py --suite ingest`  — Benchmark ingestion
4. `python run.py --suite query`   — Benchmark queries
5. `python run.py --suite arch`    — Compare models & storage
