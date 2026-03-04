#!/usr/bin/env python3
"""
Vector PoC Test Runner
======================

Usage:
    python run.py                    # Run ALL test cases
    python run.py --suite query      # Only query-performance tests
    python run.py --suite ingest     # Only ingestion tests
    python run.py --suite functional # Only functional tests
    python run.py --suite arch       # Only architecture tests
    python run.py --test TC-FC-01    # Run a single test case
    python run.py --setup            # Just setup schema + generate data
    python run.py --cleanup          # Drop all test tables
"""
from __future__ import annotations
import sys
import json
import click
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table as RichTable

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import config
import db
import schema
from datagen import ensure_datasets
from report import generate_report, save_json

console = Console()

# ── Test registry ────────────────────────────────────────────────────────
TEST_CASES = {}

def _register():
    from tests.test_query_performance import run_tc_qp_01, run_tc_qp_02, run_tc_qp_03
    from tests.test_ingestion import run_tc_iu_01, run_tc_iu_02, run_tc_iu_03
    from tests.test_functional import run_tc_fc_01, run_tc_fc_02
    from tests.test_architecture import run_tc_as_01, run_tc_as_02

    TEST_CASES.update({
        "TC-QP-01": {"fn": run_tc_qp_01, "suite": "query",      "desc": "Top-K Query Latency (50K–250K)"},
        "TC-QP-02": {"fn": run_tc_qp_02, "suite": "query",      "desc": "Top-K Query Latency (500K)"},
        "TC-QP-03": {"fn": run_tc_qp_03, "suite": "query",      "desc": "Result Accuracy vs. Baseline"},
        "TC-IU-01": {"fn": run_tc_iu_01, "suite": "ingest",     "desc": "Single Insert/Update Latency"},
        "TC-IU-02": {"fn": run_tc_iu_02, "suite": "ingest",     "desc": "Batch Ingestion Throughput"},
        "TC-IU-03": {"fn": run_tc_iu_03, "suite": "ingest",     "desc": "Data Freshness (Search-After-Write)"},
        "TC-FC-01": {"fn": run_tc_fc_01, "suite": "functional", "desc": "Dimension & Operator Support"},
        "TC-FC-02": {"fn": run_tc_fc_02, "suite": "functional", "desc": "Vector Index Creation & Usage"},
        "TC-AS-01": {"fn": run_tc_as_01, "suite": "arch",       "desc": "Inline vs. Sidecar Comparison"},
        "TC-AS-02": {"fn": run_tc_as_02, "suite": "arch",       "desc": "Storage Footprint at Scale"},
    })


# ── CLI ──────────────────────────────────────────────────────────────────

@click.command()
@click.option("--suite", type=click.Choice(["query", "ingest", "functional", "arch", "all"]),
              default="all", help="Test suite to run")
@click.option("--test", "test_id", type=str, default=None, help="Run a single test case by ID (e.g., TC-FC-01)")
@click.option("--setup", is_flag=True, help="Only setup schema and generate datasets")
@click.option("--cleanup", is_flag=True, help="Drop all test tables")
@click.option("--report/--no-report", default=True, help="Generate HTML report")
def main(suite, test_id, setup, cleanup, report):
    """Vector PoC Test Runner — CockroachDB v25.4+"""

    from datagen import has_real_dataset
    ds_label = "VectorDBBench Performance1536D500K (real-world)" if has_real_dataset() else "Synthetic (fallback)"
    console.print(Panel(
        "[bold blue]Channel Program — Vector PoC Test Suite[/bold blue]\n"
        f"CockroachDB target: {config.get_dsn().split('@')[1].split('/')[0] if '@' in config.get_dsn() else 'localhost'}\n"
        f"Dataset: {ds_label}\n"
        f"Beam size: {config.VECTOR_SEARCH_BEAM_SIZE}  |  K: {config.TOP_K}",
        title="Vector PoC", border_style="blue",
    ))

    # ── Pre-flight ───────────────────────────────────────────────────
    console.print("\n[bold]Pre-flight: Preparing datasets...[/bold]")
    ensure_datasets()
    console.print("[green]✓[/green] Datasets ready\n")

    # ── Connection check ─────────────────────────────────────────────
    console.print("[bold]Checking database connectivity...[/bold]")
    try:
        with db.connect() as conn:
            row = conn.execute("SELECT version()").fetchone()
            version = list(row.values())[0] if row else "unknown"
            console.print(f"[green]✓[/green] Connected — {version[:80]}\n")
            schema.enable_vector_extension(conn)
    except Exception as e:
        console.print(f"[red]✗[/red] Connection failed: {e}")
        console.print("\nPlease check your .env file and ensure CockroachDB is reachable.")
        sys.exit(1)

    if cleanup:
        console.print("[bold]Cleaning up test tables...[/bold]")
        with db.connect() as conn:
            schema.drop_all(conn)
        console.print("[green]✓[/green] Cleanup complete.")
        return

    if setup:
        console.print("[bold]Setting up schema...[/bold]")
        with db.connect() as conn:
            schema.reset(conn)
        console.print("[green]✓[/green] Schema ready.")
        return

    # ── Run tests ────────────────────────────────────────────────────
    _register()

    if test_id:
        test_id = test_id.upper()
        if test_id not in TEST_CASES:
            console.print(f"[red]Unknown test case: {test_id}[/red]")
            console.print(f"Available: {', '.join(TEST_CASES.keys())}")
            sys.exit(1)
        tests_to_run = {test_id: TEST_CASES[test_id]}
    elif suite == "all":
        tests_to_run = TEST_CASES
    else:
        tests_to_run = {k: v for k, v in TEST_CASES.items() if v["suite"] == suite}

    console.print(f"[bold]Running {len(tests_to_run)} test case(s)...[/bold]\n")

    all_results = []
    for tc_id, tc_info in tests_to_run.items():
        try:
            result = tc_info["fn"]()
            all_results.append(result)
        except Exception as e:
            from rich.markup import escape
            console.print(f"\n[red]ERROR in {tc_id}: {escape(str(e))}[/red]")
            all_results.append({"test_id": tc_id, "pass": False, "error": str(e)})

    # ── Summary ──────────────────────────────────────────────────────
    console.print("\n")
    console.rule("[bold blue]Test Summary")

    summary = RichTable(title="Results", show_lines=True)
    summary.add_column("Test Case", style="bold")
    summary.add_column("Description")
    summary.add_column("Result", justify="center")
    summary.add_column("Key Metric")

    for r in all_results:
        tc_id = r["test_id"]
        desc = TEST_CASES.get(tc_id, {}).get("desc", "")
        status = "[green]PASS[/green]" if r["pass"] else "[red]FAIL[/red]"

        # Extract a key metric for display
        metric = r.get("threshold", "")
        if "p95_ms" in r:
            metric = f"p95={r['p95_ms']}ms"
        elif "avg_recall" in r:
            metric = f"recall@10={r['avg_recall']}"
        elif "insert_avg_ms" in r:
            metric = f"insert avg={r['insert_avg_ms']}ms"

        summary.add_row(tc_id, desc, status, metric)

    console.print(summary)

    passed = sum(1 for r in all_results if r["pass"])
    failed = len(all_results) - passed

    if failed == 0:
        verdict = "[bold green]GO[/bold green]"
    elif failed <= 2:
        verdict = "[bold yellow]CONDITIONAL GO[/bold yellow]"
    else:
        verdict = "[bold red]NO-GO[/bold red]"

    console.print(f"\n  Passed: {passed}/{len(all_results)}")
    console.print(f"  Verdict: {verdict}\n")

    # ── Reports ──────────────────────────────────────────────────────
    json_path = save_json(all_results)
    console.print(f"  JSON results: {json_path}")

    if report:
        html_path = generate_report(all_results)
        console.print(f"  HTML report:  {html_path}")

    console.print("\n[bold green]Done![/bold green]")


if __name__ == "__main__":
    main()
