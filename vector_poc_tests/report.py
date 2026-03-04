"""
HTML report generator for PoC test results.
"""
from __future__ import annotations
import json
from string import Template
from datetime import datetime
from pathlib import Path
import config

REPORT_TEMPLATE = Template("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Vector PoC Test Report</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #f5f7fa; color: #333; padding: 2rem; }
  .container { max-width: 1000px; margin: 0 auto; }
  h1 { color: #1F4E79; margin-bottom: 0.5rem; }
  .subtitle { color: #666; margin-bottom: 2rem; }
  .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                   gap: 1rem; margin-bottom: 2rem; }
  .summary-card { background: #fff; border-radius: 8px; padding: 1.2rem;
                  box-shadow: 0 1px 3px rgba(0,0,0,0.08); text-align: center; }
  .summary-card .value { font-size: 2rem; font-weight: 700; }
  .summary-card .label { color: #888; font-size: 0.85rem; margin-top: 0.3rem; }
  .pass { color: #2e7d32; }
  .fail { color: #c62828; }
  .test-section { background: #fff; border-radius: 8px; padding: 1.5rem;
                  margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
  .test-header { display: flex; justify-content: space-between; align-items: center;
                 margin-bottom: 1rem; }
  .test-header h2 { font-size: 1.1rem; color: #1F4E79; }
  .badge { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 12px;
           font-size: 0.8rem; font-weight: 600; }
  .badge-pass { background: #e8f5e9; color: #2e7d32; }
  .badge-fail { background: #ffebee; color: #c62828; }
  table { width: 100%; border-collapse: collapse; margin-top: 0.5rem; font-size: 0.9rem; }
  th { background: #f0f4f8; padding: 0.6rem 0.8rem; text-align: left;
       border-bottom: 2px solid #ddd; font-weight: 600; color: #555; }
  td { padding: 0.5rem 0.8rem; border-bottom: 1px solid #eee; }
  .metric { font-family: 'SF Mono', Consolas, monospace; }
  pre { background: #f8f9fa; padding: 1rem; border-radius: 6px; overflow-x: auto;
        font-size: 0.82rem; margin-top: 0.5rem; }
  .threshold { color: #888; font-size: 0.85rem; }
</style>
</head>
<body>
<div class="container">
  <h1>Vector PoC Test Report</h1>
  <p class="subtitle">CockroachDB v25.4+ &mdash; Generated $timestamp</p>

  <div class="summary-grid">
    <div class="summary-card">
      <div class="value">$total_tests</div>
      <div class="label">Total Tests</div>
    </div>
    <div class="summary-card">
      <div class="value pass">$passed_tests</div>
      <div class="label">Passed</div>
    </div>
    <div class="summary-card">
      <div class="value fail">$failed_tests</div>
      <div class="label">Failed</div>
    </div>
    <div class="summary-card">
      <div class="value" style="color: $verdict_color">$verdict</div>
      <div class="label">Verdict</div>
    </div>
  </div>

  $sections

  <div class="test-section">
    <h2>Raw Results (JSON)</h2>
    <pre>$raw_json</pre>
  </div>
</div>
</body>
</html>
""")


def _test_section(result: dict) -> str:
    """Render one test result as an HTML section."""
    tid = result["test_id"]
    passed = result["pass"]
    badge_class = "badge-pass" if passed else "badge-fail"
    badge_text = "PASS" if passed else "FAIL"
    badge = f'<span class="badge {badge_class}">{badge_text}</span>'
    threshold = result.get("threshold", "")

    # Build details table from remaining keys
    skip_keys = {"test_id", "pass", "threshold", "explain_plan"}
    rows = ""
    for k, v in result.items():
        if k in skip_keys:
            continue
        if isinstance(v, (dict, list)):
            v = json.dumps(v, indent=2)
            rows += f'<tr><td><strong>{k}</strong></td><td><pre>{v}</pre></td></tr>'
        else:
            rows += f'<tr><td><strong>{k}</strong></td><td class="metric">{v}</td></tr>'

    explain = ""
    if "explain_plan" in result:
        explain = f'<h3 style="margin-top:1rem;font-size:0.95rem;">EXPLAIN Plan</h3><pre>{result["explain_plan"]}</pre>'

    return f"""
    <div class="test-section">
      <div class="test-header">
        <h2>{tid}</h2>
        {badge}
      </div>
      <p class="threshold">Threshold: {threshold}</p>
      <table>{rows}</table>
      {explain}
    </div>
    """


def generate_report(results, output_path=None):
    """Generate an HTML report from test results and return the file path."""
    total = len(results)
    passed = sum(1 for r in results if r["pass"])
    failed = total - passed

    if failed == 0:
        verdict, color = "GO", "#2e7d32"
    elif failed <= 2:
        verdict, color = "CONDITIONAL", "#f57f17"
    else:
        verdict, color = "NO-GO", "#c62828"

    sections = "\n".join(_test_section(r) for r in results)
    raw = json.dumps(results, indent=2, default=str)

    html = REPORT_TEMPLATE.substitute(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        total_tests=total,
        passed_tests=passed,
        failed_tests=failed,
        verdict=verdict,
        verdict_color=color,
        sections=sections,
        raw_json=raw,
    )

    if output_path is None:
        output_path = config.REPORTS_DIR / f"poc_report_{datetime.now():%Y%m%d_%H%M%S}.html"

    output_path.write_text(html)
    return output_path


def save_json(results, output_path=None):
    """Save raw JSON results."""
    if output_path is None:
        output_path = config.REPORTS_DIR / f"poc_results_{datetime.now():%Y%m%d_%H%M%S}.json"
    output_path.write_text(json.dumps(results, indent=2, default=str))
    return output_path
