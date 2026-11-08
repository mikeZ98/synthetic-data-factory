#!/usr/bin/env bash
# Coverage gate: runs the (network-free) test suite under branch coverage and
# requires exactly 100.00% line AND branch coverage for every measured file.
# Extra arguments are forwarded to pytest.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

export PYTHONPYCACHEPREFIX="./.cache/pycache"
export PYTHONDONTWRITEBYTECODE=1
readonly REPORT=".cache/coverage/coverage.json"

mkdir -p .cache/coverage
rm -f "$REPORT"

uv run --locked pytest \
    --cov \
    --cov-branch \
    --cov-report=term-missing \
    --cov-report="json:${REPORT}" \
    --cov-fail-under=100 \
    "$@"

uv run --locked python - "$REPORT" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)

files = report["files"]
if not files:
    sys.exit("coverage-gate: FAIL: no files were measured")

failed = False
for path, data in sorted(files.items()):
    s = data["summary"]
    lines_ok = s["missing_lines"] == 0
    branches_ok = s["missing_branches"] == 0 and s["num_partial_branches"] == 0
    line_pct = 100.0 * s["covered_lines"] / s["num_statements"] if s["num_statements"] else 100.0
    branch_pct = (
        100.0 * s["covered_branches"] / s["num_branches"] if s["num_branches"] else 100.0
    )
    status = "ok" if lines_ok and branches_ok else "FAIL"
    print(f"{status:>4}  line {line_pct:6.2f}%  branch {branch_pct:6.2f}%  {path}")
    failed |= not (lines_ok and branches_ok)

totals = report["totals"]
print(f"total {totals['percent_covered']:.2f}% across {len(files)} file(s)")
if failed or round(totals["percent_covered"], 2) != 100.00:
    sys.exit("coverage-gate: FAIL: line+branch coverage is below 100.00%")
print("coverage-gate: OK")
PY
