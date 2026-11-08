#!/usr/bin/env bash
# AST mutation gate. Mutants run against private copies of src/ under
# ./.cache/mutation/ (the repository source is never modified). Writes
# .cache/mutation/report.{md,json}; exits non-zero if any target's mutation
# score is below its configured minimum or the equivalent-mutant registry is stale.
# Extra arguments are forwarded (e.g. --workers 4).
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

export PYTHONPYCACHEPREFIX="./.cache/pycache"

uv run --locked python -m tools.mutation --config tools/mutation.toml "$@"
