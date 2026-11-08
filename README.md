# synthetic-data-factory

Curation pipeline for synthetic instruction datasets: strict data contracts,
MinHash near-duplicate removal, and Parquet export.

## Development

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+. All tool state
(uv, ruff, mypy, pytest, coverage, hypothesis, bytecode) lives under `./.cache/`.

```sh
uv sync --locked

uv run --locked ruff check
uv run --locked ruff format --check
uv run --locked mypy
uv run --locked bandit -c pyproject.toml -r src

scripts/coverage-gate.sh       # network-free tests, 100.00% line + branch coverage
scripts/audit-hygiene.sh       # no host paths, no macOS artifacts, caches isolated
scripts/audit-attribution.sh   # sole committer, Conventional Commits, no trailers
```

The test suite runs with network sockets disabled (`pytest-socket`); any test
that opens an `AF_INET`/`AF_INET6` socket fails.
