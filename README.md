# synthetic-data-factory

Curation pipeline for synthetic instruction datasets: strict data contracts,
MinHash near-duplicate removal, and Parquet export.

## Usage

```sh
uv run synthetic-data-factory data/*.jsonl more.parquet -o out/curated.parquet
# or: uv run python -m synthetic_data_factory.tui ...   (--headless --auto-exit for scripts)
```

Keys: `p` pause/resume, `c` stop gracefully (commit what was processed), `q` quit.
`SIGINT`/`SIGTERM` stop gracefully and exit 0; `SIGUSR1` stops gracefully and keeps
the UI open. A second quit or signal while stopping forces cancellation: the output
is discarded atomically and the exit code is 130.

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
