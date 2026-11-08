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

## Container

```sh
docker build -f deploy/Dockerfile -t synthetic-data-factory:v0.1.0 .
deploy/smoke-test.sh                       # non-root, no dev tooling, read-only run, graceful SIGTERM
SDF_INPUT=in.jsonl docker compose -f deploy/docker-compose.yml --profile batch up --exit-code-from curate
```

The runtime image is non-root (UID 10001), contains no pip/uv/compiler, ships
checked-hash bytecode for every module, and is bit-for-bit reproducible when
built with `SOURCE_DATE_EPOCH` and `rewrite-timestamp=true`. Compose runs it with
a read-only root filesystem, no network, no capabilities, and CPU/memory limits;
data lives in `SDF_DATA_DIR` (default `./.cache/data`).

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
scripts/run-mutation-tests.sh  # AST mutation gate (minutes; per-target minimums in tools/mutation.toml)
```

The test suite runs with network sockets disabled (`pytest-socket`); any test
that opens an `AF_INET`/`AF_INET6` socket fails.
