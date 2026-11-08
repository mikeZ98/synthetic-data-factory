# syntax=docker/dockerfile:1.7
#
# Hardened, reproducible image for synthetic-data-factory.
#
#   docker build -f deploy/Dockerfile -t synthetic-data-factory:v0.1.0 .
#
# Run with a read-only root filesystem (compose does this for you):
#   docker run --rm --read-only --tmpfs /tmp --network none --cap-drop ALL \
#     -v "$PWD/data:/data" synthetic-data-factory:v0.1.0 in.jsonl -o out/x.parquet --headless --auto-exit
#
# Reproducibility: base images are pinned by digest, uv is pinned, the
# project installs from uv.lock, bytecode is hash-based (no mtimes), and the
# user is written as fixed passwd/group lines (useradd would stamp
# /etc/shadow with the current day). Build with SOURCE_DATE_EPOCH and
# rewrite-timestamp=true for bit-identical layers (see deploy/smoke-test.sh).

ARG PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.7@sha256:240fb85ab0f263ef12f492d8476aa3a2e4e1e333f7d67fbdd923d00a506a516a

# Pinned by tag+digest via ARG (hadolint cannot resolve ARG image refs).
# hadolint ignore=DL3006
FROM ${UV_IMAGE} AS uv

# --------------------------------------------------------------------------- #
# Build: resolve the locked environment into /opt/venv and compile bytecode.   #
# --------------------------------------------------------------------------- #
# Pinned by tag+digest via ARG (hadolint cannot resolve ARG image refs).
# hadolint ignore=DL3006
FROM ${PYTHON_IMAGE} AS build

COPY --from=uv /uv /usr/local/bin/uv

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON=/usr/local/bin/python3.12 \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=0 \
    UV_CACHE_DIR=/root/.cache/uv

WORKDIR /src

# Dependencies first (cached layer keyed on the lockfile only).
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# Then the project itself, as a regular (non-editable) install. uv_cache.json is
# uv's cache-freshness marker (wall-clock time + inode): useless at runtime and
# the only non-deterministic file in the image, so it and its RECORD line go.
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable \
 && find /opt/venv -name '__pycache__' -type d -prune -exec rm -rf {} + \
 && for dist in /opt/venv/lib/python3.12/site-packages/synthetic_data_factory-*.dist-info; do \
        rm "$dist/uv_cache.json" && sed -i '/uv_cache\.json/d' "$dist/RECORD"; \
    done \
 && python -B -m compileall -f -q -j 1 --invalidation-mode checked-hash /opt/venv/lib/python3.12/site-packages

# --------------------------------------------------------------------------- #
# Runtime: slim base, no pip / uv / compilers, non-root, read-only friendly.   #
# --------------------------------------------------------------------------- #
# Pinned by tag+digest via ARG (hadolint cannot resolve ARG image refs).
# hadolint ignore=DL3006
FROM ${PYTHON_IMAGE} AS runtime

ARG VERSION=0.1.0
ARG REVISION=unknown
LABEL org.opencontainers.image.title="synthetic-data-factory" \
      org.opencontainers.image.description="Validate, deduplicate (MinHash LSH) and export synthetic instruction data" \
      org.opencontainers.image.source="https://github.com/mikeZ98/synthetic-data-factory" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}"

# The official image strips stdlib .pyc files; on a read-only root filesystem
# they could never be cached, so every start would recompile the stdlib.
# Compile them once here (hash-based, so the output is deterministic). pip is
# removed first, so nothing can install at runtime. Every helper runs with -B:
# modules a tool imports would otherwise get *timestamp* .pyc as a side effect,
# and compileall would skip them as up to date (hence also -f). -j 1 because
# marshal's FLAG_REF depends on refcounts: parallel workers, each with its own
# interning history, emit byte-different .pyc for identical sources.
RUN python -B -m pip uninstall --yes --quiet pip \
 && rm -rf /usr/local/lib/python3.12/ensurepip /usr/local/bin/pip* /root/.cache \
 && find /usr/local/lib/python3.12 -name '__pycache__' -type d -prune -exec rm -rf {} + \
 && python -B -m compileall -f -q -j 1 --invalidation-mode checked-hash /usr/local/lib/python3.12 \
 && echo 'appuser:x:10001:10001::/nonexistent:/usr/sbin/nologin' >> /etc/passwd \
 && echo 'appuser:x:10001:' >> /etc/group \
 && mkdir -p /data \
 && chown 10001:10001 /data

COPY --from=build /opt/venv /opt/venv

ENV PATH=/opt/venv/bin:$PATH \
    VIRTUAL_ENV=/opt/venv \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/nonexistent

WORKDIR /data
VOLUME ["/data"]
USER 10001:10001

ENTRYPOINT ["synthetic-data-factory"]
CMD ["--help"]
