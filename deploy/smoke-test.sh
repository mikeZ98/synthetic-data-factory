#!/usr/bin/env bash
# Container smoke test. Needs only Docker on the host: test data is generated
# by the image itself and results are checked with the image's own pyarrow.
# Data lives in a throwaway *named volume*, never a host bind mount, so the
# test behaves identically on Docker Desktop, Colima (whatever its VM mounts)
# and Linux CI, and needs no permission workarounds: Docker initialises the
# volume from the image's /data, which is owned by UID 10001.
#
#   deploy/smoke-test.sh [IMAGE]        (default: synthetic-data-factory:v0.1.0)
#
# Verifies: CLI help; non-root UID 10001; no pip/uv/compilers; checked-hash
# bytecode for every module; a full run on a read-only root filesystem with no
# network and no capabilities; and graceful SIGTERM (docker stop) mid-run:
# exit 0, a committed partial output, no .tmp files left behind.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

readonly IMAGE="${1:-synthetic-data-factory:v0.1.0}"
readonly NAME="sdf-smoke-$$"
readonly VOLUME="sdf-smoke-$$"
readonly HARDENED=(--read-only --tmpfs "/tmp:rw,noexec,nosuid,size=64m" --network none
    --cap-drop ALL --security-opt no-new-privileges:true)

step() { printf '== %s\n' "$1"; }
fail() { printf 'FAIL: %s\n' "$1" >&2; exit 1; }

cleanup() {
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    docker volume rm -f "$VOLUME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker volume create "$VOLUME" >/dev/null

run_in() { docker run --rm "${HARDENED[@]}" -v "$VOLUME:/data" "$@"; }

step "CLI help (default CMD)"
docker run --rm "$IMAGE" | grep -q "usage: synthetic-data-factory" || fail "help output"

step "non-root user"
[ "$(docker run --rm --entrypoint id "$IMAGE" -u)" = "10001" ] || fail "uid is not 10001"
[ "$(docker run --rm --entrypoint id "$IMAGE" -g)" = "10001" ] || fail "gid is not 10001"

step "no development artifacts"
docker run --rm --entrypoint sh "$IMAGE" -c '
    for tool in pip pip3 uv uvx gcc cc g++ make; do
        if command -v "$tool" >/dev/null 2>&1; then echo "found $tool"; exit 1; fi
    done
    if python -c "import pip" 2>/dev/null; then echo "pip importable"; exit 1; fi
' || fail "development tooling present in the runtime image"

step "checked-hash bytecode for every module"
docker run --rm --entrypoint python "$IMAGE" -c '
import importlib.util, pathlib, sys
roots = [pathlib.Path("/opt/venv/lib/python3.12/site-packages"), pathlib.Path("/usr/local/lib/python3.12")]
missing, wrong, total = [], [], 0
for root in roots:
    for source in root.rglob("*.py"):
        if "site-packages" in source.parts and root.name != "site-packages":
            continue
        total += 1
        cached = pathlib.Path(importlib.util.cache_from_source(str(source)))
        if not cached.exists():
            missing.append(source)
        elif int.from_bytes(cached.read_bytes()[4:8], "little") != 0b11:  # hash-based + check_source
            wrong.append(cached)
print(f"   {total} sources, missing={len(missing)}, not checked-hash={len(wrong)}", missing[:3], wrong[:3])
sys.exit(1 if missing or wrong else 0)
' || fail "bytecode is missing or not checked-hash"

step "full run: read-only rootfs, no network, no capabilities"
run_in --entrypoint python "$IMAGE" -c '
import json, uuid, datetime as dt
with open("/data/in.jsonl", "w") as f:
    for n in range(3000):
        topic = n % 2000  # 1000 exact duplicates
        f.write(json.dumps({"sample_id": str(uuid.UUID(int=n + 1, version=4)), "system_prompt": "s",
            "instruction": f"explain topic {topic} in simple words for a curious student",
            "response": f"a careful answer about topic {topic} with details", "domain": "edu",
            "created_at_utc": (dt.datetime(2026, 1, 1, tzinfo=dt.UTC) + dt.timedelta(seconds=n)).isoformat()}) + "\n")
    f.write("{broken\n")
'
run_in "$IMAGE" in.jsonl -o out/curated.parquet --row-group-size 1000 --headless --auto-exit \
    || fail "curation run exited non-zero"
rows="$(run_in --entrypoint python "$IMAGE" -c \
    'import pyarrow.parquet as pq; print(pq.read_table("/data/out/curated.parquet").num_rows)')"
[ "$rows" = "2000" ] || fail "expected 2000 deduplicated rows, got $rows"

step "graceful SIGTERM mid-run (docker stop)"
run_in --entrypoint python "$IMAGE" -c '
import json, uuid, datetime as dt
with open("/data/big.jsonl", "w") as f:
    for n in range(300_000):
        f.write(json.dumps({"sample_id": str(uuid.UUID(int=n + 1, version=4)), "system_prompt": "s",
            "instruction": f"record {n} about subject {n % 977} with several shared words", "response": "ok",
            "domain": "edu", "created_at_utc": (dt.datetime(2026, 1, 1, tzinfo=dt.UTC) + dt.timedelta(seconds=n)).isoformat()}) + "\n")
'
docker run -d --name "$NAME" "${HARDENED[@]}" -v "$VOLUME:/data" "$IMAGE" \
    big.jsonl -o out/stopped.parquet --headless >/dev/null
sleep 6
docker stop --time 60 "$NAME" >/dev/null
code="$(docker inspect --format '{{.State.ExitCode}}' "$NAME")"
[ "$code" = "0" ] || fail "graceful stop exited with $code (want 0)"
partial="$(run_in --entrypoint python "$IMAGE" -c \
    'import pyarrow.parquet as pq; print(pq.read_table("/data/out/stopped.parquet").num_rows)')"
[ "$partial" -gt 0 ] || fail "no rows committed on graceful stop"
[ "$partial" -lt 300000 ] || fail "run finished before SIGTERM; signal path not exercised"
leftovers="$(run_in --entrypoint sh "$IMAGE" -c "find /data -name '.tmp.*' | wc -l" | tr -d ' ')"
[ "$leftovers" = "0" ] || fail "$leftovers temporary files left behind"
printf '   committed %s of 300000 records before stopping\n' "$partial"

printf 'smoke-test: OK (%s)\n' "$IMAGE"
