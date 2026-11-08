"""Render the real terminal dashboard headless, for the README.

    uv run --locked python -m tools.showcase

Builds a seeded dataset that exercises every panel (near-duplicates spanning
all similarity buckets, plus one rejection of each kind), runs the actual
``CurationTUIApp`` to completion headless, and writes:

* ``docs/tui-dashboard.svg``: the screenshot exactly as Textual renders it;
* ``.cache/showcase/dashboard.txt``: the same frame as plain text.

The text frame reuses Textual's own screenshot pipeline (the compositor's full
render printed to a recording Rich console), exported as text instead of SVG.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import random
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import pyarrow as pa
import pyarrow.parquet as pq
from rich.console import Console

from synthetic_data_factory.schemas import ExportSpecification, MinHashConfig
from synthetic_data_factory.transport.reader import StreamingDatasetReader
from synthetic_data_factory.transport.schema import INSTRUCTION_SAMPLE_SCHEMA
from synthetic_data_factory.tui.app import CurationTUIApp, RunConfig

T0: Final = datetime(2026, 1, 1, tzinfo=UTC)
SIZE: Final = (118, 21)
MAX_LINE_BYTES: Final = 4_096


def _record(n: int, text: str, **overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "sample_id": str(uuid.UUID(int=n + 1, version=4)),
        "system_prompt": "You are a precise technical tutor.",
        "instruction": text,
        # Constant response: the dedup key is instruction + response, so copies
        # of an instruction are only duplicates if the response matches too.
        "response": "A worked explanation with examples.",
        "domain": "tutoring",
        "metadata": {"generator": "seed-v2", "batch": str(n // 100)},
        "created_at_utc": (T0 + timedelta(seconds=n)).isoformat(),
    }
    record.update(overrides)
    return record


def build_inputs(directory: Path, *, documents: int = 3_000, seed: int = 11) -> list[Path]:
    rng = random.Random(seed)  # noqa: S311 - showcase data, not security
    directory.mkdir(parents=True, exist_ok=True)
    texts: list[str] = []
    lines: list[str] = []
    for n in range(documents):
        if texts and rng.random() < 0.35:
            tokens = rng.choice(texts).split()
            for _ in range(rng.randint(0, 6)):  # 0-6 edits spread pairs across all buckets
                tokens[rng.randrange(len(tokens))] = f"term{rng.randrange(800)}"
            text = " ".join(tokens)
        else:
            text = " ".join(f"term{rng.randrange(800)}" for _ in range(30))
        texts.append(text)
        lines.append(json.dumps(_record(n, text)))
    lines.insert(40, "{malformed json")
    lines.insert(80, json.dumps(_record(90_000, "x" * (MAX_LINE_BYTES + 100))))  # oversized line
    lines.insert(120, json.dumps(_record(90_001, "valid text", domain="Not A Slug")))  # schema
    jsonl = directory / "samples.jsonl"
    jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # A foreign Parquet export: nullable columns and an unflagged map, containing
    # one row with a null required field and one with a duplicate metadata key.
    rows = [
        _record(95_000 + i, f"imported row {i} " + " ".join(texts[i].split()[:20]))
        for i in range(50)
    ]
    rows[3]["instruction"] = None
    columns = {name: [row[name] for row in rows] for name in INSTRUCTION_SAMPLE_SCHEMA.names}
    columns["metadata"] = [sorted(dict(m).items()) for m in columns["metadata"]]  # type: ignore[call-overload]
    columns["metadata"][7] = [("k", "1"), ("k", "2")]
    columns["created_at_utc"] = [datetime.fromisoformat(str(v)) for v in columns["created_at_utc"]]
    schema = pa.schema(
        [
            pa.field(f.name, pa.map_(pa.string(), pa.string()) if f.name == "metadata" else f.type)
            for f in INSTRUCTION_SAMPLE_SCHEMA
        ]
    )
    parquet = directory / "imported.parquet"
    pq.write_table(pa.Table.from_pydict(columns, schema=schema), parquet)
    return [jsonl, parquet]


def frame_text(app: CurationTUIApp) -> str:
    width, height = app.size
    console = Console(
        width=width, height=height, file=io.StringIO(), record=True, color_system=None
    )
    console.print(app.screen._compositor.render_update(full=True))
    return console.export_text()


async def capture(directory: Path, *, documents: int = 3_000) -> tuple[str, str]:
    inputs = build_inputs(directory, documents=documents)
    config = RunConfig(
        inputs=tuple(inputs),
        export=ExportSpecification(
            output_path="out/curated.parquet", compression="zstd", row_group_size=1_000
        ),
        minhash=MinHashConfig(threshold=0.85, num_perm=128, shingle_k=3),
        root=directory,
        batch_size=128,
        queue_maxsize=256,
    )
    readers = [
        StreamingDatasetReader.jsonl(inputs[0], batch_size=128, max_line_bytes=MAX_LINE_BYTES),
        StreamingDatasetReader.parquet(inputs[1], batch_size=128),
    ]
    app = CurationTUIApp(config, readers=readers, install_signal_handlers=False, tick_seconds=0.1)
    async with app.run_test(size=SIZE) as pilot:
        async with asyncio.timeout(120):
            while app.outcome is None:
                await pilot.pause(0.05)
        await pilot.pause(0.2)
        text, svg = frame_text(app), app.export_screenshot(title="synthetic-data-factory")
        await pilot.press("q")
    return text, svg


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.showcase")
    parser.add_argument("--work-dir", type=Path, default=Path(".cache/showcase"))
    parser.add_argument("--svg", type=Path, default=Path("docs/tui-dashboard.svg"))
    parser.add_argument("--documents", type=int, default=3_000)
    args = parser.parse_args(argv)
    text, svg = asyncio.run(capture(args.work_dir, documents=args.documents))
    args.svg.parent.mkdir(parents=True, exist_ok=True)
    args.svg.write_text(svg, encoding="utf-8")
    (args.work_dir / "dashboard.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
