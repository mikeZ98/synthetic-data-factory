"""Command-line entrypoint: ``synthetic-data-factory INPUT... --output OUT.parquet``."""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import get_args

from pydantic import ValidationError

from synthetic_data_factory.schemas import ExportSpecification, MinHashConfig
from synthetic_data_factory.schemas.export import Compression
from synthetic_data_factory.tui.app import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_QUEUE_MAXSIZE,
    CurationTUIApp,
    RunConfig,
    detect_format,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synthetic-data-factory",
        description="Validate, deduplicate (MinHash LSH) and export synthetic instruction data.",
    )
    parser.add_argument(
        "inputs", nargs="+", type=Path, metavar="INPUT", help="JSONL/NDJSON or Parquet files"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="relative destination path ending in .parquet"
    )
    parser.add_argument("--compression", choices=get_args(Compression), default="zstd")
    parser.add_argument("--row-group-size", type=int, default=100_000)
    parser.add_argument("--threshold", type=float, default=0.85, help="Jaccard dedup threshold")
    parser.add_argument("--num-perm", type=int, default=128, help="MinHash permutations (2^n)")
    parser.add_argument("--shingle-k", type=int, default=5, help="word shingle size")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--queue-size", type=int, default=DEFAULT_QUEUE_MAXSIZE)
    parser.add_argument(
        "--auto-exit", action="store_true", help="exit when the run ends instead of waiting for q"
    )
    parser.add_argument("--headless", action="store_true", help="run without a terminal display")
    return parser


def _summarize(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}" for item in error.errors()
    )


def parse_config(argv: Sequence[str] | None = None) -> tuple[RunConfig, bool]:
    """Parse and validate arguments; exits with status 2 and a message on bad input."""
    parser = build_parser()
    args = parser.parse_args(argv)
    for path in args.inputs:
        try:
            detect_format(path)
        except ValueError as exc:
            parser.error(str(exc))
        if not path.is_file():
            parser.error(f"{path}: no such file")
    for name in ("batch_size", "queue_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be a positive integer")
    try:
        export = ExportSpecification(
            output_path=args.output,
            compression=args.compression,
            row_group_size=args.row_group_size,
        )
        minhash = MinHashConfig(
            threshold=args.threshold, num_perm=args.num_perm, shingle_k=args.shingle_k
        )
    except ValidationError as exc:
        parser.error(_summarize(exc))
    config = RunConfig(
        inputs=tuple(args.inputs),
        export=export,
        minhash=minhash,
        batch_size=args.batch_size,
        queue_maxsize=args.queue_size,
        auto_exit=args.auto_exit,
    )
    return config, args.headless


def main(argv: Sequence[str] | None = None) -> int:
    config, headless = parse_config(argv)
    app = CurationTUIApp(config)
    app.run(headless=headless)
    return app.return_code or 0
