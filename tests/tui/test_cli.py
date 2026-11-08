"""CLI parsing, validation errors, and a real headless end-to-end run."""

import runpy
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from synthetic_data_factory.cli import main, parse_config
from tests.tui.test_app import UNIQUE, write_jsonl


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)  # outputs are relative to the working directory
    write_jsonl(tmp_path / "in.jsonl", 500)
    return tmp_path


def test_parse_config_builds_validated_models(workdir: Path) -> None:
    config, headless = parse_config(
        [
            "in.jsonl",
            "-o",
            "out/x.parquet",
            "--threshold",
            "0.8",
            "--num-perm",
            "64",
            "--compression",
            "gzip",
            "--batch-size",
            "32",
            "--queue-size",
            "16",
            "--headless",
        ]
    )
    assert headless and config.inputs == (Path("in.jsonl"),)
    assert (config.export.output_path, config.export.compression) == ("out/x.parquet", "gzip")
    assert (config.minhash.threshold, config.minhash.num_perm, config.minhash.shingle_k) == (
        0.8,
        64,
        5,
    )
    assert (config.batch_size, config.queue_maxsize, config.auto_exit) == (32, 16, False)


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["in.jsonl"], "required: -o/--output"),
        (["data.csv", "-o", "o.parquet"], "unsupported input type"),
        (["missing.jsonl", "-o", "o.parquet"], "no such file"),
        (["in.jsonl", "-o", "/abs/o.parquet"], "output_path"),
        (["in.jsonl", "-o", "o.csv"], "output_path"),
        (["in.jsonl", "-o", "o.parquet", "--threshold", "0"], "threshold"),
        (["in.jsonl", "-o", "o.parquet", "--num-perm", "100"], "power of two"),
        (["in.jsonl", "-o", "o.parquet", "--row-group-size", "10"], "row_group_size"),
        (["in.jsonl", "-o", "o.parquet", "--batch-size", "0"], "--batch-size"),
        (["in.jsonl", "-o", "o.parquet", "--queue-size", "-1"], "--queue-size"),
        (["in.jsonl", "-o", "o.parquet", "--compression", "lz4"], "invalid choice"),
    ],
)
def test_invalid_arguments_exit_2_with_message(
    workdir: Path, capsys: pytest.CaptureFixture[str], argv: list[str], message: str
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        parse_config(argv)
    assert excinfo.value.code == 2
    assert message in capsys.readouterr().err


def test_main_runs_headless_end_to_end(workdir: Path) -> None:
    code = main(
        [
            "in.jsonl",
            "-o",
            "out/curated.parquet",
            "--headless",
            "--auto-exit",
            "--row-group-size",
            "1000",
            "--shingle-k",
            "3",
        ]
    )
    assert code == 0
    assert pq.read_table(workdir / "out" / "curated.parquet").num_rows == UNIQUE
    assert [p for p in workdir.rglob(".tmp.*")] == []


def test_module_entrypoint(workdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    argv = [
        "synthetic_data_factory.tui",
        "in.jsonl",
        "-o",
        "m.parquet",
        "--headless",
        "--auto-exit",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    sys.modules.pop("synthetic_data_factory.tui.__main__", None)
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("synthetic_data_factory.tui", run_name="__main__")
    assert excinfo.value.code == 0
    assert (workdir / "m.parquet").is_file()
