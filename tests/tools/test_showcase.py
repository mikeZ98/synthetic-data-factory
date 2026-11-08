"""The README dashboard capture renders the real application."""

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from tools.showcase import build_inputs, capture, main


def test_inputs_exercise_every_rejection_kind(tmp_path: Path) -> None:
    jsonl, parquet = build_inputs(tmp_path, documents=200)
    lines = jsonl.read_text().splitlines()
    assert "{malformed json" in lines and any(len(line) > 4_096 for line in lines)
    table = pq.read_table(parquet)
    assert table.column("instruction").null_count == 1
    assert [("k", "1"), ("k", "2")] in table.column("metadata").to_pylist()


@pytest.mark.asyncio
async def test_capture_renders_every_panel(tmp_path: Path) -> None:
    text, svg = await capture(tmp_path, documents=400)
    for label in ("Curation", "Jaccard similarity", "Rejections", "Throughput", "COMPLETED"):
        assert label in text
    for reason in (
        "Malformed JSON",
        "Oversized line",
        "Duplicate metadata key",
        "Null required field",
    ):
        assert reason in text
    assert svg.startswith("<svg") and "synthetic-data-factory" in svg
    assert (tmp_path / "out" / "curated.parquet").is_file()


def test_main_writes_svg_and_text(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    svg = tmp_path / "docs" / "dash.svg"
    assert main(["--work-dir", str(tmp_path / "w"), "--svg", str(svg), "--documents", "300"]) == 0
    assert svg.read_text().startswith("<svg")
    assert "Rejections" in (tmp_path / "w" / "dashboard.txt").read_text()
    assert "Throughput" in capsys.readouterr().out
