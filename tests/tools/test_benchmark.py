"""The benchmark harness that produces the README tables."""

import json
from pathlib import Path

import pytest

from tools.benchmark import (
    CONFIG,
    ComplexityRow,
    MemoryRow,
    complexity,
    complexity_table,
    corpus,
    lsh_dedup,
    main,
    measure,
    memory_table,
    naive_dedup,
    write_jsonl,
)


def test_corpus_is_seeded_and_contains_duplicates() -> None:
    assert corpus(300) == corpus(300)
    assert corpus(300) != corpus(300, seed=8)
    docs = corpus(300)
    assert len(set(docs)) < len(docs)  # exact copies exist
    assert all(len(doc.split()) == 40 for doc in docs)


def test_lsh_never_drops_what_exhaustive_search_would_keep_first() -> None:
    texts = corpus(400)
    naive, naive_comparisons = naive_dedup(texts)
    lsh, lsh_comparisons = lsh_dedup(texts)
    assert lsh_comparisons < naive_comparisons
    exact_copies = {i for i, t in enumerate(texts) if t in texts[:i]}
    assert exact_copies <= naive and exact_copies <= lsh  # both catch every exact copy


def test_complexity_rows_respect_the_exhaustive_limit() -> None:
    rows = complexity([100, 200], naive_limit=100)
    assert rows[0].naive_comparisons is not None and rows[0].agreement is not None
    assert rows[1].naive_comparisons is None and rows[1].agreement is None
    table = complexity_table(rows)
    assert "| 100 |" in table and "| 200 | — | — |" in table and table.count("—") == 5


def test_complexity_without_duplicates_reports_full_agreement() -> None:
    rows = complexity([1], naive_limit=1)
    assert rows[0].agreement == 1.0 and rows[0].naive_dropped == 0


def test_memory_table_and_growth() -> None:
    row = MemoryRow(1_000, "curate", 90.0, 100.5, 700)
    assert row.growth_mib == pytest.approx(10.5)
    assert "| 1,000 | curate | 90 MiB | 100 MiB | +10 MiB | 700 |" in memory_table([row])


@pytest.mark.parametrize("mode", ["stream", "curate"])
def test_measure_runs_a_workload_in_process(tmp_path: Path, mode: str) -> None:
    source = write_jsonl(tmp_path / "c.jsonl", 120)
    result = measure(mode, source, tmp_path)  # type: ignore[arg-type]
    assert result["peak_mib"] >= result["baseline_mib"] > 0
    assert 0 < result["kept"] <= 120
    assert (tmp_path / f"{mode}.parquet").is_file()


def test_main_quick_run_writes_results(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--quick", "--work-dir", str(tmp_path)]) == 0
    payload = json.loads((tmp_path / "results.json").read_text())
    assert payload["config"] == CONFIG.model_dump()
    assert {r["mode"] for r in payload["memory"]} == {"stream", "curate"}
    assert "| N | Exhaustive time" in capsys.readouterr().out


def test_measure_child_entrypoint(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = write_jsonl(tmp_path / "c.jsonl", 20)
    assert main(["--measure", "stream", str(source), str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["kept"] == 20


def test_row_defaults() -> None:
    row = ComplexityRow(10, 0.1, 5, 1)
    assert (row.naive_seconds, row.agreement) == (None, None)
