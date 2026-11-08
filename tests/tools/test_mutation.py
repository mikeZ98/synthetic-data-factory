"""The mutation harness is itself verification code, so it is verified too."""

import ast
import dataclasses
import json
import textwrap
from pathlib import Path

import pytest

from tools.mutation import (
    Config,
    Harness,
    MutationHarnessError,
    Outcome,
    Report,
    Target,
    TargetReport,
    build_report,
    generate_mutants,
    main,
    run,
)


def mutate(source: str) -> list[tuple[str, str, str]]:
    return [
        (m.operator, m.original, m.mutated)
        for m in generate_mutants(textwrap.dedent(source), "m.py")
    ]


# --------------------------------------------------------------------------- #
# Generation: operators                                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("x = a < b", ("cmp", "a < b", "a <= b")),
        ("x = a >= b", ("cmp", "a >= b", "a > b")),
        ("x = a is None", ("cmp", "a is None", "a is not None")),
        ("x = a not in b", ("cmp", "a not in b", "a in b")),
        ("x = a + b", ("binop", "a + b", "a - b")),
        ("x = a // b", ("binop", "a // b", "a / b")),
        ("x = a | b", ("binop", "a | b", "a & b")),
        ("x += y", ("augassign", "x += y", "x -= y")),
        ("x = a and b", ("boolop", "a and b", "a or b")),
        ("x = a or b", ("boolop", "a or b", "a and b")),
        ("x = not a", ("not", "not a", "not not a")),
        ("x = a if c else b", ("negate", "a if c else b", "a if not c else b")),
        ("x = 0", ("const", "0", "1")),
        ("x = True", ("const", "True", "False")),
        ("f(a, key=k)", ("stmt", "f(a, key=k)", "None")),
        ("f(a, key=k)", ("kwarg", "f(a, key=k)", "f(a)")),
    ],
)
def test_each_operator(source: str, expected: tuple[str, str, str]) -> None:
    assert expected in mutate(source)


def test_statement_level_operators() -> None:
    source = """
    async def g():
        await h()
        while ok:
            if done:
                break
            continue
    """
    found = {(op, mutated) for op, _, mutated in mutate(source)}
    assert ("stmt", "None") in found  # await statement removed
    assert ("negate", "while not ok:\n    if done:\n        break\n    continue") in found
    assert ("loopctl", "continue") in found and ("loopctl", "break") in found


def test_swap_table_covers_every_comparison_operator() -> None:
    from tools.mutation import _CMP_SWAPS

    assert set(_CMP_SWAPS) == set(ast.cmpop.__subclasses__())


def test_multiple_comparators_mutate_independently() -> None:
    mutated = [m for op, _, m in mutate("x = a < b < c") if op == "cmp"]
    assert mutated == ["a <= b < c", "a < b <= c"]


def test_double_star_kwargs_are_not_removed() -> None:
    assert [m for op, _, m in mutate("f(**kw)") if op == "kwarg"] == []


# --------------------------------------------------------------------------- #
# Generation: what is never mutated                                           #
# --------------------------------------------------------------------------- #


def test_docstrings_annotations_decorators_fstrings_aliases_and_all_are_skipped() -> None:
    source = '''
    """Module docstring 1."""
    __all__ = ["f"]
    type Alias = int | str
    LIMIT: int = 3

    @decorator(flag=True)
    class C:
        """Class docstring 2."""

        def method(self, a: int = 1, *, b: int = 2, c: "dict[str, int]" = None) -> int | None:
            """Function docstring 3."""
            value: list[int] | None = None
            message = f"{a + 1} items"
            return a
    '''
    mutations = mutate(source)
    originals = {original for _, original, _ in mutations}
    assert originals == {"3", "1", "2"}  # LIMIT and the two int defaults only
    assert all("|" not in original for original in originals)


def test_pragma_lines_and_kwonly_without_default_are_handled() -> None:
    source = """
    def f(*, required, optional=0):
        return required + 1  # pragma: no mutate
    """
    assert mutate(source) == [("const", "0", "1")]


def test_ids_are_stable_unique_and_scoped() -> None:
    source = """
    class K:
        def f(self):
            return 1 + 1
    """
    mutants = generate_mutants(textwrap.dedent(source), "pkg/k.py")
    ids = [m.id for m in mutants]
    assert len(ids) == len(set(ids))
    assert "pkg/k.py::K.f::const::1 => 2#1" in ids and "pkg/k.py::K.f::const::1 => 2#2" in ids
    assert [m.id for m in generate_mutants(textwrap.dedent(source), "pkg/k.py")] == ids
    assert all(m.qualname == "K.f" and m.line == 4 for m in mutants)
    for mutant in mutants:
        ast.parse(mutant.source)  # every mutant is valid Python


def test_module_level_scope_name() -> None:
    assert generate_mutants("x = 1\n", "m.py")[0].qualname == "<module>"


# --------------------------------------------------------------------------- #
# Reporting                                                                   #
# --------------------------------------------------------------------------- #


def test_score_and_gate_logic() -> None:
    full = TargetReport("a.py", 100.0, generated=4, killed=3, timeout=1)
    partial = TargetReport("b.py", 90.0, generated=11, killed=9, survived=1, equivalent=1)
    empty = TargetReport("c.py", 100.0, generated=2, equivalent=2)
    assert (full.score, partial.score, empty.score) == (100.0, 90.0, 100.0)
    assert full.passed and partial.passed and empty.passed
    assert Report([full, partial, empty], []).passed
    assert not Report([full], ["stale"]).passed
    assert not Report([TargetReport("d.py", 95.0, generated=2, killed=1, survived=1)], []).passed
    table = Report([full, partial], []).table()
    assert "| `b.py` | 11 | 9 | 0 | 1 | 1 | 90.00% | 90.00% |" in table
    assert "| **total** | 15 | 12 | 1 | 1 | 1 | 92.86% | |" in table


# --------------------------------------------------------------------------- #
# End to end on a fixture project                                             #
# --------------------------------------------------------------------------- #

CALC = """
def clamp(x, lo, hi):
    if x < lo:
        return lo
    return min(x, hi)


def spin(n):
    i = 0
    while i < n:
        i += 1
    return i
"""

TESTS = """
from fixturepkg.calc import clamp, spin


def test_clamp():
    assert clamp(5, 0, 10) == 5
    assert clamp(-1, 0, 10) == 0
    assert clamp(50, 0, 10) == 10


def test_spin():
    assert spin(3) == 3
    assert spin(0) == 0
"""

# "if x < lo" -> "if x <= lo" survives: x == lo returns lo either way (equivalent).
EQUIVALENT = "fixturepkg/calc.py::clamp::cmp::x < lo => x <= lo#1"


def fixture_project(
    root: Path, *, tests: str = TESTS, equivalents: tuple[str, ...] = (EQUIVALENT,)
) -> Path:
    (root / "src" / "fixturepkg").mkdir(parents=True)
    (root / "src" / "fixturepkg" / "__init__.py").write_text("")
    (root / "src" / "fixturepkg" / "calc.py").write_text(CALC)
    (root / "src" / "fixturepkg" / "other.py").write_text("VALUE = 1\n")
    (root / "tests").mkdir()
    (root / "tests" / "test_calc.py").write_text(tests)
    (root / "pytest.ini").write_text("[pytest]\nasyncio_default_fixture_loop_scope = function\n")
    entries = "".join(f'\n[[equivalent]]\nid = "{e}"\nreason = "checked"\n' for e in equivalents)
    config = root / "mutation.toml"
    config.write_text(
        textwrap.dedent(
            """
            [mutation]
            src_dir = "src"
            package = "fixturepkg"
            min_timeout = 3.0

            [[target]]
            path = "fixturepkg/calc.py"
            tests = ["tests/test_calc.py"]
            minimum = 100.0
            """
        )
        + entries
    )
    return config


def test_end_to_end_classifies_killed_timeout_and_equivalent(tmp_path: Path) -> None:
    config_path = fixture_project(tmp_path / "proj")
    work = tmp_path / "work"
    code = main(
        [
            "--config",
            str(config_path),
            "--root",
            str(tmp_path / "proj"),
            "--workers",
            "2",
            "--work-dir",
            str(work),
        ]
    )
    assert code == 0
    payload = json.loads((work / "report.json").read_text())
    target = payload["targets"][0]
    assert payload["passed"] and payload["stale_equivalents"] == []
    assert target["equivalent"] == 1 and target["survived"] == 0
    assert target["timeout"] >= 1  # "i += 1" -> "i -= 1" never terminates
    assert target["killed"] + target["timeout"] + target["equivalent"] == target["generated"]
    assert "| `fixturepkg/calc.py` |" in (work / "report.md").read_text()
    assert (
        tmp_path / "proj" / "src" / "fixturepkg" / "calc.py"
    ).read_text() == CALC  # never touched


def test_unregistered_survivor_and_stale_entry_fail_the_gate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = fixture_project(
        tmp_path / "proj", equivalents=("fixturepkg/calc.py::gone::const::1 => 2#1",)
    )
    code = main(
        [
            "--config",
            str(config_path),
            "--root",
            str(tmp_path / "proj"),
            "--workers",
            "2",
            "--work-dir",
            str(tmp_path / "w"),
        ]
    )
    out = capsys.readouterr().out
    assert code == 1
    assert "SURVIVED fixturepkg/calc.py:3: x < lo => x <= lo" in out
    assert "STALE EQUIVALENT" in out and "mutation gate: FAIL" in out


def _config(
    root: Path, *, package: str | None = None, equivalents: dict[str, str] | None = None
) -> Config:
    config = Config.load(root / "mutation.toml", root=root)
    if package is not None:
        config = dataclasses.replace(config, package=package)
    if equivalents is not None:
        config = dataclasses.replace(config, equivalents=equivalents)
    return config


def test_failing_baseline_is_a_harness_error(tmp_path: Path) -> None:
    fixture_project(tmp_path / "proj", tests="def test_broken():\n    assert False\n")
    with pytest.raises(MutationHarnessError, match="baseline tests"):
        run(_config(tmp_path / "proj"), workers=1, work_dir=tmp_path / "w", log=lambda _: None)


def test_tests_that_do_not_import_the_target_trip_the_canary(tmp_path: Path) -> None:
    fixture_project(tmp_path / "proj", tests="def test_unrelated():\n    assert True\n")
    with pytest.raises(MutationHarnessError, match="canary"):
        run(_config(tmp_path / "proj"), workers=1, work_dir=tmp_path / "w", log=lambda _: None)


@pytest.mark.parametrize("package", ["json", "no_such_package_anywhere"])
def test_package_not_resolving_to_the_worker_copy_is_a_harness_error(
    tmp_path: Path, package: str
) -> None:
    fixture_project(tmp_path / "proj")
    harness = Harness(
        _config(tmp_path / "proj", package=package), workers=1, work_dir=tmp_path / "w"
    )
    with pytest.raises(MutationHarnessError, match="not the worker copy"):
        harness.prepare()


def test_config_defaults(tmp_path: Path) -> None:
    path = tmp_path / "m.toml"
    path.write_text(
        '[mutation]\nsrc_dir = "s"\npackage = "p"\n\n'
        '[[target]]\npath = "p/a.py"\ntests = ["t"]\nminimum = 90\n'
    )
    config = Config.load(path, root=tmp_path)
    assert config.targets == (Target("p/a.py", ("t",), 90.0),)
    assert (config.equivalents, config.pytest_args) == ({}, ())
    assert (config.timeout_factor, config.min_timeout) == (4.0, 30.0)


def test_build_report_counts_each_outcome(tmp_path: Path) -> None:
    config = _config(fixture_project(tmp_path / "proj").parent)
    mutants = generate_mutants(CALC, "fixturepkg/calc.py")[:4]
    target = config.targets[0]
    kinds: list[Outcome] = ["killed", "timeout", "survived", "survived"]
    outcomes = dict(zip([m.id for m in mutants], kinds, strict=True))
    config = _config(tmp_path / "proj", equivalents={mutants[3].id: "checked"})
    report = build_report(config, {target.path: (target, mutants, outcomes)})
    counts = report.targets[0]
    assert (counts.killed, counts.timeout, counts.survived, counts.equivalent) == (1, 1, 1, 1)
    assert counts.survivors[0]["id"] == mutants[2].id
