"""AST mutation testing for the critical modules of this repository.

How it works
------------
1. Each worker gets a private copy of ``src/`` under ``.cache/mutation/``.
   ``PYTHONPATH`` puts that copy ahead of the editable install. The repository
   source is never modified, so an interrupted run cannot leave a mutant behind.
2. Self-checks run before any mutant, so a broken harness cannot report
   "everything survived" or "everything killed":
   * import resolution: the package must import from the worker copy;
   * baseline: each target's tests must pass unmutated;
   * canary: a target replaced by ``raise`` must make its tests fail.
3. Mutants come from a deterministic walk of the AST. Docstrings, annotations,
   decorators, f-strings, ``__all__`` and lines marked ``# pragma: no mutate``
   are skipped. Each mutant runs its target's tests with ``-x`` under the
   derandomized Hypothesis ``ci`` profile, so results are reproducible. A
   timeout counts as a kill: a mutant that hangs is detected.
4. Survivors that are *semantically equivalent* to the original are listed,
   with a written justification, in the config's ``[[equivalent]]`` table. They
   leave the denominator. An entry that no longer matches a surviving mutant
   (killed, or no longer generated) fails the run, so the registry cannot rot.

Mutation score per target = (killed + timeout) / (generated - equivalent).
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import os
import queue
import shutil
import signal
import subprocess  # noqa: S404 - the harness exists to run the test suite
import sys
import time
import tomllib
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Final, Literal, cast

PRAGMA: Final = "pragma: no mutate"
DEFAULT_CONFIG: Final = Path("tools/mutation.toml")

Outcome = Literal["killed", "timeout", "survived"]

_CMP_SWAPS: Final[dict[type[ast.cmpop], type[ast.cmpop]]] = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
}
_BIN_SWAPS: Final[dict[type[ast.operator], type[ast.operator]]] = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.Div,
    ast.Div: ast.Mult,
    ast.FloorDiv: ast.Div,
    ast.Mod: ast.FloorDiv,
    ast.Pow: ast.Mult,
    ast.BitAnd: ast.BitOr,
    ast.BitOr: ast.BitAnd,
    ast.LShift: ast.RShift,
    ast.RShift: ast.LShift,
}


class MutationHarnessError(RuntimeError):
    """The harness itself is unsound for this configuration (not a test-quality finding)."""


# --------------------------------------------------------------------------- #
# Mutant generation                                                           #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Site:
    index: int  # position of the node in the deterministic walk
    variant: int  # which alternative at that node (comparator / keyword index)
    operator: str
    qualname: str
    line: int


@dataclass(frozen=True, slots=True)
class Mutant:
    id: str
    path: str
    operator: str
    qualname: str
    line: int
    original: str
    mutated: str
    source: str


def _is_docstring(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _walk(tree: ast.Module, skip_lines: frozenset[int]) -> Iterator[tuple[ast.AST, str]]:
    """Deterministic pre-order walk over mutable nodes, with their qualified scope."""

    def visit(node: ast.AST, scope: str) -> Iterator[tuple[ast.AST, str]]:
        if getattr(node, "lineno", None) in skip_lines or isinstance(node, ast.JoinedStr):
            return
        yield node, scope
        children: list[ast.AST]
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            inner = f"{scope}.{node.name}" if scope else node.name
            body = node.body[1:] if node.body and _is_docstring(node.body[0]) else node.body
            defaults: list[ast.AST] = []
            if not isinstance(node, ast.ClassDef):
                defaults = [d for d in (*node.args.defaults, *node.args.kw_defaults) if d]
            for child in (*defaults, *body):
                yield from visit(child, inner)
            return
        if isinstance(node, ast.AnnAssign):
            children = [node.target] + ([node.value] if node.value else [])
        elif (
            isinstance(node, ast.TypeAlias)
            or isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)
        ):
            return
        else:
            children = list(ast.iter_child_nodes(node))
        for child in children:
            yield from visit(child, scope)

    body = tree.body[1:] if tree.body and _is_docstring(tree.body[0]) else tree.body
    for statement in body:
        yield from visit(statement, "")


def _sites(node: ast.AST, index: int, scope: str) -> Iterator[Site]:
    line = getattr(node, "lineno", 0)

    def site(operator: str, variant: int = 0) -> Site:
        return Site(index, variant, operator, scope or "<module>", line)

    if isinstance(node, ast.Compare):  # _CMP_SWAPS covers every ast.cmpop
        for i in range(len(node.ops)):
            yield site("cmp", i)
    elif isinstance(node, ast.BinOp | ast.AugAssign) and type(node.op) in _BIN_SWAPS:
        yield site("binop" if isinstance(node, ast.BinOp) else "augassign")
    elif isinstance(node, ast.BoolOp):
        yield site("boolop")
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        yield site("not")
    elif isinstance(node, ast.If | ast.While | ast.IfExp):
        yield site("negate")
    elif isinstance(node, ast.Constant) and isinstance(node.value, int):
        yield site("const")
    elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call | ast.Await):
        yield site("stmt")
    elif isinstance(node, ast.Break | ast.Continue):
        yield site("loopctl")
    if isinstance(node, ast.Call):
        for i, keyword in enumerate(node.keywords):
            if keyword.arg is not None:
                yield site("kwarg", i)


def _apply(node: ast.AST, site: Site) -> None:
    """Mutate ``node`` in place (every operator is expressible without a parent)."""
    if isinstance(node, ast.Compare):
        node.ops[site.variant] = _CMP_SWAPS[type(node.ops[site.variant])]()
    elif isinstance(node, ast.BinOp | ast.AugAssign):
        node.op = _BIN_SWAPS[type(node.op)]()
    elif isinstance(node, ast.BoolOp):
        node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
    elif isinstance(node, ast.UnaryOp):
        node.operand = ast.UnaryOp(op=ast.Not(), operand=node.operand)  # not x -> bool(x)
    elif isinstance(node, ast.If | ast.While | ast.IfExp):
        node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
    elif isinstance(node, ast.Constant) and isinstance(node.value, int):
        node.value = (not node.value) if isinstance(node.value, bool) else node.value + 1
    elif isinstance(node, ast.Expr):
        node.value = ast.Constant(value=None)  # call statement deleted
    elif isinstance(node, ast.Break | ast.Continue):
        node.__class__ = ast.Continue if isinstance(node, ast.Break) else ast.Break
    else:  # the only remaining site kind is "kwarg", produced for ast.Call nodes
        del cast(ast.Call, node).keywords[site.variant]


def generate_mutants(source: str, path: str) -> list[Mutant]:
    """Every mutant of ``source``, in deterministic order with stable, unique ids."""
    tree = ast.parse(source)
    skip = frozenset(n for n, text in enumerate(source.splitlines(), 1) if PRAGMA in text)
    nodes = list(_walk(tree, skip))
    sites = [s for i, (node, scope) in enumerate(nodes) for s in _sites(node, i, scope)]
    seen: Counter[str] = Counter()
    mutants: list[Mutant] = []
    for site in sites:
        mutated_tree = copy.deepcopy(tree)
        mutated_nodes = [node for node, _ in _walk(mutated_tree, skip)]
        target = mutated_nodes[site.index]
        original = ast.unparse(nodes[site.index][0])
        _apply(target, site)
        mutated = ast.unparse(target)
        key = f"{path}::{site.qualname}::{site.operator}::{original} => {mutated}"
        seen[key] += 1
        mutants.append(
            Mutant(
                id=f"{key}#{seen[key]}",
                path=path,
                operator=site.operator,
                qualname=site.qualname,
                line=site.line,
                original=original,
                mutated=mutated,
                source=ast.unparse(ast.fix_missing_locations(mutated_tree)) + "\n",
            )
        )
    return mutants


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Target:
    path: str  # relative to src_dir
    tests: tuple[str, ...]
    minimum: float  # required mutation score in percent


@dataclass(frozen=True, slots=True)
class Config:
    root: Path
    src_dir: str
    package: str
    targets: tuple[Target, ...]
    equivalents: dict[str, str]
    pytest_args: tuple[str, ...] = ()
    timeout_factor: float = 4.0
    min_timeout: float = 30.0

    @classmethod
    def load(cls, path: Path, *, root: Path) -> Config:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        settings = data["mutation"]
        return cls(
            root=root,
            src_dir=settings["src_dir"],
            package=settings["package"],
            targets=tuple(
                Target(t["path"], tuple(t["tests"]), float(t["minimum"])) for t in data["target"]
            ),
            equivalents={e["id"]: e["reason"] for e in data.get("equivalent", [])},
            pytest_args=tuple(settings.get("pytest_args", ())),
            timeout_factor=float(settings.get("timeout_factor", 4.0)),
            min_timeout=float(settings.get("min_timeout", 30.0)),
        )


# --------------------------------------------------------------------------- #
# Execution                                                                   #
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Worker:
    home: Path

    @property
    def src(self) -> Path:
        return self.home / "src"

    def env(self) -> dict[str, str]:
        return {
            **os.environ,
            "PYTHONPATH": str(self.src),
            "PYTHONDONTWRITEBYTECODE": "1",  # rapid same-size rewrites must never hit stale .pyc
            "PYTHONPYCACHEPREFIX": str(self.home / "pycache"),
        }


def _run(command: Sequence[str], *, cwd: Path, env: dict[str, str], timeout: float) -> int | None:
    """Exit code, or ``None`` on timeout (the whole process group is killed)."""
    process = subprocess.Popen(  # noqa: S603 - fixed argv built from the interpreter path
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        return None


class Harness:
    def __init__(self, config: Config, *, workers: int, work_dir: Path) -> None:
        self.config = config
        self.work_dir = work_dir
        self.workers = [Worker(work_dir / f"worker-{i}") for i in range(workers)]
        self.timeouts: dict[str, float] = {}

    def prepare(self) -> None:
        shutil.rmtree(self.work_dir, ignore_errors=True)
        source = self.config.root / self.config.src_dir
        for worker in self.workers:
            shutil.copytree(source, worker.src, ignore=shutil.ignore_patterns("__pycache__"))
            self._check_import_resolution(worker)

    def _check_import_resolution(self, worker: Worker) -> None:
        probe = f"import {self.config.package} as m; print(m.__file__)"
        result = subprocess.run(  # noqa: S603 - fixed argv
            [sys.executable, "-c", probe],
            cwd=self.config.root,
            env=worker.env(),
            capture_output=True,
            text=True,
            check=False,
        )
        resolved = Path(result.stdout.strip() or "/nonexistent").resolve()
        if not resolved.is_relative_to(worker.src.resolve()):
            raise MutationHarnessError(
                f"{self.config.package} resolves to {resolved}, not the worker copy {worker.src}; "
                "mutants would be tested against the real source"
            )

    def pytest_command(self, worker: Worker, tests: Sequence[str]) -> list[str]:
        return [
            sys.executable,
            "-m",
            "pytest",
            *tests,
            "-x",
            "-q",
            "--no-header",
            f"--basetemp={worker.home / 'tmp'}",
            "-o",
            f"cache_dir={worker.home / 'pytest-cache'}",
            *self.config.pytest_args,
        ]

    def run_tests(self, worker: Worker, target: Target, timeout: float) -> int | None:
        return _run(
            self.pytest_command(worker, target.tests),
            cwd=self.config.root,
            env=worker.env(),
            timeout=timeout,
        )

    def self_check(self, target: Target) -> None:
        worker = self.workers[0]
        started = time.monotonic()
        code = self.run_tests(worker, target, timeout=3600.0)
        if code != 0:
            raise MutationHarnessError(f"baseline tests for {target.path} fail (exit {code})")
        elapsed = time.monotonic() - started
        self.timeouts[target.path] = max(
            self.config.min_timeout, elapsed * self.config.timeout_factor
        )
        path = worker.src / target.path
        original = path.read_text(encoding="utf-8")
        try:
            path.write_text("raise RuntimeError('mutation canary')\n", encoding="utf-8")
            if self.run_tests(worker, target, timeout=self.timeouts[target.path]) == 0:
                raise MutationHarnessError(
                    f"canary for {target.path} survived; tests do not import it"
                )
        finally:
            path.write_text(original, encoding="utf-8")

    def evaluate(
        self,
        target: Target,
        mutants: Sequence[Mutant],
        on_result: Callable[[Mutant, Outcome], None],
    ) -> dict[str, Outcome]:
        free: queue.Queue[Worker] = queue.Queue()
        for worker in self.workers:
            free.put(worker)
        originals = {
            w.home: (w.src / target.path).read_text(encoding="utf-8") for w in self.workers
        }

        def job(mutant: Mutant) -> tuple[str, Outcome]:
            worker = free.get()
            path = worker.src / target.path
            try:
                path.write_text(mutant.source, encoding="utf-8")
                code = self.run_tests(worker, target, timeout=self.timeouts[target.path])
            finally:
                path.write_text(originals[worker.home], encoding="utf-8")
                free.put(worker)
            outcome: Outcome = "timeout" if code is None else "survived" if code == 0 else "killed"
            on_result(mutant, outcome)
            return mutant.id, outcome

        with ThreadPoolExecutor(max_workers=len(self.workers)) as pool:
            return dict(pool.map(job, mutants))


# --------------------------------------------------------------------------- #
# Reporting                                                                   #
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class TargetReport:
    path: str
    minimum: float
    generated: int = 0
    killed: int = 0
    timeout: int = 0
    survived: int = 0
    equivalent: int = 0
    survivors: list[dict[str, object]] = field(default_factory=list)

    @property
    def score(self) -> float:
        effective = self.generated - self.equivalent
        return 100.0 * (self.killed + self.timeout) / effective if effective else 100.0

    @property
    def passed(self) -> bool:
        return self.score >= self.minimum


@dataclass(slots=True)
class Report:
    targets: list[TargetReport]
    stale_equivalents: list[str]

    @property
    def passed(self) -> bool:
        return not self.stale_equivalents and all(t.passed for t in self.targets)

    def table(self) -> str:
        lines = [
            "| File | Mutants | Killed | Timeout | Survived | Equivalent | Score | Required |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for t in self.targets:
            lines.append(
                f"| `{t.path}` | {t.generated} | {t.killed} | {t.timeout} | {t.survived} "
                f"| {t.equivalent} | {t.score:.2f}% | {t.minimum:.2f}% |"
            )
        total = TargetReport("TOTAL", 0.0)
        for t in self.targets:
            total.generated += t.generated
            total.killed += t.killed
            total.timeout += t.timeout
            total.survived += t.survived
            total.equivalent += t.equivalent
        lines.append(
            f"| **total** | {total.generated} | {total.killed} | {total.timeout} "
            f"| {total.survived} | {total.equivalent} | {total.score:.2f}% | |"
        )
        return "\n".join(lines)


def build_report(
    config: Config, results: dict[str, tuple[Target, list[Mutant], dict[str, Outcome]]]
) -> Report:
    reports: list[TargetReport] = []
    matched: set[str] = set()
    for target, mutants, outcomes in results.values():
        report = TargetReport(target.path, target.minimum, generated=len(mutants))
        for mutant in mutants:
            outcome = outcomes[mutant.id]
            if outcome == "survived" and mutant.id in config.equivalents:
                report.equivalent += 1
                matched.add(mutant.id)
            elif outcome == "survived":
                report.survived += 1
                report.survivors.append(
                    {
                        "id": mutant.id,
                        "line": mutant.line,
                        "original": mutant.original,
                        "mutated": mutant.mutated,
                    }
                )
            elif outcome == "timeout":
                report.timeout += 1
            else:
                report.killed += 1
        reports.append(report)
    stale = sorted(set(config.equivalents) - matched)
    return Report(reports, stale)


def run(
    config: Config, *, workers: int, work_dir: Path, log: Callable[[str], None] = print
) -> Report:
    harness = Harness(config, workers=workers, work_dir=work_dir)
    harness.prepare()
    results: dict[str, tuple[Target, list[Mutant], dict[str, Outcome]]] = {}
    for target in config.targets:
        source = (config.root / config.src_dir / target.path).read_text(encoding="utf-8")
        mutants = generate_mutants(source, target.path)
        harness.self_check(target)
        log(f"{target.path}: {len(mutants)} mutants (timeout {harness.timeouts[target.path]:.0f}s)")
        outcomes = harness.evaluate(
            target,
            mutants,
            lambda m, o: log(f"  {o:<8} L{m.line:<4} {m.original} => {m.mutated}"[:160]),
        )
        results[target.path] = (target, mutants, outcomes)
    return build_report(config, results)


def write_reports(report: Report, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "passed": report.passed,
        "stale_equivalents": report.stale_equivalents,
        "targets": [asdict(t) | {"score": t.score, "passed": t.passed} for t in report.targets],
    }
    (out_dir / "report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (out_dir / "report.md").write_text(report.table() + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.mutation", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--work-dir", type=Path, default=Path(".cache/mutation"))
    args = parser.parse_args(argv)
    config = Config.load(args.config, root=args.root)
    report = run(config, workers=args.workers, work_dir=args.work_dir / "workers")
    write_reports(report, args.work_dir)
    print(report.table())
    for target in report.targets:
        for survivor in target.survivors:
            where = f"{target.path}:{survivor['line']}"
            print(f"SURVIVED {where}: {survivor['original']} => {survivor['mutated']}")
            print(f"  id: {survivor['id']}")
    for stale in report.stale_equivalents:
        print(f"STALE EQUIVALENT (not a surviving mutant): {stale}")
    print("mutation gate:", "OK" if report.passed else "FAIL")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
