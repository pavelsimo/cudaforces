"""The judge: compile a submission with nvcc and run it against the test files.

Plain functions over subprocess + text comparison; no Reflex, no GPU bindings.
Run headless with: uv run python -m cudaforces.judge <slug> <solution.cu>
"""

import json
import math
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import sqlmodel

from . import generate, problems, settings
from .models import Problem, Solve, Submission

MAX_OUTPUT_CHARS = 20_000


@dataclass
class TestResult:
    index: int
    status: str  # pass | wa | re | tle
    time_ms: int
    detail: str = ""


@dataclass
class JudgeResult:
    verdict: str  # AC | WA | CE | RE | TLE | JE
    compile_output: str = ""
    tests: list[TestResult] = field(default_factory=list)

    @property
    def total_time_ms(self) -> int:
        return sum(t.time_ms for t in self.tests)


def compile_solution(workdir: Path, slug: str) -> tuple[bool, str]:
    command = [
        settings.NVCC_PATH,
        "-O2",
        f"-arch={settings.NVCC_ARCH}",
        "-I",
        str(problems.PROBLEMS_DIR),
        "solution.cu",
        str(problems.harness_path(slug)),
        "-o",
        "judge_bin",
    ]
    try:
        proc = subprocess.run(
            command,
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=settings.JUDGE_COMPILE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return False, "nvcc: compilation timed out"
    output = (proc.stderr or "") + (proc.stdout or "")
    return proc.returncode == 0, output.strip()[:MAX_OUTPUT_CHARS]


def compare_outputs(actual: str, expected: str, rtol: float, atol: float) -> str | None:
    """None when every value matches within tolerance, else a mismatch description."""
    got = actual.split()
    want = expected.split()
    if len(got) != len(want):
        return f"expected {len(want)} values, got {len(got)}"
    for i, (g, w) in enumerate(zip(got, want, strict=True)):
        try:
            gf = float(g)
        except ValueError:
            return f"value {i}: not a number: {g[:40]!r}"
        wf = float(w)
        if math.isnan(gf) or not math.isclose(gf, wf, rel_tol=rtol, abs_tol=atol):
            return f"value {i}: got {g}, expected {w}"
    return None


def run_test(binary: Path, in_path: Path, timeout_s: float) -> tuple[str, str, int]:
    """(status, output, time_ms): status is ok | re | tle; output is stdout or stderr."""
    started = time.perf_counter()
    try:
        with in_path.open() as stdin:
            proc = subprocess.run(
                [str(binary)],
                stdin=stdin,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=binary.parent,
            )
    except subprocess.TimeoutExpired:
        return "tle", "", int(timeout_s * 1000)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[:MAX_OUTPUT_CHARS]
        return "re", f"exit code {proc.returncode}: {detail}", elapsed_ms
    return "ok", proc.stdout, elapsed_ms


def test_count(slug: str) -> int:
    return len(_test_files(slug))


def _test_files(slug: str) -> list[Path]:
    directory = generate.tests_dir(slug)
    in_files = sorted(directory.glob("*.in"))
    if not in_files:
        generate.ensure_test_data(slug)
        in_files = sorted(directory.glob("*.in"))
    return in_files


def judge(
    slug: str,
    code: str,
    workdir: Path | None = None,
    max_tests: int | None = None,
    time_limit_ms: int | None = None,
) -> JudgeResult:
    rtol, atol, default_limit_ms = problems.judge_meta(slug)
    limit_ms = time_limit_ms or default_limit_ms
    workdir = (workdir or settings.DATA_DIR / "submissions" / uuid.uuid4().hex).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "solution.cu").write_text(code)

    ok, compile_output = compile_solution(workdir, slug)
    if not ok:
        return JudgeResult(verdict="CE", compile_output=compile_output)

    result = JudgeResult(verdict="AC", compile_output=compile_output)
    binary = workdir / "judge_bin"
    for index, in_path in enumerate(_test_files(slug)[:max_tests], 1):
        status, output, time_ms = run_test(binary, in_path, limit_ms / 1000)
        if status == "tle":
            result.tests.append(TestResult(index, "tle", time_ms, f"exceeded {limit_ms} ms"))
            result.verdict = "TLE"
            break
        if status == "re":
            result.tests.append(TestResult(index, "re", time_ms, output))
            result.verdict = "RE"
            break
        mismatch = compare_outputs(output, in_path.with_suffix(".out").read_text(), rtol, atol)
        if mismatch is not None:
            result.tests.append(TestResult(index, "wa", time_ms, mismatch))
            result.verdict = "WA"
            break
        result.tests.append(TestResult(index, "pass", time_ms))
    return result


def submit(session: sqlmodel.Session, user_id: int, problem: Problem, code: str) -> Submission:
    """Judge the code, record the Submission, and mark the problem solved on AC."""
    assert problem.id is not None
    try:
        result = judge(problem.slug, code)
    except Exception as error:  # noqa: BLE001 (judge failures become a JE verdict)
        result = JudgeResult(verdict="JE", compile_output=f"judge error: {error}")
    submission = Submission(
        user_id=user_id,
        problem_id=problem.id,
        code=code,
        verdict=result.verdict,
        compile_output=result.compile_output,
        results_json=results_to_json(result),
        total_time_ms=result.total_time_ms,
    )
    session.add(submission)
    session.commit()
    session.refresh(submission)
    if result.verdict == "AC" and find_solve(session, user_id, problem.id) is None:
        assert submission.id is not None
        session.add(Solve(user_id=user_id, problem_id=problem.id, submission_id=submission.id))
        session.commit()
    return submission


def find_solve(session: sqlmodel.Session, user_id: int, problem_id: int) -> Solve | None:
    return session.exec(sqlmodel.select(Solve).where(Solve.user_id == user_id, Solve.problem_id == problem_id)).first()


def solved_problem_ids(session: sqlmodel.Session, user_id: int) -> set[int]:
    solves = session.exec(sqlmodel.select(Solve).where(Solve.user_id == user_id)).all()
    return {s.problem_id for s in solves}


def results_to_json(result: JudgeResult) -> str:
    return json.dumps(
        [{"test": t.index, "status": t.status, "time_ms": t.time_ms, "detail": t.detail} for t in result.tests]
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("usage: python -m cudaforces.judge <slug> <solution.cu>")
        raise SystemExit(2)
    slug_arg, path_arg = sys.argv[1], sys.argv[2]
    outcome = judge(slug_arg, Path(path_arg).read_text())
    if outcome.compile_output:
        print(outcome.compile_output)
    for test in outcome.tests:
        line = f"test {test.index}: {test.status} ({test.time_ms} ms)"
        if test.detail:
            line += f": {test.detail}"
        print(line)
    print(f"verdict: {outcome.verdict}")
    raise SystemExit(0 if outcome.verdict == "AC" else 1)
