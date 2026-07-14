import os
import stat
from pathlib import Path

import pytest
import sqlmodel

from cudaforces import judge, problems, settings
from cudaforces.models import Identity, Problem, Solve, User

SLUG = "residual-forward"


def test_compare_outputs_exact_match() -> None:
    assert judge.compare_outputs("1.5\n2.5\n", "1.5\n2.5\n", 1e-4, 1e-5) is None


def test_compare_outputs_within_tolerance() -> None:
    assert judge.compare_outputs("1.00001\n", "1.0\n", 1e-4, 1e-5) is None


def test_compare_outputs_mismatch() -> None:
    detail = judge.compare_outputs("1.5\n9.9\n", "1.5\n2.5\n", 1e-4, 1e-5)
    assert detail == "value 1: got 9.9, expected 2.5"


def test_compare_outputs_wrong_count() -> None:
    assert judge.compare_outputs("1.5\n", "1.5\n2.5\n", 1e-4, 1e-5) == "expected 2 values, got 1"


def test_compare_outputs_nan_fails() -> None:
    assert judge.compare_outputs("nan\n", "2.5\n", 1e-4, 1e-5) is not None


def test_compare_outputs_garbage_fails() -> None:
    detail = judge.compare_outputs("Segmentation\n", "2.5\n", 1e-4, 1e-5)
    assert detail is not None and "not a number" in detail


def _write_script(path: Path, body: str) -> Path:
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.fixture
def judge_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated DATA_DIR with one canned test case for SLUG; returns tmp_path."""
    tests_dir = tmp_path / "data" / "tests" / SLUG
    tests_dir.mkdir(parents=True)
    (tests_dir / "01.in").write_text("2\n1\n2\n0.5\n0.5\n")
    (tests_dir / "01.out").write_text("1.5\n2.5\n")
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path / "data")
    return tmp_path


def _fake_nvcc(tmp_path: Path, binary_body: str) -> Path:
    """A stand-in compiler: always succeeds and emits judge_bin running binary_body."""
    return _write_script(
        tmp_path / "nvcc",
        f"printf '#!/bin/sh\\n{binary_body}\\n' > judge_bin\nchmod +x judge_bin",
    )


def test_judge_compile_error(judge_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    nvcc = _write_script(judge_env / "nvcc", 'echo "solution.cu(3): error: expected a ;" >&2\nexit 1')
    monkeypatch.setattr(settings, "NVCC_PATH", str(nvcc))
    result = judge.judge(SLUG, "bad code", workdir=judge_env / "w")
    assert result.verdict == "CE"
    assert "expected a ;" in result.compile_output
    assert result.tests == []


def test_judge_accepted(judge_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    expected = judge_env / "data" / "tests" / SLUG / "01.out"
    monkeypatch.setattr(settings, "NVCC_PATH", str(_fake_nvcc(judge_env, f"cat {expected}")))
    result = judge.judge(SLUG, "code", workdir=judge_env / "w")
    assert result.verdict == "AC"
    assert [t.status for t in result.tests] == ["pass"]


def test_judge_wrong_answer(judge_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "NVCC_PATH", str(_fake_nvcc(judge_env, "echo 1.5; echo 9.9")))
    result = judge.judge(SLUG, "code", workdir=judge_env / "w")
    assert result.verdict == "WA"
    assert "got 9.9" in result.tests[0].detail


def test_judge_runtime_error(judge_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "NVCC_PATH", str(_fake_nvcc(judge_env, "echo boom >&2; exit 3")))
    result = judge.judge(SLUG, "code", workdir=judge_env / "w")
    assert result.verdict == "RE"
    assert "exit code 3" in result.tests[0].detail


def test_judge_time_limit(judge_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "NVCC_PATH", str(_fake_nvcc(judge_env, "sleep 5")))
    result = judge.judge(SLUG, "code", workdir=judge_env / "w", time_limit_ms=300)
    assert result.verdict == "TLE"


def _make_user(db: sqlmodel.Session) -> User:
    identity = Identity(email="dev@example.com")
    db.add(identity)
    db.commit()
    assert identity.id is not None
    user = User(identity_id=identity.id, display_name="Dev")
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def test_submit_records_submission_and_solve_once(
    db: sqlmodel.Session, judge_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = judge_env / "data" / "tests" / SLUG / "01.out"
    monkeypatch.setattr(settings, "NVCC_PATH", str(_fake_nvcc(judge_env, f"cat {expected}")))
    monkeypatch.chdir(judge_env)
    problems.sync_problems(db)
    user = _make_user(db)
    problem = db.exec(sqlmodel.select(Problem).where(Problem.slug == SLUG)).one()
    assert user.id is not None

    first = judge.submit(db, user.id, problem, "code")
    second = judge.submit(db, user.id, problem, "code")
    assert first.verdict == "AC" and second.verdict == "AC"
    solves = db.exec(sqlmodel.select(Solve)).all()
    assert len(solves) == 1
    assert solves[0].submission_id == first.id
    assert judge.solved_problem_ids(db, user.id) == {problem.id}


def test_submit_judge_error_verdict(db: sqlmodel.Session, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(slug: str, code: str) -> judge.JudgeResult:
        raise RuntimeError("disk full")

    monkeypatch.setattr(judge, "judge", boom)
    problems.sync_problems(db)
    user = _make_user(db)
    problem = db.exec(sqlmodel.select(Problem).where(Problem.slug == SLUG)).one()
    assert user.id is not None
    submission = judge.submit(db, user.id, problem, "code")
    assert submission.verdict == "JE"
    assert "disk full" in submission.compile_output
    assert judge.solved_problem_ids(db, user.id) == set()


def test_judge_env_isolated_from_repo() -> None:
    # judge_env fixture must not leak: real settings restored by monkeypatch
    assert str(settings.DATA_DIR) == os.environ.get("DATA_DIR", "data")
