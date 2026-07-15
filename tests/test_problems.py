import re

import sqlmodel

from cudaforces import problems
from cudaforces.models import Problem


def _normalize_abi(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def test_registry_has_fifty_problems() -> None:
    defs = problems.all_problems()
    assert len(defs) == 50
    assert len({p.slug for p in defs}) == 50
    assert [p.position for p in defs] == list(range(50))
    difficulties = [p.difficulty for p in defs[:30]]
    assert difficulties.count("Easy") == 15
    assert difficulties.count("Medium") == 10
    assert difficulties.count("Hard") == 5


def test_every_problem_has_judge_assets() -> None:
    for index, pd in enumerate(problems.all_problems()):
        assert 'extern "C" void solve' in pd.starter_code, pd.slug
        problem_dir = problems.PROBLEMS_DIR / pd.module_name
        harness_path = problem_dir / "harness.cu"
        assert harness_path.is_file(), pd.slug
        assert (problem_dir / "ref.py").is_file(), pd.slug
        starter_abi = re.search(r'extern "C" void solve\((.*?)\)\s*\{', pd.starter_code, re.DOTALL)
        harness_abi = re.search(r'extern "C" void solve\((.*?)\)\s*;', harness_path.read_text(), re.DOTALL)
        assert starter_abi is not None and harness_abi is not None, pd.slug
        assert _normalize_abi(starter_abi.group(1)) == _normalize_abi(harness_abi.group(1)), pd.slug
        if index < 30:
            assert len(problems.ref_module(pd.slug).tests()) >= 4, pd.slug


def test_content_is_clean() -> None:
    blob = problems.CONTENT_PATH.read_text()
    assert chr(0x2014) not in blob  # no em-dashes
    assert "llm.c" not in blob  # no source attribution
    for pd in problems.all_problems():
        for para in pd.statement:
            assert para.count("$$") % 2 == 0, f"{pd.slug}: unbalanced display math in {para!r}"


def test_get_unknown_slug_raises() -> None:
    try:
        problems.get("nope")
        raise AssertionError("expected KeyError")
    except KeyError:
        pass


def test_sync_problems_is_idempotent(db: sqlmodel.Session) -> None:
    problems.sync_problems(db)
    rows = db.exec(sqlmodel.select(Problem)).all()
    assert len(rows) == 50
    residual = db.exec(sqlmodel.select(Problem).where(Problem.slug == "residual-forward")).one()
    residual_id = residual.id

    problems.sync_problems(db)
    rows = db.exec(sqlmodel.select(Problem).order_by(sqlmodel.col(Problem.position))).all()
    assert len(rows) == 50
    assert rows[0].slug == "grid-stride-saxpy"
    assert rows[0].title == "Grid-Stride SAXPY"
    assert rows[0].chapter_num == 1
    residual = db.exec(sqlmodel.select(Problem).where(Problem.slug == "residual-forward")).one()
    assert residual.id == residual_id
