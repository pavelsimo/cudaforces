import sqlmodel

from cudaforces import problems
from cudaforces.models import Problem


def test_registry_has_twenty_problems() -> None:
    defs = problems.all_problems()
    assert len(defs) == 20
    assert len({p.slug for p in defs}) == 20
    assert [p.position for p in defs] == list(range(20))


def test_every_problem_has_judge_assets() -> None:
    for pd in problems.all_problems():
        assert 'extern "C" void solve' in pd.starter_code, pd.slug
        problem_dir = problems.PROBLEMS_DIR / pd.module_name
        assert (problem_dir / "harness.cu").is_file(), pd.slug
        assert (problem_dir / "ref.py").is_file(), pd.slug


def test_content_is_clean() -> None:
    blob = problems.CONTENT_PATH.read_text()
    assert "—" not in blob  # no em-dashes
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
    assert len(rows) == 20

    problems.sync_problems(db)
    rows = db.exec(sqlmodel.select(Problem).order_by(sqlmodel.col(Problem.position))).all()
    assert len(rows) == 20
    assert rows[0].slug == "residual-forward"
    assert rows[0].title == "Residual Forward"
    assert rows[0].chapter_num == 1
