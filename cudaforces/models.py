"""Database models. State lives in records, not boolean columns."""

from datetime import UTC, datetime, timedelta

import sqlmodel

CODE_TTL = timedelta(minutes=15)
CODE_RATE_LIMIT = 10  # codes per CODE_TTL window per identity


def utcnow() -> datetime:
    """Naive UTC timestamp — SQLite stores datetimes without timezone info."""
    return datetime.now(UTC).replace(tzinfo=None)


class Identity(sqlmodel.SQLModel, table=True):
    """A global login identity — one row per email address."""

    id: int | None = sqlmodel.Field(default=None, primary_key=True)
    email: str = sqlmodel.Field(unique=True, index=True)


class User(sqlmodel.SQLModel, table=True):
    """An application user attached to an identity."""

    id: int | None = sqlmodel.Field(default=None, primary_key=True)
    identity_id: int = sqlmodel.Field(foreign_key="identity.id", index=True)
    display_name: str


class Session(sqlmodel.SQLModel, table=True):
    """An authenticated session; the random token is the credential."""

    id: int | None = sqlmodel.Field(default=None, primary_key=True)
    identity_id: int = sqlmodel.Field(foreign_key="identity.id", index=True)
    user_id: int = sqlmodel.Field(foreign_key="user.id", index=True)
    token: str = sqlmodel.Field(unique=True, index=True)
    created_at: datetime = sqlmodel.Field(default_factory=utcnow)


class MagicLinkCode(sqlmodel.SQLModel, table=True):
    """A short-lived 6-digit sign-in code."""

    id: int | None = sqlmodel.Field(default=None, primary_key=True)
    identity_id: int = sqlmodel.Field(foreign_key="identity.id", index=True)
    value: str = sqlmodel.Field(index=True)
    expires_at: datetime
    created_at: datetime = sqlmodel.Field(default_factory=utcnow)


class Problem(sqlmodel.SQLModel, table=True):
    """A CUDA kernel exercise. Content is synced from the problems registry by slug."""

    id: int | None = sqlmodel.Field(default=None, primary_key=True)
    slug: str = sqlmodel.Field(unique=True, index=True)
    position: int = sqlmodel.Field(default=0, index=True)
    title: str = ""
    difficulty: str = ""
    rating: int = 0
    chapter_id: str = ""
    chapter_num: int = 0
    chapter_title: str = ""
    chapter_desc: str = ""
    tags_json: str = "[]"
    llmc_file: str = ""
    summary: str = ""
    statement_json: str = "[]"
    requirements_json: str = "[]"
    examples_json: str = "[]"
    constraints_json: str = "[]"
    note: str = ""
    starter_code: str = ""
    rtol: float = 1e-4
    atol: float = 1e-5
    time_limit_ms: int = 5000


class Submission(sqlmodel.SQLModel, table=True):
    """One judged attempt: the code, the verdict, and per-test results."""

    id: int | None = sqlmodel.Field(default=None, primary_key=True)
    user_id: int = sqlmodel.Field(foreign_key="user.id", index=True)
    problem_id: int = sqlmodel.Field(foreign_key="problem.id", index=True)
    code: str
    verdict: str  # AC | WA | CE | RE | TLE | JE
    compile_output: str = ""
    results_json: str = "[]"
    total_time_ms: int = 0
    created_at: datetime = sqlmodel.Field(default_factory=utcnow)


class Solve(sqlmodel.SQLModel, table=True):
    """A user solved a problem — solved-ness is this record, not a boolean column."""

    __table_args__ = (sqlmodel.UniqueConstraint("user_id", "problem_id"),)

    id: int | None = sqlmodel.Field(default=None, primary_key=True)
    user_id: int = sqlmodel.Field(foreign_key="user.id", index=True)
    problem_id: int = sqlmodel.Field(foreign_key="problem.id", index=True)
    submission_id: int = sqlmodel.Field(foreign_key="submission.id")
    created_at: datetime = sqlmodel.Field(default_factory=utcnow)
