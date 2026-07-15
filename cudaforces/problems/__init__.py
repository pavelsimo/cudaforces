"""Problem registry: statements from content.json + per-problem judge assets.

Each problem owns a directory (slug with underscores) containing:
  harness.cu: main() that reads the stdin protocol, calls solve(), prints outputs
  ref.py:     NumPy reference: tests() -> list[RefCase], solve(inputs) -> outputs
"""

import functools
import importlib
import json
from pathlib import Path
from types import ModuleType

import sqlmodel

from ..models import Problem
from .types import DEFAULT_ATOL, DEFAULT_RTOL, DEFAULT_TIME_LIMIT_MS, Chapter, ProblemDef

PROBLEMS_DIR = Path(__file__).parent
CONTENT_PATH = PROBLEMS_DIR / "content.json"
JUDGE_IO_HEADER = PROBLEMS_DIR / "judge_io.h"


@functools.cache
def all_problems() -> list[ProblemDef]:
    data = json.loads(CONTENT_PATH.read_text())
    chapters = {c["id"]: Chapter(**c) for c in data["chapters"]}
    return [
        ProblemDef(
            slug=p["slug"],
            position=position,
            title=p["title"],
            difficulty=p["difficulty"],
            rating=p["rating"],
            chapter=chapters[p["chapter"]],
            tags=p["tags"],
            summary=p["summary"],
            statement=p["statement"],
            requirements=p["requirements"],
            examples=p["examples"],
            constraints=p["constraints"],
            note=p["note"],
            starter_code=p["starter"],
        )
        for position, p in enumerate(data["problems"])
    ]


def get(slug: str) -> ProblemDef:
    for problem in all_problems():
        if problem.slug == slug:
            return problem
    raise KeyError(f"unknown problem: {slug}")


def ref_module(slug: str) -> ModuleType:
    return importlib.import_module(f"cudaforces.problems.{get(slug).module_name}.ref")


def harness_path(slug: str) -> Path:
    return PROBLEMS_DIR / get(slug).module_name / "harness.cu"


def judge_meta(slug: str) -> tuple[float, float, int]:
    """(rtol, atol, time_limit_ms); ref.py module constants override defaults."""
    try:
        ref = ref_module(slug)
    except ModuleNotFoundError:
        return (DEFAULT_RTOL, DEFAULT_ATOL, DEFAULT_TIME_LIMIT_MS)
    return (
        getattr(ref, "RTOL", DEFAULT_RTOL),
        getattr(ref, "ATOL", DEFAULT_ATOL),
        getattr(ref, "TIME_LIMIT_MS", DEFAULT_TIME_LIMIT_MS),
    )


def sync_problems(session: sqlmodel.Session) -> None:
    """Upsert every ProblemDef into the problem table by slug. Idempotent."""
    existing = {p.slug: p for p in session.exec(sqlmodel.select(Problem)).all()}
    for pd in all_problems():
        rtol, atol, time_limit_ms = judge_meta(pd.slug)
        row = existing.get(pd.slug) or Problem(slug=pd.slug)
        row.position = pd.position
        row.title = pd.title
        row.difficulty = pd.difficulty
        row.rating = pd.rating
        row.chapter_id = pd.chapter.id
        row.chapter_num = pd.chapter.num
        row.chapter_title = pd.chapter.title
        row.chapter_desc = pd.chapter.desc
        row.tags_json = json.dumps(pd.tags)
        row.summary = pd.summary
        row.statement_json = json.dumps(pd.statement)
        row.requirements_json = json.dumps(pd.requirements)
        row.examples_json = json.dumps(pd.examples)
        row.constraints_json = json.dumps(pd.constraints)
        row.note = pd.note
        row.starter_code = pd.starter_code
        row.rtol = rtol
        row.atol = atol
        row.time_limit_ms = time_limit_ms
        session.add(row)
    session.commit()
