"""Problemset — the public landing page."""

import dataclasses
import json

import reflex as rx
import sqlmodel

from .. import db, judge, theme
from ..components import difficulty_badge, header, tag_pill
from ..models import Problem
from ..state import AuthState, ProgressState

DIFFICULTIES = ["All", "Easy", "Medium", "Hard"]


@dataclasses.dataclass
class ProblemRow:
    slug: str = ""
    id_label: str = ""
    title: str = ""
    summary: str = ""
    tags: list[str] = dataclasses.field(default_factory=list)
    difficulty: str = ""
    difficulty_color: str = ""
    rating: int = 0
    rating_color: str = ""
    solved: bool = False


@dataclasses.dataclass
class ChapterGroup:
    num_label: str = ""
    title: str = ""
    desc: str = ""
    problems: list[ProblemRow] = dataclasses.field(default_factory=list)


class ProblemListState(AuthState):
    query: str = ""
    difficulty: str = "All"
    _chapters: list[ChapterGroup] = []

    @rx.event
    def set_query(self, value: str) -> None:
        self.query = value

    @rx.event
    def set_difficulty(self, value: str) -> None:
        self.difficulty = value

    @rx.event
    def load(self) -> None:
        with db.session() as s:
            rows = s.exec(sqlmodel.select(Problem).order_by(sqlmodel.col(Problem.position))).all()
            solved_ids = judge.solved_problem_ids(s, self.user_id) if self.user_id else set()
        chapters: list[ChapterGroup] = []
        letter = 0
        for p in rows:
            if not chapters or chapters[-1].title != p.chapter_title:
                chapters.append(
                    ChapterGroup(num_label=f"{p.chapter_num:02d}", title=p.chapter_title, desc=p.chapter_desc)
                )
                letter = 0
            letter += 1
            chapters[-1].problems.append(
                ProblemRow(
                    slug=p.slug,
                    id_label=f"{p.chapter_num}{chr(64 + letter)}",
                    title=p.title,
                    summary=p.summary,
                    tags=json.loads(p.tags_json),
                    difficulty=p.difficulty,
                    difficulty_color=theme.difficulty_color(p.difficulty),
                    rating=p.rating,
                    rating_color=theme.rating_color(p.rating),
                    solved=p.id in solved_ids,
                )
            )
        self._chapters = chapters

    @rx.var
    def filtered_chapters(self) -> list[ChapterGroup]:
        query = self.query.strip().lower()
        result = []
        for chapter in self._chapters:
            kept = [
                row
                for row in chapter.problems
                if (self.difficulty == "All" or row.difficulty == self.difficulty)
                and (not query or query in row.title.lower() or any(query in t for t in row.tags))
            ]
            if kept:
                result.append(
                    ChapterGroup(num_label=chapter.num_label, title=chapter.title, desc=chapter.desc, problems=kept)
                )
        return result


def _difficulty_chip(value: str) -> rx.Component:
    active = ProblemListState.difficulty == value
    return rx.button(
        value,
        on_click=ProblemListState.set_difficulty(value),  # type: ignore[operator]  # partial event args
        background=rx.cond(active, "rgba(126,231,135,.18)", "transparent"),
        color=rx.cond(active, theme.ACCENT, theme.MUTED),
        border=rx.cond(active, f"1px solid {theme.ACCENT_BORDER}", f"1px solid {theme.BORDER}"),
        border_radius="99px",
        padding="4px 14px",
        height="30px",
        font_size="13px",
        cursor="pointer",
        _hover={"background": theme.HOVER_BG},
    )


def _problem_row(row: ProblemRow) -> rx.Component:
    return rx.link(
        rx.box(
            rx.text(row.id_label, color=theme.DIM, font_family=theme.MONO, font_size="13px"),
            rx.vstack(
                rx.hstack(
                    rx.text(row.title, color=theme.HEADING, font_weight="600", font_size="14px"),
                    rx.foreach(row.tags, tag_pill),
                    spacing="2",
                    align="center",
                    wrap="wrap",
                ),
                rx.text(row.summary, color=theme.FAINT, font_size="12px"),
                spacing="1",
                align="start",
            ),
            difficulty_badge(row.difficulty, row.difficulty_color),
            rx.text(
                "★ ",
                row.rating,
                color=row.rating_color,
                font_family=theme.MONO,
                font_size="13px",
                white_space="nowrap",
            ),
            rx.cond(
                row.solved,
                rx.text("✓", color=theme.ACCENT, font_weight="700", font_size="15px"),
                rx.text("", width="10px"),
            ),
            display="grid",
            grid_template_columns="44px minmax(0,1fr) 92px 64px 24px",
            align_items="center",
            gap="12px",
            padding="13px 20px",
            border_bottom=f"1px solid {theme.BORDER}",
            transition="background .12s",
            _hover={"background": theme.HOVER_BG},
        ),
        href=f"/problem/{row.slug}",
        underline="none",
        width="100%",
    )


def _chapter_section(chapter: ChapterGroup) -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.text(chapter.num_label, color=theme.ACCENT, font_family=theme.MONO, font_size="14px"),
            rx.text(chapter.title, color=theme.HEADING, font_weight="600", font_size="16px"),
            rx.text("— ", chapter.desc, color=theme.DIM, font_size="13px", display=["none", "block"]),
            spacing="2",
            align="baseline",
        ),
        rx.box(
            rx.foreach(chapter.problems, _problem_row),
            border=f"1px solid {theme.BORDER}",
            border_radius="12px",
            overflow="hidden",
            width="100%",
        ),
        spacing="2",
        width="100%",
        margin_bottom="26px",
        align="start",
    )


@rx.page(
    route="/",
    title="CudaForces — Problemset",
    on_load=[AuthState.load_user, ProblemListState.load, ProgressState.load],
)
def index() -> rx.Component:
    return rx.box(
        header(),
        rx.vstack(
            rx.heading("Problemset", color=theme.HEADING, size="8", letter_spacing="-0.02em"),
            rx.text(
                "Twenty CUDA kernel exercises covering the full GPT-2 training loop: elementwise ops, "
                "embeddings, reductions, matmuls, attention, loss, and the optimizer. Write the kernel, "
                "hit Submit, and the judge compiles it with nvcc and runs it on your GPU.",
                color=theme.MUTED,
                font_size="14px",
            ),
            rx.hstack(
                rx.input(
                    placeholder="Search problems or tags…",
                    value=ProblemListState.query,
                    on_change=ProblemListState.set_query,
                    background=theme.CARD,
                    border=f"1px solid {theme.BORDER}",
                    border_radius="9px",
                    color=theme.TEXT,
                    height="34px",
                    width="280px",
                ),
                rx.spacer(),
                *[_difficulty_chip(d) for d in DIFFICULTIES],
                spacing="2",
                width="100%",
                margin="10px 0 22px 0",
                align="center",
            ),
            rx.foreach(ProblemListState.filtered_chapters, _chapter_section),
            max_width="1060px",
            width="100%",
            margin="0 auto",
            padding="34px 22px 60px 22px",
            align="start",
        ),
        min_height="100vh",
        background=theme.BG,
    )
