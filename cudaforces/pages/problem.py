"""Problem detail — statement on the left, editor + judge console on the right."""

import asyncio
import dataclasses
import json
from urllib.parse import urlsplit

import reflex as rx
import sqlmodel
from reflex.event import EventSpec

from .. import db, judge, settings, theme
from ..components import code_editor, difficulty_badge, header, tag_pill
from ..models import Problem, Submission
from ..state import AuthState, ProgressState

LLMC_BASE = "https://github.com/karpathy/llm.c/blob/master/dev/cuda/"


@dataclasses.dataclass
class StatementBlock:
    text: str = ""
    mono: bool = False


@dataclasses.dataclass
class ExampleRow:
    input: str = ""
    output: str = ""


@dataclasses.dataclass
class ConsoleLine:
    text: str = ""
    color: str = theme.MUTED


class ProblemState(AuthState):
    problem_slug: str = ""
    title: str = ""
    difficulty: str = ""
    difficulty_color: str = theme.MUTED
    rating: int = 0
    rating_color: str = theme.MUTED
    tags: list[str] = []
    chapter_label: str = ""
    statement: list[StatementBlock] = []
    requirements: list[str] = []
    examples: list[ExampleRow] = []
    constraints: list[str] = []
    note: str = ""
    llmc_file: str = ""
    llmc_url: str = ""
    starter_code: str = ""
    code: str = ""
    console_lines: list[ConsoleLine] = []
    is_judging: bool = False
    verdict: str = ""
    is_solved: bool = False
    prev_slug: str = ""
    prev_title: str = ""
    next_slug: str = ""
    next_title: str = ""

    @rx.var
    def verdict_label(self) -> str:
        return theme.VERDICT_LABELS.get(self.verdict, "")

    @rx.var
    def verdict_color(self) -> str:
        return theme.VERDICT_COLORS.get(self.verdict, theme.MUTED)

    @rx.event
    def load(self) -> None:
        self.problem_slug = urlsplit(str(self.router.url)).path.rstrip("/").rsplit("/", 1)[-1]
        self.console_lines = []
        self.verdict = ""
        self.is_judging = False
        with db.session() as s:
            row = s.exec(sqlmodel.select(Problem).where(Problem.slug == self.problem_slug)).first()
            if row is None:
                self.title = "Problem not found"
                return
            assert row.id is not None
            self.title = row.title
            self.difficulty = row.difficulty
            self.difficulty_color = theme.difficulty_color(row.difficulty)
            self.rating = row.rating
            self.rating_color = theme.rating_color(row.rating)
            self.tags = json.loads(row.tags_json)
            self.chapter_label = f"{row.chapter_num:02d} · {row.chapter_title}"
            self.statement = [
                StatementBlock(text=text, mono=not text.rstrip().endswith((".", ":", "?", "!")))
                for text in json.loads(row.statement_json)
            ]
            self.requirements = json.loads(row.requirements_json)
            self.examples = [ExampleRow(**e) for e in json.loads(row.examples_json)]
            self.constraints = json.loads(row.constraints_json)
            self.note = row.note
            self.llmc_file = row.llmc_file
            self.llmc_url = LLMC_BASE + row.llmc_file
            self.starter_code = row.starter_code
            self.code = self._latest_code(s, row) or row.starter_code
            self.is_solved = self.user_id > 0 and judge.find_solve(s, self.user_id, row.id) is not None
            neighbors = s.exec(sqlmodel.select(Problem).order_by(sqlmodel.col(Problem.position))).all()
            position = next(i for i, p in enumerate(neighbors) if p.slug == self.problem_slug)
            self.prev_slug = neighbors[position - 1].slug if position > 0 else ""
            self.prev_title = neighbors[position - 1].title if position > 0 else ""
            self.next_slug = neighbors[position + 1].slug if position < len(neighbors) - 1 else ""
            self.next_title = neighbors[position + 1].title if position < len(neighbors) - 1 else ""

    def _latest_code(self, s: sqlmodel.Session, row: Problem) -> str | None:
        if self.user_id == 0:
            return None
        latest = s.exec(
            sqlmodel.select(Submission)
            .where(Submission.user_id == self.user_id, Submission.problem_id == row.id)
            .order_by(sqlmodel.col(Submission.id).desc())
        ).first()
        return latest.code if latest else None

    @rx.event
    def set_code(self, value: str) -> None:
        self.code = value

    @rx.event
    def reset_code(self) -> None:
        self.code = self.starter_code

    def _start_console(self) -> None:
        self.verdict = ""
        self.console_lines = [
            ConsoleLine(
                text=f"$ nvcc -O2 -arch={settings.NVCC_ARCH} solution.cu harness.cu -o judge_bin",
                color=theme.DIM,
            )
        ]

    def _render_result(self, result: judge.JudgeResult, test_total: int) -> None:
        if result.compile_output:
            for line in result.compile_output.splitlines()[:40]:
                self.console_lines.append(ConsoleLine(text=line, color=theme.FAINT))
        for test in result.tests:
            if test.status == "pass":
                self.console_lines.append(
                    ConsoleLine(
                        text=f"test {test.index}/{test_total} · OK ({test.time_ms} ms)",
                        color=theme.ACCENT,
                    )
                )
            else:
                self.console_lines.append(
                    ConsoleLine(
                        text=f"test {test.index}/{test_total} · {test.status.upper()} — {test.detail}",
                        color="#f85149",
                    )
                )
        self.verdict = result.verdict

    @rx.event(background=True)
    async def submit_code(self) -> EventSpec | None:
        async with self:
            if self.user_id == 0:
                return rx.redirect("/sign-in")
            if self.is_judging:
                return None
            self.is_judging = True
            self._start_console()
            slug, code, user_id = self.problem_slug, self.code, self.user_id
        verdict, compile_output, results_json = await asyncio.to_thread(_submit_sync, user_id, slug, code)
        async with self:
            self.is_judging = False
            result = judge.JudgeResult(verdict=verdict, compile_output=compile_output)
            result.tests = [
                judge.TestResult(r["test"], r["status"], r["time_ms"], r["detail"]) for r in json.loads(results_json)
            ]
            self._render_result(result, test_total=judge.test_count(slug))
            if verdict == "AC":
                self.is_solved = True
            progress = await self.get_state(ProgressState)
            with db.session() as s:
                progress.solved_count = len(judge.solved_problem_ids(s, user_id))
        return None

    @rx.event(background=True)
    async def run_example(self) -> EventSpec | None:
        """Run test 1 only (the statement's worked example); records nothing."""
        async with self:
            if self.user_id == 0:
                return rx.redirect("/sign-in")
            if self.is_judging:
                return None
            self.is_judging = True
            self._start_console()
            slug, code = self.problem_slug, self.code
        result = await asyncio.to_thread(judge.judge, slug, code, None, 1)
        async with self:
            self.is_judging = False
            self._render_result(result, test_total=1)
            self.verdict = ""  # a sample run is not a verdict
            summary = "sample test passed ✓" if result.verdict == "AC" else f"sample run: {result.verdict}"
            color = theme.ACCENT if result.verdict == "AC" else "#f85149"
            self.console_lines.append(ConsoleLine(text=summary, color=color))
        return None


def _submit_sync(user_id: int, slug: str, code: str) -> tuple[str, str, str]:
    """Judge and persist; returns plain values so no ORM instance escapes the session."""
    with db.session() as s:
        problem = s.exec(sqlmodel.select(Problem).where(Problem.slug == slug)).one()
        submission = judge.submit(s, user_id, problem, code)
        return submission.verdict, submission.compile_output, submission.results_json


def _section_heading(text: str) -> rx.Component:
    return rx.text(
        text,
        color=theme.HEADING,
        font_weight="600",
        font_size="14px",
        margin="18px 0 8px 0",
    )


def _statement_block(block: StatementBlock) -> rx.Component:
    return rx.cond(
        block.mono,
        rx.box(
            rx.text(block.text, font_family=theme.MONO, font_size="13px", color=theme.TEXT),
            background=theme.CARD,
            border=f"1px solid {theme.BORDER}",
            border_radius="9px",
            padding="10px 14px",
            margin="6px 0",
            overflow_x="auto",
        ),
        rx.text(block.text, color=theme.TEXT, font_size="14px", line_height="1.65", margin="6px 0"),
    )


def _example(example: ExampleRow, index: int) -> rx.Component:
    return rx.vstack(
        rx.text(f"Example {index + 1}", color=theme.HEADING, font_weight="600", font_size="13px"),
        rx.box(
            rx.text("Input", color=theme.DIM, font_size="11px", margin_bottom="4px"),
            rx.text(
                example.input,
                font_family=theme.MONO,
                font_size="12.5px",
                color=theme.TEXT,
                white_space="pre-wrap",
            ),
            background=theme.CARD,
            border=f"1px solid {theme.BORDER}",
            border_radius="9px",
            padding="10px 14px",
            width="100%",
        ),
        rx.box(
            rx.text("Output", color=theme.DIM, font_size="11px", margin_bottom="4px"),
            rx.text(
                example.output,
                font_family=theme.MONO,
                font_size="12.5px",
                color=theme.TEXT,
                white_space="pre-wrap",
            ),
            background=theme.CARD,
            border=f"1px solid {theme.BORDER}",
            border_radius="9px",
            padding="10px 14px",
            width="100%",
        ),
        spacing="2",
        width="100%",
        margin="8px 0",
        align="start",
    )


def _bullet(text: rx.Var[str]) -> rx.Component:
    return rx.hstack(
        rx.text("•", color=theme.ACCENT, font_size="13px"),
        rx.text(text, color=theme.TEXT, font_size="13.5px", line_height="1.6"),
        spacing="2",
        align="start",
    )


def _statement_panel() -> rx.Component:
    return rx.box(
        rx.link(
            "← Problemset",
            href="/",
            color=theme.DIM,
            font_size="12px",
            underline="none",
            _hover={"color": theme.TEXT},
        ),
        rx.text(
            ProblemState.chapter_label, color=theme.DIM, font_size="12px", font_family=theme.MONO, margin_top="4px"
        ),
        rx.heading(
            ProblemState.title,
            color=theme.HEADING,
            size="7",
            letter_spacing="-0.02em",
            margin="8px 0 10px 0",
        ),
        rx.hstack(
            difficulty_badge(ProblemState.difficulty, ProblemState.difficulty_color),
            rx.text(
                "★ ",
                ProblemState.rating,
                color=ProblemState.rating_color,
                font_family=theme.MONO,
                font_size="13px",
            ),
            rx.foreach(ProblemState.tags, tag_pill),
            rx.cond(
                ProblemState.is_solved,
                rx.text("✓ solved", color=theme.ACCENT, font_size="12px", font_weight="700"),
            ),
            spacing="2",
            align="center",
            wrap="wrap",
            margin_bottom="14px",
        ),
        rx.foreach(ProblemState.statement, _statement_block),
        _section_heading("Implementation Requirements"),
        rx.vstack(rx.foreach(ProblemState.requirements, _bullet), spacing="1", align="start"),
        rx.foreach(ProblemState.examples, _example),
        _section_heading("Constraints"),
        rx.vstack(rx.foreach(ProblemState.constraints, _bullet), spacing="1", align="start"),
        rx.box(
            rx.text(
                "In llm.c",
                color=theme.ACCENT,
                font_weight="600",
                font_size="12px",
                margin_bottom="4px",
            ),
            rx.text(ProblemState.note, color=theme.TEXT, font_size="13px", line_height="1.6"),
            rx.link(
                "dev/cuda/",
                ProblemState.llmc_file,
                " ↗",
                href=ProblemState.llmc_url,
                is_external=True,
                color=theme.ACCENT,
                font_family=theme.MONO,
                font_size="12px",
                underline="hover",
            ),
            background=theme.ACCENT_BG,
            border=f"1px solid {theme.ACCENT_BORDER}",
            border_radius="9px",
            padding="12px 14px",
            margin="18px 0",
            width="100%",
        ),
        rx.hstack(
            rx.cond(
                ProblemState.prev_slug != "",
                rx.link(
                    "← ",
                    ProblemState.prev_title,
                    href="/problem/" + ProblemState.prev_slug,
                    color=theme.MUTED,
                    font_size="13px",
                    underline="none",
                    _hover={"color": theme.ACCENT},
                ),
                rx.text(""),
            ),
            rx.spacer(),
            rx.cond(
                ProblemState.next_slug != "",
                rx.link(
                    ProblemState.next_title,
                    " →",
                    href="/problem/" + ProblemState.next_slug,
                    color=theme.MUTED,
                    font_size="13px",
                    underline="none",
                    _hover={"color": theme.ACCENT},
                ),
                rx.text(""),
            ),
            width="100%",
            margin_top="10px",
        ),
        padding="24px 28px 48px 28px",
        overflow_y="auto",
        height="100%",
        border_right=f"1px solid {theme.BORDER}",
    )


def _console_line(line: ConsoleLine) -> rx.Component:
    return rx.text(
        line.text,
        color=line.color,
        font_family=theme.MONO,
        font_size="12px",
        line_height="1.55",
        white_space="pre-wrap",
        word_break="break-word",
    )


def _editor_panel() -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.hstack(
                rx.box(width="8px", height="8px", border_radius="99px", background="#e3b341"),
                rx.text("solution.cu", font_family=theme.MONO, font_size="12.5px", color=theme.TEXT),
                spacing="2",
                align="center",
                background=theme.CARD,
                border=f"1px solid {theme.BORDER}",
                border_radius="8px",
                padding="5px 12px",
            ),
            rx.text("CUDA · nvcc -arch=native", color=theme.DIM, font_size="11.5px", font_family=theme.MONO),
            rx.spacer(),
            rx.text(
                "reset",
                on_click=ProblemState.reset_code,
                color=theme.DIM,
                font_size="12px",
                cursor="pointer",
                _hover={"color": theme.TEXT},
            ),
            rx.button(
                "Run",
                on_click=ProblemState.run_example,
                disabled=ProblemState.is_judging,
                background="transparent",
                color=theme.TEXT,
                border=f"1px solid {theme.BORDER_STRONG}",
                border_radius="8px",
                height="30px",
                padding="0 16px",
                font_size="13px",
                cursor="pointer",
                _hover={"background": theme.HOVER_BG},
            ),
            rx.button(
                rx.cond(ProblemState.is_judging, rx.spinner(size="1"), rx.text("Submit")),
                on_click=ProblemState.submit_code,
                disabled=ProblemState.is_judging,
                background=theme.ACCENT,
                color="#0c0f12",
                border_radius="8px",
                height="30px",
                padding="0 18px",
                font_size="13px",
                font_weight="700",
                cursor="pointer",
                _hover={"background": theme.ACCENT_HOVER},
            ),
            width="100%",
            padding="10px 14px",
            border_bottom=f"1px solid {theme.BORDER}",
            align="center",
            spacing="3",
            background=theme.PANEL,
        ),
        rx.box(
            code_editor(value=ProblemState.code, on_change=ProblemState.set_code),
            flex="1",
            width="100%",
            min_height="0",
            background=theme.PANEL,
        ),
        rx.box(
            rx.hstack(
                rx.text("Console", color=theme.DIM, font_size="11px", letter_spacing=".08em"),
                rx.spacer(),
                rx.cond(
                    ProblemState.verdict != "",
                    rx.text(
                        ProblemState.verdict_label,
                        color=ProblemState.verdict_color,
                        font_family=theme.MONO,
                        font_size="12.5px",
                        font_weight="700",
                    ),
                ),
                width="100%",
                margin_bottom="6px",
            ),
            rx.box(
                rx.cond(
                    ProblemState.console_lines,
                    rx.box(rx.foreach(ProblemState.console_lines, _console_line)),
                    rx.text(
                        "Run compiles your kernel and checks it against the worked example. "
                        "Submit runs the full test set.",
                        color=theme.DIM,
                        font_size="12px",
                        font_family=theme.MONO,
                    ),
                ),
                overflow_y="auto",
                height="150px",
            ),
            width="100%",
            padding="10px 14px",
            border_top=f"1px solid {theme.BORDER}",
            background=theme.PANEL,
        ),
        spacing="0",
        height="100%",
    )


@rx.page(
    route="/problem/[slug]",
    title="CudaForces",
    on_load=[AuthState.load_user, ProblemState.load, ProgressState.load],
)
def problem() -> rx.Component:
    return rx.box(
        header(),
        rx.box(
            _statement_panel(),
            _editor_panel(),
            display="grid",
            grid_template_columns="minmax(430px, 46%) minmax(0, 1fr)",
            height="calc(100vh - 58px)",
        ),
        background=theme.BG,
        height="100vh",
        overflow="hidden",
    )
