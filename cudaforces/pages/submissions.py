"""Submission history for the signed-in user."""

import dataclasses

import reflex as rx
import sqlmodel

from .. import db, theme
from ..components import header, verdict_badge
from ..models import Problem, Submission
from ..state import AuthState, ProgressState


@dataclasses.dataclass
class SubmissionRow:
    id: int = 0
    when: str = ""
    problem_title: str = ""
    problem_slug: str = ""
    verdict: str = ""
    verdict_label: str = ""
    verdict_color: str = ""
    total_time_ms: int = 0


class SubmissionsState(AuthState):
    rows: list[SubmissionRow] = []

    @rx.event
    def load(self) -> None:
        if self.user_id == 0:
            return
        with db.session() as s:
            pairs = s.exec(
                sqlmodel.select(Submission, Problem)
                .where(Submission.user_id == self.user_id, Submission.problem_id == Problem.id)
                .order_by(sqlmodel.col(Submission.id).desc())
            ).all()
        self.rows = [
            SubmissionRow(
                id=sub.id or 0,
                when=sub.created_at.strftime("%Y-%m-%d %H:%M"),
                problem_title=prob.title,
                problem_slug=prob.slug,
                verdict=sub.verdict,
                verdict_label=theme.VERDICT_LABELS.get(sub.verdict, sub.verdict),
                verdict_color=theme.VERDICT_COLORS.get(sub.verdict, theme.MUTED),
                total_time_ms=sub.total_time_ms,
            )
            for sub, prob in pairs
        ]


def _row(row: SubmissionRow) -> rx.Component:
    return rx.box(
        rx.text("#", row.id, color=theme.DIM, font_family=theme.MONO, font_size="12.5px"),
        rx.text(row.when, color=theme.MUTED, font_family=theme.MONO, font_size="12.5px"),
        rx.link(
            row.problem_title,
            href="/problem/" + row.problem_slug,
            color=theme.HEADING,
            font_size="13.5px",
            font_weight="600",
            underline="none",
            _hover={"color": theme.ACCENT},
        ),
        verdict_badge(row.verdict_label, row.verdict_color),
        rx.text(row.total_time_ms, " ms", color=theme.DIM, font_family=theme.MONO, font_size="12.5px"),
        display="grid",
        grid_template_columns="60px 150px minmax(0,1fr) 170px 80px",
        align_items="center",
        gap="12px",
        padding="12px 20px",
        border_bottom=f"1px solid {theme.BORDER}",
        _hover={"background": theme.HOVER_BG},
    )


@rx.page(
    route="/submissions",
    title="CudaForces · Submissions",
    on_load=[AuthState.check_auth, SubmissionsState.load, ProgressState.load],
)
def submissions() -> rx.Component:
    return rx.box(
        header(),
        rx.vstack(
            rx.heading("Submissions", color=theme.HEADING, size="8", letter_spacing="-0.02em"),
            rx.cond(
                SubmissionsState.rows,
                rx.box(
                    rx.foreach(SubmissionsState.rows, _row),
                    border=f"1px solid {theme.BORDER}",
                    border_radius="12px",
                    overflow="hidden",
                    width="100%",
                    margin_top="18px",
                ),
                rx.text(
                    "No submissions yet. Pick a problem and hit Submit.",
                    color=theme.MUTED,
                    font_size="14px",
                    margin_top="12px",
                ),
            ),
            max_width="1060px",
            width="100%",
            margin="0 auto",
            padding="34px 22px 60px 22px",
            align="start",
        ),
        min_height="100vh",
        background=theme.BG,
    )
