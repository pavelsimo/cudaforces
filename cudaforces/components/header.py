"""Sticky site header: wordmark, solve progress, auth links."""

import reflex as rx

from .. import theme
from ..state import ProgressState


def _logo() -> rx.Component:
    return rx.link(
        rx.hstack(
            rx.box(
                rx.box(background=theme.ACCENT, border_radius="1px"),
                rx.box(background=theme.ACCENT, border_radius="1px", opacity="0.55"),
                rx.box(background=theme.ACCENT, border_radius="1px", opacity="0.55"),
                rx.box(background=theme.ACCENT, border_radius="1px"),
                display="grid",
                grid_template_columns="1fr 1fr",
                gap="2px",
                width="18px",
                height="18px",
            ),
            rx.text(
                "cuda",
                rx.text.span("forces", color=theme.ACCENT, font_weight="700"),
                color=theme.HEADING,
                font_weight="600",
                font_size="17px",
                letter_spacing="-0.02em",
            ),
            spacing="2",
            align="center",
        ),
        href="/",
        underline="none",
    )


def _progress() -> rx.Component:
    return rx.hstack(
        rx.box(
            rx.box(
                background=theme.ACCENT,
                height="100%",
                border_radius="99px",
                width=ProgressState.progress_pct.to_string() + "%",
                transition="width .3s",
            ),
            background=theme.HOVER_BG,
            border=f"1px solid {theme.BORDER}",
            border_radius="99px",
            width="120px",
            height="8px",
            overflow="hidden",
        ),
        rx.text(ProgressState.solved_line, color=theme.MUTED, font_size="12px", white_space="nowrap"),
        spacing="2",
        align="center",
        display=["none", "none", "flex"],
    )


def header() -> rx.Component:
    return rx.hstack(
        _logo(),
        rx.spacer(),
        rx.cond(ProgressState.is_signed_in, _progress()),
        rx.link(
            "problemset",
            href="/",
            color=theme.MUTED,
            font_size="13px",
            underline="none",
            _hover={"color": theme.TEXT},
        ),
        rx.cond(
            ProgressState.is_signed_in,
            rx.hstack(
                rx.link(
                    "submissions",
                    href="/submissions",
                    color=theme.MUTED,
                    font_size="13px",
                    underline="none",
                    _hover={"color": theme.TEXT},
                ),
                rx.text(ProgressState.display_name, color=theme.TEXT, font_size="13px", font_weight="600"),
                rx.link(
                    "sign out",
                    on_click=ProgressState.sign_out,
                    color=theme.DIM,
                    font_size="13px",
                    underline="none",
                    cursor="pointer",
                    _hover={"color": theme.TEXT},
                ),
                spacing="4",
                align="center",
            ),
            rx.link(
                "sign in",
                href="/sign-in",
                color=theme.ACCENT,
                font_size="13px",
                font_weight="600",
                underline="none",
                _hover={"color": theme.ACCENT_HOVER},
            ),
        ),
        position="sticky",
        top="0",
        z_index="10",
        height="58px",
        padding="0 22px",
        align="center",
        spacing="5",
        background="rgba(12,15,18,.85)",
        backdrop_filter="blur(10px)",
        border_bottom=f"1px solid {theme.BORDER}",
        width="100%",
    )
