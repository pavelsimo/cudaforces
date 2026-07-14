"""Pill badges. Colors are computed server-side and passed in as plain strings."""

import reflex as rx

from .. import theme


def difficulty_badge(label: rx.Var[str] | str, color: rx.Var[str] | str) -> rx.Component:
    return rx.text(
        label,
        color=color,
        border=f"1px solid {theme.BORDER_STRONG}",
        border_radius="99px",
        padding="2px 10px",
        font_size="12px",
        font_weight="600",
        white_space="nowrap",
    )


def verdict_badge(label: rx.Var[str] | str, color: rx.Var[str] | str) -> rx.Component:
    return rx.text(
        label,
        color=color,
        font_family=theme.MONO,
        font_size="12px",
        font_weight="700",
        white_space="nowrap",
    )


def tag_pill(tag: rx.Var[str] | str) -> rx.Component:
    return rx.text(
        tag,
        color=theme.FAINT,
        background=theme.HOVER_BG,
        border=f"1px solid {theme.BORDER}",
        border_radius="99px",
        padding="1px 8px",
        font_size="11px",
        font_family=theme.MONO,
        white_space="nowrap",
    )
