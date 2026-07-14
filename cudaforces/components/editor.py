"""The CUDA code editor — the single seam for the Monaco dependency.

If reflex-monaco ever breaks against a reflex upgrade, replace the body of
code_editor with the rx.text_area fallback below (kept as _fallback_editor).
"""

from typing import Any

import reflex as rx
from reflex_monaco import monaco

from .. import theme


def code_editor(value: Any, on_change: Any, height: str = "100%") -> rx.Component:
    return monaco(
        value=value,
        on_change=on_change,
        language="cpp",
        theme="vs-dark",
        height=height,
        width="100%",
    )


def _fallback_editor(value: Any, on_change: Any, height: str = "100%") -> rx.Component:
    return rx.el.textarea(
        value=value,
        on_change=on_change,
        spell_check=False,
        font_family=theme.MONO,
        font_size="13px",
        line_height="1.6",
        color=theme.TEXT,
        background=theme.PANEL,
        border="none",
        outline="none",
        resize="none",
        padding="14px",
        width="100%",
        height=height,
    )
