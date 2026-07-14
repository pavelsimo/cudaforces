"""Application entry point."""

import reflex as rx

from . import (
    db,  # noqa: F401  — registers the SQLite PRAGMA listener
    pages,  # noqa: F401  — registers @rx.page routes
    theme,
)
from .api import api

app = rx.App(
    api_transformer=api,
    stylesheets=[theme.FONTS_URL],
    style={
        "background": theme.BG,
        "color": theme.TEXT,
        "font_family": theme.FONT,
        "::selection": {"background": "rgba(126,231,135,.25)"},
    },
)
