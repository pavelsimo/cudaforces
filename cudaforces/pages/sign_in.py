"""Sign-in page: request a magic link code by email."""

import reflex as rx

from .. import settings
from ..state import AuthState


def _guest_login() -> list[rx.Component]:
    """Rendered only in dev (no SMTP configured); settings are compile-time constants."""
    if not settings.GUEST_LOGIN_ENABLED:
        return []
    return [
        rx.divider(),
        rx.button(
            "Continue as guest",
            on_click=AuthState.sign_in_as_guest,
            variant="outline",
            width="100%",
        ),
    ]


@rx.page(route="/sign-in", title="Sign in · CudaForces")
def sign_in() -> rx.Component:
    return rx.container(
        rx.vstack(
            rx.heading("Sign in", size="7"),
            rx.text("Enter your email and we'll send you a verification code."),
            rx.form(
                rx.vstack(
                    rx.input(name="email", placeholder="you@example.com", type="email", required=True, width="100%"),
                    rx.button("Send code", type="submit", width="100%"),
                    spacing="3",
                ),
                on_submit=AuthState.send_code,
                width="100%",
            ),
            rx.cond(AuthState.error != "", rx.text(AuthState.error, color_scheme="red")),
            *_guest_login(),
            spacing="4",
            padding_top="4em",
        ),
        size="1",
    )
