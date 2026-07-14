"""Home page — requires authentication."""

import reflex as rx

from ..state import AuthState


@rx.page(route="/", title="CudaForces", on_load=AuthState.check_auth)
def index() -> rx.Component:
    return rx.container(
        rx.vstack(
            rx.heading("CudaForces", size="8"),
            rx.text("Codeforces-style CUDA kernel problemset with a local nvcc judge"),
            rx.text("Signed in as ", rx.text.strong(AuthState.display_name)),
            rx.button("Sign out", on_click=AuthState.sign_out, variant="soft"),
            spacing="4",
            padding_top="4em",
        ),
        size="2",
    )
