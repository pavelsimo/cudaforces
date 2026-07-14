"""Authentication state shared by all pages."""

from typing import Any

import reflex as rx
import sqlmodel
from reflex.event import EventSpec

from . import auth, db, mailer
from .models import User


class AuthState(rx.State):
    session_token: str = rx.Cookie("", name="session_token", same_site="lax")
    email: str = ""
    error: str = ""
    display_name: str = ""
    user_id: int = 0

    @rx.var
    def is_signed_in(self) -> bool:
        return self.user_id > 0

    def _resolve_user(self) -> None:
        with db.session() as s:
            session = auth.find_session_by_token(s, self.session_token)
            if session is None:
                self.user_id = 0
                self.display_name = ""
                return
            user = s.get(User, session.user_id)
            self.user_id = session.user_id
            self.display_name = user.display_name if user else ""

    @rx.event
    def load_user(self) -> None:
        """on_load for public pages — resolves the cookie without redirecting."""
        self._resolve_user()

    @rx.event
    def check_auth(self) -> EventSpec | None:
        """on_load guard for protected pages."""
        self._resolve_user()
        if self.user_id == 0:
            return rx.redirect("/sign-in")
        return None

    @rx.event
    def send_code(self, form_data: dict[str, Any]) -> EventSpec | None:
        email = auth.normalize_email(str(form_data.get("email", "")))
        if not auth.valid_email(email):
            self.error = "Enter a valid email address."
            return None
        with db.session() as s:
            identity = auth.find_or_create_identity(s, email)
            code = auth.issue_code(s, identity)
        if code is None:
            self.error = "Too many codes requested. Try again in a few minutes."
            return None
        mailer.send_magic_link_code(email, code.value)
        self.email = email
        self.error = ""
        return rx.redirect("/verify")

    @rx.event
    def confirm_code(self, form_data: dict[str, Any]) -> EventSpec | None:
        value = str(form_data.get("code", "")).strip()
        with db.session() as s:
            session = auth.verify_code(s, self.email, value)
        if session is None:
            self.error = "Invalid or expired code."
            return None
        self.session_token = session.token
        self.error = ""
        return rx.redirect("/")

    @rx.event
    def sign_out(self) -> EventSpec:
        with db.session() as s:
            auth.terminate_session(s, self.session_token)
        self.session_token = ""
        self.display_name = ""
        self.user_id = 0
        return rx.redirect("/")


class ProgressState(AuthState):
    """Solved-progress numbers shown in the site header."""

    solved_count: int = 0
    total_count: int = 0

    @rx.var
    def progress_pct(self) -> int:
        return round(100 * self.solved_count / self.total_count) if self.total_count else 0

    @rx.var
    def solved_line(self) -> str:
        return f"{self.solved_count} / {self.total_count} solved"

    @rx.event
    def load(self) -> None:
        from . import judge  # local import — judge pulls in generate/numpy
        from .models import Problem

        with db.session() as s:
            self.total_count = len(s.exec(sqlmodel.select(Problem.id)).all())
            self.solved_count = len(judge.solved_problem_ids(s, self.user_id)) if self.user_id else 0
