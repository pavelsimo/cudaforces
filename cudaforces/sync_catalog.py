"""Synchronize the problem catalog without creating development users."""

from . import db, problems


def sync_catalog() -> None:
    with db.session() as session:
        problems.sync_problems(session)


if __name__ == "__main__":
    sync_catalog()
