from logging.config import fileConfig

import sqlmodel

from alembic import context
from cudaforces import (
    db,
    models,  # noqa: F401  (populate SQLModel.metadata)
)

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = sqlmodel.SQLModel.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=str(db.engine.url),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    with db.engine.connect() as connection:
        # batch mode rebuilds tables (copy + drop + rename); FK enforcement must be
        # off on this connection or dropping a referenced table fails
        connection.exec_driver_sql("PRAGMA foreign_keys=OFF")
        connection.commit()  # end the autobegun transaction so alembic owns its own
        context.configure(connection=connection, target_metadata=target_metadata, render_as_batch=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
