"""drop problem llmc_file

Revision ID: 52a602b01910
Revises: 1c36d9c34623
Create Date: 2026-07-15 10:50:59.566923

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "52a602b01910"
down_revision: str | None = "1c36d9c34623"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("problem", schema=None) as batch_op:
        batch_op.drop_column("llmc_file")


def downgrade() -> None:
    with op.batch_alter_table("problem", schema=None) as batch_op:
        batch_op.add_column(sa.Column("llmc_file", sa.VARCHAR(), nullable=False, server_default=""))
