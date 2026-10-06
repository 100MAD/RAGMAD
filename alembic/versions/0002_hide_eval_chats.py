"""hide evaluation chats from the sidebar

Revision ID: 0002_hide_eval_chats
Revises: 0001_initial
Create Date: 2026-10-06

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002_hide_eval_chats"
down_revision: Union[str, Sequence[str], None] = "0001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "chats",
        sa.Column("hidden", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.execute("UPDATE chats SET hidden = true WHERE title = 'Evaluation'")


def downgrade() -> None:
    op.drop_column("chats", "hidden")
