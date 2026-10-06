"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-10-06

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "chats",
        sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_table(
        "documents",
        sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
        sa.Column("chat_id", sa.Uuid(), nullable=False),
        sa.Column("filename", sa.String(length=500), nullable=False),
        sa.Column("content_type", sa.String(length=200), nullable=False),
        sa.Column("size", sa.Integer(), nullable=False),
        sa.Column("storage_key", sa.String(length=1000), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["chat_id"], ["chats.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_documents_chat_id", "documents", ["chat_id"])
    op.create_index("ix_documents_status", "documents", ["status"])
    op.create_table(
        "messages",
        sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
        sa.Column("chat_id", sa.Uuid(), nullable=False),
        sa.Column("role", sa.String(length=20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("sources", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["chat_id"], ["chats.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_messages_chat_id", "messages", ["chat_id"])


def downgrade() -> None:
    op.drop_index("ix_messages_chat_id", table_name="messages")
    op.drop_table("messages")
    op.drop_index("ix_documents_status", table_name="documents")
    op.drop_index("ix_documents_chat_id", table_name="documents")
    op.drop_table("documents")
    op.drop_table("chats")
