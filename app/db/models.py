"""
SQLAlchemy models for conversation history (Phase 5).

One table, deliberately simple: every message (user or assistant turn)
for every conversation, in order. This is the "Memory" box's Postgres
side in the architecture diagram — full, durable conversation history.
Semantic/long-term recall (facts, similarity search) is ChromaDB's job,
handled separately in app/services/memory/vector_store.py.
"""
from datetime import datetime, timezone

from sqlalchemy import DateTime, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class MessageRecord(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)  # "user" | "assistant" | "system"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    __table_args__ = (
        # Every query in this phase filters by conversation_id and
        # orders by recency — this index serves both at once.
        Index("ix_messages_conversation_id_created_at", "conversation_id", "created_at"),
    )
