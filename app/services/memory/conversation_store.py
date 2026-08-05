"""
Conversation (short-term) memory store — Phase 5.

Persists full turn-by-turn history per conversation_id to Postgres, so
a conversation survives across separate calls/processes without the
caller needing to carry `history` themselves (Phase 4's limitation).

`ConversationMemory` is a structural Protocol (same pattern as
TranscriptionClient in app/services/transcription.py) so the LangGraph
agent depends on "anything that can get/append messages for a
conversation" rather than this concrete Postgres implementation. Tests
inject a simple in-memory fake instead of standing up a real database.
"""
from __future__ import annotations

from typing import Protocol

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.logging_config import get_logger
from app.db.models import MessageRecord

logger = get_logger(__name__)

_ROLE_TO_MESSAGE_CLASS = {
    "user": HumanMessage,
    "assistant": AIMessage,
    "system": SystemMessage,
}
_MESSAGE_TYPE_TO_ROLE = {"human": "user", "ai": "assistant", "system": "system"}


class ConversationMemory(Protocol):
    async def get_recent_messages(self, conversation_id: str, limit: int) -> list[BaseMessage]: ...

    async def append_messages(self, conversation_id: str, messages: list[BaseMessage]) -> None: ...


class PostgresConversationStore:
    """Real implementation, backed by the `messages` table. Also works
    against SQLite in tests (see tests/test_conversation_store.py) —
    the schema is simple enough that both dialects behave identically
    here."""

    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession]):
        self._sessionmaker = sessionmaker

    async def get_recent_messages(self, conversation_id: str, limit: int = 20) -> list[BaseMessage]:
        """Returns the most recent `limit` messages, oldest-first (the
        order an LLM prompt expects), so the caller doesn't need to
        remember to reverse it."""
        async with self._sessionmaker() as session:
            stmt = (
                select(MessageRecord)
                .where(MessageRecord.conversation_id == conversation_id)
                .order_by(MessageRecord.created_at.desc(), MessageRecord.id.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            records = list(result.scalars().all())

        records.reverse()  # we queried newest-first to LIMIT correctly; flip back to chronological

        messages: list[BaseMessage] = []
        for record in records:
            cls = _ROLE_TO_MESSAGE_CLASS.get(record.role)
            if cls is None:
                logger.warning("Unknown stored role '%s' for conversation=%s, skipping", record.role, conversation_id)
                continue
            messages.append(cls(content=record.content))

        return messages

    async def append_messages(self, conversation_id: str, messages: list[BaseMessage]) -> None:
        if not messages:
            return

        records = []
        for message in messages:
            role = _MESSAGE_TYPE_TO_ROLE.get(message.type)
            if role is None:
                logger.warning("Unknown message type '%s', storing as 'user'", message.type)
                role = "user"
            content = message.content if isinstance(message.content, str) else str(message.content)
            records.append(MessageRecord(conversation_id=conversation_id, role=role, content=content))

        async with self._sessionmaker() as session:
            session.add_all(records)
            await session.commit()

        logger.debug("Persisted %d message(s) for conversation=%s", len(records), conversation_id)
