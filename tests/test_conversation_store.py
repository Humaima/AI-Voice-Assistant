"""
Tests for PostgresConversationStore (Phase 5).

Uses SQLite in-memory (via aiosqlite) instead of a real Postgres
server — same SQLAlchemy ORM models and async session API, so this
exercises real SQL behavior (ordering, filtering, persistence across
queries within a connection) without needing Docker/Postgres running.
The schema here is simple enough that SQLite and Postgres behave
identically for these tests.
"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.db.session import build_engine, build_sessionmaker, create_all_tables
from app.services.memory.conversation_store import PostgresConversationStore


@pytest.fixture
async def store():
    """Fresh in-memory SQLite database per test — StaticPool keeps the
    same in-memory DB alive across the multiple connections our async
    sessionmaker opens (plain in-memory SQLite is per-connection by
    default and would otherwise look empty on the second query)."""
    from sqlalchemy.pool import StaticPool

    engine = build_engine("sqlite+aiosqlite:///:memory:", poolclass=StaticPool)
    await create_all_tables(engine)
    sessionmaker = build_sessionmaker(engine)
    yield PostgresConversationStore(sessionmaker)
    await engine.dispose()


class TestPostgresConversationStore:
    async def test_empty_conversation_returns_empty_list(self, store):
        messages = await store.get_recent_messages("conv-1", limit=20)
        assert messages == []

    async def test_append_then_retrieve_preserves_order(self, store):
        await store.append_messages(
            "conv-1",
            [HumanMessage(content="hello"), AIMessage(content="hi there")],
        )

        messages = await store.get_recent_messages("conv-1", limit=20)

        assert len(messages) == 2
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].content == "hello"
        assert isinstance(messages[1], AIMessage)
        assert messages[1].content == "hi there"

    async def test_multiple_appends_accumulate_in_order(self, store):
        await store.append_messages("conv-1", [HumanMessage(content="first")])
        await store.append_messages("conv-1", [AIMessage(content="first reply")])
        await store.append_messages("conv-1", [HumanMessage(content="second")])

        messages = await store.get_recent_messages("conv-1", limit=20)

        assert [m.content for m in messages] == ["first", "first reply", "second"]

    async def test_limit_returns_most_recent_messages_in_chronological_order(self, store):
        for i in range(5):
            await store.append_messages("conv-1", [HumanMessage(content=f"message {i}")])

        messages = await store.get_recent_messages("conv-1", limit=2)

        # Should be the 2 MOST RECENT, but still returned oldest-first.
        assert [m.content for m in messages] == ["message 3", "message 4"]

    async def test_conversations_are_isolated(self, store):
        await store.append_messages("conv-1", [HumanMessage(content="in conv 1")])
        await store.append_messages("conv-2", [HumanMessage(content="in conv 2")])

        conv1_messages = await store.get_recent_messages("conv-1", limit=20)
        conv2_messages = await store.get_recent_messages("conv-2", limit=20)

        assert [m.content for m in conv1_messages] == ["in conv 1"]
        assert [m.content for m in conv2_messages] == ["in conv 2"]

    async def test_system_role_roundtrips(self, store):
        await store.append_messages("conv-1", [SystemMessage(content="a system note")])

        messages = await store.get_recent_messages("conv-1", limit=20)

        assert len(messages) == 1
        assert isinstance(messages[0], SystemMessage)

    async def test_append_empty_list_is_a_no_op(self, store):
        await store.append_messages("conv-1", [])
        messages = await store.get_recent_messages("conv-1", limit=20)
        assert messages == []
