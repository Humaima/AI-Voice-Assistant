"""
Agent service — the boundary between the LangGraph graph
(app/agents/graph.py) and the rest of the app. Converts between the
API-friendly ChatMessageModel and LangChain's BaseMessage types, owns
the compiled graph instance, and (Phase 5) lazily constructs the real
Postgres/ChromaDB-backed memory stores so persistence "just works" for
callers that only pass a conversation_id.
"""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.agents.graph import build_graph
from app.agents.state import AgentState
from app.core.logging_config import get_logger
from app.db.session import build_engine, build_sessionmaker
from app.models.agent import AgentResult, ChatMessageModel
from app.services.memory.conversation_store import ConversationMemory, PostgresConversationStore
from app.services.memory.vector_store import ChromaVectorStore, VectorMemory, build_chroma_client

logger = get_logger(__name__)

_ROLE_TO_MESSAGE_CLASS = {
    "user": HumanMessage,
    "assistant": AIMessage,
    "system": SystemMessage,
}
_MESSAGE_TYPE_TO_ROLE = {"human": "user", "ai": "assistant", "system": "system"}


def messages_from_models(models: list[ChatMessageModel]) -> list[BaseMessage]:
    result = []
    for m in models:
        cls = _ROLE_TO_MESSAGE_CLASS.get(m.role)
        if cls is None:
            logger.warning("Unknown message role '%s', treating as user turn", m.role)
            cls = HumanMessage
        result.append(cls(content=m.content))
    return result


def messages_to_models(messages: list[BaseMessage]) -> list[ChatMessageModel]:
    return [
        ChatMessageModel(
            role=_MESSAGE_TYPE_TO_ROLE.get(m.type, m.type),
            content=m.content if isinstance(m.content, str) else str(m.content),
        )
        for m in messages
    ]


class AgentService:
    """Owns one compiled graph instance. Construct once (e.g. as a
    module-level singleton via get_agent_service below) rather than per
    request — compiling the graph has a small fixed cost, and there's
    no per-request state to isolate since AgentState is passed fresh
    into .ainvoke() each call.

    `memory_store`/`vector_store` are optional: pass them to get real
    persistence (production, or a test using fakes); leave them unset
    to get Phase 4's caller-carries-history behavior, useful for quick
    prompt iteration without a database."""

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        memory_store: ConversationMemory | None = None,
        vector_store: VectorMemory | None = None,
    ):
        self._memory_store = memory_store
        self._vector_store = vector_store
        self._graph = build_graph(llm, memory_store=memory_store, vector_store=vector_store)

    @property
    def memory_store(self) -> ConversationMemory | None:
        """The Postgres-backed conversation store this service is
        configured with, if any. Public so other callers (Phase 8's
        streaming path, which bypasses the graph for response
        generation) can reuse the same store instance rather than
        constructing a second one — see app/api/audio.py's
        voice-reply-stream-test endpoint."""
        return self._memory_store

    @property
    def vector_store(self) -> VectorMemory | None:
        """The ChromaDB-backed semantic store this service is
        configured with, if any. Same reasoning as memory_store above."""
        return self._vector_store

    async def run(
        self,
        conversation_id: str,
        transcript: str,
        history: list[BaseMessage] | None = None,
    ) -> AgentResult:
        """When a memory_store is configured, history is loaded from it
        automatically by the graph's load_memory node — any explicit
        `history` passed here is ignored, since the store is the
        authoritative source once persistence is turned on. Without a
        memory_store, `history` is used as-is (Phase 4 behavior)."""
        initial_messages: list[BaseMessage] = [] if self._memory_store is not None else (history or [])

        initial_state: AgentState = {
            "conversation_id": conversation_id,
            "transcript": transcript,
            "messages": initial_messages,
            "response": "",
            "note": "",
        }

        result_state = await self._graph.ainvoke(initial_state)

        return AgentResult(
            response_text=result_state["response"],
            history=messages_to_models(result_state["messages"]),
        )


_agent_service_singleton: AgentService | None = None


def get_agent_service() -> AgentService:
    """Lazy singleton — the real Groq-backed graph, Postgres engine,
    and ChromaDB client are only constructed on first use, so importing
    this module never requires GROQ_API_KEY or a reachable database to
    be set up (Postgres/ChromaDB connection settings default to
    localhost, see app/core/config.py). Real memory stores are wired in
    by default here; pass explicit stores to AgentService directly if
    you want Phase 4's no-persistence behavior instead (as most tests
    in tests/test_agent.py do, via fakes)."""
    global _agent_service_singleton
    if _agent_service_singleton is None:
        engine = build_engine()
        sessionmaker = build_sessionmaker(engine)
        memory_store = PostgresConversationStore(sessionmaker)

        chroma_client = build_chroma_client()
        vector_store = ChromaVectorStore(chroma_client)

        _agent_service_singleton = AgentService(memory_store=memory_store, vector_store=vector_store)

    return _agent_service_singleton
