"""
The LangGraph agent — matches the "LangGraph" box in the architecture
diagram: Memory <-> Conversational Chain -> Response -> (Phase 6)
Generate Ava's Note.

Phase 5 gave `load_memory`/`update_memory` real bodies (Postgres +
ChromaDB). Phase 6 adds two response-quality steps:
- `generate_response`'s output now runs through
  `sanitize_for_speech` (app/agents/text_processing.py) — a defensive
  backstop in case the LLM ignores the system prompt's "no markdown"
  instruction.
- A new `generate_note` node produces a short caption summarizing the
  reply (the diagram's "Generate Ava's Note" step) — sent alongside
  the voice note on WhatsApp so the user can tell what it's about
  without playing the audio, similar to a notification preview.

Both memory stores remain optional (`None` by default), unchanged from
Phase 5 — this keeps every earlier test passing.
"""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.llm import get_llm
from app.agents.prompts import NOTE_SYSTEM_PROMPT, SYSTEM_PROMPT
from app.agents.state import AgentState
from app.agents.text_processing import sanitize_for_speech
from app.core.logging_config import get_logger
from app.services.memory.conversation_store import ConversationMemory
from app.services.memory.vector_store import VectorMemory

logger = get_logger(__name__)

# How many most-recent turns to load verbatim from Postgres. Kept
# small on purpose — this is short-term/recency memory; older-but-
# relevant context is ChromaDB's job (see _load_memory_node below).
RECENT_HISTORY_LIMIT = 20

# How many semantically relevant older snippets to pull from ChromaDB.
SEMANTIC_RECALL_K = 3

# Fallback note length (in words) if the note-generation LLM call
# fails — see _generate_note_node's except branch.
NOTE_FALLBACK_MAX_WORDS = 12


async def _load_memory_node(
    state: AgentState,
    memory_store: ConversationMemory | None,
    vector_store: VectorMemory | None,
) -> dict:
    """Populates `messages` from Postgres (if configured) and prepends
    a system note with semantically relevant older context from
    ChromaDB (if configured). Falls back to Phase 4's no-op when
    neither store is set — `messages` then stays whatever the caller
    passed into the initial state."""
    conversation_id = state["conversation_id"]

    if memory_store is None:
        logger.debug("load_memory | conversation=%s | no memory_store configured, skipping", conversation_id)
        return {"conversation_id": conversation_id}

    recent = await memory_store.get_recent_messages(conversation_id, limit=RECENT_HISTORY_LIMIT)
    logger.debug("load_memory | conversation=%s | loaded %d recent messages", conversation_id, len(recent))

    new_messages: list = list(recent)

    if vector_store is not None:
        relevant = vector_store.search(conversation_id, state["transcript"], k=SEMANTIC_RECALL_K)
        if relevant:
            recall_note = "Earlier in this conversation, relevant context came up: " + " | ".join(relevant)
            new_messages.append(SystemMessage(content=recall_note))
            logger.debug(
                "load_memory | conversation=%s | added %d semantic recall snippet(s)",
                conversation_id,
                len(relevant),
            )

    return {"conversation_id": conversation_id, "messages": new_messages}


async def _generate_response_node(state: AgentState, llm: BaseChatModel) -> dict:
    """The 'Conversational Chain' box: builds system prompt + prior
    history + this turn's transcript, and calls Llama-3.3-70B via Groq.
    The raw reply is run through `sanitize_for_speech` before being
    stored anywhere — a defensive backstop in case the model ignores
    the system prompt's "no markdown" instruction. Returns the new
    user/assistant turn as messages to append (via the add_messages
    reducer — see app/agents/state.py) plus the plain `response`
    string for callers that just want the reply text."""
    history = state.get("messages", [])
    prompt_messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *history,
        HumanMessage(content=state["transcript"]),
    ]

    ai_message = await llm.ainvoke(prompt_messages)
    raw_text = ai_message.content if isinstance(ai_message.content, str) else str(ai_message.content)
    response_text = sanitize_for_speech(raw_text)

    logger.info(
        "generate_response | conversation=%s | transcript_chars=%d | response_chars=%d",
        state["conversation_id"],
        len(state["transcript"]),
        len(response_text),
    )

    return {
        "messages": [HumanMessage(content=state["transcript"]), AIMessage(content=response_text)],
        "response": response_text,
    }


async def _generate_note_node(state: AgentState, llm: BaseChatModel) -> dict:
    """The 'Generate Ava's Note' box from the diagram: a short caption
    summarizing `response`, meant to accompany the voice note as
    WhatsApp message text. Runs as its own small LLM call rather than
    asking for both response + note in one call, so a bad/verbose note
    can never leak into or distort the actual spoken response.

    Falls back to a simple truncation of the response itself if the
    LLM call fails for any reason — a missing/bad caption is a minor
    cosmetic issue, not worth failing the whole turn over when the
    actual response already generated successfully."""
    response_text = state.get("response", "")
    if not response_text:
        return {"note": ""}

    try:
        note_message = await llm.ainvoke(
            [SystemMessage(content=NOTE_SYSTEM_PROMPT), HumanMessage(content=response_text)]
        )
        raw_note = (
            note_message.content if isinstance(note_message.content, str) else str(note_message.content)
        )
        note = sanitize_for_speech(raw_note)
    except Exception as exc:
        logger.warning(
            "generate_note | conversation=%s | note generation failed, using fallback: %s",
            state["conversation_id"],
            exc,
        )
        words = response_text.split()
        note = " ".join(words[:NOTE_FALLBACK_MAX_WORDS])
        if len(words) > NOTE_FALLBACK_MAX_WORDS:
            note += "..."

    logger.debug("generate_note | conversation=%s | note=%r", state["conversation_id"], note)
    return {"note": note}


async def _update_memory_node(
    state: AgentState,
    memory_store: ConversationMemory | None,
    vector_store: VectorMemory | None,
) -> dict:
    """Persists this turn (the last human+ai pair appended by
    generate_response) to Postgres and ChromaDB. Relies on the graph's
    linear shape — update_memory always runs after generate_response
    (with generate_note in between, which doesn't touch `messages`),
    so the last 2 entries in state['messages'] are guaranteed to be
    this turn's new messages, not older history."""
    conversation_id = state["conversation_id"]
    all_messages = state.get("messages", [])

    if memory_store is None and vector_store is None:
        logger.debug("update_memory | conversation=%s | no stores configured, skipping", conversation_id)
        return {"conversation_id": conversation_id}

    new_turn = all_messages[-2:] if len(all_messages) >= 2 else all_messages

    if memory_store is not None:
        await memory_store.append_messages(conversation_id, new_turn)

    if vector_store is not None:
        for message in new_turn:
            role = "assistant" if isinstance(message, AIMessage) else "user"
            content = message.content if isinstance(message.content, str) else str(message.content)
            vector_store.add_memory(conversation_id, role, content)

    logger.debug("update_memory | conversation=%s | persisted %d message(s)", conversation_id, len(new_turn))
    return {"conversation_id": conversation_id}


def build_graph(
    llm: BaseChatModel | None = None,
    memory_store: ConversationMemory | None = None,
    vector_store: VectorMemory | None = None,
) -> CompiledStateGraph:
    """Compile the agent graph. All three dependencies are injectable:
    `llm` for tests/alternate models, `memory_store`/`vector_store` so
    callers that don't need persistence (unit tests, quick prompt
    iteration) can skip standing up Postgres/ChromaDB entirely."""
    chat_model = llm or get_llm()

    graph = StateGraph(AgentState)

    async def _load_memory(state: AgentState) -> dict:
        return await _load_memory_node(state, memory_store, vector_store)

    async def _generate_response(state: AgentState) -> dict:
        return await _generate_response_node(state, chat_model)

    async def _generate_note(state: AgentState) -> dict:
        return await _generate_note_node(state, chat_model)

    async def _update_memory(state: AgentState) -> dict:
        return await _update_memory_node(state, memory_store, vector_store)

    graph.add_node("load_memory", _load_memory)
    graph.add_node("generate_response", _generate_response)
    graph.add_node("generate_note", _generate_note)
    graph.add_node("update_memory", _update_memory)

    graph.add_edge(START, "load_memory")
    graph.add_edge("load_memory", "generate_response")
    graph.add_edge("generate_response", "generate_note")
    graph.add_edge("generate_note", "update_memory")
    graph.add_edge("update_memory", END)

    return graph.compile()
