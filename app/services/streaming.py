"""
Audio Streaming (Phase 8).

Reduces server-side latency by overlapping "waiting for the rest of
the LLM's response" with "already synthesizing speech for the part
that's done" — instead of the Phase 4-7 sequence (wait for the full
text response, THEN synthesize the full audio clip), this streams
Ava's reply sentence by sentence, kicking off TTS for each sentence as
soon as it's ready.

IMPORTANT CONTEXT — what this does and doesn't change: WhatsApp
delivers voice notes as complete audio files, not a live stream, so
this doesn't change what an end WhatsApp user experiences once Phase 9
sends the reply — they still get one complete voice note. What it DOES
change is wall-clock time on the server between receiving a transcript
and having the full reply ready, and it lays the groundwork for any
future consumer that CAN play audio progressively (e.g. a live web
interface). The debug endpoints in app/api/audio.py let you measure
this improvement directly (see the README's Phase 8 testing steps).

DESIGN NOTE — why this isn't a LangGraph node: LangGraph's node model
returns one dict update per node execution; it's not a natural fit for
a node that needs to progressively yield partial results to an outer
async generator while it's still running. Rather than force that
abstraction, this module bypasses the graph for response generation
specifically, but reuses graph.py's load_memory/generate_note/
update_memory node functions directly for everything else — they're
plain async functions, not tied to LangGraph internals, so there's
almost no duplicated logic between the streaming and non-streaming
paths. Only generate_response has a genuinely different implementation
here (streaming vs. one blocking call).
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.graph import _generate_note_node, _load_memory_node, _update_memory_node
from app.agents.prompts import SYSTEM_PROMPT
from app.agents.state import AgentState
from app.agents.text_processing import sanitize_for_speech
from app.core.logging_config import get_logger
from app.services.memory.conversation_store import ConversationMemory
from app.services.memory.vector_store import VectorMemory
from app.services.sentence_chunker import SentenceChunker
from app.services.tts import TTSService

logger = get_logger(__name__)


def _partial_state(**overrides: Any) -> AgentState:
    """graph.py's node functions expect a full AgentState (a TypedDict
    with every key required), but each node only actually reads a
    couple of keys — this fills in harmless defaults for the rest so
    we can call those functions directly here without constructing a
    full real state (which doesn't exist yet at this point in the
    streaming flow anyway)."""
    base: dict[str, Any] = {
        "conversation_id": "",
        "transcript": "",
        "messages": [],
        "response": "",
        "note": "",
    }
    base.update(overrides)
    return cast(AgentState, base)


@dataclass
class SentenceEvent:
    """One completed sentence-group from a streamed LLM response, with
    timing relative to when generation started. Used both to drive
    per-sentence TTS in `stream_voice_reply` and, on its own (without
    any TTS), as the payload for the metrics-only debug endpoint."""

    text: str
    elapsed_seconds: float


async def stream_response_sentences(
    llm: BaseChatModel,
    prompt_messages: list,
    min_chunk_chars: int = 20,
) -> AsyncIterator[SentenceEvent]:
    """Streams the LLM's response token-by-token (`llm.astream`),
    yielding each completed sentence-group as soon as it's ready
    instead of waiting for the full response. This is the piece that
    makes the latency win possible — everything downstream (TTS,
    persistence) can start working on the first sentence while later
    ones are still being generated.
    """
    start = time.monotonic()
    chunker = SentenceChunker(min_chunk_chars=min_chunk_chars)

    async for chunk in llm.astream(prompt_messages):
        delta = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
        for sentence in chunker.feed(delta):
            yield SentenceEvent(sentence, time.monotonic() - start)

    leftover = chunker.flush()
    if leftover:
        yield SentenceEvent(leftover, time.monotonic() - start)


async def _build_prompt_messages(
    conversation_id: str,
    transcript: str,
    memory_store: ConversationMemory | None,
    vector_store: VectorMemory | None,
) -> list:
    """Loads memory via graph.py's `_load_memory_node` (reused
    directly, not reimplemented) and builds the same
    [system, *history, transcript] shape `_generate_response_node`
    uses, so streamed and non-streamed replies see identical context."""
    load_state = _partial_state(conversation_id=conversation_id, transcript=transcript, messages=[])
    loaded = await _load_memory_node(load_state, memory_store, vector_store)
    history = loaded.get("messages", [])
    return [SystemMessage(content=SYSTEM_PROMPT), *history, HumanMessage(content=transcript)]


async def stream_voice_reply(
    conversation_id: str,
    transcript: str,
    llm: BaseChatModel,
    tts_service: TTSService,
    memory_store: ConversationMemory | None = None,
    vector_store: VectorMemory | None = None,
) -> AsyncIterator[bytes]:
    """The Phase 8 fast path: streams synthesized audio out sentence by
    sentence. Persists memory/note at the end using graph.py's
    generate_note/update_memory node functions directly, so streamed
    replies are stored identically to non-streamed ones.

    Note on sanitization: each sentence is sanitized independently as
    it's synthesized (vs. the non-streaming path sanitizing the whole
    response at once). This means markdown that happens to span a
    sentence-chunk boundary won't be caught — an accepted tradeoff for
    the latency win; worth knowing if a response somehow produces
    misplaced formatting only in the streaming path.
    """
    prompt_messages = await _build_prompt_messages(conversation_id, transcript, memory_store, vector_store)

    full_text_parts: list[str] = []

    async for event in stream_response_sentences(llm, prompt_messages):
        full_text_parts.append(event.text)
        clean_sentence = sanitize_for_speech(event.text)
        if not clean_sentence:
            continue

        logger.debug(
            "stream_voice_reply | conversation=%s | sentence ready at %.2fs (%d chars)",
            conversation_id,
            event.elapsed_seconds,
            len(clean_sentence),
        )
        # TTSService.synthesize is a blocking/sync call (real network
        # I/O) — run it off the event loop so other requests aren't
        # blocked while this one waits on ElevenLabs.
        audio_chunk = await asyncio.to_thread(tts_service.synthesize, clean_sentence)
        yield audio_chunk

    full_text = sanitize_for_speech(" ".join(p.strip() for p in full_text_parts if p.strip()))

    note_result = await _generate_note_node(
        _partial_state(conversation_id=conversation_id, response=full_text), llm
    )

    await _update_memory_node(
        _partial_state(
            conversation_id=conversation_id,
            messages=[HumanMessage(content=transcript), AIMessage(content=full_text)],
        ),
        memory_store,
        vector_store,
    )

    logger.info(
        "stream_voice_reply | conversation=%s | complete | response_chars=%d | note=%r",
        conversation_id,
        len(full_text),
        note_result.get("note", ""),
    )
