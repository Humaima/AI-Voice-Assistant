"""
Debug routes for exercising the LangGraph agent directly with text
input, without needing an actual voice note. Useful for iterating on
prompts/persona (app/agents/prompts.py) quickly, and (Phase 8) for
measuring streaming's latency improvement.
"""
import asyncio
import time

from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.agents.llm import get_llm
from app.agents.prompts import SYSTEM_PROMPT
from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.models.agent import AgentChatRequest, AgentResult
from app.models.streaming import LatencyComparisonResult, SentenceTiming, StreamMetricsResult
from app.services.agent_service import AgentService, get_agent_service, messages_from_models
from app.services.streaming import stream_response_sentences, stream_voice_reply
from app.services.tts import TTSService

logger = get_logger(__name__)
settings = get_settings()
router = APIRouter()


@router.post("/chat-test", response_model=AgentResult)
async def chat_test(request: AgentChatRequest):
    """Send a transcript (as if it came from Phase 3), get back Ava's
    reply. Conversation history is now (Phase 5) persisted server-side
    in Postgres/ChromaDB by conversation_id — you no longer need to
    pass `history` back yourself; just reuse the same conversation_id
    across calls to continue a conversation. `history` in the request
    is only honored if persistent memory isn't configured (see
    AgentService docstring) and is otherwise ignored.

    Requires `python -m app.db.bootstrap` to have been run once, and
    Postgres/ChromaDB reachable (docker compose up).
    """
    if not settings.groq_api_key:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not set. Add it to your .env file to use the agent.",
        )

    history_messages = messages_from_models(request.history)

    try:
        service = get_agent_service()
        result = await service.run(request.conversation_id, request.transcript, history_messages)
    except Exception as exc:  # LangGraph/Groq/DB/Chroma can raise a variety of error types here
        logger.error("Agent run failed for conversation=%s: %s", request.conversation_id, exc)
        raise HTTPException(
            status_code=502,
            detail=(
                f"Agent failed: {exc}. If this is a connection error, make sure Postgres and "
                "ChromaDB are running (docker compose up) and that `python -m app.db.bootstrap` "
                "has been run once."
            ),
        ) from exc

    return result


class StreamTestRequest(BaseModel):
    conversation_id: str
    transcript: str


@router.post("/stream-metrics-test", response_model=StreamMetricsResult)
async def stream_metrics_test(request: StreamTestRequest):
    """Streams Ava's response and returns timing for each sentence as
    it became ready — no TTS involved, no memory loaded. This isolates
    exactly one thing: how quickly sentences become available from the
    LLM as it streams, which is the foundation the latency win in
    Phase 8 is built on. Cheap and fast to call repeatedly since it
    never touches ElevenLabs, Postgres, or ChromaDB.
    """
    if not settings.groq_api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set. Add it to your .env file.")

    prompt_messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=request.transcript)]
    llm = get_llm()

    sentences: list[SentenceTiming] = []
    total_elapsed = 0.0
    try:
        async for event in stream_response_sentences(llm, prompt_messages):
            sentences.append(SentenceTiming(text=event.text, elapsed_seconds=round(event.elapsed_seconds, 3)))
            total_elapsed = event.elapsed_seconds
    except Exception as exc:
        logger.error("Streaming failed for conversation=%s: %s", request.conversation_id, exc)
        raise HTTPException(status_code=502, detail=f"Streaming failed: {exc}") from exc

    full_text = " ".join(s.text for s in sentences)
    return StreamMetricsResult(
        sentences=sentences, full_text=full_text, total_elapsed_seconds=round(total_elapsed, 3)
    )


@router.post("/latency-comparison-test", response_model=LatencyComparisonResult)
async def latency_comparison_test(request: StreamTestRequest):
    """Runs the SAME transcript through both the blocking (Phase 4-7)
    and streaming (Phase 8) paths and reports timing for each, so the
    improvement is a number you can look at instead of something you
    have to take on faith. Deliberately side-effect-free — neither
    path here touches Postgres/ChromaDB (memory_store/vector_store are
    left unset on both), so this is safe and cheap to run repeatedly
    as a pure benchmark, and comparing the two is apples-to-apples
    (no memory I/O time mixed into either number).

    - `blocking_total_ms`: time for the old approach — wait for the
      complete response, then synthesize the complete audio in one
      TTS call.
    - `streaming_time_to_first_audio_ms`: time until the FIRST audio
      chunk is ready in the new approach — this is what actually
      drives perceived latency for anyone waiting on a reply.
    - `streaming_total_ms`: time for the streaming approach to finish
      entirely (all sentences synthesized) — usually close to, or a
      bit more than, blocking_total_ms, since it makes more TTS calls
      overall. The win is in `streaming_time_to_first_audio_ms`, not
      this number.
    """
    if not settings.groq_api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set. Add it to your .env file.")
    if not settings.elevenlabs_api_key:
        raise HTTPException(status_code=500, detail="ELEVENLABS_API_KEY is not set. Add it to your .env file.")

    llm = get_llm()
    tts_service = TTSService()

    # --- Blocking path: full response, then one TTS call ---
    blocking_start = time.monotonic()
    try:
        blocking_service = AgentService(llm=llm)  # no memory_store/vector_store — side-effect-free
        blocking_result = await blocking_service.run(request.conversation_id, request.transcript)
        await asyncio.to_thread(tts_service.synthesize, blocking_result.response_text)
    except Exception as exc:
        logger.error("Blocking path failed for conversation=%s: %s", request.conversation_id, exc)
        raise HTTPException(status_code=502, detail=f"Blocking path failed: {exc}") from exc
    blocking_total_ms = (time.monotonic() - blocking_start) * 1000

    # --- Streaming path: sentence-by-sentence ---
    streaming_start = time.monotonic()
    time_to_first_audio_ms: float | None = None
    try:
        async for _audio_chunk in stream_voice_reply(
            request.conversation_id, request.transcript, llm, tts_service
        ):
            if time_to_first_audio_ms is None:
                time_to_first_audio_ms = (time.monotonic() - streaming_start) * 1000
    except Exception as exc:
        logger.error("Streaming path failed for conversation=%s: %s", request.conversation_id, exc)
        raise HTTPException(status_code=502, detail=f"Streaming path failed: {exc}") from exc
    streaming_total_ms = (time.monotonic() - streaming_start) * 1000

    if time_to_first_audio_ms is None:
        # Response was empty enough that no sentence ever flushed — treat
        # "first audio" as the whole streaming duration in that edge case.
        time_to_first_audio_ms = streaming_total_ms

    return LatencyComparisonResult(
        blocking_total_ms=round(blocking_total_ms, 1),
        streaming_time_to_first_audio_ms=round(time_to_first_audio_ms, 1),
        streaming_total_ms=round(streaming_total_ms, 1),
        speedup_to_first_audio=round(blocking_total_ms / time_to_first_audio_ms, 2),
    )
