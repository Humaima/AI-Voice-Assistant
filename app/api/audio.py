"""
Debug routes for the voice pipeline (Phases 2-3-4-6-7-8): lets you POST
an audio file and see each stage of the pipeline run, without waiting
for Phase 9's Twilio webhook to exist. Useful for manual testing with
curl or the /docs Swagger UI.

These routes are NOT the production WhatsApp entrypoint — that's built
in Phase 9 and will call these services directly from the webhook
handler instead of via file upload.
"""
import base64
import io

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.agents.llm import get_llm
from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.services.agent_service import get_agent_service, messages_from_models
from app.services.audio_buffer import AudioBuffer
from app.services.audio_processor import AudioValidationError, process_voice_note
from app.services.streaming import stream_voice_reply
from app.services.transcription import TranscriptionError, TranscriptionService
from app.services.tts import AUDIO_MEDIA_TYPES, TTSError, TTSService

logger = get_logger(__name__)
settings = get_settings()
router = APIRouter()


@router.post("/process-test")
async def process_audio_test(file: UploadFile = File(...)):
    """Upload a voice note (ogg/mp3/wav/m4a/amr) and get back the
    pipeline's processing result: validated, converted to 16kHz mono,
    normalized, silence-trimmed, and chunked."""
    audio_bytes = await file.read()
    declared_format = (file.filename or "").rsplit(".", 1)[-1] if file.filename else ""

    if not declared_format:
        # Fall back to content-type's subtype, e.g. "audio/ogg" -> "ogg"
        declared_format = (file.content_type or "").split("/")[-1]

    try:
        result, segments = process_voice_note(audio_bytes, declared_format)
    except AudioValidationError as exc:
        logger.warning("Voice note rejected: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    buffer = AudioBuffer.from_segments(conversation_id="test-conversation", segments=segments)

    return {
        "result": result.model_dump(),
        "buffer_total_duration_ms": buffer.total_duration_ms(),
        "buffer_chunk_count": len(buffer.chunks),
    }


@router.post("/transcribe-test")
async def transcribe_audio_test(file: UploadFile = File(...)):
    """Upload a voice note and run the full Phase 2 + Phase 3 pipeline:
    validate/convert/chunk the audio, then transcribe every chunk with
    Whisper-large-v3 on Groq and stitch the result into one transcript.

    Requires GROQ_API_KEY to be set in .env — this endpoint makes a
    real call to Groq's API, unlike /audio/process-test which is fully
    local.
    """
    if not settings.groq_api_key:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not set. Add it to your .env file to use transcription.",
        )

    audio_bytes = await file.read()
    declared_format = (file.filename or "").rsplit(".", 1)[-1] if file.filename else ""
    if not declared_format:
        declared_format = (file.content_type or "").split("/")[-1]

    try:
        _, segments = process_voice_note(audio_bytes, declared_format)
    except AudioValidationError as exc:
        logger.warning("Voice note rejected before transcription: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    buffer = AudioBuffer.from_segments(conversation_id="test-conversation", segments=segments)

    try:
        transcription_service = TranscriptionService()
        transcript = transcription_service.transcribe_buffer(buffer)
    except TranscriptionError as exc:
        logger.error("Transcription failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return transcript.model_dump()


@router.post("/converse-test")
async def converse_test(
    file: UploadFile = File(...),
    conversation_id: str = Form(default="test-conversation"),
):
    """Full pipeline test: voice note in -> transcription -> agent
    response out, as text. This chains Phases 2, 3, and 4 — TTS
    (Phase 7) turns `response_text` into speech, not built yet.

    Note: like /audio/transcribe-test, this has no server-side memory
    across calls (Phase 5). Each call to this endpoint starts a fresh
    conversation.
    """
    if not settings.groq_api_key:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not set. Add it to your .env file.",
        )

    audio_bytes = await file.read()
    declared_format = (file.filename or "").rsplit(".", 1)[-1] if file.filename else ""
    if not declared_format:
        declared_format = (file.content_type or "").split("/")[-1]

    try:
        _, segments = process_voice_note(audio_bytes, declared_format)
    except AudioValidationError as exc:
        logger.warning("Voice note rejected before transcription: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    buffer = AudioBuffer.from_segments(conversation_id=conversation_id, segments=segments)

    try:
        transcription = TranscriptionService().transcribe_buffer(buffer)
    except TranscriptionError as exc:
        logger.error("Transcription failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    try:
        agent_result = await get_agent_service().run(
            conversation_id=conversation_id,
            transcript=transcription.full_text,
            history=messages_from_models([]),
        )
    except Exception as exc:
        logger.error("Agent run failed for conversation=%s: %s", conversation_id, exc)
        raise HTTPException(
            status_code=502,
            detail=(
                f"Agent failed: {exc}. If this is a connection error, make sure Postgres and "
                "ChromaDB are running (docker compose up) and that `python -m app.db.bootstrap` "
                "has been run once."
            ),
        ) from exc

    return {
        "transcript": transcription.full_text,
        "response_text": agent_result.response_text,
        "note": getattr(agent_result, "note", ""),
    }


@router.post("/voice-reply-test")
async def voice_reply_test(
    file: UploadFile = File(...),
    conversation_id: str = Form(default="test-conversation"),
):
    """The full loop, matching the entire architecture diagram: voice
    note in -> transcribe (Phase 3) -> agent response + note (Phase 4/6)
    -> synthesize speech (Phase 7) -> audio out. This is the first
    endpoint where you can actually *hear* Ava reply.

    The response body is the synthesized audio itself (playable
    directly in Swagger UI). Since HTTP headers must be ASCII, the
    transcript/response/note text — which may contain any Unicode
    character — is returned base64-encoded in X-Transcript-B64,
    X-Response-Text-B64, and X-Note-B64 headers. Decode them to read
    what was actually said; most HTTP clients (curl -i, browser
    devtools) show response headers alongside the body already.
    """
    if not settings.groq_api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set. Add it to your .env file.")
    if not settings.elevenlabs_api_key:
        raise HTTPException(status_code=500, detail="ELEVENLABS_API_KEY is not set. Add it to your .env file.")

    audio_bytes = await file.read()
    declared_format = (file.filename or "").rsplit(".", 1)[-1] if file.filename else ""
    if not declared_format:
        declared_format = (file.content_type or "").split("/")[-1]

    try:
        _, segments = process_voice_note(audio_bytes, declared_format)
    except AudioValidationError as exc:
        logger.warning("Voice note rejected before transcription: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    buffer = AudioBuffer.from_segments(conversation_id=conversation_id, segments=segments)

    try:
        transcription = TranscriptionService().transcribe_buffer(buffer)
    except TranscriptionError as exc:
        logger.error("Transcription failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    try:
        agent_result = await get_agent_service().run(
            conversation_id=conversation_id,
            transcript=transcription.full_text,
            history=messages_from_models([]),
        )
    except Exception as exc:
        logger.error("Agent run failed for conversation=%s: %s", conversation_id, exc)
        raise HTTPException(
            status_code=502,
            detail=(
                f"Agent failed: {exc}. If this is a connection error, make sure Postgres and "
                "ChromaDB are running (docker compose up) and that `python -m app.db.bootstrap` "
                "has been run once."
            ),
        ) from exc

    try:
        reply_audio = TTSService().synthesize(agent_result.response_text)
    except TTSError as exc:
        logger.error("TTS synthesis failed for conversation=%s: %s", conversation_id, exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    media_type = AUDIO_MEDIA_TYPES.get(settings.tts_output_format, "audio/mpeg")
    headers = {
        "X-Transcript-B64": base64.b64encode(transcription.full_text.encode("utf-8")).decode("ascii"),
        "X-Response-Text-B64": base64.b64encode(agent_result.response_text.encode("utf-8")).decode("ascii"),
        "X-Note-B64": base64.b64encode(getattr(agent_result, "note", "").encode("utf-8")).decode("ascii"),
    }
    return StreamingResponse(io.BytesIO(reply_audio), media_type=media_type, headers=headers)


@router.post("/voice-reply-stream-test")
async def voice_reply_stream_test(
    file: UploadFile = File(...),
    conversation_id: str = Form(default="test-conversation"),
):
    """Phase 8's true-streaming version of /voice-reply-test: the
    response body is genuinely streamed over HTTP as each sentence is
    synthesized (chunked transfer encoding), rather than the whole
    audio file being assembled server-side before anything is sent.
    The final audio you get is equivalent either way — this is about
    *when* bytes start arriving, not what they contain.

    IMPORTANT LIMITATION, honestly documented rather than worked
    around: HTTP headers must be sent before the response body starts,
    but the full response/note text isn't known until streaming
    finishes. So unlike /voice-reply-test, this endpoint can only put
    the transcript (known upfront, from Phase 3) in a response header
    — X-Transcript-B64. The response text and note are logged
    server-side (visible in your uvicorn console) instead; there's no
    standard, broadly-supported way to attach trailing metadata after
    a streamed HTTP body completes. If you need response_text/note
    programmatically, use /voice-reply-test instead — the tradeoff for
    true streaming here is losing that upfront metadata.

    Uses real Postgres/ChromaDB memory (same as /voice-reply-test) —
    unlike the side-effect-free /agent/latency-comparison-test
    benchmark endpoint.
    """
    if not settings.groq_api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set. Add it to your .env file.")
    if not settings.elevenlabs_api_key:
        raise HTTPException(status_code=500, detail="ELEVENLABS_API_KEY is not set. Add it to your .env file.")

    audio_bytes = await file.read()
    declared_format = (file.filename or "").rsplit(".", 1)[-1] if file.filename else ""
    if not declared_format:
        declared_format = (file.content_type or "").split("/")[-1]

    try:
        _, segments = process_voice_note(audio_bytes, declared_format)
    except AudioValidationError as exc:
        logger.warning("Voice note rejected before transcription: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    buffer = AudioBuffer.from_segments(conversation_id=conversation_id, segments=segments)

    try:
        transcription = TranscriptionService().transcribe_buffer(buffer)
    except TranscriptionError as exc:
        logger.error("Transcription failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    logger.info(
        "voice-reply-stream-test | conversation=%s | transcript=%r",
        conversation_id,
        transcription.full_text,
    )

    llm = get_llm()
    tts_service = TTSService()
    memory_service = get_agent_service()  # only used for its configured stores, not .run()

    media_type = AUDIO_MEDIA_TYPES.get(settings.tts_output_format, "audio/mpeg")
    headers = {
        "X-Transcript-B64": base64.b64encode(transcription.full_text.encode("utf-8")).decode("ascii"),
    }
    return StreamingResponse(
        stream_voice_reply(
            conversation_id,
            transcription.full_text,
            llm,
            tts_service,
            memory_store=memory_service.memory_store,
            vector_store=memory_service.vector_store,
        ),
        media_type=media_type,
        headers=headers,
    )
