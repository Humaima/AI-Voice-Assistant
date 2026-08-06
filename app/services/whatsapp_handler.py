"""
WhatsApp message handling — the actual product this whole project has
been building toward: a real inbound WhatsApp voice note, run through
the full pipeline (Phases 2-3-4-5-6-7), replied to with a real
outbound WhatsApp voice note.

Phase 10 adds `process_and_send_reply_async`: the pipeline (audio
processing + transcription + LLM + TTS) can take 10+ seconds end to
end — too close to Twilio's webhook timeout to safely run inside a
synchronous webhook response (this was directly observed in testing:
processing times approaching 13 seconds). The production path now
acknowledges Twilio immediately with an empty TwiML response and
processes the message in a FastAPI background task, sending the real
reply afterward via Twilio's REST API — see app/api/webhook.py for how
the two are wired together.

`handle_incoming_whatsapp_message` (the original, synchronous,
TwiML-reply path) is kept for the instant "no voice note attached"
case, where there's no meaningful processing time and no timeout risk.

All entrypoints are plain functions accepting injected services
(rather than reaching for globals/singletons internally) so they're
testable without a real Twilio account, real audio, or real
Groq/ElevenLabs credentials — app/api/webhook.py wires up the real
services and calls these; tests call them directly with fakes.

Why the non-streaming pipeline, not Phase 8's streaming one: Twilio
sends outbound WhatsApp media as ONE fetchable URL, not a live stream
— there's no way to progressively deliver audio to a WhatsApp user, so
there's no latency benefit to streaming here. The complete audio file
has to exist somewhere before Twilio can fetch any of it.
"""
from __future__ import annotations

import time
from typing import cast

from twilio.twiml.messaging_response import Message, MessagingResponse

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.core.metrics import (
    pipeline_stage_errors_total,
    whatsapp_messages_total,
    whatsapp_reply_duration_seconds,
)
from app.services.agent_service import AgentService
from app.services.audio_buffer import AudioBuffer
from app.services.audio_processor import AudioValidationError, process_voice_note
from app.services.media_store import build_media_url, save_media
from app.services.transcription import TranscriptionError, TranscriptionService
from app.services.tts import AUDIO_MEDIA_TYPES, TTSError, TTSService
from app.services.twilio_client import MediaDownloadError, MediaDownloader, MessageSender

logger = get_logger(__name__)
settings = get_settings()

_NO_VOICE_NOTE_REPLY = (
    "Hey! I'm Ava — I reply to voice notes. Send me one and I'll get back to you with a voice reply."
)
_PROCESSING_FAILED_REPLY = "Sorry, I had trouble with that voice note — could you try sending it again?"


def _guess_format_from_content_type(content_type: str) -> str:
    """WhatsApp voice notes normally arrive as audio/ogg (opus codec).
    Falls back to the subtype for anything else Twilio might forward."""
    return content_type.split("/")[-1].split(";")[0].strip()


def _safe_caption(text: str, max_chars: int = 60) -> str:
    """Fallback caption when note generation produced nothing (Phase 6's
    graph already has its own fallback for when the note LLM call
    raises — this covers the separate case where it succeeds but
    sanitizes down to an empty string). Truncates on a word boundary
    with an ellipsis rather than a raw character slice, which can cut
    off mid-word with no indication anything was trimmed."""
    text = text.strip()
    if not text:
        return "Here's my reply"
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(" ", 1)[0]
    return f"{truncated}..." if truncated else f"{text[:max_chars]}..."


async def _generate_voice_reply(
    from_number: str,
    media_url: str,
    media_content_type: str,
    *,
    media_downloader: MediaDownloader,
    agent_service: AgentService,
    tts_service: TTSService,
) -> tuple[str, str | None]:
    """The actual pipeline: download -> validate/process audio ->
    transcribe -> agent (memory + response + note) -> synthesize
    speech -> host the result. Returns (message_text, media_public_url)
    — media_public_url is None if anything failed along the way, in
    which case message_text is a user-facing explanation rather than a
    real reply.

    Deliberately returns data rather than building a reply itself —
    both callers (the synchronous TwiML builder and the async REST
    sender) need this same result delivered in different ways, so
    delivery is their job, not this function's.
    """
    try:
        audio_bytes = await media_downloader.download(media_url)
    except MediaDownloadError as exc:
        logger.error("whatsapp | from=%s | media download failed: %s", from_number, exc)
        pipeline_stage_errors_total.labels(stage="media_download").inc()
        return _PROCESSING_FAILED_REPLY, None

    declared_format = _guess_format_from_content_type(media_content_type)

    try:
        _, segments = process_voice_note(audio_bytes, declared_format)
    except AudioValidationError as exc:
        logger.warning("whatsapp | from=%s | audio validation failed: %s", from_number, exc)
        pipeline_stage_errors_total.labels(stage="audio_validation").inc()
        return (
            "I couldn't quite process that voice note — it might be too long, too quiet, "
            "or in a format I don't support. Mind trying again?",
            None,
        )

    buffer = AudioBuffer.from_segments(conversation_id=from_number, segments=segments)

    try:
        transcription = TranscriptionService().transcribe_buffer(buffer)
    except TranscriptionError as exc:
        logger.error("whatsapp | from=%s | transcription failed: %s", from_number, exc)
        pipeline_stage_errors_total.labels(stage="transcription").inc()
        return _PROCESSING_FAILED_REPLY, None

    try:
        agent_result = await agent_service.run(conversation_id=from_number, transcript=transcription.full_text)
    except Exception as exc:
        logger.error("whatsapp | from=%s | agent run failed: %s", from_number, exc)
        pipeline_stage_errors_total.labels(stage="agent").inc()
        return _PROCESSING_FAILED_REPLY, None

    try:
        reply_audio = tts_service.synthesize(agent_result.response_text)
    except TTSError as exc:
        logger.error("whatsapp | from=%s | TTS synthesis failed: %s", from_number, exc)
        pipeline_stage_errors_total.labels(stage="tts").inc()
        # The text reply IS ready even though speech synthesis failed —
        # send it as plain text rather than failing the whole turn.
        return agent_result.response_text or _PROCESSING_FAILED_REPLY, None

    media_type_for_storage = AUDIO_MEDIA_TYPES.get(settings.tts_output_format, "audio/mpeg")
    filename = save_media(reply_audio, media_type_for_storage)

    try:
        media_public_url = build_media_url(filename)
    except ValueError as exc:
        logger.error("whatsapp | from=%s | %s", from_number, exc)
        return agent_result.response_text, None

    logger.info(
        "whatsapp | from=%s | transcript=%r | response_chars=%d | note=%r | media_url=%s",
        from_number,
        transcription.full_text,
        len(agent_result.response_text),
        agent_result.note,
        media_public_url,
    )

    caption = agent_result.note or _safe_caption(agent_result.response_text)
    return caption, media_public_url


async def handle_incoming_whatsapp_message(
    from_number: str,
    num_media: int,
    media_url: str | None,
    media_content_type: str | None,
    *,
    media_downloader: MediaDownloader,
    agent_service: AgentService,
    tts_service: TTSService,
) -> str:
    """Synchronous TwiML-reply path. Fine for the instant "no voice
    note attached" case (no meaningful processing time, no timeout
    risk) — NOT recommended for real voice-note traffic in production,
    where the full pipeline can take 10+ seconds and risks exceeding
    Twilio's webhook timeout. Production voice-note handling should use
    process_and_send_reply_async instead (see app/api/webhook.py).

    `from_number` (Twilio's `From` field, e.g. "whatsapp:+15551234567")
    doubles as the conversation_id — a stable, natural per-user
    identity for Phase 5's memory.
    """
    response = MessagingResponse()

    if num_media < 1 or not media_url or not media_content_type:
        logger.info("whatsapp | from=%s | no media attached, sending instructions", from_number)
        whatsapp_messages_total.labels(outcome="no_media_reply").inc()
        response.message(_NO_VOICE_NOTE_REPLY)
        return str(response)

    if not media_content_type.startswith("audio/"):
        logger.info(
            "whatsapp | from=%s | non-audio media (%s), sending instructions",
            from_number,
            media_content_type,
        )
        whatsapp_messages_total.labels(outcome="no_media_reply").inc()
        response.message(_NO_VOICE_NOTE_REPLY)
        return str(response)

    message_text, media_public_url = await _generate_voice_reply(
        from_number,
        media_url,
        media_content_type,
        media_downloader=media_downloader,
        agent_service=agent_service,
        tts_service=tts_service,
    )

    message = cast(Message, response.message(message_text))
    if media_public_url:
        message.media(media_public_url)
    return str(response)


async def process_and_send_reply_async(
    from_number: str,
    media_url: str,
    media_content_type: str,
    *,
    media_downloader: MediaDownloader,
    agent_service: AgentService,
    tts_service: TTSService,
    message_sender: MessageSender,
) -> None:
    """The production path (Phase 10): runs the full pipeline and sends
    the result via Twilio's REST API, rather than a synchronous TwiML
    webhook reply. Meant to run as a FastAPI BackgroundTask, scheduled
    AFTER the webhook has already returned an immediate ack to Twilio —
    see app/api/webhook.py. This is what actually removes the timeout
    risk: Twilio's webhook response time is now near-instant regardless
    of how long transcription/the LLM/TTS take.

    Returns nothing — there's no HTTP response left to shape by this
    point, since the webhook already responded. Any failure here means
    the user gets no reply at all, which is why it's logged loudly
    rather than silently swallowed.
    """
    start = time.monotonic()

    message_text, media_public_url = await _generate_voice_reply(
        from_number,
        media_url,
        media_content_type,
        media_downloader=media_downloader,
        agent_service=agent_service,
        tts_service=tts_service,
    )

    whatsapp_reply_duration_seconds.observe(time.monotonic() - start)
    whatsapp_messages_total.labels(
        outcome="voice_note_processed" if media_public_url else "pipeline_error"
    ).inc()

    try:
        await message_sender.send(to=from_number, body=message_text, media_url=media_public_url)
    except Exception as exc:
        logger.error("whatsapp | from=%s | failed to send async reply: %s", from_number, exc)
        pipeline_stage_errors_total.labels(stage="media_send").inc()
