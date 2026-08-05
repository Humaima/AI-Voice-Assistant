"""
Voice Generation (Phase 7).

Matches step 4 of the architecture diagram: Ava's text response gets
synthesized into speech via ElevenLabs (eleven_flash_v2_5), producing
the audio that gets buffered (step 5) and sent back over WhatsApp.

Same shape as app/services/transcription.py: a structural Protocol so
tests can inject a fake client instead of hitting the real ElevenLabs
API, plus retry/backoff for transient errors.
"""
from __future__ import annotations

import time
from typing import Any, Protocol

import httpx
from elevenlabs.core.api_error import ApiError

from app.agents.text_processing import sanitize_for_speech
from app.core.config import get_settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Exceptions worth retrying: transient network/server-side issues.
# ApiError with a 4xx status (bad request, auth, invalid voice_id) is
# NOT retried — same reasoning as transcription.py: retrying a
# guaranteed failure just burns time and rate-limit budget.
_RETRYABLE_NETWORK_EXCEPTIONS = (httpx.ConnectError, httpx.TimeoutException, httpx.ReadTimeout)


class TTSError(RuntimeError):
    """Raised when speech synthesis fails — either a non-retryable API
    error, retries were exhausted, or the input text failed validation
    (empty, or over tts_max_chars)."""


# Maps ElevenLabs output_format strings to their HTTP media type, for
# API routes that stream the resulting audio back. Public (not
# module-private) since both app/api/tts.py and app/api/audio.py need it.
AUDIO_MEDIA_TYPES = {
    "mp3_22050_32": "audio/mpeg",
    "mp3_44100_32": "audio/mpeg",
    "mp3_44100_64": "audio/mpeg",
    "mp3_44100_96": "audio/mpeg",
    "mp3_44100_128": "audio/mpeg",
    "mp3_44100_192": "audio/mpeg",
    "pcm_16000": "audio/pcm",
    "pcm_22050": "audio/pcm",
    "pcm_24000": "audio/pcm",
    "pcm_44100": "audio/pcm",
    "ulaw_8000": "audio/basic",
}


class _TextToSpeechAPI(Protocol):
    def convert(self, voice_id: str, *, text: str, model_id: Any, output_format: Any) -> Any: ...


class TTSClient(Protocol):
    """Structural interface for anything shaped like `ElevenLabs` —
    i.e. exposes `.text_to_speech.convert(...)`. Lets tests inject a
    lightweight fake client without subclassing the real SDK class."""

    @property
    def text_to_speech(self) -> _TextToSpeechAPI: ...


def build_elevenlabs_client() -> TTSClient:
    from elevenlabs.client import ElevenLabs

    return ElevenLabs(api_key=settings.elevenlabs_api_key)


class TTSService:
    """Thin wrapper around the ElevenLabs API with retry/backoff and
    input validation. The client is injectable so tests can supply a
    fake one instead of hitting the real API."""

    def __init__(self, client: TTSClient | None = None):
        self._client = client or build_elevenlabs_client()
        self.max_retries = settings.tts_max_retries
        self.backoff_base = settings.tts_backoff_base_seconds

    def synthesize(self, text: str) -> bytes:
        """Convert `text` to speech, returning raw audio bytes in
        settings.tts_output_format (mp3 by default). Runs the text
        through sanitize_for_speech first as defense-in-depth — Phase
        6 already sanitizes agent responses, but this makes TTSService
        safe to call directly with arbitrary text too."""
        clean_text = sanitize_for_speech(text)

        if not clean_text:
            raise TTSError("Cannot synthesize empty text")

        if len(clean_text) > settings.tts_max_chars:
            raise TTSError(
                f"Text too long for synthesis: {len(clean_text)} chars "
                f"(max {settings.tts_max_chars}). This usually means the "
                "agent's response was unexpectedly long — check the prompt."
            )

        return self._synthesize_with_retry(clean_text)

    def _synthesize_with_retry(self, text: str) -> bytes:
        last_exc: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                chunks = self._client.text_to_speech.convert(
                    voice_id=settings.elevenlabs_voice_id,
                    text=text,
                    model_id=settings.elevenlabs_model,
                    output_format=settings.tts_output_format,
                )
                audio_bytes = b"".join(chunks)
                if not audio_bytes:
                    raise TTSError("ElevenLabs returned empty audio")
                return audio_bytes

            except _RETRYABLE_NETWORK_EXCEPTIONS as exc:
                last_exc = exc
                if attempt == self.max_retries:
                    break
                wait_s = self.backoff_base**attempt
                logger.warning(
                    "TTS synthesis attempt %d/%d failed (%s), retrying in %.1fs",
                    attempt,
                    self.max_retries,
                    type(exc).__name__,
                    wait_s,
                )
                time.sleep(wait_s)

            except ApiError as exc:
                status = exc.status_code or 0
                if status >= 500 or status == 429:
                    # Server error or rate limit — worth retrying.
                    last_exc = exc
                    if attempt == self.max_retries:
                        break
                    wait_s = self.backoff_base**attempt
                    logger.warning(
                        "TTS synthesis attempt %d/%d failed (status=%s), retrying in %.1fs",
                        attempt,
                        self.max_retries,
                        status,
                        wait_s,
                    )
                    time.sleep(wait_s)
                else:
                    # 4xx (bad voice_id, auth failure, invalid params) — not retryable.
                    logger.error("Non-retryable ElevenLabs API error (status=%s): %s", status, exc)
                    raise TTSError(f"ElevenLabs rejected the synthesis request: {exc}") from exc

            except TTSError:
                # Already the right type (e.g. "empty audio" above) — let it
                # propagate as-is rather than getting caught by the
                # catch-all below and double-wrapped.
                raise

            except Exception as exc:
                # Defensive backstop: any exception type we didn't
                # explicitly anticipate (a client library raising
                # something outside its documented error hierarchy,
                # for instance) should still surface as TTSError, not
                # leak an arbitrary exception type to callers. Not
                # retried, since we don't know its nature well enough
                # to assume retrying would help.
                logger.error("Unexpected error during TTS synthesis: %s", exc)
                raise TTSError(f"Unexpected error during speech synthesis: {exc}") from exc

        raise TTSError(f"Speech synthesis failed after {self.max_retries} attempts: {last_exc}") from last_exc
