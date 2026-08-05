"""
Speech Recognition (Phase 3).

Matches step 3 of the architecture diagram: each buffered audio chunk
(Phase 2's AudioBuffer) gets sent to Whisper-large-v3 on Groq, and the
per-chunk transcripts are stitched back into one continuous transcript
that Phase 4 (LangGraph agent) can treat as if the audio were never
split at all.

Why stitching needs care: chunks were built with a 500ms overlap
(app/services/audio_processor.py's chunk_audio) so words aren't cut off
at a chunk boundary. That overlap means the tail of chunk N's
transcript and the head of chunk N+1's transcript often contain the
same few words — naive concatenation would duplicate them. We detect
that overlap with a fuzzy text match and trim it.
"""
from __future__ import annotations

import difflib
import time
from typing import Any, Protocol

import groq

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.models.transcription import ChunkTranscript, TranscriptionResult
from app.services.audio_buffer import AudioBuffer, BufferedChunk

logger = get_logger(__name__)
settings = get_settings()

# Exceptions worth retrying: transient network/server-side issues.
# Auth/bad-request errors (401/400/etc) are not retried — retrying a
# malformed request just burns time and rate-limit budget for the same
# guaranteed failure.
_RETRYABLE_EXCEPTIONS = (
    groq.RateLimitError,
    groq.APIConnectionError,
    groq.APITimeoutError,
    groq.InternalServerError,
)


class _TranscriptionsAPI(Protocol):
    def create(
        self,
        *,
        file: Any,
        model: Any,
        response_format: Any,
        temperature: Any,
    ) -> Any: ...


class _AudioAPI(Protocol):
    @property
    def transcriptions(self) -> _TranscriptionsAPI: ...


class TranscriptionClient(Protocol):
    """Structural interface for anything shaped like `groq.Groq` —
    i.e. exposes `.audio.transcriptions.create(...)`. This is what lets
    tests inject a lightweight fake client (see FakeGroqClient in
    tests/test_transcription.py) without subclassing the real SDK
    class, while still being fully type-checked.

    `audio`/`transcriptions` are declared as read-only properties
    rather than plain attributes: Pyright treats plain Protocol
    attributes as invariant (the implementer's type must match
    exactly), which would reject any fake client whose nested `audio`
    object isn't the literal same class groq.Groq uses. Properties are
    covariant, so any object with a structurally compatible `.audio`
    satisfies this — which is the actual guarantee we need."""

    @property
    def audio(self) -> _AudioAPI: ...


class TranscriptionError(RuntimeError):
    """Raised when a chunk can't be transcribed — either a
    non-retryable API error, or retries were exhausted. Callers (the
    debug endpoint now, the LangGraph agent node in Phase 4) should
    catch this and decide how to degrade gracefully."""


class TranscriptionService:
    """Thin wrapper around the Groq Whisper API with retry/backoff and
    multi-chunk stitching. The client is injectable so tests can supply
    a fake one instead of hitting the real API."""

    def __init__(self, client: TranscriptionClient | None = None):
        self._client: TranscriptionClient = client or groq.Groq(api_key=settings.groq_api_key)
        self.max_retries = settings.transcription_max_retries
        self.backoff_base = settings.transcription_backoff_base_seconds

    def _transcribe_bytes(self, wav_bytes: bytes, filename: str) -> dict:
        """Call the Groq API with retry/backoff for transient errors.
        Returns the raw response as a dict."""
        last_exc: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.audio.transcriptions.create(
                    file=(filename, wav_bytes),
                    model=settings.groq_whisper_model,
                    response_format="verbose_json",
                    temperature=settings.transcription_temperature,
                )
                return response.model_dump() if hasattr(response, "model_dump") else dict(response)

            except _RETRYABLE_EXCEPTIONS as exc:
                last_exc = exc
                if attempt == self.max_retries:
                    break
                wait_s = self.backoff_base**attempt
                logger.warning(
                    "Groq transcription attempt %d/%d failed (%s), retrying in %.1fs",
                    attempt,
                    self.max_retries,
                    type(exc).__name__,
                    wait_s,
                )
                time.sleep(wait_s)

            except groq.APIStatusError as exc:
                # 4xx like bad request or auth failure — retrying won't help.
                logger.error("Non-retryable Groq API error: %s", exc)
                raise TranscriptionError(f"Groq rejected the transcription request: {exc}") from exc

        raise TranscriptionError(
            f"Transcription failed after {self.max_retries} attempts: {last_exc}"
        ) from last_exc

    def transcribe_chunk(self, chunk: BufferedChunk) -> ChunkTranscript:
        """Transcribe a single buffered chunk."""
        wav_bytes = chunk.to_wav_bytes()
        raw = self._transcribe_bytes(wav_bytes, f"chunk_{chunk.index}.wav")

        text = (raw.get("text") or "").strip()
        segments = raw.get("segments") or []
        avg_logprob = None
        if segments:
            # Average across segments gives a rough per-chunk confidence signal.
            logprobs = [s.get("avg_logprob") for s in segments if s.get("avg_logprob") is not None]
            avg_logprob = sum(logprobs) / len(logprobs) if logprobs else None

        return ChunkTranscript(
            chunk_index=chunk.index,
            text=text,
            language=raw.get("language"),
            duration_s=raw.get("duration"),
            avg_logprob=avg_logprob,
        )

    def transcribe_buffer(self, buffer: AudioBuffer) -> TranscriptionResult:
        """Transcribe every chunk in a buffer, in order, and stitch the
        results into one continuous transcript."""
        start = time.monotonic()
        chunk_transcripts: list[ChunkTranscript] = []

        for chunk in buffer.chunks:
            transcript = self.transcribe_chunk(chunk)
            chunk_transcripts.append(transcript)
            buffer.mark_consumed(chunk.index)

            if not transcript.text:
                logger.warning(
                    "Chunk %d of conversation %s produced empty transcript",
                    chunk.index,
                    buffer.conversation_id,
                )

        full_text = stitch_transcripts(
            [c.text for c in chunk_transcripts],
            overlap_words=settings.transcription_stitch_overlap_words,
        )

        language = next((c.language for c in chunk_transcripts if c.language), None)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        logger.info(
            "Transcribed conversation=%s | chunks=%d | chars=%d | elapsed_ms=%d",
            buffer.conversation_id,
            len(chunk_transcripts),
            len(full_text),
            elapsed_ms,
        )

        return TranscriptionResult(
            full_text=full_text,
            language=language,
            chunks=chunk_transcripts,
            chunk_count=len(chunk_transcripts),
            processing_time_ms=elapsed_ms,
        )


def stitch_transcripts(texts: list[str], overlap_words: int = 6) -> str:
    """Join per-chunk transcripts into one string, trimming duplicated
    words that show up at chunk boundaries because chunks overlap by
    design (see chunk_audio's audio_chunk_overlap_ms)."""
    non_empty = [t.strip() for t in texts if t.strip()]
    if not non_empty:
        return ""

    stitched = non_empty[0]
    for next_text in non_empty[1:]:
        stitched = _merge_overlap(stitched, next_text, max_overlap_words=overlap_words)

    return stitched.strip()


def _merge_overlap(a: str, b: str, max_overlap_words: int) -> str:
    """Find the longest word-boundary overlap between the tail of `a`
    and the head of `b` (up to max_overlap_words), and merge without
    duplicating it. Falls back to plain concatenation if no meaningful
    overlap is found."""
    a_words = a.split()
    b_words = b.split()
    max_check = min(max_overlap_words, len(a_words), len(b_words))

    best_k = 0
    for k in range(max_check, 0, -1):
        a_tail = " ".join(a_words[-k:]).lower()
        b_head = " ".join(b_words[:k]).lower()
        similarity = difflib.SequenceMatcher(None, a_tail, b_head).ratio()
        if similarity > 0.8:
            best_k = k
            break

    if best_k:
        return a + " " + " ".join(b_words[best_k:])
    return a + " " + b
