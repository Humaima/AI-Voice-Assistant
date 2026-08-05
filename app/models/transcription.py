"""
Transcription domain models (Phase 3).

These are what Phase 4 (LangGraph agent) will consume as input — the
whole point of assembling `TranscriptionResult.full_text` is that the
agent shouldn't need to know the audio was ever chunked at all.
"""
from pydantic import BaseModel


class ChunkTranscript(BaseModel):
    """Raw transcription result for a single audio chunk, before
    stitching. Kept around for debugging/observability — if the final
    stitched transcript looks wrong, this is where you'd look to find
    which chunk introduced the problem."""

    chunk_index: int
    text: str
    language: str | None = None
    duration_s: float | None = None
    avg_logprob: float | None = None  # Whisper's confidence signal, when available


class TranscriptionResult(BaseModel):
    """What Phase 3 hands off to Phase 4."""

    full_text: str
    language: str | None = None
    chunks: list[ChunkTranscript]
    chunk_count: int
    processing_time_ms: int
