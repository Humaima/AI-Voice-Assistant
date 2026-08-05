"""
Audio domain models.

These carry metadata through the pipeline (validation -> conversion ->
normalization -> silence removal -> chunking -> buffer) so every stage
knows what it's holding without re-inspecting raw bytes.
"""
from enum import Enum

from pydantic import BaseModel, Field


class SupportedAudioFormat(str, Enum):
    """WhatsApp voice notes arrive as OGG/Opus. We also accept a few
    others so the pipeline isn't hard-locked to one client's encoding
    (e.g. Twilio can sometimes deliver AMR on older Android senders)."""

    OGG = "ogg"
    OPUS = "opus"
    MP3 = "mp3"
    WAV = "wav"
    M4A = "m4a"
    AMR = "amr"


class AudioMetadata(BaseModel):
    format: SupportedAudioFormat
    duration_ms: int
    sample_rate: int
    channels: int
    size_bytes: int


class AudioChunk(BaseModel):
    """A single chunk of a longer voice note, ready to be sent to
    Whisper (Phase 3). start_ms/end_ms are offsets into the original
    audio, kept for logging/debugging and potential re-stitching."""

    index: int
    start_ms: int
    end_ms: int
    duration_ms: int
    sample_rate: int = 16000
    channels: int = 1

    model_config = {"arbitrary_types_allowed": True}


class ProcessedAudioResult(BaseModel):
    """What the pipeline hands off to Phase 3 (Speech Recognition)."""

    original_metadata: AudioMetadata
    chunk_count: int
    chunks_metadata: list[AudioChunk]
    total_processed_duration_ms: int
    silence_removed_ms: int = Field(
        description="How much silence was trimmed, useful for monitoring audio quality"
    )
