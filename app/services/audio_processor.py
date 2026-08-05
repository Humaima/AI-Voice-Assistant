"""
Voice Input Pipeline (Phase 2).

Matches step 1-2 of the architecture diagram: a raw WhatsApp voice note
comes in, gets validated, converted to 16kHz mono WAV, normalized,
stripped of silence, and split into chunks that get handed to an
AudioBuffer — which Phase 3 (Whisper transcription) will read from.

All functions operate on pydub.AudioSegment so we get ffmpeg-backed
decoding of whatever codec WhatsApp/Twilio hands us (OGG/Opus, AMR,
etc.) for free, without hand-rolling codec detection.
"""
from __future__ import annotations

import io

from pydub import AudioSegment
from pydub.effects import normalize as pydub_normalize
from pydub.silence import detect_nonsilent

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.models.audio import AudioChunk, AudioMetadata, ProcessedAudioResult, SupportedAudioFormat

logger = get_logger(__name__)
settings = get_settings()

# Magic-byte / extension hints pydub can decode. We don't trust the
# client-supplied content-type alone since it's easy to spoof or mislabel.
_SUPPORTED_FORMATS = {f.value for f in SupportedAudioFormat}


class AudioValidationError(ValueError):
    """Raised when incoming audio fails validation — bad format, too
    long, too large, or corrupt/empty. Callers (the webhook handler in
    Phase 9) should catch this and reply to the user with a friendly
    'couldn't process that voice note' message rather than a 500."""


def validate_format(audio_bytes: bytes, declared_format: str) -> str:
    """Confirm the format is one we support and the bytes aren't empty.
    Returns the normalized format string pydub expects for `from_file`."""
    fmt = declared_format.lower().lstrip(".")
    if fmt == "opus":
        fmt = "ogg"  # opus voice notes are carried in an ogg container

    if fmt not in _SUPPORTED_FORMATS:
        raise AudioValidationError(
            f"Unsupported audio format '{declared_format}'. Supported: {sorted(_SUPPORTED_FORMATS)}"
        )

    if not audio_bytes:
        raise AudioValidationError("Received empty audio payload")

    if len(audio_bytes) > settings.audio_max_size_bytes:
        raise AudioValidationError(
            f"Audio file too large: {len(audio_bytes)} bytes "
            f"(max {settings.audio_max_size_bytes})"
        )

    return fmt


def load_audio(audio_bytes: bytes, fmt: str) -> AudioSegment:
    """Decode raw bytes into a pydub AudioSegment. This is the one
    place ffmpeg gets invoked, so decode failures surface here with a
    clear error rather than deep inside a later processing step."""
    try:
        return AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
    except Exception as exc:  # pydub/ffmpeg raise a mix of exception types
        # ffmpeg's own exception message includes its full build config
        # and stderr dump, which is noise for API consumers and would
        # leak internals if passed straight through. Log the full detail
        # server-side, return a clean message to the caller.
        logger.warning("ffmpeg failed to decode audio as '%s': %s", fmt, exc)
        raise AudioValidationError(
            f"Could not decode audio as '{fmt}' — the file may be corrupted or not valid audio."
        ) from exc


def convert_to_target_wav(segment: AudioSegment) -> AudioSegment:
    """Resample to 16kHz mono — what Whisper (Phase 3) expects. Doing
    this once up front means every downstream step works with a known,
    consistent format."""
    return segment.set_frame_rate(settings.audio_target_sample_rate).set_channels(
        settings.audio_target_channels
    )


def normalize_audio(segment: AudioSegment) -> AudioSegment:
    """Bring quiet/loud voice notes to a consistent volume so Whisper
    doesn't struggle with faint recordings."""
    return pydub_normalize(segment)


def remove_silence(segment: AudioSegment) -> tuple[AudioSegment, int]:
    """Trim leading/trailing silence and collapse long internal gaps.
    Returns (trimmed_segment, ms_of_silence_removed) — the latter is
    useful telemetry (Phase 10 dashboards) for spotting bad recordings.
    """
    nonsilent_ranges = detect_nonsilent(
        segment,
        min_silence_len=settings.audio_min_silence_len_ms,
        silence_thresh=settings.audio_silence_thresh_db,
    )

    if not nonsilent_ranges:
        # Entirely silent clip — nothing to transcribe. Let the caller
        # decide how to handle this (Phase 9 might reply "I didn't
        # catch that, could you send it again?").
        raise AudioValidationError("Audio contains no detectable speech (all silence)")

    original_duration = len(segment)

    # Stitch together only the non-silent spans, with a small padding
    # so words aren't clipped right at the boundary.
    padding_ms = 100
    stitched = AudioSegment.empty()
    for start, end in nonsilent_ranges:
        padded_start = max(0, start - padding_ms)
        padded_end = min(original_duration, end + padding_ms)
        stitched += segment[padded_start:padded_end]

    silence_removed = original_duration - len(stitched)
    return stitched, max(silence_removed, 0)


def chunk_audio(segment: AudioSegment) -> list[AudioSegment]:
    """Split into overlapping chunks for streaming to Whisper (Phase 3)
    and for lower end-to-end latency, matching the diagram's dashed
    'Chunking Voice Note' segments. Short clips return a single chunk."""
    chunk_len = settings.audio_chunk_length_ms
    overlap = settings.audio_chunk_overlap_ms
    total_len = len(segment)

    if total_len <= chunk_len:
        return [segment]

    chunks = []
    start = 0
    step = chunk_len - overlap
    while start < total_len:
        end = min(start + chunk_len, total_len)
        chunks.append(segment[start:end])
        if end == total_len:
            break
        start += step

    return chunks


def process_voice_note(audio_bytes: bytes, declared_format: str) -> tuple[ProcessedAudioResult, list[AudioSegment]]:
    """Full Phase 2 pipeline: validate -> load -> convert -> normalize
    -> trim silence -> chunk. Returns metadata (for logging/API
    responses) alongside the actual AudioSegment chunks (for Phase 3
    to send to Whisper) — kept separate since AudioSegment isn't JSON
    serializable and shouldn't leak into API responses."""

    fmt = validate_format(audio_bytes, declared_format)
    original_size = len(audio_bytes)

    raw_segment = load_audio(audio_bytes, fmt)

    if len(raw_segment) > settings.audio_max_duration_ms:
        raise AudioValidationError(
            f"Audio too long: {len(raw_segment)}ms (max {settings.audio_max_duration_ms}ms)"
        )

    original_metadata = AudioMetadata(
        format=SupportedAudioFormat.OGG if fmt == "ogg" else SupportedAudioFormat(fmt),
        duration_ms=len(raw_segment),
        sample_rate=raw_segment.frame_rate,
        channels=raw_segment.channels,
        size_bytes=original_size,
    )

    segment = convert_to_target_wav(raw_segment)
    segment = normalize_audio(segment)
    segment, silence_removed_ms = remove_silence(segment)
    chunks = chunk_audio(segment)

    chunk_start = 0
    chunks_metadata = []
    for i, chunk in enumerate(chunks):
        chunk_end = chunk_start + len(chunk)
        chunks_metadata.append(
            AudioChunk(
                index=i,
                start_ms=chunk_start,
                end_ms=chunk_end,
                duration_ms=len(chunk),
                sample_rate=settings.audio_target_sample_rate,
                channels=settings.audio_target_channels,
            )
        )
        chunk_start += len(chunk) - settings.audio_chunk_overlap_ms

    result = ProcessedAudioResult(
        original_metadata=original_metadata,
        chunk_count=len(chunks),
        chunks_metadata=chunks_metadata,
        total_processed_duration_ms=len(segment),
        silence_removed_ms=silence_removed_ms,
    )

    logger.info(
        "Processed voice note | format=%s | original_ms=%d | processed_ms=%d | chunks=%d | silence_removed_ms=%d",
        fmt,
        original_metadata.duration_ms,
        result.total_processed_duration_ms,
        result.chunk_count,
        silence_removed_ms,
    )

    return result, chunks
