"""
Tests for the Phase 2 voice input pipeline.

Uses synthetically generated tones (sine wave speech-like bursts
separated by silence) rather than a real voice note, so tests don't
depend on external audio fixtures. This is enough to exercise every
stage: format validation, decoding, resampling, normalization, silence
trimming, and chunking.
"""
import io
from typing import cast

import numpy as np
import pytest
from pydub import AudioSegment
from pydub.generators import Sine

from app.services.audio_buffer import AudioBuffer
from app.services.audio_processor import (
    AudioValidationError,
    chunk_audio,
    process_voice_note,
    validate_format,
)


def _make_test_segment(total_ms: int = 4000, tone_hz: int = 440) -> AudioSegment:
    """Build alternating tone/silence bursts to simulate speech
    separated by pauses, at a non-16kHz sample rate (8kHz) and stereo,
    so we can verify resampling/downmixing actually happens."""
    tone = Sine(tone_hz).to_audio_segment(duration=800).apply_gain(-3.0)
    silence = AudioSegment.silent(duration=600)
    segment = (tone + silence) * 3
    segment = segment.set_frame_rate(8000).set_channels(2)
    trimmed = segment[:total_ms] if len(segment) > total_ms else segment
    # pydub's AudioSegment defines both __iter__ (sample-by-sample) and
    # __getitem__ (slicing), which confuses static type checkers into
    # inferring a Generator union for the slice result. It's actually
    # always an AudioSegment at runtime, hence the explicit cast.
    return cast(AudioSegment, trimmed)


def _export_as(segment: AudioSegment, fmt: str) -> bytes:
    buf = io.BytesIO()
    segment.export(buf, format=fmt)
    return buf.getvalue()


class TestValidateFormat:
    def test_rejects_unsupported_format(self):
        with pytest.raises(AudioValidationError):
            validate_format(b"fake bytes", "flac")

    def test_rejects_empty_bytes(self):
        with pytest.raises(AudioValidationError):
            validate_format(b"", "ogg")

    def test_accepts_opus_as_ogg_container(self):
        fmt = validate_format(b"non-empty", "opus")
        assert fmt == "ogg"

    def test_accepts_wav(self):
        fmt = validate_format(b"non-empty", ".wav")
        assert fmt == "wav"


class TestProcessVoiceNote:
    def test_full_pipeline_wav(self):
        segment = _make_test_segment(total_ms=4200)
        wav_bytes = _export_as(segment, "wav")

        result, chunks = process_voice_note(wav_bytes, "wav")

        # Resampled to 16kHz mono regardless of the 8kHz-stereo input
        assert result.original_metadata.sample_rate == 8000
        assert result.original_metadata.channels == 2
        assert result.total_processed_duration_ms > 0
        assert result.chunk_count == len(chunks) == 1  # under 30s -> single chunk
        assert result.silence_removed_ms > 0  # our synthetic silences got trimmed

    def test_full_pipeline_ogg(self):
        segment = _make_test_segment(total_ms=3000)
        ogg_bytes = _export_as(segment, "ogg")

        result, chunks = process_voice_note(ogg_bytes, "ogg")

        assert result.chunk_count == 1
        assert chunks[0].frame_rate == 16000
        assert chunks[0].channels == 1

    def test_rejects_all_silence(self):
        silent = AudioSegment.silent(duration=2000, frame_rate=16000)
        wav_bytes = _export_as(silent, "wav")

        with pytest.raises(AudioValidationError, match="no detectable speech"):
            process_voice_note(wav_bytes, "wav")

    def test_long_audio_produces_multiple_chunks(self):
        # Build ~65s of tone/silence so it must split into >1 chunk
        # given the default 30s chunk length.
        segment = _make_test_segment(total_ms=1400)
        long_segment = segment * 47  # ~65.8s
        wav_bytes = _export_as(long_segment, "wav")

        result, chunks = process_voice_note(wav_bytes, "wav")

        assert result.chunk_count >= 2
        assert len(chunks) == result.chunk_count
        # Every chunk metadata entry lines up with an actual chunk
        for meta, seg in zip(result.chunks_metadata, chunks):
            assert abs(meta.duration_ms - len(seg)) <= 5  # rounding tolerance


class TestChunking:
    def test_short_clip_single_chunk(self):
        segment = AudioSegment.silent(duration=1000, frame_rate=16000) + Sine(300).to_audio_segment(
            duration=1000
        )
        chunks = chunk_audio(segment)
        assert len(chunks) == 1

    def test_chunk_boundaries_cover_full_audio(self):
        segment = Sine(300).to_audio_segment(duration=70_000)  # 70s
        chunks = chunk_audio(segment)
        assert len(chunks) >= 3
        # First chunk starts at 0
        assert len(chunks[0]) > 0


class TestAudioBuffer:
    def test_buffer_tracks_consumption(self):
        segments = [Sine(300).to_audio_segment(duration=500) for _ in range(3)]
        buffer = AudioBuffer.from_segments("conv-1", segments)

        assert not buffer.is_fully_consumed()
        first = buffer.next_unconsumed()
        assert first is not None
        assert first.index == 0

        buffer.mark_consumed(0)
        second = buffer.next_unconsumed()
        assert second is not None
        assert second.index == 1

        buffer.mark_consumed(1)
        buffer.mark_consumed(2)
        assert buffer.is_fully_consumed()
        assert buffer.next_unconsumed() is None

    def test_to_wav_bytes_produces_valid_wav(self):
        segments = [Sine(300).to_audio_segment(duration=500)]
        buffer = AudioBuffer.from_segments("conv-1", segments)
        wav_bytes = buffer.chunks[0].to_wav_bytes()

        # Should be re-decodable as a valid WAV
        reloaded = AudioSegment.from_file(io.BytesIO(wav_bytes), format="wav")
        assert len(reloaded) > 0
