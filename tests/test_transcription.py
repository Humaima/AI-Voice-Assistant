"""
Tests for the Phase 3 transcription service.

Uses a fake Groq client instead of the real API — this repo's sandbox
(and CI, most likely) has no route to api.groq.com, and more
importantly we want to test retry/backoff and stitching logic
deterministically without depending on real network calls or a real
API key.
"""
from types import SimpleNamespace
from typing import Any

import groq
import httpx
import pytest

from app.services.audio_buffer import AudioBuffer, BufferedChunk
from app.services.transcription import (
    TranscriptionError,
    TranscriptionService,
    stitch_transcripts,
)
from pydub import AudioSegment
from pydub.generators import Sine


def _fake_httpx_response(status_code: int) -> httpx.Response:
    """groq's exception classes require a real httpx.Response (they
    read response.request internally), so build a minimal but valid
    one rather than a SimpleNamespace."""
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/audio/transcriptions")
    return httpx.Response(status_code=status_code, request=request)


def _fake_response(text: str, language: str = "en", duration: float = 2.0) -> SimpleNamespace:
    """Mimics the shape of groq's transcription response closely enough
    for TranscriptionService to extract text/language/segments from it."""
    return SimpleNamespace(
        model_dump=lambda: {
            "text": text,
            "language": language,
            "duration": duration,
            "segments": [{"avg_logprob": -0.2}],
        }
    )


class _FakeTranscriptionsAPI:
    """Explicitly typed stand-in for groq's `client.audio.transcriptions`
    — matches app.services.transcription._TranscriptionsAPI structurally
    (via a concrete `create` method) so FakeGroqClient satisfies the
    TranscriptionClient Protocol without needing casts or `# type: ignore`."""

    def __init__(self, outer: "FakeGroqClient"):
        self._outer = outer

    def create(self, **kwargs: Any) -> Any:
        return self._outer._record_and_respond(**kwargs)


class _FakeAudioAPI:
    """Stand-in for groq's `client.audio`."""

    def __init__(self, outer: "FakeGroqClient"):
        self.transcriptions = _FakeTranscriptionsAPI(outer)


class FakeGroqClient:
    """Stand-in for groq.Groq. `responses` is a list where each item is
    either a response to return or an Exception instance to raise, one
    per call to .create(), in order. Call count/args are recorded for
    assertions.

    Built from explicit classes (rather than nested SimpleNamespace)
    so it structurally satisfies TranscriptionClient for static type
    checking, exactly the way the real groq.Groq client does.
    """

    def __init__(self, responses: list):
        self._responses = list(responses)
        self.call_args_list: list[dict] = []
        self.audio = _FakeAudioAPI(self)

    def _record_and_respond(self, **kwargs: Any) -> Any:
        self.call_args_list.append(kwargs)
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _make_buffer(n_chunks: int = 1, conversation_id: str = "conv-1") -> AudioBuffer:
    segments = [Sine(300).to_audio_segment(duration=500) for _ in range(n_chunks)]
    return AudioBuffer.from_segments(conversation_id, segments)


class TestStitchTranscripts:
    def test_empty_list(self):
        assert stitch_transcripts([]) == ""

    def test_single_chunk(self):
        assert stitch_transcripts(["hello there"]) == "hello there"

    def test_no_overlap_just_concatenates(self):
        result = stitch_transcripts(["hello there", "how are you"])
        assert result == "hello there how are you"

    def test_trims_duplicated_boundary_words(self):
        # "do you know the" repeated across the chunk boundary, as would
        # happen with the 500ms audio overlap producing duplicated words.
        result = stitch_transcripts(
            ["Do you know the neural", "the neural maze podcast"]
        )
        assert result == "Do you know the neural maze podcast"

    def test_skips_empty_chunks(self):
        result = stitch_transcripts(["hello", "", "  ", "world"])
        assert result == "hello world"

    def test_all_empty_returns_empty_string(self):
        assert stitch_transcripts(["", "  ", ""]) == ""


class TestTranscribeChunk:
    def test_successful_transcription(self):
        client = FakeGroqClient([_fake_response("hello world")])
        service = TranscriptionService(client=client)
        buffer = _make_buffer(1)

        result = service.transcribe_chunk(buffer.chunks[0])

        assert result.text == "hello world"
        assert result.language == "en"
        assert result.chunk_index == 0
        assert result.avg_logprob == pytest.approx(-0.2)

    def test_sends_correct_model_and_params(self):
        client = FakeGroqClient([_fake_response("test")])
        service = TranscriptionService(client=client)
        buffer = _make_buffer(1)

        service.transcribe_chunk(buffer.chunks[0])

        call = client.call_args_list[0]
        assert call["model"] == "whisper-large-v3"
        assert call["response_format"] == "verbose_json"
        assert call["temperature"] == 0.0


class TestRetryBehavior:
    def test_retries_on_rate_limit_then_succeeds(self):
        client = FakeGroqClient(
            [
                groq.RateLimitError("rate limited", response=_fake_httpx_response(429), body=None),
                _fake_response("recovered after retry"),
            ]
        )
        service = TranscriptionService(client=client)
        service.backoff_base = 0.01  # keep test fast
        buffer = _make_buffer(1)

        result = service.transcribe_chunk(buffer.chunks[0])

        assert result.text == "recovered after retry"
        assert len(client.call_args_list) == 2

    def test_exhausts_retries_and_raises(self):
        err = groq.APIConnectionError(request=httpx.Request("POST", "https://api.groq.com/openai/v1/audio/transcriptions"))
        client = FakeGroqClient([err, err, err])
        service = TranscriptionService(client=client)
        service.max_retries = 3
        service.backoff_base = 0.01

        buffer = _make_buffer(1)

        with pytest.raises(TranscriptionError, match="Transcription failed after 3 attempts"):
            service.transcribe_chunk(buffer.chunks[0])

        assert len(client.call_args_list) == 3

    def test_non_retryable_error_fails_immediately(self):
        err = groq.BadRequestError(
            "bad request", response=_fake_httpx_response(400), body=None
        )
        client = FakeGroqClient([err])
        service = TranscriptionService(client=client)
        buffer = _make_buffer(1)

        with pytest.raises(TranscriptionError, match="Groq rejected the transcription request"):
            service.transcribe_chunk(buffer.chunks[0])

        # Only one attempt — no retry for a 4xx.
        assert len(client.call_args_list) == 1


class TestTranscribeBuffer:
    def test_multi_chunk_stitches_and_marks_consumed(self):
        client = FakeGroqClient(
            [
                _fake_response("Do you know the neural"),
                _fake_response("the neural maze podcast"),
            ]
        )
        service = TranscriptionService(client=client)
        buffer = _make_buffer(2, conversation_id="conv-multi")

        result = service.transcribe_buffer(buffer)

        assert result.full_text == "Do you know the neural maze podcast"
        assert result.chunk_count == 2
        assert result.language == "en"
        assert buffer.is_fully_consumed()

    def test_single_failed_chunk_raises_and_stops_processing(self):
        err = groq.BadRequestError(
            "bad request", response=_fake_httpx_response(400), body=None
        )
        client = FakeGroqClient([_fake_response("first chunk ok"), err])
        service = TranscriptionService(client=client)
        buffer = _make_buffer(2)

        with pytest.raises(TranscriptionError):
            service.transcribe_buffer(buffer)

        # First chunk succeeded and was marked consumed before the failure.
        assert buffer.chunks[0].consumed is True
        assert buffer.chunks[1].consumed is False
