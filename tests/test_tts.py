"""
Tests for the Phase 7 TTS service.

Uses a fake ElevenLabs client instead of the real API — no network
access to api.elevenlabs.io in this sandbox (or most CI environments),
and more importantly we want retry/backoff and validation logic tested
deterministically. Same pattern as tests/test_transcription.py's
FakeGroqClient.
"""
from typing import Any

import httpx
import pytest
from elevenlabs.core.api_error import ApiError

from app.core.config import get_settings
from app.services.tts import TTSError, TTSService

settings = get_settings()


class _FakeTextToSpeechAPI:
    """Explicitly typed stand-in for `client.text_to_speech` — matches
    app.services.tts._TextToSpeechAPI structurally via a concrete
    `convert` method, so FakeElevenLabsClient satisfies the TTSClient
    Protocol without casts or `# type: ignore`."""

    def __init__(self, outer: "FakeElevenLabsClient"):
        self._outer = outer

    def convert(self, voice_id: str, *, text: str, model_id: Any, output_format: Any):
        return self._outer._record_and_respond(voice_id=voice_id, text=text, model_id=model_id, output_format=output_format)


class FakeElevenLabsClient:
    """Stand-in for elevenlabs.client.ElevenLabs. `responses` is a list
    where each item is either an iterable of bytes chunks to return, or
    an Exception to raise, one per call to .convert(), in order."""

    def __init__(self, responses: list):
        self._responses = list(responses)
        self.call_args_list: list[dict] = []
        self.text_to_speech = _FakeTextToSpeechAPI(self)

    def _record_and_respond(self, **kwargs: Any):
        self.call_args_list.append(kwargs)
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return iter(item)


def _fake_httpx_response(status_code: int) -> httpx.Response:
    request = httpx.Request("POST", "https://api.elevenlabs.io/v1/text-to-speech/voice123")
    return httpx.Response(status_code=status_code, request=request)


class TestSynthesizeSuccess:
    def test_returns_joined_audio_bytes(self):
        client = FakeElevenLabsClient([[b"chunk1", b"chunk2", b"chunk3"]])
        service = TTSService(client=client)

        audio = service.synthesize("Hello there")

        assert audio == b"chunk1chunk2chunk3"

    def test_sends_correct_voice_model_and_format(self):
        client = FakeElevenLabsClient([[b"audio"]])
        service = TTSService(client=client)

        service.synthesize("Test text")

        call = client.call_args_list[0]
        assert call["voice_id"] == settings.elevenlabs_voice_id
        assert call["model_id"] == settings.elevenlabs_model
        assert call["output_format"] == settings.tts_output_format

    def test_sanitizes_text_before_sending(self):
        client = FakeElevenLabsClient([[b"audio"]])
        service = TTSService(client=client)

        service.synthesize("**Bold** text here")

        sent_text = client.call_args_list[0]["text"]
        assert "**" not in sent_text
        assert sent_text == "Bold text here"


class TestSynthesizeValidation:
    def test_rejects_empty_text(self):
        service = TTSService(client=FakeElevenLabsClient([]))
        with pytest.raises(TTSError, match="empty"):
            service.synthesize("")

    def test_rejects_whitespace_only_text(self):
        service = TTSService(client=FakeElevenLabsClient([]))
        with pytest.raises(TTSError, match="empty"):
            service.synthesize("   ")

    def test_rejects_text_over_max_chars(self):
        service = TTSService(client=FakeElevenLabsClient([]))
        long_text = "word " * (settings.tts_max_chars // 4)  # comfortably over the limit
        with pytest.raises(TTSError, match="too long"):
            service.synthesize(long_text)

    def test_accepts_text_at_exactly_max_chars(self):
        client = FakeElevenLabsClient([[b"audio"]])
        service = TTSService(client=client)
        text = "a" * settings.tts_max_chars

        audio = service.synthesize(text)

        assert audio == b"audio"

    def test_empty_audio_response_raises(self):
        client = FakeElevenLabsClient([[b""]])
        service = TTSService(client=client)
        with pytest.raises(TTSError, match="empty audio"):
            service.synthesize("Hello")


class TestRetryBehavior:
    def test_retries_on_connection_error_then_succeeds(self):
        client = FakeElevenLabsClient([httpx.ConnectError("connection failed"), [b"recovered audio"]])
        service = TTSService(client=client)
        service.backoff_base = 0.01  # keep test fast

        audio = service.synthesize("Hello")

        assert audio == b"recovered audio"
        assert len(client.call_args_list) == 2

    def test_retries_on_timeout(self):
        client = FakeElevenLabsClient([httpx.TimeoutException("timed out"), [b"ok"]])
        service = TTSService(client=client)
        service.backoff_base = 0.01

        audio = service.synthesize("Hello")

        assert audio == b"ok"

    def test_retries_on_rate_limit_429(self):
        err = ApiError(status_code=429, body={"detail": "rate limited"})
        client = FakeElevenLabsClient([err, [b"ok after backoff"]])
        service = TTSService(client=client)
        service.backoff_base = 0.01

        audio = service.synthesize("Hello")

        assert audio == b"ok after backoff"

    def test_retries_on_server_error_5xx(self):
        err = ApiError(status_code=503, body={"detail": "service unavailable"})
        client = FakeElevenLabsClient([err, [b"ok"]])
        service = TTSService(client=client)
        service.backoff_base = 0.01

        audio = service.synthesize("Hello")

        assert audio == b"ok"

    def test_exhausts_retries_and_raises(self):
        err = httpx.ConnectError("still failing")
        client = FakeElevenLabsClient([err, err, err])
        service = TTSService(client=client)
        service.max_retries = 3
        service.backoff_base = 0.01

        with pytest.raises(TTSError, match="Speech synthesis failed after 3 attempts"):
            service.synthesize("Hello")

        assert len(client.call_args_list) == 3

    def test_non_retryable_4xx_fails_immediately(self):
        err = ApiError(status_code=401, body={"detail": "invalid api key"})
        client = FakeElevenLabsClient([err])
        service = TTSService(client=client)

        with pytest.raises(TTSError, match="ElevenLabs rejected the synthesis request"):
            service.synthesize("Hello")

        # Only one attempt — no retry for a 4xx.
        assert len(client.call_args_list) == 1

    def test_bad_voice_id_422_fails_immediately(self):
        err = ApiError(status_code=422, body={"detail": "invalid voice_id"})
        client = FakeElevenLabsClient([err])
        service = TTSService(client=client)

        with pytest.raises(TTSError):
            service.synthesize("Hello")

        assert len(client.call_args_list) == 1
