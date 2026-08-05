"""
Tests for Phase 9's WhatsApp message handling — the actual business
logic of turning an inbound voice note into an outbound TwiML reply.

Runs the REAL pipeline pieces (audio processing, AudioBuffer, real
AgentService with a real compiled graph) with fakes only at the true
external boundaries: media download (no real Twilio), the LLM (no
real Groq), and TTS (no real ElevenLabs). This mirrors how the rest of
this project tests things — real internal logic, fake external I/O.
"""
import io

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pydub.generators import Sine

from app.services.agent_service import AgentService
from app.services.tts import TTSService
from app.services.twilio_client import MediaDownloadError
from app.services.whatsapp_handler import handle_incoming_whatsapp_message, process_and_send_reply_async


class FakeMessageSender:
    """Records every outbound message .send() was called with — lets
    tests assert exactly what would have been sent via Twilio's REST
    API, without a real Twilio account."""

    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.sent_messages: list[dict] = []

    async def send(self, to: str, body: str, media_url: str | None = None) -> None:
        if self.should_fail:
            raise RuntimeError("simulated Twilio REST API failure")
        self.sent_messages.append({"to": to, "body": body, "media_url": media_url})


class FakeMediaDownloader:
    def __init__(self, audio_bytes: bytes | None = None, error: Exception | None = None):
        self._audio_bytes = audio_bytes
        self._error = error
        self.requested_urls: list[str] = []

    async def download(self, media_url: str) -> bytes:
        self.requested_urls.append(media_url)
        if self._error:
            raise self._error
        assert self._audio_bytes is not None
        return self._audio_bytes


class FakeConversationMemory:
    def __init__(self):
        self.store: dict[str, list] = {}

    async def get_recent_messages(self, conversation_id: str, limit: int):
        return self.store.get(conversation_id, [])[-limit:]

    async def append_messages(self, conversation_id: str, messages: list) -> None:
        self.store.setdefault(conversation_id, []).extend(messages)


class _FakeTextToSpeechAPI:
    def __init__(self, outer: "FakeTTSClient"):
        self._outer = outer

    def convert(self, voice_id: str, *, text: str, model_id, output_format):
        if self._outer.should_fail:
            raise RuntimeError("simulated TTS failure")
        self._outer.synthesized_texts.append(text)
        return iter([b"fake mp3 audio bytes"])


class FakeTTSClient:
    def __init__(self, should_fail: bool = False):
        self.synthesized_texts: list[str] = []
        self.should_fail = should_fail
        self.text_to_speech = _FakeTextToSpeechAPI(self)


def _make_valid_ogg_voice_note() -> bytes:
    """A short, valid, real OGG audio clip — enough to pass Phase 2's
    validation and produce a real transcribable(-shaped) buffer."""
    tone = Sine(440).to_audio_segment(duration=1200).apply_gain(-3.0)
    buf = io.BytesIO()
    tone.export(buf, format="ogg")
    return buf.getvalue()


def _make_corrupt_audio_bytes() -> bytes:
    return b"this is not valid audio data at all"


@pytest.fixture
def agent_service_factory():
    """Builds an AgentService with a fake LLM (always returns preset
    responses) and a fresh in-memory conversation store, so each test
    can control exactly what "Ava" says without real Groq calls."""

    def _build(responses: list[str], memory_store=None):
        llm = FakeListChatModel(responses=responses)
        return AgentService(llm=llm, memory_store=memory_store)

    return _build


class TestNoMediaAttached:
    async def test_text_only_message_gets_instructions_reply(self, agent_service_factory):
        twiml = await handle_incoming_whatsapp_message(
            from_number="whatsapp:+15551234567",
            num_media=0,
            media_url=None,
            media_content_type=None,
            media_downloader=FakeMediaDownloader(),
            agent_service=agent_service_factory(["unused"]),
            tts_service=TTSService(client=FakeTTSClient()),
        )
        assert "voice note" in twiml.lower()
        assert "<Media>" not in twiml

    async def test_non_audio_media_gets_instructions_reply(self, agent_service_factory):
        twiml = await handle_incoming_whatsapp_message(
            from_number="whatsapp:+15551234567",
            num_media=1,
            media_url="https://api.twilio.com/media/IMG123",
            media_content_type="image/jpeg",
            media_downloader=FakeMediaDownloader(),
            agent_service=agent_service_factory(["unused"]),
            tts_service=TTSService(client=FakeTTSClient()),
        )
        assert "voice note" in twiml.lower()
        assert "<Media>" not in twiml


class TestMediaDownloadFailure:
    async def test_download_failure_gets_friendly_error_reply(self, agent_service_factory):
        downloader = FakeMediaDownloader(error=MediaDownloadError("simulated network failure"))

        twiml = await handle_incoming_whatsapp_message(
            from_number="whatsapp:+15551234567",
            num_media=1,
            media_url="https://api.twilio.com/media/ME123",
            media_content_type="audio/ogg",
            media_downloader=downloader,
            agent_service=agent_service_factory(["unused"]),
            tts_service=TTSService(client=FakeTTSClient()),
        )

        assert "trouble" in twiml.lower()
        assert "<Media>" not in twiml
        assert downloader.requested_urls == ["https://api.twilio.com/media/ME123"]


class TestInvalidAudio:
    async def test_corrupt_audio_gets_friendly_format_error_reply(self, agent_service_factory):
        downloader = FakeMediaDownloader(audio_bytes=_make_corrupt_audio_bytes())

        twiml = await handle_incoming_whatsapp_message(
            from_number="whatsapp:+15551234567",
            num_media=1,
            media_url="https://api.twilio.com/media/ME123",
            media_content_type="audio/ogg",
            media_downloader=downloader,
            agent_service=agent_service_factory(["unused"]),
            tts_service=TTSService(client=FakeTTSClient()),
        )

        assert "<Media>" not in twiml
        assert "again" in twiml.lower()


class TestSuccessfulFlow:
    async def test_valid_voice_note_produces_media_reply(self, agent_service_factory, monkeypatch):
        downloader = FakeMediaDownloader(audio_bytes=_make_valid_ogg_voice_note())
        tts_client = FakeTTSClient()

        monkeypatch.setattr(
            "app.services.whatsapp_handler.build_media_url",
            lambda filename: f"https://example.ngrok-free.app/media/{filename}",
        )

        # Groq Whisper isn't faked here (transcription is a real network
        # call), but process_voice_note + TranscriptionService.transcribe_buffer
        # would need a real API key to actually run — instead we monkeypatch
        # TranscriptionService to avoid a real network call while still
        # exercising the REAL audio-processing pipeline up to that point.
        class FakeTranscriptionResult:
            full_text = "hey what's the weather like"

        class FakeTranscriptionService:
            def transcribe_buffer(self, buffer):
                return FakeTranscriptionResult()

        monkeypatch.setattr(
            "app.services.whatsapp_handler.TranscriptionService", FakeTranscriptionService
        )

        twiml = await handle_incoming_whatsapp_message(
            from_number="whatsapp:+15551234567",
            num_media=1,
            media_url="https://api.twilio.com/media/ME123",
            media_content_type="audio/ogg",
            media_downloader=downloader,
            agent_service=agent_service_factory(["It's sunny and warm today!", "Shares the weather"]),
            tts_service=TTSService(client=tts_client),
        )

        assert "<Media>https://example.ngrok-free.app/media/" in twiml
        assert "Shares the weather" in twiml  # the note, used as message body
        assert tts_client.synthesized_texts == ["It's sunny and warm today!"]

    async def test_conversation_id_is_the_whatsapp_from_number(self, agent_service_factory, monkeypatch):
        """Memory should be keyed by the WhatsApp sender's number —
        confirms conversation continuity works per-user automatically."""
        downloader = FakeMediaDownloader(audio_bytes=_make_valid_ogg_voice_note())
        memory_store = FakeConversationMemory()

        monkeypatch.setattr(
            "app.services.whatsapp_handler.build_media_url", lambda filename: "https://x.test/media/f.mp3"
        )

        class FakeTranscriptionResult:
            full_text = "hello there"

        class FakeTranscriptionService:
            def transcribe_buffer(self, buffer):
                return FakeTranscriptionResult()

        monkeypatch.setattr(
            "app.services.whatsapp_handler.TranscriptionService", FakeTranscriptionService
        )

        await handle_incoming_whatsapp_message(
            from_number="whatsapp:+15559998888",
            num_media=1,
            media_url="https://api.twilio.com/media/ME123",
            media_content_type="audio/ogg",
            media_downloader=downloader,
            agent_service=agent_service_factory(["Hi there!", "Greets them"], memory_store=memory_store),
            tts_service=TTSService(client=FakeTTSClient()),
        )

        assert "whatsapp:+15559998888" in memory_store.store
        assert memory_store.store["whatsapp:+15559998888"][0].content == "hello there"


class TestTTSFailureFallback:
    async def test_tts_failure_falls_back_to_text_reply(self, agent_service_factory, monkeypatch):
        downloader = FakeMediaDownloader(audio_bytes=_make_valid_ogg_voice_note())

        class FakeTranscriptionResult:
            full_text = "test message"

        class FakeTranscriptionService:
            def transcribe_buffer(self, buffer):
                return FakeTranscriptionResult()

        monkeypatch.setattr(
            "app.services.whatsapp_handler.TranscriptionService", FakeTranscriptionService
        )

        twiml = await handle_incoming_whatsapp_message(
            from_number="whatsapp:+15551234567",
            num_media=1,
            media_url="https://api.twilio.com/media/ME123",
            media_content_type="audio/ogg",
            media_downloader=downloader,
            agent_service=agent_service_factory(["This is the text reply", "A note"]),
            tts_service=TTSService(client=FakeTTSClient(should_fail=True)),
        )

        assert "<Media>" not in twiml
        assert "This is the text reply" in twiml


class TestMissingPublicBaseUrlFallback:
    async def test_missing_base_url_falls_back_to_text_reply(self, agent_service_factory, monkeypatch):
        downloader = FakeMediaDownloader(audio_bytes=_make_valid_ogg_voice_note())

        class FakeTranscriptionResult:
            full_text = "test message"

        class FakeTranscriptionService:
            def transcribe_buffer(self, buffer):
                return FakeTranscriptionResult()

        monkeypatch.setattr(
            "app.services.whatsapp_handler.TranscriptionService", FakeTranscriptionService
        )

        def _raise_missing_url(filename):
            raise ValueError("PUBLIC_BASE_URL is not set.")

        monkeypatch.setattr("app.services.whatsapp_handler.build_media_url", _raise_missing_url)

        twiml = await handle_incoming_whatsapp_message(
            from_number="whatsapp:+15551234567",
            num_media=1,
            media_url="https://api.twilio.com/media/ME123",
            media_content_type="audio/ogg",
            media_downloader=downloader,
            agent_service=agent_service_factory(["Text-only fallback works", "A note"]),
            tts_service=TTSService(client=FakeTTSClient()),
        )

        assert "<Media>" not in twiml
        assert "Text-only fallback works" in twiml


class TestProcessAndSendReplyAsync:
    """Phase 10's production path: same pipeline as
    handle_incoming_whatsapp_message, but delivers via a MessageSender
    (Twilio's REST API) instead of returning TwiML — used when the
    webhook has already responded and this runs as a background task."""

    async def test_successful_voice_note_sends_message_with_media(self, agent_service_factory, monkeypatch):
        downloader = FakeMediaDownloader(audio_bytes=_make_valid_ogg_voice_note())
        sender = FakeMessageSender()

        monkeypatch.setattr(
            "app.services.whatsapp_handler.build_media_url",
            lambda filename: f"https://example.ngrok-free.app/media/{filename}",
        )

        class FakeTranscriptionResult:
            full_text = "hey what's the weather like"

        class FakeTranscriptionService:
            def transcribe_buffer(self, buffer):
                return FakeTranscriptionResult()

        monkeypatch.setattr("app.services.whatsapp_handler.TranscriptionService", FakeTranscriptionService)

        await process_and_send_reply_async(
            from_number="whatsapp:+15551234567",
            media_url="https://api.twilio.com/media/ME123",
            media_content_type="audio/ogg",
            media_downloader=downloader,
            agent_service=agent_service_factory(["Here's my answer", "A short note"]),
            tts_service=TTSService(client=FakeTTSClient()),
            message_sender=sender,
        )

        assert len(sender.sent_messages) == 1
        sent = sender.sent_messages[0]
        assert sent["to"] == "whatsapp:+15551234567"
        assert sent["media_url"] is not None
        assert sent["media_url"].startswith("http")

    async def test_download_failure_still_sends_a_fallback_text_message(self, agent_service_factory):
        """Unlike the synchronous TwiML path, there's no HTTP response
        left to shape by the time this runs — the ONLY way the user
        finds out something went wrong is if this function still calls
        message_sender.send() with a fallback message."""
        downloader = FakeMediaDownloader(error=MediaDownloadError("simulated network failure"))
        sender = FakeMessageSender()

        await process_and_send_reply_async(
            from_number="whatsapp:+15551234567",
            media_url="https://api.twilio.com/media/ME123",
            media_content_type="audio/ogg",
            media_downloader=downloader,
            agent_service=agent_service_factory(["unused"]),
            tts_service=TTSService(client=FakeTTSClient()),
            message_sender=sender,
        )

        assert len(sender.sent_messages) == 1
        assert sender.sent_messages[0]["media_url"] is None
        assert "trouble" in sender.sent_messages[0]["body"].lower()

    async def test_tts_failure_sends_text_only_fallback(self, agent_service_factory, monkeypatch):
        downloader = FakeMediaDownloader(audio_bytes=_make_valid_ogg_voice_note())
        sender = FakeMessageSender()

        class FakeTranscriptionResult:
            full_text = "hey what's the weather like"

        class FakeTranscriptionService:
            def transcribe_buffer(self, buffer):
                return FakeTranscriptionResult()

        monkeypatch.setattr("app.services.whatsapp_handler.TranscriptionService", FakeTranscriptionService)

        await process_and_send_reply_async(
            from_number="whatsapp:+15551234567",
            media_url="https://api.twilio.com/media/ME123",
            media_content_type="audio/ogg",
            media_downloader=downloader,
            agent_service=agent_service_factory(["This is the text reply", "A note"]),
            tts_service=TTSService(client=FakeTTSClient(should_fail=True)),
            message_sender=sender,
        )

        assert len(sender.sent_messages) == 1
        assert sender.sent_messages[0]["media_url"] is None
        assert "This is the text reply" in sender.sent_messages[0]["body"]

    async def test_send_failure_is_caught_and_logged_not_raised(self, agent_service_factory, caplog):
        """If Twilio's REST API itself fails (network issue, invalid
        number, etc.), this must not raise — it's running in a
        background task with nothing to catch an unhandled exception,
        which would otherwise be silently swallowed by the task
        runner with zero visibility. Logging loudly is the only
        recourse left at this point."""
        downloader = FakeMediaDownloader(audio_bytes=_make_valid_ogg_voice_note())
        sender = FakeMessageSender(should_fail=True)

        await process_and_send_reply_async(
            from_number="whatsapp:+15551234567",
            media_url="https://api.twilio.com/media/ME123",
            media_content_type="audio/ogg",
            media_downloader=downloader,
            agent_service=agent_service_factory(["A reply", "A note"]),
            tts_service=TTSService(client=FakeTTSClient()),
            message_sender=sender,
        )
        # No exception propagated — that's the assertion. Also confirm
        # the failure was actually logged, not silently swallowed.
        assert "failed to send async reply" in caplog.text
