"""
Tests for Twilio helpers: webhook signature validation and inbound
media download (Phase 9), plus outbound REST message sending
(Phase 10).

Signature validation is tested against Twilio's REAL algorithm (via
RequestValidator.compute_signature — the same code Twilio itself uses)
rather than a hand-rolled fake, so these tests would actually catch a
regression in how we call the validator. Media download is tested
against the REAL TwilioMediaDownloader class using httpx.MockTransport
(a fake network transport, not a fake class) so the actual HTTP-calling
code path gets exercised. Outbound sending is tested against the REAL
TwilioMessageSender class with a fake twilio.rest.Client stand-in,
since the SDK itself doesn't offer an httpx-style mock transport hook.
"""
import httpx
import pytest
from twilio.request_validator import RequestValidator

from app.core.config import get_settings
from app.services.twilio_client import (
    MediaDownloadError,
    TwilioMediaDownloader,
    TwilioMessageSender,
    TwilioSignatureError,
    validate_twilio_signature,
)

settings = get_settings()

_TEST_URL = "https://example.ngrok-free.app/webhook/whatsapp"
_TEST_PARAMS = {"From": "whatsapp:+15551234567", "Body": "", "NumMedia": "1"}


def _real_signature(url: str = _TEST_URL, params: dict | None = None) -> str:
    """Computes a genuinely valid Twilio signature the same way Twilio
    itself would, using whatever TWILIO_AUTH_TOKEN is currently
    configured (settings.twilio_auth_token — set via .env.example's
    placeholder in these tests)."""
    validator = RequestValidator(settings.twilio_auth_token)
    return validator.compute_signature(url, params or _TEST_PARAMS)


@pytest.fixture(autouse=True)
def _fake_auth_token(monkeypatch):
    """.env.example's TWILIO_AUTH_TOKEN default is blank, and
    validate_twilio_signature correctly refuses to validate anything
    without one — so signature tests need a real (fake-but-non-empty)
    token configured, same as a real deployment would have."""
    monkeypatch.setattr(settings, "twilio_auth_token", "test_auth_token_for_signature_tests")


class TestValidateTwilioSignature:
    def test_valid_signature_passes(self):
        signature = _real_signature()
        # Should not raise.
        validate_twilio_signature(_TEST_URL, _TEST_PARAMS, signature)

    def test_wrong_signature_raises(self):
        with pytest.raises(TwilioSignatureError, match="did not match"):
            validate_twilio_signature(_TEST_URL, _TEST_PARAMS, "totally-wrong-signature")

    def test_missing_signature_raises(self):
        with pytest.raises(TwilioSignatureError, match="missing"):
            validate_twilio_signature(_TEST_URL, _TEST_PARAMS, None)

    def test_empty_signature_raises(self):
        with pytest.raises(TwilioSignatureError, match="missing"):
            validate_twilio_signature(_TEST_URL, _TEST_PARAMS, "")

    def test_signature_for_different_url_fails(self):
        """A signature computed for one URL should not validate
        against a different URL — this is what catches the
        PUBLIC_BASE_URL misconfiguration case described in the
        function's docstring."""
        signature = _real_signature(url="https://example.ngrok-free.app/webhook/whatsapp")
        with pytest.raises(TwilioSignatureError):
            validate_twilio_signature(
                "https://a-different-url.ngrok-free.app/webhook/whatsapp", _TEST_PARAMS, signature
            )

    def test_signature_for_different_params_fails(self):
        signature = _real_signature(params={"From": "whatsapp:+15551234567", "Body": "original"})
        tampered_params = {"From": "whatsapp:+15551234567", "Body": "tampered"}
        with pytest.raises(TwilioSignatureError):
            validate_twilio_signature(_TEST_URL, tampered_params, signature)

    def test_missing_auth_token_raises_clear_error(self, monkeypatch):
        monkeypatch.setattr(settings, "twilio_auth_token", "")
        with pytest.raises(TwilioSignatureError, match="TWILIO_AUTH_TOKEN"):
            validate_twilio_signature(_TEST_URL, _TEST_PARAMS, "any-signature")


class TestTwilioMediaDownloader:
    async def test_successful_download_returns_bytes(self):
        transport = httpx.MockTransport(lambda request: httpx.Response(200, content=b"fake audio bytes"))
        downloader = TwilioMediaDownloader(transport=transport)

        result = await downloader.download("https://api.twilio.com/media/ME123")

        assert result == b"fake audio bytes"

    async def test_sends_basic_auth_with_account_credentials(self, monkeypatch):
        monkeypatch.setattr(settings, "twilio_account_sid", "AC_test_sid")
        monkeypatch.setattr(settings, "twilio_auth_token", "test_token")

        captured_requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_requests.append(request)
            return httpx.Response(200, content=b"audio")

        transport = httpx.MockTransport(handler)
        downloader = TwilioMediaDownloader(transport=transport)

        await downloader.download("https://api.twilio.com/media/ME123")

        assert "authorization" in captured_requests[0].headers

    async def test_http_error_status_raises_media_download_error(self):
        transport = httpx.MockTransport(lambda request: httpx.Response(404, content=b"not found"))
        downloader = TwilioMediaDownloader(transport=transport)

        with pytest.raises(MediaDownloadError, match="Failed to download"):
            await downloader.download("https://api.twilio.com/media/does-not-exist")

    async def test_connection_error_raises_media_download_error(self):
        def handler(request: httpx.Request):
            raise httpx.ConnectError("connection refused", request=request)

        transport = httpx.MockTransport(handler)
        downloader = TwilioMediaDownloader(transport=transport)

        with pytest.raises(MediaDownloadError):
            await downloader.download("https://api.twilio.com/media/ME123")


class _FakeMessagesResource:
    def __init__(self, outer: "_FakeTwilioRestClient"):
        self._outer = outer

    def create(self, **kwargs):
        self._outer.create_calls.append(kwargs)
        return object()  # real SDK returns a MessageInstance; unused by our code


class _FakeTwilioRestClient:
    """Stand-in for twilio.rest.Client — just records what
    .messages.create() was called with, no real network access."""

    def __init__(self):
        self.create_calls: list[dict] = []
        self.messages = _FakeMessagesResource(self)


class TestTwilioMessageSender:
    async def test_sends_with_correct_to_from_and_body(self):
        client = _FakeTwilioRestClient()
        sender = TwilioMessageSender(client=client)

        await sender.send(to="whatsapp:+15551234567", body="hello there")

        assert len(client.create_calls) == 1
        call = client.create_calls[0]
        assert call["to"] == "whatsapp:+15551234567"
        assert call["from_"] == settings.twilio_whatsapp_number
        assert call["body"] == "hello there"
        assert "media_url" not in call

    async def test_includes_media_url_as_a_list_when_provided(self):
        client = _FakeTwilioRestClient()
        sender = TwilioMessageSender(client=client)

        await sender.send(to="whatsapp:+15551234567", body="here's the audio", media_url="https://example.com/a.mp3")

        call = client.create_calls[0]
        assert call["media_url"] == ["https://example.com/a.mp3"]

    async def test_omits_media_url_key_entirely_when_none(self):
        """Twilio's SDK treats an explicitly-passed media_url=None
        differently from the kwarg being absent — omitting it entirely
        when there's no media avoids sending a spurious empty value."""
        client = _FakeTwilioRestClient()
        sender = TwilioMessageSender(client=client)

        await sender.send(to="whatsapp:+15551234567", body="text only", media_url=None)

        call = client.create_calls[0]
        assert "media_url" not in call
