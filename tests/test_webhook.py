"""
HTTP-level tests for Phase 9's webhook and media routes.

Deliberately shallow compared to tests/test_whatsapp_handler.py — the
real business logic is already thoroughly tested there against the
plain `handle_incoming_whatsapp_message` function. These tests exist to
confirm the HTTP layer itself works: routing, Twilio's actual
form-encoded request shape gets parsed correctly, and signature
validation is actually wired in before any real work happens.
"""
from twilio.request_validator import RequestValidator

from app.core.config import get_settings

settings = get_settings()


def _client(monkeypatch):
    """Builds a TestClient with a known TWILIO_AUTH_TOKEN and
    PUBLIC_BASE_URL configured, since both are required for signature
    validation to even be checkable."""
    monkeypatch.setattr(settings, "twilio_auth_token", "test_webhook_auth_token")
    monkeypatch.setattr(settings, "public_base_url", "https://example.ngrok-free.app")

    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)


def _signed_request(client, form_data: dict[str, str]):
    url = "https://example.ngrok-free.app/webhook/whatsapp"
    validator = RequestValidator(settings.twilio_auth_token)
    signature = validator.compute_signature(url, form_data)
    return client.post(
        "/webhook/whatsapp",
        data=form_data,
        headers={"X-Twilio-Signature": signature},
    )


class TestWebhookSignatureValidation:
    def test_missing_signature_header_returns_403(self, monkeypatch):
        client = _client(monkeypatch)
        response = client.post(
            "/webhook/whatsapp",
            data={"From": "whatsapp:+15551234567", "NumMedia": "0"},
        )
        assert response.status_code == 403

    def test_wrong_signature_returns_403(self, monkeypatch):
        client = _client(monkeypatch)
        response = client.post(
            "/webhook/whatsapp",
            data={"From": "whatsapp:+15551234567", "NumMedia": "0"},
            headers={"X-Twilio-Signature": "not-a-real-signature"},
        )
        assert response.status_code == 403

    def test_valid_signature_passes_validation(self, monkeypatch):
        """Confirms a correctly-signed request gets past the signature
        check (proceeding to the next guard — missing GROQ_API_KEY —
        rather than being rejected as unauthorized). Not testing the
        full pipeline here; that's tests/test_whatsapp_handler.py's job."""
        client = _client(monkeypatch)
        monkeypatch.setattr(settings, "groq_api_key", "")

        response = _signed_request(client, {"From": "whatsapp:+15551234567", "NumMedia": "0"})

        # 500 (missing GROQ_API_KEY) proves signature validation passed —
        # a 403 would mean it never got that far.
        assert response.status_code == 500
        assert "GROQ_API_KEY" in response.json()["detail"]

    def test_tampered_form_field_fails_validation(self, monkeypatch):
        """A signature computed for one set of form values shouldn't
        validate if even one field is changed afterward — this is
        what actually protects against a forged request."""
        client = _client(monkeypatch)
        url = "https://example.ngrok-free.app/webhook/whatsapp"
        validator = RequestValidator(settings.twilio_auth_token)
        original_data = {"From": "whatsapp:+15551234567", "NumMedia": "0"}
        signature = validator.compute_signature(url, original_data)

        tampered_data = {"From": "whatsapp:+19998887777", "NumMedia": "0"}
        response = client.post(
            "/webhook/whatsapp",
            data=tampered_data,
            headers={"X-Twilio-Signature": signature},
        )
        assert response.status_code == 403


class TestSuccessfulResponseFormat:
    """Regression coverage for a real bug: the webhook originally
    returned Content-Type: application/xml instead of text/xml. Twilio
    silently discards TwiML responses with the wrong content type
    rather than erroring — the webhook still returns a clean 200, so
    nothing about the HTTP exchange itself looks wrong, but Twilio
    never acts on the response (no reply sent, no media ever fetched).
    This exact failure mode is invisible without asserting the header
    directly, which is why this test exists."""

    def test_no_media_response_has_correct_twiml_content_type(self, monkeypatch):
        client = _client(monkeypatch)
        monkeypatch.setattr(settings, "groq_api_key", "fake_key_present")
        monkeypatch.setattr(settings, "elevenlabs_api_key", "fake_key_present")

        # The no-media path never actually calls agent_service/tts_service
        # (handle_incoming_whatsapp_message returns early), but the route
        # constructs them eagerly as call arguments regardless of that —
        # stub them out so this test doesn't need real Postgres/ChromaDB/
        # ElevenLabs clients to be constructible.
        import app.api.webhook as webhook_module

        monkeypatch.setattr(webhook_module, "get_agent_service", lambda: object())
        monkeypatch.setattr(webhook_module, "TTSService", lambda: object())

        response = _signed_request(client, {"From": "whatsapp:+15551234567", "NumMedia": "0"})

        assert response.status_code == 200
        # Must be exactly "text/xml" (with optional charset suffix) —
        # NOT "application/xml", which looks equivalent but Twilio
        # silently rejects.
        assert response.headers["content-type"].startswith("text/xml")
        assert "<Response>" in response.text
        assert "voice notes" in response.text.lower()


class TestWebhookMissingCredentials:
    def test_missing_groq_key_returns_clean_500(self, monkeypatch):
        client = _client(monkeypatch)
        monkeypatch.setattr(settings, "groq_api_key", "")
        monkeypatch.setattr(settings, "elevenlabs_api_key", "fake_key")

        response = _signed_request(client, {"From": "whatsapp:+15551234567", "NumMedia": "0"})

        assert response.status_code == 500
        assert "GROQ_API_KEY" in response.json()["detail"]

    def test_missing_elevenlabs_key_returns_clean_500(self, monkeypatch):
        client = _client(monkeypatch)
        monkeypatch.setattr(settings, "groq_api_key", "fake_key")
        monkeypatch.setattr(settings, "elevenlabs_api_key", "")

        response = _signed_request(client, {"From": "whatsapp:+15551234567", "NumMedia": "0"})

        assert response.status_code == 500
        assert "ELEVENLABS_API_KEY" in response.json()["detail"]


class TestMediaRoute:
    def test_get_nonexistent_media_returns_404(self, monkeypatch):
        client = _client(monkeypatch)
        response = client.get("/media/does-not-exist.mp3")
        assert response.status_code == 404

    def test_get_existing_media_returns_audio(self, monkeypatch, tmp_path):
        monkeypatch.setattr(settings, "media_storage_dir", str(tmp_path))
        client = _client(monkeypatch)

        from app.services.media_store import save_media

        filename = save_media(b"fake mp3 content", "audio/mpeg")

        response = client.get(f"/media/{filename}")

        assert response.status_code == 200
        assert response.content == b"fake mp3 content"
        assert response.headers["content-type"] == "audio/mpeg"

    def test_path_traversal_attempt_returns_404_not_file_contents(self, monkeypatch):
        client = _client(monkeypatch)
        response = client.get("/media/..%2F..%2Fetc%2Fpasswd")
        assert response.status_code in (404, 400)  # FastAPI/Starlette may reject the path itself


class TestVoiceNoteSchedulesBackgroundTask:
    """Phase 10: a message WITH audio media should get an immediate,
    empty TwiML ack (not the full pipeline result inline), with the
    actual processing scheduled as a background task instead. Starlette's
    TestClient runs background tasks synchronously as part of the
    request/response cycle in tests (unlike real deployment, where they
    run after the response has already been sent) — so we can assert
    both the immediate response shape AND that the task ran, in one call.
    """

    def test_audio_message_gets_empty_ack_and_schedules_processing(self, monkeypatch):
        client = _client(monkeypatch)
        monkeypatch.setattr(settings, "groq_api_key", "fake_key_present")
        monkeypatch.setattr(settings, "elevenlabs_api_key", "fake_key_present")
        monkeypatch.setattr("app.api.webhook.get_agent_service", lambda: object())
        monkeypatch.setattr("app.api.webhook.TTSService", lambda: object())
        monkeypatch.setattr("app.api.webhook.TwilioMessageSender", lambda: object())

        calls = []

        async def fake_process_and_send_reply_async(**kwargs):
            calls.append(kwargs)

        monkeypatch.setattr(
            "app.api.webhook.process_and_send_reply_async", fake_process_and_send_reply_async
        )

        response = _signed_request(
            client,
            {
                "From": "whatsapp:+15551234567",
                "NumMedia": "1",
                "MediaUrl0": "https://api.twilio.com/media/ME123",
                "MediaContentType0": "audio/ogg",
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/xml")
        # Empty ack — no <Message> at all, since the real reply is sent
        # separately later via the REST API, not in this response.
        assert "<Message>" not in response.text
        assert "<Response" in response.text

        # The background task was actually scheduled and ran (in the
        # test client) with the right arguments.
        assert len(calls) == 1
        assert calls[0]["from_number"] == "whatsapp:+15551234567"
        assert calls[0]["media_url"] == "https://api.twilio.com/media/ME123"

    def test_no_media_message_does_not_schedule_background_task(self, monkeypatch):
        """The trivial instant-reply path shouldn't go through the
        background-task machinery at all — confirms the two code paths
        in the webhook are correctly distinguished by has_audio_media."""
        client = _client(monkeypatch)
        monkeypatch.setattr(settings, "groq_api_key", "fake_key_present")
        monkeypatch.setattr(settings, "elevenlabs_api_key", "fake_key_present")

        calls = []

        async def fake_process_and_send_reply_async(**kwargs):
            calls.append(kwargs)

        monkeypatch.setattr(
            "app.api.webhook.process_and_send_reply_async", fake_process_and_send_reply_async
        )
        monkeypatch.setattr("app.api.webhook.get_agent_service", lambda: object())
        monkeypatch.setattr("app.api.webhook.TTSService", lambda: object())

        response = _signed_request(client, {"From": "whatsapp:+15551234567", "NumMedia": "0"})

        assert response.status_code == 200
        assert "<Message>" in response.text  # the synchronous instructions reply
        assert len(calls) == 0
