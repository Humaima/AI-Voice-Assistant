"""
Twilio helpers: webhook signature validation and inbound media
download (Phase 9), plus outbound REST message sending (Phase 10).

All three are security/correctness-critical or involve real network
I/O, and are kept separate from the webhook route itself
(app/api/webhook.py) so they're independently testable — signature
validation especially, since Twilio's signing algorithm is
deterministic and worth testing directly rather than trusting it
blindly.
"""
from __future__ import annotations

import asyncio
from typing import Any, Protocol

import httpx
from twilio.request_validator import RequestValidator
from twilio.rest import Client as TwilioRestClient

from app.core.config import get_settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)
settings = get_settings()


class TwilioSignatureError(ValueError):
    """Raised when a webhook request's X-Twilio-Signature doesn't
    match what Twilio's algorithm computes for the request — meaning
    the request either didn't genuinely come from Twilio, or the
    public URL Twilio believes it called doesn't match what we
    validated against (a common misconfiguration, not just an attack)."""


def validate_twilio_signature(url: str, form_params: dict[str, str], signature: str | None) -> None:
    """Raises TwilioSignatureError if the signature is missing or
    doesn't match. `url` must be the EXACT public URL Twilio called —
    if PUBLIC_BASE_URL in .env doesn't match your real ngrok/production
    URL, validation will fail even for genuine Twilio requests, which
    is a configuration bug worth surfacing clearly rather than silently
    skipping validation."""
    if not settings.twilio_auth_token:
        raise TwilioSignatureError(
            "TWILIO_AUTH_TOKEN is not set — cannot validate webhook signatures. "
            "Set it in .env before accepting real WhatsApp traffic."
        )

    if not signature:
        raise TwilioSignatureError("Request is missing the X-Twilio-Signature header")

    validator = RequestValidator(settings.twilio_auth_token)
    if not validator.validate(url, form_params, signature):
        raise TwilioSignatureError(
            "X-Twilio-Signature did not match. Either this request didn't genuinely come from "
            "Twilio, or PUBLIC_BASE_URL in .env doesn't match the URL Twilio actually called "
            "(check your ngrok URL hasn't changed)."
        )


class MessageSender(Protocol):
    """Structural interface for sending an outbound WhatsApp message via
    Twilio's REST API (as opposed to a synchronous TwiML webhook
    reply). Phase 10 introduces this: the webhook now acknowledges
    Twilio immediately and processes the pipeline in a background task,
    sending the actual reply via this REST call once ready — see
    app/services/whatsapp_handler.py's process_and_reply_async for why."""

    async def send(self, to: str, body: str, media_url: str | None = None) -> None: ...


class _MessagesAPI(Protocol):
    def create(self, *, to: Any, from_: Any, body: Any, media_url: Any = ...) -> Any: ...


class _TwilioClientLike(Protocol):
    """Structural interface for anything shaped like `twilio.rest.Client`
    — i.e. exposes `.messages.create(...)`. Same reasoning as
    TranscriptionClient (app/services/transcription.py) and TTSClient
    (app/services/tts.py): typing TwilioMessageSender's constructor
    against this Protocol instead of the concrete SDK class lets tests
    inject a lightweight fake client without subclassing Twilio's
    actual Client, while staying fully type-checked. `Any` for
    create()'s kwargs for the same reason TTSClient uses Any — Twilio's
    real method signature uses narrow types (Union[str, object] sentinel
    defaults) that would otherwise break contravariance against a
    fake's simpler signature."""

    @property
    def messages(self) -> _MessagesAPI: ...


class TwilioMessageSender:
    """Real implementation, using twilio.rest.Client. The SDK's client
    is synchronous (blocking network I/O under the hood) — callers
    should run `.send()` via asyncio.to_thread, same pattern already
    used for TTSService.synthesize elsewhere in this project."""

    def __init__(self, client: _TwilioClientLike | None = None):
        self._client: _TwilioClientLike = client or TwilioRestClient(
            settings.twilio_account_sid, settings.twilio_auth_token
        )

    async def send(self, to: str, body: str, media_url: str | None = None) -> None:
        kwargs: dict = {
            "to": to,
            "from_": settings.twilio_whatsapp_number,
            "body": body,
        }
        if media_url:
            kwargs["media_url"] = [media_url]
        await asyncio.to_thread(self._client.messages.create, **kwargs)


class MediaDownloadError(RuntimeError):
    """Raised when fetching inbound media from Twilio's servers fails."""


class MediaDownloader(Protocol):
    """Structural interface for downloading inbound WhatsApp media —
    same Protocol-injectable pattern used throughout this project
    (TranscriptionClient, TTSClient, etc.) so tests can inject a fake
    downloader instead of making a real HTTP call to Twilio."""

    async def download(self, media_url: str) -> bytes: ...


class TwilioMediaDownloader:
    """Real implementation: Twilio's inbound media URLs require Basic
    Auth with your Account SID/Auth Token — they're not publicly
    fetchable without credentials, unlike the media WE host for
    outbound replies (app/api/media.py, deliberately public since
    Twilio's servers need to fetch it without our credentials).

    `transport` is injectable so tests can exercise this REAL class
    (not just a fake standing in for it) against a fake HTTP transport
    (httpx.MockTransport) instead of making a genuine network call."""

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None):
        self._transport = transport

    async def download(self, media_url: str) -> bytes:
        auth = (settings.twilio_account_sid, settings.twilio_auth_token)
        async with httpx.AsyncClient(timeout=30.0, transport=self._transport) as client:
            try:
                response = await client.get(media_url, auth=auth)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise MediaDownloadError(f"Failed to download media from Twilio: {exc}") from exc
        return response.content
