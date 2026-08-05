"""
The real production entrypoint: Twilio calls this URL every time
someone sends a WhatsApp message to your Twilio number. Replaces the
file-upload debug endpoints (/audio/*) as the actual product — those
remain useful for testing individual pipeline stages, but this is what
a real user's phone actually talks to.

Twilio POSTs application/x-www-form-urlencoded (not JSON) — hence
Form(...) parameters below, matching the shape Twilio actually sends,
not a hypothetical JSON body.

Phase 10: voice-note messages are now processed as a FastAPI
BackgroundTask rather than synchronously within the webhook response.
The full pipeline (audio processing + transcription + LLM + TTS) was
observed taking up to ~13 seconds in real testing — too close to
Twilio's webhook timeout to risk running inline. The webhook now
returns an empty TwiML ack almost instantly, and the real reply is
sent afterward via Twilio's REST API (see
app/services/whatsapp_handler.py's process_and_send_reply_async and
app/services/twilio_client.py's TwilioMessageSender). Text-only "no
voice note attached" replies stay synchronous — building that string
takes microseconds, so there's no timeout risk to design around.
"""
from fastapi import APIRouter, BackgroundTasks, Form, HTTPException, Request, Response
from twilio.twiml.messaging_response import MessagingResponse

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.services.agent_service import get_agent_service
from app.services.tts import TTSService
from app.services.twilio_client import (
    TwilioMediaDownloader,
    TwilioMessageSender,
    TwilioSignatureError,
    validate_twilio_signature,
)
from app.services.whatsapp_handler import handle_incoming_whatsapp_message, process_and_send_reply_async

logger = get_logger(__name__)
settings = get_settings()
router = APIRouter()

# Content-Type MUST be exactly "text/xml" for every TwiML response
# below — Twilio's parser silently discards responses with any other
# content type (even the seemingly-equivalent "application/xml"),
# rather than erroring. A discarded response looks identical to a
# successful one from your server's perspective (still a 200 OK) — the
# only symptom is Twilio never acting on it.
_TWIML_MEDIA_TYPE = "text/xml"


@router.post("/whatsapp")
async def whatsapp_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    From: str = Form(...),  # noqa: N803 — matches Twilio's actual field name exactly
    NumMedia: int = Form(default=0),
    MediaUrl0: str | None = Form(default=None),
    MediaContentType0: str | None = Form(default=None),
):
    """Twilio's inbound WhatsApp webhook. Configure this URL (your
    PUBLIC_BASE_URL + /webhook/whatsapp) in the Twilio Console under
    your WhatsApp sender's "When a message comes in" setting — see the
    README's Phase 9/10 setup steps.
    """
    form = await request.form()
    form_params = {key: str(value) for key, value in form.items()}
    signature = request.headers.get("X-Twilio-Signature")
    webhook_url = f"{settings.public_base_url.rstrip('/')}/webhook/whatsapp"

    try:
        validate_twilio_signature(webhook_url, form_params, signature)
    except TwilioSignatureError as exc:
        logger.warning("Rejected webhook request: %s", exc)
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    if not settings.groq_api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set.")
    if not settings.elevenlabs_api_key:
        raise HTTPException(status_code=500, detail="ELEVENLABS_API_KEY is not set.")

    has_audio_media = NumMedia >= 1 and MediaUrl0 and MediaContentType0 and MediaContentType0.startswith("audio/")

    if not has_audio_media:
        # No real processing to do — building this string takes
        # microseconds, so the synchronous TwiML path is fine here.
        twiml = await handle_incoming_whatsapp_message(
            from_number=From,
            num_media=NumMedia,
            media_url=MediaUrl0,
            media_content_type=MediaContentType0,
            media_downloader=TwilioMediaDownloader(),
            agent_service=get_agent_service(),
            tts_service=TTSService(),
        )
        return Response(content=twiml, media_type=_TWIML_MEDIA_TYPE)

    # Real voice note: schedule the full pipeline as a background task
    # and acknowledge Twilio immediately, well under its timeout. The
    # actual reply is sent separately via Twilio's REST API once ready
    # — see process_and_send_reply_async.
    assert MediaUrl0 is not None and MediaContentType0 is not None  # narrowed by has_audio_media above

    background_tasks.add_task(
        process_and_send_reply_async,
        from_number=From,
        media_url=MediaUrl0,
        media_content_type=MediaContentType0,
        media_downloader=TwilioMediaDownloader(),
        agent_service=get_agent_service(),
        tts_service=TTSService(),
        message_sender=TwilioMessageSender(),
    )

    logger.info("whatsapp | from=%s | voice note received, processing in background", From)
    empty_ack = str(MessagingResponse())  # "<Response></Response>" — received, no immediate reply
    return Response(content=empty_ack, media_type=_TWIML_MEDIA_TYPE)
