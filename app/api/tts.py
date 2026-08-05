"""
Debug route for Phase 7: send text, get back playable audio — lets you
test ElevenLabs synthesis in isolation before chaining it onto the
full voice pipeline in /audio/voice-reply-test.
"""
import io

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.services.tts import AUDIO_MEDIA_TYPES, TTSError, TTSService

logger = get_logger(__name__)
settings = get_settings()
router = APIRouter()


class SynthesizeRequest(BaseModel):
    text: str


@router.post("/synthesize-test")
async def synthesize_test(request: SynthesizeRequest):
    """Send {"text": "..."} and get back playable audio. In Swagger UI
    (/docs), the response renders with a built-in audio player — click
    Execute and press play directly on the result."""
    if not settings.elevenlabs_api_key:
        raise HTTPException(
            status_code=500,
            detail="ELEVENLABS_API_KEY is not set. Add it to your .env file to use TTS.",
        )

    try:
        audio_bytes = TTSService().synthesize(request.text)
    except TTSError as exc:
        logger.error("TTS synthesis failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    media_type = AUDIO_MEDIA_TYPES.get(settings.tts_output_format, "audio/mpeg")
    return StreamingResponse(io.BytesIO(audio_bytes), media_type=media_type)
