"""
Serves generated reply audio files back out over HTTP, so Twilio (or
anyone testing manually) can fetch them by URL — see
app/services/media_store.py for why this exists.
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from app.services.media_store import load_media
from app.services.tts import AUDIO_MEDIA_TYPES

router = APIRouter()

_MEDIA_TYPE_BY_EXTENSION = {
    "mp3": "audio/mpeg",
    "pcm": "audio/pcm",
    "ulaw": "audio/basic",
}


@router.get("/{filename}")
async def get_media(filename: str):
    audio_bytes = load_media(filename)
    if audio_bytes is None:
        raise HTTPException(status_code=404, detail="Media file not found (it may have expired)")

    extension = filename.rsplit(".", 1)[-1] if "." in filename else ""
    media_type = _MEDIA_TYPE_BY_EXTENSION.get(extension) or next(iter(AUDIO_MEDIA_TYPES.values()))

    return Response(content=audio_bytes, media_type=media_type)
