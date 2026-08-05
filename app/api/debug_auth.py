"""
Debug endpoint access control (Phase 10).

The /audio/*, /agent/*, /tts/* routers exist for testing individual
pipeline stages (see each router's docstring) — every call can spend
real Groq/ElevenLabs credits. That's a reasonable tradeoff while
developing locally behind your own firewall/ngrok tunnel, but on a
public deployment, anyone who finds the URL could run up your bill
just by hitting these repeatedly.

`require_debug_access` is a FastAPI dependency applied to those three
routers. Behavior depends on ENVIRONMENT:
- Not "production" (i.e. local dev): always allowed, no token needed —
  matches every earlier phase's testing workflow unchanged.
- "production": requires an exact match on the X-Debug-Token header
  against DEBUG_API_TOKEN. If DEBUG_API_TOKEN is empty, debug routes
  are refused entirely in production — the safe default, since an
  empty required token is a common way access control accidentally
  ends up doing nothing.
"""
from fastapi import Header, HTTPException

from app.core.config import get_settings

settings = get_settings()


async def require_debug_access(x_debug_token: str | None = Header(default=None)) -> None:
    if settings.environment != "production":
        return

    if not settings.debug_api_token:
        raise HTTPException(
            status_code=404,
            detail="Debug endpoints are disabled in production (DEBUG_API_TOKEN not set).",
        )

    if x_debug_token != settings.debug_api_token:
        raise HTTPException(status_code=404, detail="Not found")
