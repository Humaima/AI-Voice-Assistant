"""
FastAPI application entrypoint.

Phase 1 scope: app boots, logging is configured, health check works,
and settings load correctly from .env. Routers for the voice webhook
(Phase 9), LangGraph agent (Phase 4-6), etc. get included here as they
are built in later phases — the `include_router` calls below are left
as commented placeholders so the wiring point is obvious.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import get_settings
from app.core.logging_config import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s in %s mode", settings.app_name, settings.environment)
    # Phase 10: open DB pool / redis pool / chroma client here and
    # attach to app.state, then close them in the block after `yield`.
    yield
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title="Conversational Voice AI Assistant",
    description="Voice-in, voice-out WhatsApp assistant built with LangGraph + Groq + ElevenLabs.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["system"])
async def health_check():
    """Basic liveness probe used by Docker/orchestrator health checks."""
    return {"status": "ok", "app": settings.app_name, "environment": settings.environment}


from fastapi import Depends  # noqa: E402

from app.api.agent import router as agent_router  # noqa: E402
from app.api.audio import router as audio_router  # noqa: E402
from app.api.debug_auth import require_debug_access  # noqa: E402
from app.api.media import router as media_router  # noqa: E402
from app.api.metrics import router as metrics_router  # noqa: E402
from app.api.tts import router as tts_router  # noqa: E402
from app.api.webhook import router as webhook_router  # noqa: E402

# Phase 10: debug routers require X-Debug-Token when ENVIRONMENT=production
# (require_debug_access is a no-op in dev) — see app/api/debug_auth.py.
_debug_deps = [Depends(require_debug_access)]

app.include_router(audio_router, prefix="/audio", tags=["audio-debug"], dependencies=_debug_deps)  # Phase 2/3
app.include_router(agent_router, prefix="/agent", tags=["agent-debug"], dependencies=_debug_deps)  # Phase 4
app.include_router(tts_router, prefix="/tts", tags=["tts-debug"], dependencies=_debug_deps)  # Phase 7
app.include_router(media_router, prefix="/media", tags=["media"])  # Phase 9 — public; Twilio must reach it
app.include_router(webhook_router, prefix="/webhook", tags=["whatsapp"])  # Phase 9 — public; Twilio calls it
app.include_router(metrics_router, tags=["observability"])  # Phase 10
