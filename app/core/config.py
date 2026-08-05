"""
Centralized application configuration.

Everything the app reads from the environment goes through this single
Settings object, so no module ever calls os.environ directly. That keeps
config discoverable (one file to read) and testable (override Settings()
in tests instead of monkeypatching env vars everywhere).
"""
from functools import lru_cache
from urllib.parse import quote_plus

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- App ---
    app_name: str = "voice-ai-assistant"
    environment: str = "development"
    log_level: str = "INFO"

    # --- Groq ---
    groq_api_key: str = ""
    groq_whisper_model: str = "whisper-large-v3"
    groq_llm_model: str = "llama-3.3-70b-versatile"

    # --- ElevenLabs ---
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = ""
    elevenlabs_model: str = "eleven_flash_v2_5"

    # --- WhatsApp (Meta Cloud API) ---
    whatsapp_verify_token: str = ""
    whatsapp_access_token: str = ""
    whatsapp_phone_number_id: str = ""
    whatsapp_api_version: str = "v20.0"

    # --- Twilio ---
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_whatsapp_number: str = ""

    # --- WhatsApp webhook (Phase 9) ---
    # The public HTTPS URL Twilio can reach this server at (e.g. your
    # ngrok URL during local dev, or your real domain in production).
    # Needed for two things: (1) validating that inbound webhook
    # requests genuinely came from Twilio — the signature check is
    # computed over the exact URL Twilio believes it called — and
    # (2) building the public URL for generated reply audio, since
    # Twilio fetches outbound media from a URL, not inline bytes.
    public_base_url: str = ""
    # Local directory generated reply audio is temporarily saved to,
    # served back out via GET /media/{filename}. A real production
    # deployment would use cloud storage (S3 etc.) instead — local
    # disk is fine for this project's current scope (Phase 10 is where
    # deployment concerns like this get revisited).
    media_storage_dir: str = "media_storage"
    # How long generated reply audio files are kept before cleanup.
    media_file_ttl_seconds: int = 3600

    # --- Postgres ---
    # `database_url` is DERIVED from these components below rather than
    # read as its own literal string from .env — a prior version had
    # DATABASE_URL as a separate hardcoded value that duplicated
    # POSTGRES_PASSWORD, which silently drifted out of sync whenever
    # only one of the two got edited (this is exactly the
    # "password authentication failed" bug you'd hit if you changed
    # one but not the other). Only POSTGRES_* below needs editing now.
    postgres_user: str = "voiceai"
    postgres_password: str = "voiceai_pw"
    postgres_host: str = "localhost"
    postgres_port: int = 5433  # not Postgres's usual 5432 — see .env.example for why
    postgres_db: str = "voiceai_db"
    # Escape hatch for advanced cases (e.g. a managed cloud Postgres
    # with its own connection string) — if explicitly set, this wins
    # over the derived value instead of being combined with it.
    database_url_override: str = ""
    database_url: str = ""  # computed in model_validator below; don't set directly in .env

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"

    # --- ChromaDB ---
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection: str = "conversation_memory"

    # --- LangSmith ---
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "voice-ai-assistant"

    # --- Audio pipeline (Phase 2) ---
    audio_target_sample_rate: int = 16000  # Whisper expects 16kHz
    audio_target_channels: int = 1         # mono
    audio_max_duration_ms: int = 5 * 60 * 1000   # reject voice notes over 5 min
    audio_max_size_bytes: int = 16 * 1024 * 1024  # 16MB, matches WhatsApp's own cap
    audio_chunk_length_ms: int = 30_000     # 30s chunks sent to Whisper
    audio_chunk_overlap_ms: int = 500       # small overlap avoids cutting words at boundaries
    audio_silence_thresh_db: int = -40      # dBFS below which audio is considered silence
    audio_min_silence_len_ms: int = 400     # silence must last this long to be trimmed

    # --- Production hardening (Phase 10) ---
    # The /audio/*, /agent/*, /tts/* routers are debug/testing tools —
    # each call can spend real Groq/ElevenLabs credits. Fine to leave
    # open while developing locally; on a public deployment, anyone
    # who finds the URL could run up your bill. When ENVIRONMENT is
    # "production", these routes require this exact token in an
    # X-Debug-Token header. Leave DEBUG_API_TOKEN empty to disable the
    # debug routers entirely in production (safest default) rather
    # than accidentally leaving them reachable with a guessable token.
    debug_api_token: str = ""

    # --- Media storage backend (Phase 10) ---
    # "local" (default, matches Phase 9) writes to MEDIA_STORAGE_DIR on
    # local disk — fine for development, but most deployment platforms
    # run containers with ephemeral filesystems, so files written to
    # local disk can silently vanish on redeploy/restart/scale-out.
    # "s3" uses an S3-compatible bucket instead (AWS S3, Cloudflare R2,
    # Backblaze B2, MinIO, etc. — anything speaking the S3 API).
    media_storage_backend: str = "local"
    s3_bucket_name: str = ""
    s3_region: str = "us-east-1"
    # Leave empty for real AWS S3; set for S3-compatible providers
    # (e.g. Cloudflare R2's account-specific endpoint).
    s3_endpoint_url: str = ""
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    # --- Transcription (Phase 3) ---
    transcription_max_retries: int = 3
    transcription_backoff_base_seconds: float = 1.5
    transcription_temperature: float = 0.0   # 0 = deterministic, best for transcription accuracy
    transcription_stitch_overlap_words: int = 6  # how many boundary words to check for dedup

    # --- Text-to-Speech (Phase 7) ---
    tts_max_retries: int = 3
    tts_backoff_base_seconds: float = 1.5
    tts_output_format: str = "mp3_44100_128"  # ElevenLabs format string
    # eleven_flash_v2_5 handles long input fine, but Ava's replies are
    # meant to be short (see prompts.py) — this is a safety ceiling to
    # fail loudly on a runaway response rather than silently truncate
    # or send an enormous, expensive synthesis request.
    tts_max_chars: int = 2000

    @model_validator(mode="after")
    def _compute_database_url(self) -> "Settings":
        if self.database_url_override:
            self.database_url = self.database_url_override
        else:
            # URL-encode the password: safe even if it contains
            # characters like @ or / that would otherwise break the
            # connection string.
            user = quote_plus(self.postgres_user)
            password = quote_plus(self.postgres_password)
            self.database_url = (
                f"postgresql+asyncpg://{user}:{password}"
                f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Settings are cached — env is read once per process, not per request."""
    return Settings()
