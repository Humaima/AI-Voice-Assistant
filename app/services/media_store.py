"""
Media storage: local disk (Phase 9) or S3-compatible cloud storage
(Phase 10).

Twilio's WhatsApp API fetches outbound media from a public URL — you
can't attach audio bytes directly to a webhook response. This module
saves generated reply audio somewhere reachable and hands back a
filename/key; the public URL is built separately (build_media_url).

Why S3 matters, not just "nice to have": most deployment platforms
(Render, Railway, Fly.io, most container platforms) run your app in
containers with EPHEMERAL local disks — files written locally can
silently vanish on redeploy, restart, or when scaled to multiple
instances (each instance would have its own separate local disk, so a
file saved by one instance wouldn't exist on another that happens to
serve the GET /media/{filename} request). Local disk
(MEDIA_STORAGE_BACKEND=local) is genuinely fine for local development,
where none of that applies — but switch to S3 before deploying for
real.

Backend selection is via settings.media_storage_backend ("local" |
"s3"). The public function API (save_media/load_media/build_media_url/
cleanup_expired_files) stays the same regardless of backend, so
callers (app/services/whatsapp_handler.py, app/api/media.py) don't
need to know or care which one is active.

Design note on blocking I/O: both backends do blocking network/disk
I/O synchronously (boto3 has no native asyncio support, matching the
same synchronous-client pattern already used for Groq/ElevenLabs
elsewhere in this project). Called from async code without
`asyncio.to_thread`, this blocks the event loop for the duration of
the call — an accepted tradeoff at this project's current scale (a
personal/learning project, not high-concurrency production traffic),
consistent with how transcription/TTS calls are already handled. If
you deploy this somewhere that needs to handle many concurrent
WhatsApp conversations, wrapping these calls in `asyncio.to_thread`
(or moving background processing to a real task queue like Celery/RQ/
arq instead of FastAPI's BackgroundTasks) would be the next step —
noted here rather than silently deferred.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Protocol

from app.core.config import get_settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)
settings = get_settings()

_EXTENSION_BY_MEDIA_TYPE = {
    "audio/mpeg": "mp3",
    "audio/pcm": "pcm",
    "audio/basic": "ulaw",
}


def _new_filename(media_type: str) -> str:
    extension = _EXTENSION_BY_MEDIA_TYPE.get(media_type, "bin")
    return f"{uuid.uuid4().hex}.{extension}"


class MediaStorage(Protocol):
    """Structural interface both backends implement — same
    Protocol-injectable pattern used throughout this project so tests
    can inject a fake instead of touching real disk or real S3."""

    def save(self, audio_bytes: bytes, media_type: str) -> str: ...
    def load(self, filename: str) -> bytes | None: ...
    def build_url(self, filename: str) -> str: ...
    def cleanup_expired(self) -> int: ...


class LocalDiskMediaStorage:
    """Phase 9's original implementation. Fine for local development —
    see this module's docstring for why it's not what a real
    deployment should use."""

    def _storage_dir(self) -> Path:
        path = Path(settings.media_storage_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _resolve_safe_path(self, filename: str) -> Path | None:
        """Resolves `filename` against the storage directory and
        rejects anything that would escape it (e.g. `../../etc/passwd`)
        — the filename in GET /media/{filename} comes straight from
        the URL path, so this is a real, not theoretical, check."""
        storage_dir = self._storage_dir().resolve()
        candidate = (storage_dir / filename).resolve()
        if storage_dir not in candidate.parents and candidate != storage_dir:
            return None
        if candidate.parent != storage_dir:
            return None
        return candidate

    def save(self, audio_bytes: bytes, media_type: str) -> str:
        filename = _new_filename(media_type)
        file_path = self._storage_dir() / filename
        file_path.write_bytes(audio_bytes)
        logger.debug("Saved media file %s locally (%d bytes, %s)", filename, len(audio_bytes), media_type)
        return filename

    def load(self, filename: str) -> bytes | None:
        file_path = self._resolve_safe_path(filename)
        if file_path is None or not file_path.is_file():
            return None
        return file_path.read_bytes()

    def build_url(self, filename: str) -> str:
        if not settings.public_base_url:
            raise ValueError(
                "PUBLIC_BASE_URL is not set. Twilio needs a public URL to fetch reply audio from — "
                "set this to your ngrok URL (see README) or real domain."
            )
        base = settings.public_base_url.rstrip("/")
        return f"{base}/media/{filename}"

    def cleanup_expired(self) -> int:
        """Deletes files older than media_file_ttl_seconds. Not
        scheduled automatically — call this periodically yourself, or
        wire it to a scheduler. Returns the number of files removed."""
        cutoff = time.time() - settings.media_file_ttl_seconds
        removed = 0
        for file_path in self._storage_dir().iterdir():
            if file_path.is_file() and file_path.stat().st_mtime < cutoff:
                file_path.unlink()
                removed += 1
        if removed:
            logger.info("Cleaned up %d expired local media file(s)", removed)
        return removed


class S3MediaStorage:
    """Phase 10: S3-compatible cloud storage. Works with real AWS S3
    (leave S3_ENDPOINT_URL empty) or any S3-compatible provider —
    Cloudflare R2, Backblaze B2, MinIO for self-hosting, etc. — by
    setting S3_ENDPOINT_URL to that provider's endpoint.

    `client` is injectable so tests can supply a fake instead of a
    real boto3 client / real AWS credentials."""

    def __init__(self, client: Any = None):
        self._client = client or self._build_client()

    def _build_client(self):
        import boto3

        kwargs: dict = {"region_name": settings.s3_region}
        if settings.aws_access_key_id:
            kwargs["aws_access_key_id"] = settings.aws_access_key_id
            kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
        if settings.s3_endpoint_url:
            kwargs["endpoint_url"] = settings.s3_endpoint_url
        return boto3.client("s3", **kwargs)

    def save(self, audio_bytes: bytes, media_type: str) -> str:
        if not settings.s3_bucket_name:
            raise ValueError("S3_BUCKET_NAME is not set — required when MEDIA_STORAGE_BACKEND=s3.")

        filename = _new_filename(media_type)
        self._client.put_object(
            Bucket=settings.s3_bucket_name,
            Key=filename,
            Body=audio_bytes,
            ContentType=media_type,
        )
        logger.debug("Saved media file %s to S3 (%d bytes, %s)", filename, len(audio_bytes), media_type)
        return filename

    def load(self, filename: str) -> bytes | None:
        try:
            obj = self._client.get_object(Bucket=settings.s3_bucket_name, Key=filename)
        except self._client.exceptions.NoSuchKey:
            return None
        except Exception:
            # botocore raises a generic ClientError for some "not
            # found" cases depending on provider/bucket config, not
            # always the specific NoSuchKey subclass — treat any
            # fetch failure as "not found" rather than crashing the
            # request, matching load_media's documented contract.
            return None
        return obj["Body"].read()

    def build_url(self, filename: str) -> str:
        """Returns the direct public object URL — assumes the bucket
        (or its objects) are configured for public read access. For a
        private bucket, generate presigned URLs instead
        (self._client.generate_presigned_url(...)); not done here to
        keep the default path simple, since Twilio just needs SOME
        URL it can fetch without additional credentials."""
        if not settings.s3_bucket_name:
            raise ValueError("S3_BUCKET_NAME is not set — required when MEDIA_STORAGE_BACKEND=s3.")
        if settings.s3_endpoint_url:
            return f"{settings.s3_endpoint_url.rstrip('/')}/{settings.s3_bucket_name}/{filename}"
        return f"https://{settings.s3_bucket_name}.s3.{settings.s3_region}.amazonaws.com/{filename}"

    def cleanup_expired(self) -> int:
        """Not implemented here — real S3 usage should use a bucket
        lifecycle rule (Object Lifecycle Management) to expire objects
        automatically, which is more reliable than an app-level sweep
        (works even if the app isn't running) and doesn't require
        listing the whole bucket. Configure that in your cloud
        provider's console/IaC instead of relying on this method."""
        return 0


def _get_storage() -> MediaStorage:
    """Constructed fresh on every call (not cached) so tests that
    monkeypatch settings.media_storage_backend between cases always
    get a backend reflecting the current settings — real client
    construction (boto3.client(...)) is a cheap, local, no-network-call
    operation, so there's no real cost to not caching this."""
    if settings.media_storage_backend == "s3":
        return S3MediaStorage()
    return LocalDiskMediaStorage()


def save_media(audio_bytes: bytes, media_type: str) -> str:
    return _get_storage().save(audio_bytes, media_type)


def load_media(filename: str) -> bytes | None:
    return _get_storage().load(filename)


def build_media_url(filename: str) -> str:
    return _get_storage().build_url(filename)


def cleanup_expired_files() -> int:
    return _get_storage().cleanup_expired()
