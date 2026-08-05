"""
Tests for Phase 9's media storage — the local-disk save/serve
mechanism for generated reply audio.
"""
import time

import pytest

from app.core.config import get_settings
from app.services.media_store import (
    S3MediaStorage,
    build_media_url,
    cleanup_expired_files,
    load_media,
    save_media,
)

settings = get_settings()


@pytest.fixture
def temp_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "media_storage_dir", str(tmp_path))
    return tmp_path


class TestSaveAndLoadMedia:
    def test_save_then_load_roundtrips_exact_bytes(self, temp_storage):
        filename = save_media(b"fake mp3 bytes", "audio/mpeg")
        assert load_media(filename) == b"fake mp3 bytes"

    def test_save_uses_correct_extension_for_known_media_types(self, temp_storage):
        assert save_media(b"x", "audio/mpeg").endswith(".mp3")
        assert save_media(b"x", "audio/pcm").endswith(".pcm")
        assert save_media(b"x", "audio/basic").endswith(".ulaw")

    def test_save_falls_back_to_bin_extension_for_unknown_type(self, temp_storage):
        filename = save_media(b"x", "application/unknown")
        assert filename.endswith(".bin")

    def test_each_save_gets_a_unique_filename(self, temp_storage):
        first = save_media(b"content one", "audio/mpeg")
        second = save_media(b"content two", "audio/mpeg")
        assert first != second

    def test_load_nonexistent_file_returns_none(self, temp_storage):
        assert load_media("does-not-exist.mp3") is None


class TestPathTraversalProtection:
    """The filename in GET /media/{filename} comes straight from the
    URL path — these aren't theoretical attacks, they're what an
    actual malicious or malformed request would look like."""

    def test_rejects_parent_directory_traversal(self, temp_storage):
        assert load_media("../../etc/passwd") is None

    def test_rejects_absolute_path(self, temp_storage):
        assert load_media("/etc/passwd") is None

    def test_rejects_nested_traversal_attempt(self, temp_storage):
        assert load_media("subdir/../../secret.txt") is None

    def test_legitimate_filename_still_works(self, temp_storage):
        filename = save_media(b"real content", "audio/mpeg")
        assert load_media(filename) == b"real content"


class TestBuildMediaUrl:
    def test_builds_correct_url_with_base_url_configured(self, monkeypatch):
        monkeypatch.setattr(settings, "public_base_url", "https://example.ngrok-free.app")
        url = build_media_url("abc123.mp3")
        assert url == "https://example.ngrok-free.app/media/abc123.mp3"

    def test_strips_trailing_slash_from_base_url(self, monkeypatch):
        monkeypatch.setattr(settings, "public_base_url", "https://example.ngrok-free.app/")
        url = build_media_url("abc123.mp3")
        assert url == "https://example.ngrok-free.app/media/abc123.mp3"

    def test_raises_clear_error_when_base_url_not_configured(self, monkeypatch):
        monkeypatch.setattr(settings, "public_base_url", "")
        with pytest.raises(ValueError, match="PUBLIC_BASE_URL"):
            build_media_url("abc123.mp3")


class TestCleanupExpiredFiles:
    def test_removes_files_older_than_ttl(self, temp_storage, monkeypatch):
        monkeypatch.setattr(settings, "media_file_ttl_seconds", 0)
        filename = save_media(b"old content", "audio/mpeg")
        time.sleep(0.05)  # ensure mtime is measurably in the past relative to a 0s TTL

        removed = cleanup_expired_files()

        assert removed == 1
        assert load_media(filename) is None

    def test_keeps_files_within_ttl(self, temp_storage, monkeypatch):
        monkeypatch.setattr(settings, "media_file_ttl_seconds", 3600)
        filename = save_media(b"fresh content", "audio/mpeg")

        removed = cleanup_expired_files()

        assert removed == 0
        assert load_media(filename) == b"fresh content"

    def test_returns_zero_for_empty_directory(self, temp_storage):
        assert cleanup_expired_files() == 0


class _FakeNoSuchKey(Exception):
    pass


class _FakeS3Exceptions:
    NoSuchKey = _FakeNoSuchKey


class FakeS3Client:
    """Stand-in for a boto3 S3 client — records put_object calls and
    serves get_object from an in-memory dict, so S3MediaStorage (the
    real class, not a fake standing in for it) is tested end-to-end
    without real AWS credentials or network access."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.put_calls: list[dict] = []
        self.exceptions = _FakeS3Exceptions()

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, ContentType: str):  # noqa: N803
        self.put_calls.append({"Bucket": Bucket, "Key": Key, "ContentType": ContentType})
        self.objects[Key] = Body

    def get_object(self, *, Bucket: str, Key: str):  # noqa: N803
        if Key not in self.objects:
            raise self.exceptions.NoSuchKey(f"Key not found: {Key}")
        return {"Body": _FakeBody(self.objects[Key])}


class _FakeBody:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


class TestS3MediaStorage:
    def test_save_then_load_roundtrips_exact_bytes(self, monkeypatch):
        monkeypatch.setattr(settings, "s3_bucket_name", "test-bucket")
        client = FakeS3Client()
        storage = S3MediaStorage(client=client)

        filename = storage.save(b"fake mp3 bytes", "audio/mpeg")

        assert storage.load(filename) == b"fake mp3 bytes"

    def test_save_uses_correct_extension(self, monkeypatch):
        monkeypatch.setattr(settings, "s3_bucket_name", "test-bucket")
        storage = S3MediaStorage(client=FakeS3Client())

        assert storage.save(b"x", "audio/mpeg").endswith(".mp3")

    def test_save_sends_correct_bucket_and_content_type(self, monkeypatch):
        monkeypatch.setattr(settings, "s3_bucket_name", "test-bucket")
        client = FakeS3Client()
        storage = S3MediaStorage(client=client)

        storage.save(b"x", "audio/mpeg")

        assert client.put_calls[0]["Bucket"] == "test-bucket"
        assert client.put_calls[0]["ContentType"] == "audio/mpeg"

    def test_save_without_bucket_configured_raises_clear_error(self, monkeypatch):
        monkeypatch.setattr(settings, "s3_bucket_name", "")
        storage = S3MediaStorage(client=FakeS3Client())

        with pytest.raises(ValueError, match="S3_BUCKET_NAME"):
            storage.save(b"x", "audio/mpeg")

    def test_load_nonexistent_key_returns_none(self, monkeypatch):
        monkeypatch.setattr(settings, "s3_bucket_name", "test-bucket")
        storage = S3MediaStorage(client=FakeS3Client())

        assert storage.load("nonexistent.mp3") is None

    def test_build_url_with_real_aws(self, monkeypatch):
        monkeypatch.setattr(settings, "s3_bucket_name", "my-bucket")
        monkeypatch.setattr(settings, "s3_region", "eu-west-1")
        monkeypatch.setattr(settings, "s3_endpoint_url", "")
        storage = S3MediaStorage(client=FakeS3Client())

        url = storage.build_url("abc123.mp3")

        assert url == "https://my-bucket.s3.eu-west-1.amazonaws.com/abc123.mp3"

    def test_build_url_with_s3_compatible_endpoint(self, monkeypatch):
        monkeypatch.setattr(settings, "s3_bucket_name", "my-bucket")
        monkeypatch.setattr(settings, "s3_endpoint_url", "https://abc123.r2.cloudflarestorage.com")
        storage = S3MediaStorage(client=FakeS3Client())

        url = storage.build_url("abc123.mp3")

        assert url == "https://abc123.r2.cloudflarestorage.com/my-bucket/abc123.mp3"

    def test_build_url_without_bucket_configured_raises_clear_error(self, monkeypatch):
        monkeypatch.setattr(settings, "s3_bucket_name", "")
        storage = S3MediaStorage(client=FakeS3Client())

        with pytest.raises(ValueError, match="S3_BUCKET_NAME"):
            storage.build_url("abc123.mp3")

    def test_cleanup_expired_returns_zero_not_implemented(self, monkeypatch):
        """S3 cleanup is deliberately delegated to bucket lifecycle
        rules rather than implemented here — see the class docstring."""
        storage = S3MediaStorage(client=FakeS3Client())
        assert storage.cleanup_expired() == 0


class TestBackendSelection:
    def test_local_backend_selected_by_default(self, temp_storage):
        filename = save_media(b"local content", "audio/mpeg")
        assert load_media(filename) == b"local content"

    def test_s3_backend_selected_when_configured(self, monkeypatch):
        """Confirms the free-function API (save_media/load_media)
        actually dispatches to S3MediaStorage when
        media_storage_backend is set to "s3" — not just that
        S3MediaStorage works in isolation."""
        monkeypatch.setattr(settings, "media_storage_backend", "s3")
        monkeypatch.setattr(settings, "s3_bucket_name", "test-bucket")

        fake_client = FakeS3Client()
        monkeypatch.setattr(
            "app.services.media_store.S3MediaStorage._build_client", lambda self: fake_client
        )

        filename = save_media(b"s3 content", "audio/mpeg")

        assert fake_client.objects[filename] == b"s3 content"
        assert load_media(filename) == b"s3 content"
