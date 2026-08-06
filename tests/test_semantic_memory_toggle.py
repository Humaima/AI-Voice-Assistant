"""
Tests for the ENABLE_SEMANTIC_MEMORY toggle — lets a deployment run
with Postgres-only memory and skip ChromaDB entirely. Added when a
real deployment scenario came up: some platforms (e.g. Render's free
tier) charge extra for running a second Docker service, which
ChromaDB requires.

get_agent_service() is a module-level singleton, so these tests reset
it via monkeypatch before each call — otherwise a cached instance from
an earlier test (built under different settings) would leak in.
"""
import app.services.agent_service as agent_service_module
from app.core.config import get_settings
from app.services.agent_service import get_agent_service

settings = get_settings()


def _reset_singleton(monkeypatch):
    monkeypatch.setattr(agent_service_module, "_agent_service_singleton", None)


class TestSemanticMemoryToggle:
    def test_disabled_skips_chromadb_entirely(self, monkeypatch):
        _reset_singleton(monkeypatch)
        monkeypatch.setattr(settings, "enable_semantic_memory", False)
        monkeypatch.setattr(settings, "groq_api_key", "fake_key_for_test")

        service = get_agent_service()

        assert service.vector_store is None
        # Postgres memory should still be configured — only the
        # ChromaDB piece is skipped.
        assert service.memory_store is not None

    def test_enabled_calls_build_chroma_client(self, monkeypatch):
        """chromadb.HttpClient connects eagerly at construction (no
        live Chroma server available in this test environment), so
        this confirms build_chroma_client gets called when enabled —
        the actual connection behavior is ChromaVectorStore's own
        concern, covered in tests/test_vector_store.py."""
        _reset_singleton(monkeypatch)
        monkeypatch.setattr(settings, "enable_semantic_memory", True)
        monkeypatch.setattr(settings, "groq_api_key", "fake_key_for_test")

        called = {"value": False}

        class _FakeChromaClient:
            def get_or_create_collection(self, **kwargs):
                return object()

        def _fake_build_chroma_client():
            called["value"] = True
            return _FakeChromaClient()

        monkeypatch.setattr(agent_service_module, "build_chroma_client", _fake_build_chroma_client)

        service = get_agent_service()

        assert called["value"] is True
        assert service.vector_store is not None

    def test_disabled_never_calls_build_chroma_client(self, monkeypatch):
        """Stronger check than just asserting vector_store is None —
        confirms the ChromaDB client constructor is never even
        invoked, so a deployment without ChromaDB reachable won't hit
        any connection attempt at all, not even a lazy/deferred one."""
        _reset_singleton(monkeypatch)
        monkeypatch.setattr(settings, "enable_semantic_memory", False)
        monkeypatch.setattr(settings, "groq_api_key", "fake_key_for_test")

        called = {"value": False}

        def _fail_if_called():
            called["value"] = True
            raise AssertionError("build_chroma_client should not be called when disabled")

        monkeypatch.setattr(agent_service_module, "build_chroma_client", _fail_if_called)

        get_agent_service()

        assert called["value"] is False
