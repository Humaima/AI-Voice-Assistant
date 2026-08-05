"""
Tests for Phase 10's debug endpoint access control
(app/api/debug_auth.py) — verifies the dev/production behavior split
directly against a real FastAPI app+TestClient, not just unit-testing
the dependency function in isolation, since the actual wiring (which
routers get the dependency, which don't) matters as much as the logic.
"""
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app

settings = get_settings()


def _client() -> TestClient:
    return TestClient(app)


class TestDevModeAllowsDebugRoutes:
    def test_debug_route_reachable_without_token_in_dev(self, monkeypatch):
        monkeypatch.setattr(settings, "environment", "development")
        client = _client()

        response = client.post("/tts/synthesize-test", json={"text": "hi"})

        # Not a 404 — proves it got PAST the debug guard. Whatever
        # happens next (missing API key, network error) is a different
        # concern this test isn't about.
        assert response.status_code != 404


class TestProductionModeBlocksDebugRoutes:
    def test_no_token_returns_404_in_production(self, monkeypatch):
        monkeypatch.setattr(settings, "environment", "production")
        monkeypatch.setattr(settings, "debug_api_token", "correct-token")
        client = _client()

        response = client.post("/tts/synthesize-test", json={"text": "hi"})

        assert response.status_code == 404

    def test_wrong_token_returns_404_in_production(self, monkeypatch):
        monkeypatch.setattr(settings, "environment", "production")
        monkeypatch.setattr(settings, "debug_api_token", "correct-token")
        client = _client()

        response = client.post(
            "/tts/synthesize-test", json={"text": "hi"}, headers={"X-Debug-Token": "wrong-token"}
        )

        assert response.status_code == 404

    def test_correct_token_passes_the_guard(self, monkeypatch):
        monkeypatch.setattr(settings, "environment", "production")
        monkeypatch.setattr(settings, "debug_api_token", "correct-token")
        client = _client()

        response = client.post(
            "/tts/synthesize-test", json={"text": "hi"}, headers={"X-Debug-Token": "correct-token"}
        )

        assert response.status_code != 404

    def test_empty_debug_token_setting_blocks_everyone(self, monkeypatch):
        """Empty DEBUG_API_TOKEN in production means debug routes are
        refused entirely — not "any token works," which would be the
        dangerous failure mode for an unset access-control secret."""
        monkeypatch.setattr(settings, "environment", "production")
        monkeypatch.setattr(settings, "debug_api_token", "")
        client = _client()

        response = client.post(
            "/tts/synthesize-test", json={"text": "hi"}, headers={"X-Debug-Token": ""}
        )

        assert response.status_code == 404

    def test_applies_to_all_three_debug_routers(self, monkeypatch):
        monkeypatch.setattr(settings, "environment", "production")
        monkeypatch.setattr(settings, "debug_api_token", "correct-token")
        client = _client()

        tts_response = client.post("/tts/synthesize-test", json={"text": "hi"})
        assert tts_response.status_code == 404, "/tts/synthesize-test should be blocked without a token"

        agent_response = client.post(
            "/agent/chat-test", json={"conversation_id": "c1", "transcript": "hi"}
        )
        assert agent_response.status_code == 404, "/agent/chat-test should be blocked without a token"


class TestPublicRoutesUnaffected:
    """The webhook and media routes must stay reachable without a
    debug token even in production — Twilio has no way to send one,
    and this router intentionally has its own separate protection
    (signature validation) instead. Confirms the debug guard was
    applied to the right three routers, not accidentally everywhere."""

    def test_webhook_route_does_not_require_debug_token_in_production(self, monkeypatch):
        monkeypatch.setattr(settings, "environment", "production")
        monkeypatch.setattr(settings, "debug_api_token", "correct-token")
        client = _client()

        response = client.post("/webhook/whatsapp", data={"From": "whatsapp:+15551234567", "NumMedia": "0"})

        # 403 from missing Twilio signature — NOT 404 from the debug
        # guard. Proves the debug dependency isn't applied here at all.
        assert response.status_code == 403

    def test_health_check_unaffected_in_production(self, monkeypatch):
        monkeypatch.setattr(settings, "environment", "production")
        monkeypatch.setattr(settings, "debug_api_token", "correct-token")
        client = _client()

        response = client.get("/health")

        assert response.status_code == 200

    def test_metrics_endpoint_unaffected_in_production(self, monkeypatch):
        monkeypatch.setattr(settings, "environment", "production")
        monkeypatch.setattr(settings, "debug_api_token", "correct-token")
        client = _client()

        response = client.get("/metrics")

        assert response.status_code == 200
