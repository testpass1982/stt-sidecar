"""TDD Green: CORS headers on all endpoints."""
from fastapi.testclient import TestClient
from tui_app import app, API_KEY
from unittest.mock import patch
import httpx

client = TestClient(app)

def _mock_resp(**overrides):
    defaults = dict(status_code=200, json_data={"count": 0, "next": None, "results": []})
    defaults.update(overrides)
    return httpx.Response(defaults["status_code"], json=defaults["json_data"])

DC_PATH = "stt_sidecar.django_client.httpx.Client.get"
ORIGIN = "http://127.0.0.1:8000"


def test_cors_preflight_reindex():
    """OPTIONS preflight from a different origin gets 200 + CORS headers."""
    r = client.options("/reindex", headers={
        "origin": ORIGIN,
        "access-control-request-method": "POST",
    })
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == ORIGIN


def test_cors_preflight_health():
    r = client.options("/health", headers={
        "origin": ORIGIN,
        "access-control-request-method": "GET",
    })
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == ORIGIN


def test_cors_get_health():
    r = client.get("/health", headers={"origin": ORIGIN})
    assert r.headers.get("access-control-allow-origin") == ORIGIN


def test_cors_get_search():
    r = client.get("/search?q=test&project_id=1", headers={"origin": ORIGIN})
    assert r.headers.get("access-control-allow-origin") == ORIGIN


def test_cors_post_reindex():
    with patch(DC_PATH, return_value=_mock_resp()):
        r = client.post(
            "/reindex",
            json={"project_id": 10},
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "origin": ORIGIN,
            },
        )
        assert r.status_code == 200
        assert r.headers.get("access-control-allow-origin") == ORIGIN


def test_cors_post_reindex_unauth():
    r = client.post(
        "/reindex",
        json={"project_id": 10},
        headers={"origin": ORIGIN},
    )
    # 401 is still returned, but CORS headers must be present
    assert r.status_code == 401
    assert r.headers.get("access-control-allow-origin") == ORIGIN


def test_cors_get_graph():
    r = client.get("/graph?project_id=10", headers={"origin": ORIGIN})
    assert r.headers.get("access-control-allow-origin") == ORIGIN


def test_cors_preflight_allows_post():
    r = client.options("/reindex", headers={
        "origin": ORIGIN,
        "access-control-request-method": "POST",
    })
    methods = r.headers.get("access-control-allow-methods", "")
    assert "POST" in methods


def test_cors_preflight_allow_credentials():
    r = client.options("/reindex", headers={
        "origin": ORIGIN,
        "access-control-request-method": "POST",
    })
    assert r.headers.get("access-control-allow-credentials") == "true"
