"""TDD Red→Green: FastAPI endpoints — /search, /reindex, /graph."""
import httpx
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from tui_app import app, API_KEY

client = TestClient(app)

# shared mock for django_client calls
MOCK_EMPTY = httpx.Response(200, json={"count": 0, "next": None, "results": []})
DC_PATH = "stt_sidecar.django_client.httpx.Client.get"


def test_health_still_works():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data
    assert data["search_engine_ready"] is True
    assert "ollama_available" in data
    assert "embed_model" in data
    assert "vec_db_initialized" in data


def test_search_endpoint_exists():
    r = client.get("/search?q=authentication&project_id=10")
    assert r.status_code == 200


def test_search_returns_json_with_results():
    r = client.get("/search?q=login&project_id=10")
    data = r.json()
    assert "results" in data
    assert isinstance(data["results"], list)


def test_search_missing_params():
    r = client.get("/search")
    assert r.status_code == 422


def test_reindex_requires_auth():
    r = client.post("/reindex", json={"project_id": 10})
    assert r.status_code == 401


def test_reindex_with_valid_auth():
    with patch(DC_PATH, return_value=MOCK_EMPTY):
        r = client.post(
            "/reindex",
            json={"project_id": 10},
            headers={"Authorization": f"Bearer {API_KEY}"},
        )
        assert r.status_code == 200


def test_reindex_returns_task_count():
    with patch(DC_PATH, return_value=MOCK_EMPTY):
        r = client.post(
            "/reindex",
            json={"project_id": 10},
            headers={"Authorization": f"Bearer {API_KEY}"},
        )
        data = r.json()
        assert "task_count" in data
        assert isinstance(data["task_count"], int)


def test_reindex_missing_project_id():
    r = client.post(
        "/reindex",
        json={},
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    assert r.status_code == 422


def test_graph_endpoint_exists():
    r = client.get("/graph?project_id=10")
    assert r.status_code == 200


def test_graph_returns_nodes_and_edges():
    with patch(DC_PATH, return_value=MOCK_EMPTY):
        client.post(
            "/reindex",
            json={"project_id": 10},
            headers={"Authorization": f"Bearer {API_KEY}"},
        )
    r = client.get("/graph?project_id=10")
    data = r.json()
    assert "nodes" in data
    assert "edges" in data
