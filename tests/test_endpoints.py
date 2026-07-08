"""TDD Red→Green: FastAPI endpoints — /search, /reindex, /graph."""
import pytest
from fastapi.testclient import TestClient

from tui_app import app, API_KEY

# FastAPI TestClient runs lifespan automatically
client = TestClient(app)


def test_health_still_works():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data
    assert data["search_engine_ready"] is True


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
    r = client.post(
        "/reindex",
        json={"project_id": 10},
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    assert r.status_code == 200


def test_reindex_returns_task_count():
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
    # First reindex so there's data
    client.post(
        "/reindex",
        json={"project_id": 10},
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    r = client.get("/graph?project_id=10")
    data = r.json()
    assert "nodes" in data
    assert "edges" in data
