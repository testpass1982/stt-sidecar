"""TDD Red→Green: django_client.py — fetch tasks from Django API."""
import pytest
from unittest.mock import patch, MagicMock
import httpx


@pytest.fixture
def mock_empty():
    """Mock httpx.Client.get to return empty results."""
    resp = httpx.Response(200, json={"count": 0, "next": None, "results": []})
    with patch.object(httpx.Client, "get", return_value=resp) as mock:
        yield mock


def test_fetch_returns_list_of_tasks(mock_empty):
    """fetch_project_tasks returns a list."""
    from stt_sidecar.django_client import fetch_project_tasks
    tasks = fetch_project_tasks("https://tasks.webworx.ru", 10, "test-key")
    assert isinstance(tasks, list)


def test_fetch_empty_project_returns_empty_list(mock_empty):
    from stt_sidecar.django_client import fetch_project_tasks
    tasks = fetch_project_tasks("https://tasks.webworx.ru", 99999, "test-key")
    assert tasks == []


def test_each_task_has_required_fields():
    """Every task returned has id, title, description."""
    resp = httpx.Response(200, json={
        "count": 2,
        "next": None,
        "results": [
            {"id": 1, "title": "Login", "description": "Auth page"},
            {"id": 2, "title": "Logout", "description": ""},
        ],
    })
    with patch.object(httpx.Client, "get", return_value=resp):
        from stt_sidecar.django_client import fetch_project_tasks
        tasks = fetch_project_tasks("https://tasks.webworx.ru", 10, "key")
        for t in tasks:
            assert "id" in t
            assert "title" in t
            assert "description" in t


def test_fetch_sends_correct_headers():
    """Request includes X-STT-Api-Key header."""
    resp = httpx.Response(200, json={"count": 0, "next": None, "results": []})

    with patch.object(httpx.Client, "get", return_value=resp) as mock_get:
        from stt_sidecar.django_client import fetch_project_tasks
        fetch_project_tasks("https://tasks.webworx.ru", 10, "my-secret-key")

        _, kwargs = mock_get.call_args
        headers = kwargs.get("headers", {})
        assert headers.get("X-STT-Api-Key") == "my-secret-key", \
            f"got headers: {headers}"


def test_fetch_handles_pagination():
    """Follows next page links and merges results."""
    call_count = [0]

    def mock_get(url, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return httpx.Response(200, json={
                "count": 3,
                "next": "http://test/api/v1/tasks/?page=2&project=10",
                "results": [
                    {"id": 1, "title": "Task 1", "description": "First"},
                    {"id": 2, "title": "Task 2", "description": "Second"},
                ],
            })
        else:
            return httpx.Response(200, json={
                "count": 3,
                "next": None,
                "results": [
                    {"id": 3, "title": "Task 3", "description": "Third"},
                ],
            })

    with patch.object(httpx.Client, "get", side_effect=mock_get):
        from stt_sidecar.django_client import fetch_project_tasks
        tasks = fetch_project_tasks("http://test", 10, "key")
        assert len(tasks) == 3
        assert call_count[0] == 2


def test_fetch_includes_parent_id():
    """Task dict includes parent_id when available."""
    resp = httpx.Response(200, json={
        "count": 1,
        "next": None,
        "results": [
            {"id": 2, "title": "Child", "description": "desc", "parent": 1},
        ],
    })
    with patch.object(httpx.Client, "get", return_value=resp):
        from stt_sidecar.django_client import fetch_project_tasks
        tasks = fetch_project_tasks("http://test", 10, "key")
        assert tasks[0]["parent_id"] == 1


def test_fetch_403_raises_permission_error():
    """403 response raises PermissionError."""
    resp = httpx.Response(403, json={"detail": "Invalid API key"})
    with patch.object(httpx.Client, "get", return_value=resp):
        from stt_sidecar.django_client import fetch_project_tasks
        with pytest.raises(PermissionError):
            fetch_project_tasks("http://test", 10, "bad-key")


def test_fetch_404_raises_value_error():
    """404 response raises ValueError."""
    resp = httpx.Response(404, json={"detail": "Not found"})
    with patch.object(httpx.Client, "get", return_value=resp):
        from stt_sidecar.django_client import fetch_project_tasks
        with pytest.raises(ValueError):
            fetch_project_tasks("http://test", 999, "key")


def test_fetch_500_raises_connection_error():
    """500 response raises ConnectionError."""
    resp = httpx.Response(500, json={"detail": "Server error"})
    with patch.object(httpx.Client, "get", return_value=resp):
        from stt_sidecar.django_client import fetch_project_tasks
        with pytest.raises(ConnectionError):
            fetch_project_tasks("http://test", 10, "key")


def test_fetch_handles_null_description():
    """A null description becomes empty string (not 'null')."""
    resp = httpx.Response(200, json={
        "count": 1,
        "next": None,
        "results": [
            {"id": 5, "title": "No desc", "description": None},
        ],
    })
    with patch.object(httpx.Client, "get", return_value=resp):
        from stt_sidecar.django_client import fetch_project_tasks
        tasks = fetch_project_tasks("http://test", 10, "key")
        assert tasks[0]["description"] == ""