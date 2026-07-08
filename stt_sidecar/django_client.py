"""django_client.py — fetch tasks from webworx Django API for indexing."""

from typing import Any
import httpx

PAGE_SIZE = 500
TIMEOUT_SEC = 30


def fetch_project_tasks(
    base_url: str,
    project_id: int,
    api_key: str,
) -> list[dict[str, Any]]:
    """Fetch all tasks for a project from the Django REST API.

    Returns a list of dicts compatible with SearchEngine.reindex_project():
      {id, title, description, parent_id, labels, comments}
    """
    headers = {
        "X-STT-Api-Key": api_key,
        "Accept": "application/json",
    }
    url = f"{base_url.rstrip('/')}/api/v1/tasks/"
    params = {"project": project_id, "page_size": PAGE_SIZE}

    all_tasks: list[dict[str, Any]] = []
    is_first = True

    with httpx.Client(timeout=TIMEOUT_SEC) as client:
        while url:
            req_params = params if is_first else {}
            is_first = False
            resp = client.get(url, headers=headers, params=req_params)
            # ponytail: params only on first request; subsequent URLs include params

            if resp.status_code == 401 or resp.status_code == 403:
                raise PermissionError(
                    f"Django API returned {resp.status_code} for project {project_id}. "
                    f"Check STT_API_KEY."
                )
            if resp.status_code == 404:
                raise ValueError(f"Project {project_id} not found at {url}")
            if not resp.is_success:
                raise ConnectionError(
                    f"Django API returned {resp.status_code} for {url}: {resp.text[:200]}"
                )

            data = resp.json()
            results = data.get("results", [])

            for task in results:
                all_tasks.append({
                    "id": task["id"],
                    "title": task.get("title", ""),
                    "description": task.get("description", "") or "",
                    "parent_id": task.get("parent"),  # None if no parent
                })

            # Pagination: follow `next` link
            url = data.get("next")
            params = {}  # subsequent URLs are fully qualified

    return all_tasks
