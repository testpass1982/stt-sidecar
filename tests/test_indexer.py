"""TDD Red: SearchEngine.reindex_project + semantic_search integration.
SearchEngine exists, but reindex_project with full indexing flow isn't tested yet."""
import pytest
from stt_sidecar.search_engine import SearchEngine


@pytest.fixture
def engine():
    e = SearchEngine(":memory:")
    yield e
    e.close()


@pytest.fixture
def sample_tasks():
    return [
        {"id": 1, "title": "Login page", "description": "User authentication via email and password"},
        {"id": 2, "title": "Registration", "description": "New user signup with email verification"},
        {"id": 3, "title": "Password reset", "description": "Forgot password flow with email token"},
        {"id": 4, "title": "Dashboard", "description": "Main dashboard showing project stats"},
        {"id": 5, "title": "Logout", "description": "Clear session and redirect to login"},
        {"id": 6, "title": "Billing", "description": "Monthly subscription payment processing"},
    ]


class TestReindex:
    def test_reindex_returns_task_count(self, engine, sample_tasks):
        count = engine.reindex_project(10, sample_tasks)
        assert count == 6

    def test_reindex_idempotent(self, engine, sample_tasks):
        engine.reindex_project(10, sample_tasks)
        count2 = engine.reindex_project(10, sample_tasks)
        assert count2 == 6  # same count after re-index

    def test_metadata_after_reindex(self, engine, sample_tasks):
        engine.reindex_project(10, sample_tasks)
        row = engine.db.execute(
            "SELECT task_count, graph_built FROM project_meta WHERE project_id = 10"
        ).fetchone()
        assert row is not None
        assert row["task_count"] == 6

    def test_empty_tasks_list(self, engine):
        count = engine.reindex_project(42, [])
        assert count == 0

    def test_projects_are_separate(self, engine, sample_tasks):
        engine.reindex_project(10, sample_tasks)
        engine.reindex_project(20, [{"id": 99, "title": "Other", "description": "different project"}])

        r10 = engine.db.execute("SELECT COUNT(*) as c FROM tasks_vectors WHERE project_id=10").fetchone()
        r20 = engine.db.execute("SELECT COUNT(*) as c FROM tasks_vectors WHERE project_id=20").fetchone()
        assert r10["c"] == 6
        assert r20["c"] == 1


class TestSemanticSearch:
    def test_search_finds_relevant_task(self, engine, sample_tasks):
        engine.reindex_project(10, sample_tasks)
        results = engine.semantic_search("authentication", 10, top_k=5)
        assert len(results) >= 1
        # "login" and "authentication" should be semantically close
        titles = {r["title"] for r in results}
        assert "Login page" in titles, f"expected 'Login page' in {titles}"

    def test_search_empty_query_returns_empty(self, engine, sample_tasks):
        engine.reindex_project(10, sample_tasks)
        results = engine.semantic_search("", 10)
        assert results == []

    def test_search_no_match(self, engine, sample_tasks):
        engine.reindex_project(10, sample_tasks)
        results = engine.semantic_search("zzzzyyyyyyxxxxx", 10, top_k=3)
        # Still returns something — everything is distant but something is closest
        assert isinstance(results, list)

    def test_search_wrong_project_returns_empty(self, engine, sample_tasks):
        engine.reindex_project(10, sample_tasks)
        results = engine.semantic_search("authentication", 999, top_k=5)
        assert len(results) == 0

    def test_search_returns_distance(self, engine, sample_tasks):
        engine.reindex_project(10, sample_tasks)
        results = engine.semantic_search("dashboard charts", 10, top_k=3)
        for r in results:
            assert "task_id" in r
            assert "title" in r
            assert "distance" in r
            assert isinstance(r["distance"], float)

    def test_russian_search(self, engine, sample_tasks):
        engine.reindex_project(10, sample_tasks)
        results = engine.semantic_search("вход в систему", 10, top_k=5)
        titles = {r["title"] for r in results}
        assert "Login page" in titles or "Logout" in titles, f"russian 'вход' should find login/logout, got {titles}"
