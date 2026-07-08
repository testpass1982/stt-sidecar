"""TDD Red: knowledge graph operations — build_graph + get_graph.
SearchEngine.graph_edges exists, but build_graph/get_graph aren't tested."""
import pytest
from stt_sidecar.search_engine import SearchEngine


@pytest.fixture
def engine():
    e = SearchEngine(":memory:")
    yield e
    e.close()


@pytest.fixture
def tasks_with_parents():
    """Tasks with parent/child relationships and #ref mentions."""
    return [
        {"id": 1, "title": "Project setup", "description": "Initialize repo and CI pipeline"},
        {"id": 2, "title": "Login API", "description": "JWT auth endpoint. See #1 for setup"},
        {"id": 3, "title": "Registration API", "description": "User registration. References #2"},
        {"id": 4, "title": "Dashboard", "description": "Main dashboard UI. Depends on #2 and #3"},
        {"id": 5, "title": "Logout", "description": "Session cleanup. Related to #2"},
    ]


class TestBuildGraph:
    def test_build_graph_creates_parent_edges(self, engine, tasks_with_parents):
        engine.reindex_project(10, tasks_with_parents)
        engine.build_graph(10, tasks_with_parents)

        edges = engine.db.execute(
            "SELECT COUNT(*) as c FROM graph_edges WHERE project_id=10"
        ).fetchone()
        assert edges["c"] > 0

    def test_build_graph_creates_ref_edges(self, engine, tasks_with_parents):
        engine.reindex_project(10, tasks_with_parents)
        engine.build_graph(10, tasks_with_parents)

        ref_edges = engine.db.execute(
            "SELECT * FROM graph_edges WHERE project_id=10 AND relation='reference'"
        ).fetchall()
        assert len(ref_edges) >= 1
        # #1 mentioned in task #2 description
        refs = {(r["source_task_id"], r["target_task_id"]) for r in ref_edges}
        assert (2, 1) in refs, f"task #2 references #1, edges={refs}"

    def test_build_graph_updates_meta(self, engine, tasks_with_parents):
        engine.reindex_project(10, tasks_with_parents)
        engine.build_graph(10, tasks_with_parents)

        meta = engine.db.execute(
            "SELECT graph_built FROM project_meta WHERE project_id=10"
        ).fetchone()
        assert meta["graph_built"] > 0

    def test_build_graph_no_parents_no_crash(self, engine):
        tasks = [{"id": 1, "title": "Lone task", "description": "No relations"}]
        engine.reindex_project(10, tasks)
        # Should not crash
        engine.build_graph(10, tasks)

    def test_build_graph_empty_tasks(self, engine):
        engine.reindex_project(10, [])
        engine.build_graph(10, [])
        c = engine.db.execute("SELECT COUNT(*) as c FROM graph_edges WHERE project_id=10").fetchone()
        assert c["c"] == 0


class TestGetGraph:
    def test_get_graph_returns_nodes_and_edges(self, engine, tasks_with_parents):
        engine.reindex_project(10, tasks_with_parents)
        engine.build_graph(10, tasks_with_parents)

        g = engine.get_graph(10)
        assert "nodes" in g
        assert "edges" in g
        assert len(g["nodes"]) >= 1
        assert len(g["edges"]) >= 1

    def test_get_graph_returns_node_title(self, engine, tasks_with_parents):
        engine.reindex_project(10, tasks_with_parents)
        engine.build_graph(10, tasks_with_parents)

        g = engine.get_graph(10)
        titles = {n["title"] for n in g["nodes"]}
        assert "Login API" in titles or "Project setup" in titles

    def test_get_graph_no_data_returns_empty(self, engine):
        g = engine.get_graph(999)
        assert g == {"nodes": [], "edges": []}

    def test_get_graph_edge_has_relation(self, engine, tasks_with_parents):
        engine.reindex_project(10, tasks_with_parents)
        engine.build_graph(10, tasks_with_parents)

        g = engine.get_graph(10)
        relations = {e["relation"] for e in g["edges"]}
        assert "reference" in relations
