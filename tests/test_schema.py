"""TDD Red: schema creation via SearchEngine — should fail, no SearchEngine yet."""
import pytest


def test_schema_creates_all_tables():
    """SearchEngine.__init__ creates tasks_vectors, graph_edges, project_meta."""
    from stt_sidecar.search_engine import SearchEngine
    import sqlite3

    engine = SearchEngine(":memory:")
    db = engine.db if hasattr(engine, 'db') else sqlite3.connect(":memory:")

    tables = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    names = {r[0] for r in tables}

    assert "tasks_vectors" in names, "missing tasks_vectors virtual table"
    assert "graph_edges" in names, "missing graph_edges table"
    assert "project_meta" in names, "missing project_meta table"


def test_graph_edges_schema():
    """graph_edges has all required columns."""
    from stt_sidecar.search_engine import SearchEngine

    engine = SearchEngine(":memory:")
    cols = engine.db.execute("PRAGMA table_info('graph_edges')").fetchall()
    col_names = {c[1] for c in cols}

    for required in ("project_id", "source_task_id", "target_task_id", "relation", "weight", "metadata"):
        assert required in col_names, f"missing column: {required}"


def test_project_meta_schema():
    """project_meta has all required columns."""
    from stt_sidecar.search_engine import SearchEngine

    engine = SearchEngine(":memory:")
    cols = engine.db.execute("PRAGMA table_info('project_meta')").fetchall()
    col_names = {c[1] for c in cols}

    for required in ("project_id", "task_count", "indexed_at", "graph_built", "ollama_status"):
        assert required in col_names, f"missing column: {required}"
