"""TDD Red-turned-Green: sqlite-vec core — direct vec0 tests."""
import sqlite3, struct, pytest
from pathlib import Path

TEST_DB = Path(__file__).parent / "test_search.db"

def _vec_to_blob(v):
    return struct.pack(f"{len(v)}f", *v)

def _create_vec0(con):
    con.execute("CREATE VIRTUAL TABLE tasks_vectors USING vec0("
                "  project_id INTEGER,"
                "  task_id INTEGER PRIMARY KEY,"
                "  title TEXT,"
                "  description TEXT,"
                "  vector FLOAT[768]"
                ")")

@pytest.fixture(autouse=True)
def clean_db():
    if TEST_DB.exists():
        TEST_DB.unlink()
    yield
    if TEST_DB.exists():
        TEST_DB.unlink()

def _get_con():
    import sqlite_vec
    con = sqlite3.connect(str(TEST_DB))
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    return con


def test_vec0_table_creation():
    con = _get_con()
    _create_vec0(con)
    rows = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tasks_vectors'").fetchall()
    assert len(rows) == 1
    con.close()


def test_insert_and_cosine_search():
    con = _get_con()
    _create_vec0(con)

    v1 = _vec_to_blob([0.1] * 768)
    v2 = _vec_to_blob([0.9] * 768)
    q = _vec_to_blob([0.1] * 768)  # query identical to v1

    con.execute("INSERT INTO tasks_vectors (project_id, task_id, title, description, vector) "
                "VALUES (10, 1, 'Login', 'Login page', ?)", (v1,))
    con.execute("INSERT INTO tasks_vectors (project_id, task_id, title, description, vector) "
                "VALUES (10, 2, 'Logout', 'Logout button', ?)", (v2,))

    rows = con.execute(
        "SELECT task_id, distance FROM tasks_vectors "
        "WHERE project_id = 10 AND vector MATCH ? "
        "ORDER BY distance LIMIT 2", (q,)
    ).fetchall()

    assert len(rows) >= 1
    assert rows[0][0] == 1  # identical vector → task 1 first
    assert rows[0][1] < 0.01  # distance near zero
    con.close()


def test_empty_index_returns_nothing():
    con = _get_con()
    _create_vec0(con)
    q = _vec_to_blob([0.5] * 768)
    rows = con.execute(
        "SELECT task_id FROM tasks_vectors WHERE project_id=10 AND vector MATCH ? LIMIT 5",
        (q,)
    ).fetchall()
    assert len(rows) == 0
    con.close()


def test_persist_across_connections():
    con = _get_con()
    _create_vec0(con)
    v = _vec_to_blob([0.42] * 768)
    con.execute("INSERT INTO tasks_vectors (project_id, task_id, title, description, vector) "
                "VALUES (7, 99, 'Test', 'Persist test', ?)", (v,))
    con.commit()
    con.close()

    con2 = _get_con()
    rows = con2.execute(
        "SELECT task_id FROM tasks_vectors WHERE project_id=7 AND vector MATCH ? LIMIT 5",
        (v,)
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == 99
    con2.close()


def test_invalid_project_id_returns_empty():
    con = _get_con()
    _create_vec0(con)
    v = _vec_to_blob([0.1] * 768)
    con.execute("INSERT INTO tasks_vectors (project_id, task_id, title, description, vector) "
                "VALUES (10, 1, 'Only in 10', 'desc', ?)", (v,))
    con.commit()

    q = _vec_to_blob([0.1] * 768)
    rows = con.execute(
        "SELECT task_id FROM tasks_vectors WHERE project_id=99 AND vector MATCH ? LIMIT 5",
        (q,)
    ).fetchall()
    assert len(rows) == 0
    con.close()
