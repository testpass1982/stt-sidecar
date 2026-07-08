"""Shared fixtures for TDD phase."""
import pytest
import sqlite3
from pathlib import Path

TEST_DB = Path(__file__).parent / "test_search.db"


@pytest.fixture
def fresh_db():
    """Provide a clean sqlite3 connection with sqlite-vec loaded."""
    import sqlite_vec
    if TEST_DB.exists():
        TEST_DB.unlink()
    con = sqlite3.connect(str(TEST_DB))
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    yield con
    con.close()
    if TEST_DB.exists():
        TEST_DB.unlink()
