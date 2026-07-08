"""search_engine.py — SQLite-vec + Ollama embeddings + knowledge graph."""

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional
from urllib.request import Request, urlopen

VECTOR_DIM = 768  # nomic-embed-text

OLLAMA_URL = "http://localhost:11434/api/embeddings"


def _ollama_embed(text: str) -> list[float]:
    """Embed text via Ollama nomic-embed-text. Returns 768-d vector."""
    if not text or not text.strip():
        return [0.0] * VECTOR_DIM
    body = json.dumps({"model": "nomic-embed-text", "prompt": text}).encode()
    req = Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        return data.get("embedding", [0.0] * VECTOR_DIM)
    except Exception:
        return [0.0] * VECTOR_DIM


def _load_vec0(con: sqlite3.Connection):
    """Load the sqlite-vec extension into an already-opened connection."""
    import sqlite_vec

    con.enable_load_extension(True)
    sqlite_vec.load(con)


def _init_schema(con: sqlite3.Connection):
    """Create all required tables if they don't exist."""
    _load_vec0(con)
    con.executescript("""
        CREATE VIRTUAL TABLE IF NOT EXISTS tasks_vectors USING vec0(
            project_id INTEGER,
            task_id INTEGER PRIMARY KEY,
            title TEXT,
            description TEXT,
            vector FLOAT[768]
        );

        CREATE TABLE IF NOT EXISTS graph_edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            source_task_id INTEGER NOT NULL,
            target_task_id INTEGER NOT NULL,
            relation TEXT NOT NULL,
            weight REAL DEFAULT 1.0,
            metadata TEXT DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS project_meta (
            project_id INTEGER PRIMARY KEY,
            task_count INTEGER DEFAULT 0,
            indexed_at TIMESTAMP,
            graph_built INTEGER DEFAULT 0,
            ollama_status TEXT DEFAULT 'unknown'
        );
    """)
    con.commit()


# ponytail: vec0 uses a custom binary format for vectors.
#           We serialize [float] to bytes via struct for INSERT/query args.
def _vec_to_blob(v: list[float]) -> bytes:
    """Serialize a list of floats to the binary format sqlite-vec expects."""
    import struct

    return struct.pack(f"{len(v)}f", *v)


def _blob_to_list(b: bytes) -> list[float]:
    import struct

    return list(struct.unpack(f"{len(b)//4}f", b))


class SearchEngine:
    """Semantic search over project tasks using SQLite-vec + Ollama embeddings."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.db: sqlite3.Connection = sqlite3.connect(db_path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        _init_schema(self.db)

    def embed(self, text: str) -> list[float]:
        """Compute 768-d embedding via Ollama nomic-embed-text.

        Empty string returns a zero vector (no Ollama call).
        """
        return _ollama_embed(text)

    # ── Reindex ──────────────────────────────────────────────────────────
    def reindex_project(self, project_id: int, tasks: list[dict[str, Any]]) -> int:
        """Clear existing index for a project and rebuild from tasks.

        Each task dict must have: id, title, description (optional).
        Returns the number of tasks indexed.
        """
        cur = self.db.cursor()

        # Clear existing data for this project
        cur.execute("DELETE FROM tasks_vectors WHERE project_id = ?", (project_id,))
        cur.execute("DELETE FROM graph_edges WHERE project_id = ?", (project_id,))
        cur.execute("DELETE FROM project_meta WHERE project_id = ?", (project_id,))

        count = 0
        for task in tasks:
            tid = task.get("id")
            title = task.get("title", "")
            desc = task.get("description", "") or ""
            text_for_embed = (title + " " + desc).strip()
            if not text_for_embed:
                continue
            v = _ollama_embed(text_for_embed)
            # vec0 INSERT uses blob for vector column
            cur.execute(
                "INSERT INTO tasks_vectors (project_id, task_id, title, description, vector) "
                "VALUES (?, ?, ?, ?, ?)",
                (project_id, tid, title, desc, _vec_to_blob(v)),
            )
            count += 1

        # Save project meta
        cur.execute(
            "INSERT INTO project_meta (project_id, task_count, indexed_at, ollama_status) "
            "VALUES (?, ?, ?, ?)",
            (project_id, count, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "ok"),
        )
        self.db.commit()
        return count

    # ── Semantic search ──────────────────────────────────────────────────
    def semantic_search(
        self, query: str, project_id: int, top_k: int = 20
    ) -> list[dict[str, Any]]:
        """Search tasks by semantic similarity.

        Returns list of {task_id, title, description, distance}.
        """
        if not query or not query.strip():
            return []
        v = _ollama_embed(query)
        q_blob = _vec_to_blob(v)

        cur = self.db.cursor()
        rows = cur.execute(
            "SELECT task_id, title, description, distance "
            "FROM tasks_vectors "
            "WHERE project_id = ? AND vector MATCH ? "
            "ORDER BY distance "
            "LIMIT ?",
            (project_id, q_blob, top_k),
        ).fetchall()

        return [
            {"task_id": r["task_id"], "title": r["title"], "description": r["description"], "distance": r["distance"]}
            for r in rows
        ]

    # ── Graph ────────────────────────────────────────────────────────────
    def build_graph(self, project_id: int, tasks: list[dict[str, Any]]):
        """Build task-link graph: parent/child + @ref mentions.

        ponytail: no LLM entity extraction in v1 — only structural links.
        Add LLM extraction when index quality needs improvement.
        """
        cur = self.db.cursor()

        edges = 0

        # 1. Parent/child from task.parent_id
        for task in tasks:
            parent_id = task.get("parent_id") or task.get("parent")
            if parent_id:
                cur.execute(
                    "INSERT INTO graph_edges (project_id, source_task_id, target_task_id, relation, weight) "
                    "VALUES (?, ?, ?, 'parent_child', 1.0)",
                    (project_id, parent_id, task["id"]),
                )
                edges += 1

        # 2. Cross-references from description: #NNN
        import re

        ref_re = re.compile(r"#(\d+)")
        for task in tasks:
            desc = task.get("description", "") or ""
            for m in ref_re.finditer(desc):
                ref_id = int(m.group(1))
                if ref_id != task["id"]:
                    cur.execute(
                        "INSERT INTO graph_edges (project_id, source_task_id, target_task_id, relation, weight) "
                        "VALUES (?, ?, ?, 'reference', 0.8)",
                        (project_id, task["id"], ref_id),
                    )
                    edges += 1

        # Update project_meta
        cur.execute(
            "UPDATE project_meta SET graph_built = ? WHERE project_id = ?",
            (edges, project_id),
        )
        self.db.commit()

    def get_graph(self, project_id: int) -> dict[str, Any]:
        """Return {nodes, edges} for a project's knowledge graph."""
        cur = self.db.cursor()

        # Collect unique task IDs referenced in edges
        edge_rows = cur.execute(
            "SELECT * FROM graph_edges WHERE project_id = ?", (project_id,)
        ).fetchall()

        task_ids = set()
        edges = []
        for r in edge_rows:
            task_ids.add(r["source_task_id"])
            task_ids.add(r["target_task_id"])
            edges.append({
                "id": r["id"],
                "source": r["source_task_id"],
                "target": r["target_task_id"],
                "relation": r["relation"],
                "weight": r["weight"],
            })

        # Get node info from tasks_vectors
        nodes = []
        for tid in sorted(task_ids):
            row = cur.execute(
                "SELECT title, description FROM tasks_vectors WHERE task_id = ?", (tid,)
            ).fetchone()
            if row:
                nodes.append({
                    "id": tid,
                    "title": row["title"],
                    "description": row["description"][:80] if row["description"] else "",
                })

        return {"nodes": nodes, "edges": edges}

    def close(self):
        self.db.close()
