"""E1: a write that fails outside an explicit transaction must leave nothing behind.

Python's sqlite3 opens an implicit BEGIN on the first DML; if a write method raised
after inserting rows and nothing rolled back, the next unrelated write's COMMIT used
to publish the partial rows.
"""

import sqlite3

import pytest

from ai_db.errors import AiDbStorageError
from ai_db.storage.models import ChunkRecord, FileRecord
from ai_db.storage.sqlite_backend import SQLiteBackend


@pytest.fixture
def backend(tmp_path):
    b = SQLiteBackend(str(tmp_path / "tx.db"))
    b.initialize()
    yield b
    b.close()


def _count_chunks(db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        return int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
    finally:
        conn.close()


def test_failed_insert_is_rolled_back_not_committed_later(backend):
    backend.upsert_file(FileRecord("/r/a.py", "sha", 1.0, 1, "p"))
    bad = ChunkRecord("/r/a.py", "code", "def a", 1, 2, "def a(): pass", "p")
    bad.parent_index = 5  # out of range: raises after the rows were inserted
    with pytest.raises(AiDbStorageError):
        backend.insert_chunks([bad])

    assert not backend.conn.in_transaction
    backend.set_state("unrelated", "write")  # used to COMMIT the orphan chunk
    assert _count_chunks(backend.db_path) == 0


def test_failed_replace_keeps_previous_chunks(backend):
    backend.upsert_file(FileRecord("/r/a.py", "sha", 1.0, 1, "p"))
    backend.insert_chunks([ChunkRecord("/r/a.py", "code", "def a", 1, 2, "def a(): pass", "p")])
    bad = ChunkRecord("/r/a.py", "code", "def b", 1, 2, "def b(): pass", "p")
    bad.parent_index = 9
    with pytest.raises(AiDbStorageError):
        backend.replace_file_chunks("/r/a.py", [bad])

    backend.set_state("unrelated", "write")
    assert [c.name for c in backend.get_chunks_for_file("/r/a.py")] == ["def a"]


def test_delete_skill_by_file_removes_fts_and_vectors(backend):
    from ai_db.storage.models import SkillRecord

    backend.upsert_skill(SkillRecord("deploy", "ship to prod", "/s/deploy.md", "deploy release",
                                     "sha", 1.0, "deploy the service", "p"))
    backend.upsert_skill_vector("deploy", "p", [1.0, 0.0])
    assert [n for n, _ in backend.search_skills("deploy", ["p"])] == ["deploy"]

    backend.delete_skill(filepath="/s/deploy.md", project="p")  # no name: by file

    assert backend.search_skills("deploy", ["p"]) == []
    assert backend.conn.execute("SELECT COUNT(*) FROM skill_vectors").fetchone()[0] == 0
