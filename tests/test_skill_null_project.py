"""Legacy skills rows with a NULL project are folded into 'global' when the schema opens."""

from __future__ import annotations

from ai_db.storage.sqlite_backend import SQLiteBackend


def _skill(conn, project):
    conn.execute("INSERT INTO skills (name, description, filepath, triggers, sha256, last_modified, project) "
                 "VALUES (?, 'd', '/s/SKILL.md', '', 'x', 0, ?)", (f"s-{project}", project))


def test_null_project_skills_are_normalised(tmp_path):
    db = str(tmp_path / "k.db")
    b = SQLiteBackend(db)
    b.initialize()
    conn = b.conn
    _skill(conn, "global")
    conn.execute("INSERT INTO skills (name, description, filepath, triggers, sha256, last_modified, project) "
                 "VALUES ('s-global', 'd', '/s/SKILL.md', '', 'x', 0, NULL)")  # duplicate of the global row
    conn.execute("INSERT INTO skills (name, description, filepath, triggers, sha256, last_modified, project) "
                 "VALUES ('lonely', 'd', '/l/SKILL.md', '', 'x', 0, NULL)")
    conn.commit()
    b.close()
    b = SQLiteBackend(db)
    b.initialize()
    rows = sorted(tuple(r) for r in b.conn.execute("SELECT name, project FROM skills"))
    assert rows == [("lonely", "global"), ("s-global", "global")]
    assert {s.project for s in b.skills_without_vectors()} == {"global"}
    b.close()
