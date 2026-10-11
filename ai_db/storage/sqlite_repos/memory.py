"""Skills, saved contexts, and their recall vectors.

A mixin of :class:`ai_db.storage.sqlite_backend.SQLiteBackend`: every repository shares
the backend's connection and ``transaction()`` (see ``_base.SQLiteRepository``)."""

from __future__ import annotations

import json
import sqlite3
import zlib
from typing import Any

from ai_db.constants import (
    CONTEXT_FTS_MAX_CHARS,
    SKILL_CONTENT_MAX_CHARS,
    ZLIB_LEVEL,
)
from ai_db.errors import AiDbQueryError
from ai_db.search.query_builder import (
    base_terms,
    build_fts,
    expand_terms,
)
from ai_db.storage.models import (
    ContextRecord,
    SkillRecord,
)
from ai_db.storage.sqlite_repos._base import (
    SQLiteRepository,
    _decompress,
    _loads,
    _writes,
    cosine_distance,
    pack_vector,
    unpack_vector,
)


class MemoryRepository(SQLiteRepository):
    def get_skills(self, allowed_projects: list[str] | None = None) -> list[SkillRecord]:
        self._check_closed()
        if allowed_projects is not None and len(allowed_projects) == 0:
            return []
        cur = self.conn.cursor()
        if allowed_projects is not None:
            placeholders = ",".join("?" for _ in allowed_projects)
            cur.execute(
                f"SELECT name, description, filepath, triggers, sha256, last_modified, project FROM skills WHERE project IN ({placeholders})",
                list(allowed_projects)
            )
        else:
            cur.execute("SELECT name, description, filepath, triggers, sha256, last_modified, project FROM skills")
        return [
            SkillRecord(
                name=r["name"],
                description=r["description"],
                filepath=r["filepath"],
                triggers=r["triggers"],
                sha256=r["sha256"],
                last_modified=r["last_modified"],
                project=r["project"]
            )
            for r in cur.fetchall()
        ]

    def get_skills_by_project(self, project: str) -> dict[str, tuple[str, str]]:
        self._check_closed()
        cur = self.conn.cursor()
        cur.execute("SELECT filepath, name, sha256 FROM skills WHERE project = ?", (project,))
        return {r["filepath"]: (r["name"], r["sha256"]) for r in cur.fetchall()}

    @_writes
    def upsert_skill(self, skill: SkillRecord) -> None:
        self._check_closed()
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO skills (name, description, filepath, triggers, sha256, last_modified, project)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name, project) DO UPDATE SET
                description = excluded.description,
                filepath = excluded.filepath,
                triggers = excluded.triggers,
                sha256 = excluded.sha256,
                last_modified = excluded.last_modified
            """,
            (skill.name, skill.description, skill.filepath, skill.triggers, skill.sha256, skill.last_modified, skill.project)
        )
        cur.execute("DELETE FROM fts_skills WHERE name = ? AND project = ?", (skill.name, skill.project))
        cur.execute(
            "INSERT INTO fts_skills (name, description, triggers, content, project) VALUES (?, ?, ?, ?, ?)",
            (skill.name, skill.description, skill.triggers, skill.content[:SKILL_CONTENT_MAX_CHARS], skill.project)
        )

    @_writes
    def delete_skill(self, filepath: str, project: str, name: str | None = None) -> None:
        self._check_closed()
        cur = self.conn.cursor()
        if name:
            names = [name]
        else:
            names = [r["name"] for r in cur.execute(
                "SELECT name FROM skills WHERE filepath = ? AND project = ?", (filepath, project))]
        # A skill lives in three tables; deleting only the row left FTS hits for skills
        # that no longer exist.
        for n in names:
            cur.execute("DELETE FROM skills WHERE name = ? AND project = ?", (n, project))
            cur.execute("DELETE FROM fts_skills WHERE name = ? AND project = ?", (n, project))
            cur.execute("DELETE FROM skill_vectors WHERE name = ? AND project = ?", (n, project))

    def search_skills(
        self,
        query: str,
        allowed_projects: list[str] | None = None,
        limit: int = 20,
    ) -> list[tuple[str, float]]:
        self._check_closed()
        if allowed_projects is not None and len(allowed_projects) == 0:
            return []
        terms = expand_terms(query)
        if not terms:
            return []
        cur = self.conn.cursor()
        fts_query = build_fts(terms, base_terms(query))

        clauses = ["fts_skills MATCH ?"]
        params: list[Any] = [fts_query]

        if allowed_projects is not None:
            placeholders = ",".join("?" for _ in allowed_projects)
            clauses.append(f"project IN ({placeholders})")
            params.extend(allowed_projects)

        where = " AND ".join(clauses)
        params.append(limit)

        try:
            cur.execute(
                f"""
                SELECT name, project, bm25(fts_skills) as rank
                FROM fts_skills
                WHERE {where}
                ORDER BY rank
                LIMIT ?
                """,
                params
            )
        except sqlite3.OperationalError as exc:
            raise AiDbQueryError(fts_query, exc) from exc
        return [(r["name"], -float(r["rank"])) for r in cur.fetchall()]

    def sync_skills(
        self,
        skill_dirs: list[str] | None = None,
        project: str = "global",
        verbose: bool = True,
    ) -> dict[str, int]:
        """Discover and index skills from skill directories into the skills table."""
        from ai_db.search.skills import SkillRouter

        router = SkillRouter(db=self, db_path=self.db_path)
        return router.sync_skills(skill_dirs=skill_dirs, project=project, verbose=verbose)

    @_writes
    def save_context(self, context: ContextRecord) -> int:
        self._check_closed()
        cur = self.conn.cursor()
        zcontent = zlib.compress(context.full_notes.encode("utf-8"), level=ZLIB_LEVEL)
        active_json = json.dumps(context.active_files)
        tasks_json = json.dumps(context.open_tasks)

        cur.execute(
            """
            INSERT INTO contexts (session_id, project, title, summary, active_files, open_tasks, timestamp, zcontent)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, project) DO UPDATE SET
                title = excluded.title,
                summary = excluded.summary,
                active_files = excluded.active_files,
                open_tasks = excluded.open_tasks,
                timestamp = excluded.timestamp,
                zcontent = excluded.zcontent
            RETURNING id
            """,
            (context.session_id, context.project, context.title, context.summary, active_json, tasks_json, context.timestamp, zcontent)
        )
        row = cur.fetchone()
        context_id = int(row["id"]) if row is not None else 0

        cur.execute("DELETE FROM fts_contexts WHERE session_id = ? AND project = ?", (context.session_id, context.project))
        cur.execute(
            "INSERT INTO fts_contexts (session_id, project, title, summary, content) VALUES (?, ?, ?, ?, ?)",
            (context.session_id, context.project, context.title or "", context.summary, context.full_notes[:CONTEXT_FTS_MAX_CHARS])
        )
        return context_id

    def get_context(
        self,
        session_id: str | None = None,
        allowed_projects: list[str] | None = None,
    ) -> ContextRecord | None:
        self._check_closed()
        if allowed_projects is not None and len(allowed_projects) == 0:
            return None
        cur = self.conn.cursor()
        clauses = []
        params: list[Any] = []

        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)

        if allowed_projects is not None:
            placeholders = ",".join("?" for _ in allowed_projects)
            clauses.append(f"project IN ({placeholders})")
            params.extend(allowed_projects)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        order_limit = "" if session_id else "ORDER BY timestamp DESC LIMIT 1"

        cur.execute(
            f"SELECT session_id, project, title, summary, active_files, open_tasks, timestamp, zcontent FROM contexts {where} {order_limit}",
            params
        )
        row = cur.fetchone()
        if not row:
            return None
        full_notes = _decompress(row["zcontent"], f"context {row['session_id']}")
        active_files = _loads(row["active_files"], "context.active_files") if row["active_files"] else []
        open_tasks = _loads(row["open_tasks"], "context.open_tasks") if row["open_tasks"] else []

        return ContextRecord(
            session_id=row["session_id"],
            project=row["project"],
            title=row["title"] or "",
            summary=row["summary"],
            active_files=active_files,
            open_tasks=open_tasks,
            timestamp=float(row["timestamp"]),
            full_notes=full_notes
        )

    def list_contexts(self, allowed_projects: list[str] | None = None) -> list[dict[str, Any]]:
        self._check_closed()
        if allowed_projects is not None and len(allowed_projects) == 0:
            return []
        cur = self.conn.cursor()
        clauses = []
        params: list[Any] = []

        if allowed_projects is not None:
            placeholders = ",".join("?" for _ in allowed_projects)
            clauses.append(f"project IN ({placeholders})")
            params.extend(allowed_projects)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        cur.execute(
            f"SELECT session_id, project, title, summary, timestamp, active_files, open_tasks FROM contexts {where} ORDER BY timestamp DESC",
            params
        )
        results = []
        for r in cur.fetchall():
            af = _loads(r["active_files"], "context.active_files") if r["active_files"] else []
            ot = _loads(r["open_tasks"], "context.open_tasks") if r["open_tasks"] else []
            results.append({
                "session_id": r["session_id"],
                "project": r["project"],
                "title": r["title"] or "",
                "summary": r["summary"],
                "timestamp": r["timestamp"],
                "active_files": af,
                "open_tasks": ot
            })
        return results

    def search_contexts(
        self,
        query: str,
        allowed_projects: list[str] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        self._check_closed()
        if allowed_projects is not None and len(allowed_projects) == 0:
            return []
        terms = expand_terms(query)
        if not terms:
            return []
        cur = self.conn.cursor()
        fts_query = build_fts(terms, base_terms(query))

        clauses = ["fts_contexts MATCH ?"]
        params: list[Any] = [fts_query]

        if allowed_projects is not None:
            placeholders = ",".join("?" for _ in allowed_projects)
            clauses.append(f"fts_contexts.project IN ({placeholders})")
            params.extend(allowed_projects)

        where = " AND ".join(clauses)
        params.append(top_k)

        try:
            cur.execute(
                f"""
                SELECT fts_contexts.session_id, fts_contexts.project, fts_contexts.title,
                       fts_contexts.summary, bm25(fts_contexts) as rank, contexts.timestamp
                FROM fts_contexts
                JOIN contexts ON fts_contexts.session_id = contexts.session_id AND fts_contexts.project = contexts.project
                WHERE {where}
                ORDER BY rank
                LIMIT ?
                """,
                params
            )
        except sqlite3.OperationalError as exc:
            raise AiDbQueryError(fts_query, exc) from exc
        return [
            {
                "session_id": r["session_id"],
                "project": r["project"],
                "title": r["title"],
                "summary": r["summary"],
                "score": round(abs(float(r["rank"])), 3),
                "timestamp": r["timestamp"]
            }
            for r in cur.fetchall()
        ]

    @_writes
    def delete_context(
        self,
        session_id: str,
        project: str | None = None,
    ) -> bool:
        """Delete a context record by session_id and project.

        Returns True if a record was deleted, False if no record matched.
        """
        self._check_closed()
        if not session_id:
            raise ValueError("session_id is required for delete_context")

        if project is None:
            project = "global"

        cur = self.conn.cursor()
        # Delete from both contexts and fts_contexts
        cur.execute(
            "DELETE FROM contexts WHERE session_id = ? AND project = ?",
            (session_id, project)
        )
        deleted_rows = cur.rowcount
        cur.execute(
            "DELETE FROM fts_contexts WHERE session_id = ? AND project = ?",
            (session_id, project)
        )
        # Also delete associated context vectors if they exist
        cur.execute(
            """DELETE FROM context_vectors
               WHERE context_id NOT IN (SELECT id FROM contexts)"""
        )
        return deleted_rows > 0

    @_writes
    def upsert_skill_vector(self, name: str, project: str, embedding: list[float]) -> None:
        """Store (or replace) one skill embedding."""
        self._check_closed()
        self.conn.execute(
            "INSERT OR REPLACE INTO skill_vectors (name, project, embedding) VALUES (?, ?, ?)",
            (name, project, pack_vector(embedding)),
        )

    @_writes
    def delete_skill_vectors(self, name: str, project: str) -> None:
        """Drop the embedding for a removed skill."""
        self._check_closed()
        self.conn.execute(
            "DELETE FROM skill_vectors WHERE name = ? AND project = ?", (name, project))

    @_writes
    def upsert_context_vector(self, context_id: int, embedding: list[float]) -> None:
        """Store (or replace) one context embedding, keyed by contexts.id."""
        self._check_closed()
        self.conn.execute(
            "INSERT OR REPLACE INTO context_vectors (context_id, embedding) VALUES (?, ?)",
            (context_id, pack_vector(embedding)),
        )

    def search_skills_vector(
        self,
        vector: list[float],
        allowed_projects: list[str] | None = None,
        limit: int = 20,
    ) -> list[tuple[str, str, float]]:
        """Skills by cosine distance to ``vector``: ``(name, project, distance)``, closest first.

        Brute force: a user has tens of skills, so an ANN index would cost more
        than it saves and would pin the vector width at DDL time.
        """
        self._check_closed()
        if allowed_projects is not None and len(allowed_projects) == 0:
            return []
        sql = "SELECT name, project, embedding FROM skill_vectors"
        params: list[Any] = []
        if allowed_projects is not None:
            sql += f" WHERE project IN ({','.join('?' * len(allowed_projects))})"
            params.extend(allowed_projects)
        rows = self.conn.execute(sql, params).fetchall()
        scored = [
            (r["name"], r["project"], cosine_distance(vector, unpack_vector(r["embedding"])))
            for r in rows
        ]
        scored.sort(key=lambda t: t[2])
        return [(n, p, round(d, 4)) for n, p, d in scored[:limit]]

    def skills_without_vectors(self) -> list[SkillRecord]:
        self._check_closed()
        rows = self.conn.execute(
            "SELECT s.name, s.description, s.filepath, s.triggers, s.sha256, s.last_modified, s.project "
            "FROM skills s LEFT JOIN skill_vectors v ON v.name = s.name AND v.project = s.project "
            "WHERE v.name IS NULL").fetchall()
        return [SkillRecord(name=r["name"], description=r["description"], filepath=r["filepath"],
                            triggers=r["triggers"], sha256=r["sha256"],
                            last_modified=r["last_modified"], project=r["project"]) for r in rows]

    def contexts_without_vectors(self) -> list[tuple[int, str, str]]:
        self._check_closed()
        rows = self.conn.execute(
            "SELECT c.id, c.title, c.summary FROM contexts c "
            "LEFT JOIN context_vectors v ON v.context_id = c.id WHERE v.context_id IS NULL").fetchall()
        return [(int(r["id"]), r["title"] or "", r["summary"]) for r in rows]

    @_writes
    def clear_recall_vectors(self) -> None:
        self._check_closed()
        self.conn.execute("DELETE FROM skill_vectors")
        self.conn.execute("DELETE FROM context_vectors")

    def search_contexts_vector(
        self,
        vector: list[float],
        allowed_projects: list[str] | None = None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Contexts by cosine distance to ``vector`` (closest first)."""
        self._check_closed()
        if allowed_projects is not None and len(allowed_projects) == 0:
            return []
        sql = """
            SELECT c.session_id, c.project, c.title, c.summary, c.timestamp, v.embedding
            FROM context_vectors v
            JOIN contexts c ON c.id = v.context_id
        """
        params: list[Any] = []
        if allowed_projects is not None:
            sql += f" WHERE c.project IN ({','.join('?' * len(allowed_projects))})"
            params.extend(allowed_projects)
        rows = self.conn.execute(sql, params).fetchall()
        scored = [
            {
                "session_id": r["session_id"],
                "project": r["project"],
                "title": r["title"],
                "summary": r["summary"],
                "distance": round(cosine_distance(vector, unpack_vector(r["embedding"])), 4),
                "timestamp": r["timestamp"],
            }
            for r in rows
        ]
        scored.sort(key=lambda d: d["distance"])
        return scored[:top_k]
