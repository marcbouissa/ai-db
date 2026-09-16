import json
import time
import zlib
import sqlite3
from typing import List, Dict, Any, Optional
from ai_db.utils import get_allowed_projects, tokenize

class ContextMemory:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def save_context(self, session_id: str, summary: str, project: Optional[str] = None,
                     title: Optional[str] = None, active_files: Optional[List[str]] = None,
                     open_tasks: Optional[List[str]] = None, full_notes: Optional[str] = None) -> Dict[str, Any]:
        """Saves or updates a chat session context snapshot for a project."""
        if not project:
            project = "global"

        cur = self.conn.cursor()
        now = time.time()
        files_json = json.dumps(active_files or [])
        tasks_json = json.dumps(open_tasks or [])
        title_str = title or f"Session {session_id}"
        notes_str = full_notes or summary

        # Compress full text blob
        zblob = zlib.compress(notes_str.encode("utf-8"), level=9)

        # Upsert into contexts
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
            """,
            (session_id, project, title_str, summary, files_json, tasks_json, now, zblob)
        )

        # Update FTS index
        cur.execute("DELETE FROM fts_contexts WHERE session_id = ? AND project = ?", (session_id, project))
        cur.execute(
            """INSERT INTO fts_contexts (session_id, project, title, summary, content)
               VALUES (?, ?, ?, ?, ?)""",
            (session_id, project, title_str, summary, notes_str)
        )

        self.conn.commit()
        return {
            "session_id": session_id,
            "project": project,
            "title": title_str,
            "summary": summary,
            "active_files": active_files or [],
            "open_tasks": open_tasks or [],
            "timestamp": now
        }

    def get_context(self, session_id: Optional[str] = None, project: Optional[str] = None,
                    allowed_projects: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """Retrieves the latest or specified session context for the allowed project scope."""
        allowed = get_allowed_projects(project or "global", allowed_projects)
        placeholders = ",".join("?" for _ in allowed)

        cur = self.conn.cursor()
        if session_id:
            cur.execute(
                f"""SELECT session_id, project, title, summary, active_files, open_tasks, timestamp, zcontent
                   FROM contexts
                   WHERE session_id = ? AND project IN ({placeholders})
                   ORDER BY timestamp DESC LIMIT 1""",
                [session_id] + allowed
            )
        else:
            # Pick most recently saved context in project scope
            cur.execute(
                f"""SELECT session_id, project, title, summary, active_files, open_tasks, timestamp, zcontent
                   FROM contexts
                   WHERE project IN ({placeholders})
                   ORDER BY timestamp DESC LIMIT 1""",
                allowed
            )

        row = cur.fetchone()
        if not row:
            return None

        try:
            full_notes = zlib.decompress(row["zcontent"]).decode("utf-8", errors="replace")
        except Exception:
            full_notes = ""

        try:
            files = json.loads(row["active_files"]) if row["active_files"] else []
        except Exception:
            files = []

        try:
            tasks = json.loads(row["open_tasks"]) if row["open_tasks"] else []
        except Exception:
            tasks = []

        return {
            "session_id": row["session_id"],
            "project": row["project"],
            "title": row["title"],
            "summary": row["summary"],
            "active_files": files,
            "open_tasks": tasks,
            "timestamp": row["timestamp"],
            "full_notes": full_notes
        }

    def list_contexts(self, project: Optional[str] = None, allowed_projects: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Lists saved contexts for the project scope."""
        allowed = get_allowed_projects(project or "global", allowed_projects)
        placeholders = ",".join("?" for _ in allowed)

        cur = self.conn.cursor()
        cur.execute(
            f"""SELECT session_id, project, title, summary, active_files, open_tasks, timestamp
               FROM contexts
               WHERE project IN ({placeholders})
               ORDER BY timestamp DESC""",
            allowed
        )
        results = []
        for r in cur.fetchall():
            try:
                files = json.loads(r["active_files"]) if r["active_files"] else []
                tasks = json.loads(r["open_tasks"]) if r["open_tasks"] else []
            except Exception:
                files, tasks = [], []

            results.append({
                "session_id": r["session_id"],
                "project": r["project"],
                "title": r["title"],
                "summary": r["summary"],
                "active_files_count": len(files),
                "open_tasks_count": len(tasks),
                "timestamp": r["timestamp"]
            })
        return results

    def query_contexts(self, query_text: str, project: Optional[str] = None,
                       allowed_projects: Optional[List[str]] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        """Searches across saved session contexts using BM25."""
        tokens = tokenize(query_text)
        if not tokens:
            return []

        allowed = get_allowed_projects(project or "global", allowed_projects)
        placeholders = ",".join("?" for _ in allowed)
        fts_query = " OR ".join(tokens)

        cur = self.conn.cursor()
        try:
            cur.execute(
                f"""
                SELECT session_id, project, title, summary, bm25(fts_contexts) as rank
                FROM fts_contexts
                WHERE fts_contexts MATCH ? AND project IN ({placeholders})
                ORDER BY rank
                LIMIT ?
                """,
                [fts_query] + allowed + [top_k]
            )
            rows = cur.fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []
