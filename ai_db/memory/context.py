import time
from typing import Any

from ai_db.storage.models import ContextRecord
from ai_db.utils import get_allowed_projects, tokenize


class ContextMemory:
    def __init__(self, db: Any = None, conn: Any = None):
        self.db = db if db is not None else conn
        self.cross_project: dict[str, list[str]] = {}

    def save_context(
        self, session_id: str, summary: str, project: str | None = None,
        title: str | None = None, active_files: list[str] | None = None,
        open_tasks: list[str] | None = None, full_notes: str | None = None
    ) -> dict[str, Any]:
        """Saves or updates a chat session context snapshot for a project."""
        if not project:
            project = "global"

        now = time.time()
        title_str = title or f"Session {session_id}"
        notes_str = full_notes or summary

        record = ContextRecord(
            session_id=session_id,
            project=project,
            title=title_str,
            summary=summary,
            active_files=active_files or [],
            open_tasks=open_tasks or [],
            timestamp=now,
            full_notes=notes_str
        )
        self.db.save_context(record)

        return {
            "session_id": session_id,
            "project": project,
            "title": title_str,
            "summary": summary,
            "active_files": active_files or [],
            "open_tasks": open_tasks or [],
            "timestamp": now
        }

    def get_context(
        self, session_id: str | None = None, project: str | None = None,
        allowed_projects: list[str] | None = None
    ) -> dict[str, Any] | None:
        """Retrieves the latest or specified session context for the allowed project scope."""
        allowed = get_allowed_projects(project or "global", allowed_projects, self.cross_project)
        record = self.db.get_context(session_id=session_id, allowed_projects=allowed)
        if not record:
            return None

        return {
            "session_id": record.session_id,
            "project": record.project,
            "title": record.title,
            "summary": record.summary,
            "active_files": list(record.active_files),
            "open_tasks": list(record.open_tasks),
            "timestamp": record.timestamp,
            "full_notes": record.full_notes
        }

    def list_contexts(
        self, project: str | None = None, allowed_projects: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Lists saved contexts for the project scope."""
        allowed = get_allowed_projects(project or "global", allowed_projects, self.cross_project)
        raw_list = self.db.list_contexts(allowed_projects=allowed)
        results = []
        for r in raw_list:
            af = r.get("active_files") or []
            ot = r.get("open_tasks") or []
            results.append({
                "session_id": r["session_id"],
                "project": r["project"],
                "title": r.get("title") or "",
                "summary": r["summary"],
                "active_files": af,
                "open_tasks": ot,
                "active_files_count": len(af),
                "open_tasks_count": len(ot),
                "timestamp": r["timestamp"]
            })
        return results

    def query_contexts(
        self, query_text: str, project: str | None = None,
        allowed_projects: list[str] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        """Searches across saved session contexts using BM25."""
        tokens = tokenize(query_text)
        if not tokens:
            return []

        allowed = get_allowed_projects(project or "global", allowed_projects, self.cross_project)
        hits: list[dict[str, Any]] = self.db.search_contexts(tokens, allowed_projects=allowed, top_k=top_k)
        return hits
