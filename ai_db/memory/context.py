import os
import time
from typing import Any

from ai_db.logger import _logger
from ai_db.storage.models import ContextRecord
from ai_db.utils import get_allowed_projects


class ContextMemory:
    def __init__(self, db: Any = None, conn: Any = None, embedder: Any = None):
        self.db = db if db is not None else conn
        self.cross_project: dict[str, list[str]] = {}
        # Set by VectorDB in hybrid mode; None (lexical) means no vectors.
        self.embedder = embedder

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
        context_id = self.db.save_context(record)
        self._embed_context(record, context_id)

        return {
            "session_id": session_id,
            "project": project,
            "title": title_str,
            "summary": summary,
            "active_files": active_files or [],
            "open_tasks": open_tasks or [],
            "timestamp": now
        }

    def _embed_context(self, record: ContextRecord, context_id: int) -> None:
        """Store the context embedding (hybrid mode only).

        Text is ``title + summary`` per spec. Failures are swallowed on purpose:
        a missing vector must never stop a context from being saved, and the
        lexical path stays fully functional without one.
        """
        if self.embedder is None or not context_id:
            return
        try:
            text = f"{record.title or ''}\n{record.summary}".strip()
            if not text:
                return
            self.db.upsert_context_vector(context_id, self.embedder.embed_documents([text])[0])
        except Exception as exc:  # noqa: BLE001 - vectorisation is best-effort
            _logger.debug("context embedding skipped for %s: %s", record.session_id, exc)

    def get_context(
        self, session_id: str | None = None, project: str | None = None,
        allowed_projects: list[str] | None = None
    ) -> dict[str, Any] | None:
        """Retrieves the latest or specified session context for the allowed project scope.

        Includes:
        - saved_at: human-readable age of the context
        - active_files with "missing" flag for files that no longer exist
        """
        allowed = get_allowed_projects(project or "global", allowed_projects, self.cross_project)
        record = self.db.get_context(session_id=session_id, allowed_projects=allowed)
        if not record:
            return None

        # Calculate human-readable age
        now = time.time()
        age_seconds = now - record.timestamp
        if age_seconds < 60:
            saved_at = f"{int(age_seconds)}s ago"
        elif age_seconds < 3600:
            saved_at = f"{int(age_seconds // 60)}m ago"
        elif age_seconds < 86400:
            saved_at = f"{int(age_seconds // 3600)}h ago"
        else:
            saved_at = f"{int(age_seconds // 86400)}d ago"

        # Flag missing files
        active_files_with_status = []
        for fpath in record.active_files:
            exists = os.path.exists(fpath)
            active_files_with_status.append({
                "path": fpath,
                "missing": not exists
            })

        return {
            "session_id": record.session_id,
            "project": record.project,
            "title": record.title,
            "summary": record.summary,
            "active_files": active_files_with_status,
            "open_tasks": list(record.open_tasks),
            "timestamp": record.timestamp,
            "saved_at": saved_at,
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
        if not query_text or not query_text.strip():
            return []

        allowed = get_allowed_projects(project or "global", allowed_projects, self.cross_project)
        hits: list[dict[str, Any]] = self.db.search_contexts(query_text, allowed_projects=allowed, top_k=top_k)
        return hits

    def delete_context(
        self, session_id: str, project: str | None = None
    ) -> bool:
        """Delete a context by session_id and project.

        Returns True if the context was deleted, False if not found.
        """
        if not session_id:
            raise ValueError("session_id is required")

        return self.db.delete_context(session_id=session_id, project=project or "global")
