import time
from typing import List, Dict, Any, Optional
from ai_db.utils import get_allowed_projects, tokenize
from ai_db.storage.models import ContextRecord


class ContextMemory:
    def __init__(self, db: Any = None, conn: Any = None):
        self.db = db if db is not None else conn

    def save_context(
        self, session_id: str, summary: str, project: Optional[str] = None,
        title: Optional[str] = None, active_files: Optional[List[str]] = None,
        open_tasks: Optional[List[str]] = None, full_notes: Optional[str] = None
    ) -> Dict[str, Any]:
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
        self, session_id: Optional[str] = None, project: Optional[str] = None,
        allowed_projects: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Retrieves the latest or specified session context for the allowed project scope."""
        allowed = get_allowed_projects(project or "global", allowed_projects)
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
        self, project: Optional[str] = None, allowed_projects: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Lists saved contexts for the project scope."""
        allowed = get_allowed_projects(project or "global", allowed_projects)
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
        self, query_text: str, project: Optional[str] = None,
        allowed_projects: Optional[List[str]] = None, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Searches across saved session contexts using BM25."""
        tokens = tokenize(query_text)
        if not tokens:
            return []

        allowed = get_allowed_projects(project or "global", allowed_projects)
        return self.db.search_contexts(tokens, allowed_projects=allowed, top_k=top_k)
