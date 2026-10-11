import os
import time
from typing import Any

from ai_db.constants import CONTEXT_RECALL_CANDIDATES, RRF_K
from ai_db.ext.recall import RecallProvider
from ai_db.storage.models import ContextRecord
from ai_db.utils import get_allowed_projects


class ContextMemory:
    def __init__(self, db: Any = None, conn: Any = None,
                 recall: list[RecallProvider] | None = None):
        self.db = db if db is not None else conn
        self.cross_project: dict[str, list[str]] = {}
        # Recall extensions (recall.providers), set by VectorDB; empty = BM25 only.
        self.recall: list[RecallProvider] = list(recall or [])

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
        for provider in self.recall:
            provider.index_context(context_id, title_str, summary)

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
        """Search saved session contexts: BM25, fused (RRF) with every recall provider."""
        if not query_text or not query_text.strip():
            return []

        allowed = get_allowed_projects(project or "global", allowed_projects, self.cross_project)
        if not self.recall:
            hits: list[dict[str, Any]] = self.db.search_contexts(
                query_text, allowed_projects=allowed, top_k=top_k)
            return hits

        ranked: list[tuple[str, list[dict[str, Any]]]] = [
            ("bm25", self.db.search_contexts(query_text, allowed_projects=allowed,
                                             top_k=CONTEXT_RECALL_CANDIDATES))]
        for provider in self.recall:
            ranked.append((provider.name, [
                {"session_id": h.session_id, "project": h.project, "title": h.title,
                 "summary": h.summary, "timestamp": h.timestamp}
                for h in provider.rank_contexts(query_text, allowed, CONTEXT_RECALL_CANDIDATES)]))
        fused: dict[tuple[str, str], dict[str, Any]] = {}
        for source, hits in ranked:
            for rank, hit in enumerate(hits, 1):
                key = (hit["session_id"], hit["project"])
                row = fused.setdefault(key, {**hit, "score": 0.0, "match": []})
                row["score"] += 1.0 / (RRF_K + rank)
                row["match"].append(source)
        results = sorted(fused.values(), key=lambda r: r["score"], reverse=True)[:top_k]
        for r in results:
            r["score"] = round(r["score"], 4)
        return results

    def delete_context(
        self, session_id: str, project: str | None = None
    ) -> bool:
        """Delete a context by session_id and project.

        Returns True if the context was deleted, False if not found.
        """
        if not session_id:
            raise ValueError("session_id is required")

        return bool(self.db.delete_context(session_id=session_id, project=project or "global"))
