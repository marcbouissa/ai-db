import os
from typing import List, Dict, Any, Optional
from ai_db.constants import CANDIDATE_POOL
from ai_db.search.ranking import Ranker
from ai_db.search.retriever import LexicalRetriever, Retriever
from ai_db.utils import get_allowed_projects


class QueryEngine:
    def __init__(self, db: Any = None, conn: Any = None):
        self.db = db if db is not None else conn
        self.cross_project: Dict[str, List[str]] = {}
        # Replaced once by VectorDB according to retrieval.mode.
        self.retriever: Retriever = LexicalRetriever(self.db)
        self.ranker = Ranker(self.db)

    def query(
        self, search_text: str, top_k: int = 5, relative_to: Optional[str] = None,
        project: Optional[str] = None, allowed_projects: Optional[List[str]] = None,
        languages: Optional[List[str]] = None, chunk_types: Optional[List[str]] = None,
        modified_since: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        allowed = get_allowed_projects(project or "global", allowed_projects, self.cross_project)
        filters = {"allowed_projects": allowed, "languages": languages,
                   "chunk_types": chunk_types, "modified_since": modified_since}
        search_results = self.search(search_text, filters, top_k)

        results = []
        for r in search_results:
            path_display = r.filepath
            if relative_to:
                try:
                    path_display = os.path.relpath(path_display, relative_to)
                except Exception:
                    pass

            snippet = getattr(r, "snippet", "") or ""
            score = round(float(getattr(r, "score", 1.0)), 8)
            chunk_id = getattr(r, "chunk_id", None) or getattr(r, "id", None)

            results.append({
                "chunk_id": chunk_id,
                "file": path_display,
                "abs_path": r.filepath,
                "name": r.name,
                "qualified_name": getattr(r, "qualified_name", "") or r.name,
                "language": getattr(r, "language", ""),
                "type": r.chunk_type,
                "project": r.project,
                "lines": f"L{r.start_line}-{r.end_line}",
                "score": score,
                "snippet": snippet[:280].strip() + ("..." if len(snippet) > 280 else "")
            })

        return results[:top_k]

    def search(self, text: str, filters: Dict[str, Any], top_k: int) -> List[Any]:
        """Ranked SearchResults for ``text`` (retriever candidates, then final ranking)."""
        pool = max(top_k, CANDIDATE_POOL)
        candidates = self.retriever.candidates(text, filters, pool)
        return self.ranker.rank(text, candidates, filters.get("allowed_projects"))[:top_k]

    def query_symbol(
        self, name: str, relative_to: Optional[str] = None,
        project: Optional[str] = None, allowed_projects: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        allowed = get_allowed_projects(project or "global", allowed_projects, self.cross_project)
        symbols = self.db.query_symbols(name=name, allowed_projects=allowed, limit=50)

        results = []
        for s in symbols:
            path_display = s.filepath
            if relative_to:
                try:
                    path_display = os.path.relpath(path_display, relative_to)
                except Exception:
                    pass
            results.append({
                "name": s.name,
                "symbol_type": s.symbol_type,
                "file": path_display,
                "abs_path": s.filepath,
                "line": s.line,
                "signature": s.signature,
                "project": s.project
            })
        return results

    def check_syntax(
        self, target_path: Optional[str] = None, relative_to: Optional[str] = None,
        project: Optional[str] = None, allowed_projects: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        allowed = get_allowed_projects(project or "global", allowed_projects, self.cross_project)
        errors = self.db.get_syntax_errors(target_path=target_path, allowed_projects=allowed)

        results = []
        for err in errors:
            path_display = err.filepath
            if relative_to:
                try:
                    path_display = os.path.relpath(path_display, relative_to)
                except Exception:
                    pass
            results.append({
                "file": path_display,
                "abs_path": err.filepath,
                "line": err.line,
                "col": err.col,
                "message": err.message,
                "project": err.project
            })
        return results

    def query_callers(
        self, symbol_name: str, relative_to: Optional[str] = None,
        project: Optional[str] = None,
        allowed_projects: Optional[List[str]] = None,
        top_k: int = 100
    ) -> List[Dict[str, Any]]:
        """F2: Find all call sites, imports, and inheritance refs to a given symbol name."""
        allowed = get_allowed_projects(project or "global", allowed_projects, self.cross_project)
        callers = self.db.query_symbol_callers(callee_name=symbol_name, allowed_projects=allowed, limit=top_k)

        results = []
        for r in callers:
            path = os.path.relpath(r.caller_filepath, relative_to) if relative_to else r.caller_filepath
            results.append({
                "file": path,
                "caller": r.caller_name,
                "line": r.caller_line,
                "ref_type": r.ref_type,
                "project": r.project,
            })
        return results

    def query_annotations(
        self, kind: Optional[str] = None,
        filepath: Optional[str] = None,
        project: Optional[str] = None,
        allowed_projects: Optional[List[str]] = None,
        top_k: int = 200
    ) -> List[Dict[str, Any]]:
        """F10: List TODO/FIXME/HACK tags and docstrings, optionally filtered by kind or file."""
        allowed = get_allowed_projects(project or "global", allowed_projects, self.cross_project)
        annotations = self.db.query_annotations(kind=kind, filepath=filepath, allowed_projects=allowed, limit=top_k)

        return [
            {
                "filepath": a.filepath,
                "line": a.line,
                "kind": a.kind,
                "symbol": a.symbol,
                "content": a.content,
                "project": a.project,
            }
            for a in annotations
        ]
