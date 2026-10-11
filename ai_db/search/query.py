import os
from typing import Any

from ai_db.constants import CANDIDATE_POOL, DEFAULT_DOC_WEIGHT, DEFAULT_RERANK_TOP_N
from ai_db.search.ranking import Ranker
from ai_db.search.retriever import LexicalRetriever, Retriever
from ai_db.utils import get_allowed_projects


def _display_path(path: str, relative_to: str) -> str:
    """Path relative to ``relative_to``; on Windows, paths on another drive have no
    relative form and are shown absolute."""
    try:
        return os.path.relpath(path, relative_to)
    except ValueError:
        return path


class QueryEngine:
    def __init__(self, db: Any = None, conn: Any = None,
                 doc_weight: float = DEFAULT_DOC_WEIGHT, rerank_top_n: int = DEFAULT_RERANK_TOP_N):
        self.db = db if db is not None else conn
        self.cross_project: dict[str, list[str]] = {}
        # Replaced once by VectorDB according to retrieval.mode.
        self.retriever: Retriever = LexicalRetriever(self.db)
        # VectorDB passes index.doc_weight / rerank.top_n (and swaps in a reranker).
        self.ranker = Ranker(self.db, doc_weight=doc_weight, rerank_top_n=rerank_top_n)

    def query(
        self, search_text: str, top_k: int = 5, relative_to: str | None = None,
        project: str | None = None, allowed_projects: list[str] | None = None,
        languages: list[str] | None = None, chunk_types: list[str] | None = None,
        modified_since: float | None = None,
    ) -> list[dict[str, Any]]:
        allowed = get_allowed_projects(project or "global", allowed_projects, self.cross_project)
        filters = {"allowed_projects": allowed, "languages": languages,
                   "chunk_types": chunk_types, "modified_since": modified_since}
        search_results = self.search(search_text, filters, top_k)

        results = []
        for r in search_results:
            path_display = r.filepath
            if relative_to:
                path_display = _display_path(path_display, relative_to)

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

    def search(self, text: str, filters: dict[str, Any], top_k: int) -> list[Any]:
        """Ranked SearchResults for ``text`` (retriever candidates, then final ranking)."""
        pool = max(top_k, CANDIDATE_POOL)
        candidates = self.retriever.candidates(text, filters, pool)
        return self.ranker.rank(text, candidates, filters.get("allowed_projects"))[:top_k]

    def query_symbol(
        self, name: str, relative_to: str | None = None,
        project: str | None = None, allowed_projects: list[str] | None = None,
        prefix: bool = False,
    ) -> list[dict[str, Any]]:
        """Symbols named ``name`` (``match: "exact"``). Prefix matches (``match: "prefix"``)
        are included only when nothing matches exactly, or when ``prefix`` is set."""
        allowed = get_allowed_projects(project or "global", allowed_projects, self.cross_project)
        symbols = self.db.query_symbols(name=name, allowed_projects=allowed, limit=50, include_prefix=prefix)
        if not symbols and not prefix:
            symbols = self.db.query_symbols(name=name, allowed_projects=allowed, limit=50, include_prefix=True)

        results = []
        for s in symbols:
            path_display = s.filepath
            if relative_to:
                path_display = _display_path(path_display, relative_to)
            results.append({
                "name": s.name,
                "symbol_type": s.symbol_type,
                "file": path_display,
                "abs_path": s.filepath,
                "line": s.line,
                "signature": s.signature,
                "project": s.project,
                "match": "exact" if s.name == name else "prefix",
            })
        return results

    def check_syntax(
        self, target_path: str | None = None, relative_to: str | None = None,
        project: str | None = None, allowed_projects: list[str] | None = None
    ) -> list[dict[str, Any]]:
        allowed = get_allowed_projects(project or "global", allowed_projects, self.cross_project)
        errors = self.db.get_syntax_errors(target_path=target_path, allowed_projects=allowed)

        results = []
        for err in errors:
            path_display = err.filepath
            if relative_to:
                path_display = _display_path(path_display, relative_to)
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
        self, symbol_name: str, relative_to: str | None = None,
        project: str | None = None,
        allowed_projects: list[str] | None = None,
        top_k: int = 100
    ) -> dict[str, Any]:
        """F2: Find call sites, imports, and inheritance refs to a symbol.

        ``symbol_name`` may be qualified (``Class.method``: only call sites whose receiver
        resolved to that class) or bare (every call of that name; ``groups`` then shows the
        per-class breakdown). Returns ``{"total", "returned", "truncated", "results", "groups"?}``.
        """
        allowed = get_allowed_projects(project or "global", allowed_projects, self.cross_project)
        callers, total = self.db.query_symbol_callers(callee_name=symbol_name, allowed_projects=allowed, limit=top_k)

        results = []
        for r in callers:
            path = os.path.relpath(r.caller_filepath, relative_to) if relative_to else r.caller_filepath
            results.append({
                "file": path,
                "caller": r.caller_name,
                "line": r.call_line if r.call_line else r.caller_line,
                "caller_line": r.caller_line,
                "col": r.call_col,
                "ref_type": r.ref_type,
                "qualifier": r.callee_qualifier,
                "resolved": r.callee_resolved,
                "project": r.project,
            })
        out: dict[str, Any] = {
            "total": total, "returned": len(results), "truncated": total > len(results), "results": results,
        }
        if "." not in symbol_name:
            counts = self.db.count_symbol_callers_by_resolved(symbol_name, allowed)
            out["groups"] = {(k if k is not None else "unresolved"): v for k, v in sorted(
                counts.items(), key=lambda kv: (kv[0] is None, kv[0] or ""))}
        return out

    def query_annotations(
        self, kind: str | None = None,
        filepath: str | None = None,
        project: str | None = None,
        allowed_projects: list[str] | None = None,
        top_k: int = 200
    ) -> list[dict[str, Any]]:
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
