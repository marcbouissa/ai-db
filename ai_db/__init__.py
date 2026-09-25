"""
ai_db: High-Performance Vector & Code Intelligence Database.
Code intelligence and retrieval for AI agents.
"""

import time
from typing import Any

__version__ = "0.1.0"

from ai_db.analyzer.engine import AnalyzerEngine
from ai_db.analyzer.formatters import Formatters, format_as_sexp, format_as_stub
from ai_db.analyzer.references import ReferenceStore
from ai_db.config import AppConfig, config_path, load_config
from ai_db.constants import (
    DEFAULT_CONFIG_FILE,
    DEFAULT_DB_FILE,
    DEFAULT_SKILL_DIRS,
    HARD_IGNORE_DIRS,
    INDEXABLE_EXTENSIONS,
    VENDOR_NOISE_EXTENSIONS,
)
from ai_db.errors import AiDbConfigError, AiDbError, AiDbQueryError
from ai_db.memory.context import ContextMemory
from ai_db.parser.ast_visitor import extract_file_outline
from ai_db.parser.chunker import chunk_file
from ai_db.parser.linters import validate_python_syntax
from ai_db.parser.ts_graph import extract_graph, language_for
from ai_db.search.indexer import Indexer
from ai_db.search.query import QueryEngine
from ai_db.search.skills import SkillRouter
from ai_db.storage.backend import StorageBackend, VectorCapable
from ai_db.storage.database import Database
from ai_db.storage.factory import StorageBackendFactory
from ai_db.storage.state import get_session_state, set_session_state
from ai_db.utils import (
    compute_sha256,
    detect_project_name,
    get_allowed_projects,
    should_index_path,
    strip_code_bloat,
    tokenize,
)
from ai_db.watcher import run_watch as _run_watch


class VectorDB:
    """Unified Facade for ai-db, maintaining 100% backward compatibility."""

    backend: StorageBackend
    db: StorageBackend
    conn: Any
    db_path: str
    telemetry_tracker: Any

    def __init__(self, db_path: str | StorageBackend | None = None,
                 config: AppConfig | None = None):
        self.config = config if config is not None else load_config()
        if isinstance(db_path, StorageBackend):
            self.backend = db_path
        else:
            self.backend = StorageBackendFactory.from_config(self.config, db_path=db_path)
            self.backend.initialize()

        self.db = self.backend
        self.db_path = getattr(self.backend, "db_path", str(db_path or DEFAULT_DB_FILE))
        self.conn = getattr(self.backend, "conn", None)
        self.context_memory = ContextMemory(db=self.backend)
        self.indexer = Indexer(db=self.backend)
        self.indexer.set_ignore_patterns(self.config.index.ignore)
        self.query_engine = QueryEngine(db=self.backend)
        self.skill_router = SkillRouter(db=self.backend, db_path=self.db_path)
        self.analyzer_engine = AnalyzerEngine(db=self.backend)
        self.formatters = Formatters
        for component in (self.context_memory, self.query_engine, self.skill_router):
            component.cross_project = self.config.cross_project
        self._configure_retrieval()
        # Last hook: any change to the index (chunks, vectors, graph) invalidates cached results.
        self.indexer.post_sync_hooks.append(
            lambda changed: self.backend.bump_index_generation() if changed else None)
        from ai_db.telemetry.tracker import TelemetryTracker
        self.telemetry_tracker = TelemetryTracker(conn=self.conn, db_path=self.db_path)

    last_cache_hit: bool = False

    def _configure_retrieval(self) -> None:
        """Pick the single retriever for ``retrieval.mode`` (no runtime switching)."""
        from ai_db.embed.registry import build_embedder
        from ai_db.rerank.registry import build_reranker
        from ai_db.search.ranking import Ranker
        from ai_db.search.retriever import HybridRetriever, LexicalRetriever

        self.reranker = build_reranker(self.config.rerank)
        self.query_engine.ranker = Ranker(self.backend, self.reranker, doc_weight=self.config.index.doc_weight)

        self.embedder = build_embedder(self.config.embedding)
        if self.config.retrieval_mode == "hybrid":
            if self.embedder is None:
                raise AiDbConfigError("retrieval.mode 'hybrid' requires an embedding provider")
            if "vector" not in self.backend.capabilities():
                raise AiDbConfigError(
                    f"retrieval.mode 'hybrid' needs a backend with the 'vector' capability; "
                    f"{self.backend.backend_name} has {sorted(self.backend.capabilities())}")
            if not isinstance(self.backend, VectorCapable):
                raise AiDbConfigError(
                    f"{self.backend.backend_name} declares 'vector' but does not implement VectorCapable")
            self.backend.ensure_vector_index(self.embedder.dim, self.embedder.model_id)
            self.query_engine.retriever = HybridRetriever(self.backend, self.embedder)
            self.indexer.post_sync_hooks.append(lambda _changed: self.embed_missing())
        else:
            self.query_engine.retriever = LexicalRetriever(self.backend)

    def retrieval_signature(self) -> dict[str, Any]:
        return {
            "mode": self.config.retrieval_mode,
            "embedding_model": self.embedder.model_id if self.embedder else None,
            "rerank_model": self.reranker.model_id if self.reranker else None,
        }

    def _cached(self, tool: str, query: str, params: dict[str, Any], compute: Any) -> Any:
        """Return the cached result for this exact request at the current index generation,
        computing and storing it on a miss."""
        from ai_db.search.cache import cache_key

        t0 = time.perf_counter()
        key = cache_key(tool, query, params, self.retrieval_signature())
        gen = self.backend.get_index_generation()
        hit = self.backend.get_query_cache(key, gen)
        self.last_cache_hit = hit is not None
        stages: dict[str, float] = {}
        if hit is not None:
            result = hit
        else:
            self.query_engine.retriever.last_timings = {}
            self.query_engine.ranker.last_timings = {}
            result = compute()
            self.backend.set_query_cache(key, gen, result)
            stages = {**self.query_engine.retriever.last_timings,
                      **self.query_engine.ranker.last_timings}
        self._log_query(tool, query, params.get("mode"), (time.perf_counter() - t0) * 1000,
                        hit is not None, stages, result)
        return result

    def _log_query(self, tool: str, query: str, mode: str | None, total_ms: float,
                   cache_hit: bool, stages: dict[str, float], result: Any) -> None:
        if tool == "investigate":
            top = [[e["qualified_name"], e["score"]] for e in result["evidence"][:10]]
        else:
            top = [[h["chunk_id"], h["score"]] for h in result[:10]]
        sig = self.retrieval_signature()
        self.backend.log_query({
            "timestamp": time.time(), "tool": tool, "query": query, "mode": mode,
            "total_ms": round(total_ms, 3), "cache_hit": cache_hit,
            "stages": {k: round(v, 3) for k, v in stages.items()},
            "providers": {"retriever": sig["mode"], "embedding": sig["embedding_model"],
                          "rerank": sig["rerank_model"]},
            "top": top,
        })

    def embed_missing(self) -> int:
        """Embed chunks that have no vector yet (hybrid mode only)."""
        from ai_db.embed.indexing import embed_missing

        if self.embedder is None:
            raise AiDbConfigError("embed_missing needs an embedding provider")
        batch = int(self.config.embedding.options.get("batch_size", 64))
        return embed_missing(self.backend, self.embedder, batch_size=batch)

    # Storage & DB management
    def status(self) -> dict[str, Any]:
        return self.db.status()

    def optimize(self, prune_missing: bool = True, default_format: str | None = None) -> dict[str, Any]:
        return self.db.optimize(prune_missing=prune_missing, default_format=default_format)

    def get_session_state(self, key: str) -> Any | None:
        if hasattr(self.backend, "get_state"):
            return self.backend.get_state(key)
        return get_session_state(self.conn, key)

    def set_session_state(self, key: str, value: Any):
        if hasattr(self.backend, "set_state"):
            return self.backend.set_state(key, value)
        return set_session_state(self.conn, key, value)

    def close(self):
        self.db.close()

    # Indexing
    def scan_directory(self, root_dir: str) -> list[str]:
        return self.indexer.scan_directory(root_dir)

    def sync(self, root_dir: str, project: str | None = None, verbose: bool = True) -> dict[str, int]:
        return self.indexer.sync(root_dir, project=project, verbose=verbose)

    def _index_file(self, filepath: str, file_hash: str, project: str = "global"):
        return self.indexer._index_file(filepath, file_hash, project=project)

    def prune_file(self, filepath: str):
        result = self.indexer.prune_file(filepath)
        self.backend.bump_index_generation()
        return result

    def sync_paths(self, root_dir: str, paths: list[str], project: str | None = None) -> dict[str, int]:
        """Incrementally sync only ``paths`` (from a file watcher) under ``root_dir``."""
        return self.indexer.sync_paths(root_dir, paths, project=project)

    # Search & Code Query
    def query(self, search_text: str, top_k: int = 5, relative_to: str | None = None,
              project: str | None = None, allowed_projects: list[str] | None = None,
              **kwargs: Any) -> list[dict[str, Any]]:
        k = kwargs.get("top", top_k)
        t0 = time.perf_counter()
        params = {"top_k": k, "relative_to": relative_to, "project": project,
                  "allowed_projects": allowed_projects, "languages": kwargs.get("languages"),
                  "chunk_types": kwargs.get("chunk_types"),
                  "modified_since": kwargs.get("modified_since")}
        results = self._cached("query", search_text, params, lambda: self.query_engine.query(
            search_text, top_k=k, relative_to=relative_to, project=project,
            allowed_projects=allowed_projects, languages=params["languages"],
            chunk_types=params["chunk_types"], modified_since=params["modified_since"]))
        latency_ms = (time.perf_counter() - t0) * 1000.0
        backend_name = "sqlite_wal" if self.backend.backend_name == "sqlite" else self.backend.backend_name
        self.telemetry_tracker.record_query(backend=backend_name, latency_ms=latency_ms, results_count=len(results))
        hits: list[dict[str, Any]] = results
        return hits

    def query_symbol(self, name: str, relative_to: str | None = None,
                     project: str | None = None, allowed_projects: list[str] | None = None) -> list[dict[str, Any]]:
        return self.query_engine.query_symbol(name, relative_to=relative_to,
                                              project=project, allowed_projects=allowed_projects)

    def check_syntax(self, target_path: str | None = None, relative_to: str | None = None,
                     project: str | None = None, allowed_projects: list[str] | None = None) -> list[dict[str, Any]]:
        return self.query_engine.check_syntax(target_path, relative_to=relative_to,
                                              project=project, allowed_projects=allowed_projects)

    # Skill discovery & routing
    def sync_skills(self, skill_dirs: list[str] | None = None, project: str = "global", verbose: bool = True) -> dict[str, int]:
        return self.skill_router.sync_skills(skill_dirs=skill_dirs, project=project, verbose=verbose)

    def route_skills(self, prompt: str, top_k: int = 3,
                     project: str | None = None, allowed_projects: list[str] | None = None,
                     min_confidence: float | None = None) -> list[dict[str, Any]]:
        return self.skill_router.route_skills(prompt, top_k=top_k, project=project,
                                              allowed_projects=allowed_projects,
                                              min_confidence=min_confidence)

    # Context Memory
    def save_context(self, session_id: str, summary: str, project: str | None = None,
                     title: str | None = None, active_files: list[str] | None = None,
                     open_tasks: list[str] | None = None, full_notes: str | None = None) -> dict[str, Any]:
        return self.context_memory.save_context(session_id, summary, project=project,
                                                title=title, active_files=active_files,
                                                open_tasks=open_tasks, full_notes=full_notes)

    def get_context(self, session_id: str | None = None, project: str | None = None,
                    allowed_projects: list[str] | None = None) -> dict[str, Any] | None:
        return self.context_memory.get_context(session_id=session_id, project=project, allowed_projects=allowed_projects)

    def list_contexts(self, project: str | None = None, allowed_projects: list[str] | None = None) -> list[dict[str, Any]]:
        return self.context_memory.list_contexts(project=project, allowed_projects=allowed_projects)

    def query_contexts(self, query_text: str, project: str | None = None,
                       allowed_projects: list[str] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        return self.context_memory.query_contexts(query_text, project=project, allowed_projects=allowed_projects, top_k=top_k)

    # Output Formatters
    @staticmethod
    def format_as_stub(data: dict[str, Any]) -> str:
        return format_as_stub(data)

    @staticmethod
    def format_as_sexp(data: dict[str, Any]) -> str:
        return format_as_sexp(data)

    # Analyzer Engine (RFC: tokenopt-analyzer v2)
    def _store_analysis_ref(self, filepath: str, name: str, start_line: int, end_line: int, kind: str, body_text: str) -> str:
        return self.analyzer_engine._store_analysis_ref(filepath, name, start_line, end_line, kind, body_text)

    def expand_ref(self, ref_id: str, depth: str = "full", span: tuple[int, int] | None = None) -> dict[str, Any] | None:
        return self.analyzer_engine.expand_ref(ref_id, depth=depth, span=span)

    def _diff_spans(self, filepath: str, current_content: str, since: str | None) -> dict[str, Any]:
        return self.analyzer_engine._diff_spans(filepath, current_content, since=since)

    def analyze_file(self, filepath: str, depth: str = "structure",
                     span: tuple[int, int] | None = None,
                     focus: str | None = None,
                     q: str | None = None,
                     since: str | None = None,
                     ctx_lines: int = 10,
                     bypass_cache: bool = False,
                     no_cache: bool = False) -> dict[str, Any]:
        result = self.analyzer_engine.analyze_file(filepath, depth=depth, span=span, focus=focus, q=q,
                                                   since=since, ctx_lines=ctx_lines,
                                                   bypass_cache=bypass_cache, no_cache=no_cache)
        meta = result.get("meta", {})
        is_hit = bool(meta.get("cached", False))
        tokens_in = meta.get("tokens_in", 0)
        tokens_out = meta.get("tokens_out", 0)
        tokens_saved = max(0, tokens_in - tokens_out) if is_hit else 0
        self.telemetry_tracker.record_cache_access(hit=is_hit, tokens_saved=tokens_saved)
        return result

    def analyze_batch(self, targets: list[str], depth: str = "structure",
                      q: str | None = None, focus: str | None = None,
                      span: tuple[int, int] | None = None,
                      since: str | None = None,
                      max_out: int | None = None,
                      cursor: str | None = None,
                      ctx_lines: int = 10) -> dict[str, Any]:
        return self.analyzer_engine.analyze_batch(targets, depth=depth, q=q, focus=focus, span=span,
                                                 since=since, max_out=max_out, cursor=cursor, ctx_lines=ctx_lines)

    def locate_targets(self, q: str, scope: str = ".", k: int = 5) -> list[dict[str, Any]]:
        return self.analyzer_engine.locate_targets(q=q, scope=scope, k=k)

    # Investigation (replaces the agent's analysis loop)
    def investigate(self, query: str, budget_tokens: int = 8000, mode: str = "explain",
                    **kwargs: Any) -> dict[str, Any]:
        from ai_db.analysis.investigate import Investigator
        params = {"budget_tokens": budget_tokens, "mode": mode, **kwargs}
        pack: dict[str, Any] = self._cached("investigate", query, params, lambda: Investigator(self).investigate(
            query, budget_tokens=budget_tokens, mode=mode, **kwargs).to_dict())
        return pack

    # F2: Cross-reference / callers
    def query_callers(self, symbol_name: str, relative_to: str | None = None,
                      project: str | None = None,
                      allowed_projects: list[str] | None = None) -> list[dict[str, Any]]:
        return self.query_engine.query_callers(symbol_name, relative_to=relative_to,
                                               project=project, allowed_projects=allowed_projects)

    # F5: File diff against stored snapshot or git ref
    def diff_file(self, filepath: str, since: str | None = "last") -> dict[str, Any]:
        import os as _os
        abs_path = _os.path.abspath(_os.path.expanduser(filepath))
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        return self.analyzer_engine.ref_store._diff_spans(abs_path, content, since=since)

    # F10: Annotations / TODOs
    def query_annotations(self, kind: str | None = None,
                          filepath: str | None = None,
                          project: str | None = None,
                          allowed_projects: list[str] | None = None) -> list[dict[str, Any]]:
        return self.query_engine.query_annotations(kind=kind, filepath=filepath,
                                                   project=project, allowed_projects=allowed_projects)


def run_watch(db_path: str | None, target_dir: str, debounce_ms: int = 300):
    return _run_watch(VectorDB, db_path, target_dir, debounce_ms)

__all__ = [
    "DEFAULT_CONFIG_FILE",
    "DEFAULT_DB_FILE",
    "DEFAULT_SKILL_DIRS",
    "HARD_IGNORE_DIRS",
    "INDEXABLE_EXTENSIONS",
    "VENDOR_NOISE_EXTENSIONS",
    "AiDbConfigError",
    "AiDbError",
    "AiDbQueryError",
    "AnalyzerEngine",
    "AppConfig",
    "ContextMemory",
    "Database",
    "Formatters",
    "Indexer",
    "QueryEngine",
    "ReferenceStore",
    "SkillRouter",
    "StorageBackend",
    "StorageBackendFactory",
    "VectorDB",
    "__version__",
    "chunk_file",
    "compute_sha256",
    "config_path",
    "detect_project_name",
    "extract_file_outline",
    "extract_graph",
    "format_as_sexp",
    "format_as_stub",
    "get_allowed_projects",
    "get_session_state",
    "language_for",
    "load_config",
    "run_watch",
    "set_session_state",
    "should_index_path",
    "strip_code_bloat",
    "tokenize",
    "validate_python_syntax",
]
