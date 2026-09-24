"""
ai_db: High-Performance Vector & Code Intelligence Database.
Modular architecture with zero external dependencies (Python standard library only).
"""

import os
import time
from typing import List, Dict, Any, Tuple, Optional, Union

__version__ = "0.1.0"

from ai_db.constants import (
    DEFAULT_DB_FILE,
    DEFAULT_CONFIG_FILE,
    DEFAULT_SKILL_DIRS,
    INDEXABLE_EXTENSIONS,
    HARD_IGNORE_DIRS,
    VENDOR_NOISE_EXTENSIONS
)

from ai_db.errors import AiDbError, AiDbConfigError, AiDbQueryError
from ai_db.config import AppConfig, load_config, config_path
from ai_db.utils import (
    detect_project_name,
    get_allowed_projects,
    compute_sha256,
    tokenize,
    strip_code_bloat,
    should_index_path
)

from ai_db.parser.syntax import validate_python_syntax
from ai_db.parser.ast_visitor import extract_symbols, extract_file_outline
from ai_db.parser.chunker import chunk_file

from ai_db.storage.backend import StorageBackend
from ai_db.storage.factory import StorageBackendFactory
from ai_db.storage.database import Database
from ai_db.storage.state import get_session_state, set_session_state
from ai_db.memory.context import ContextMemory
from ai_db.search.indexer import Indexer
from ai_db.search.query import QueryEngine
from ai_db.search.skills import SkillRouter
from ai_db.analyzer.formatters import format_as_stub, format_as_sexp, Formatters
from ai_db.analyzer.references import ReferenceStore
from ai_db.analyzer.engine import AnalyzerEngine
from ai_db.watcher import run_watch as _run_watch

class VectorDB:
    """Unified Facade for ai-db, maintaining 100% backward compatibility."""

    backend: StorageBackend
    db: StorageBackend
    conn: Any
    db_path: str
    telemetry_tracker: Optional[Any]

    def __init__(self, db_path: Optional[Union[str, StorageBackend]] = None,
                 config: Optional[AppConfig] = None):
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
        self.query_engine = QueryEngine(db=self.backend)
        self.skill_router = SkillRouter(db=self.backend, db_path=self.db_path)
        self.analyzer_engine = AnalyzerEngine(db=self.backend)
        self.formatters = Formatters
        for component in (self.context_memory, self.query_engine, self.skill_router):
            component.cross_project = self.config.cross_project
        self._configure_retrieval()
        try:
            from ai_db.telemetry.tracker import TelemetryTracker
            self.telemetry_tracker = TelemetryTracker(conn=self.conn, db_path=self.db_path)
        except Exception:
            self.telemetry_tracker = None

    def _configure_retrieval(self) -> None:
        """Pick the single retriever for ``retrieval.mode`` (no runtime switching)."""
        from ai_db.embed.registry import build_embedder
        from ai_db.search.retriever import HybridRetriever, LexicalRetriever

        self.embedder = build_embedder(self.config.embedding)
        if self.config.retrieval_mode == "hybrid":
            if self.embedder is None:
                raise AiDbConfigError("retrieval.mode 'hybrid' requires an embedding provider")
            if "vector" not in self.backend.capabilities():
                raise AiDbConfigError(
                    f"retrieval.mode 'hybrid' needs a backend with the 'vector' capability; "
                    f"{self.backend.backend_name} has {sorted(self.backend.capabilities())}")
            self.backend.ensure_vector_index(self.embedder.dim, self.embedder.model_id)
            self.query_engine.retriever = HybridRetriever(self.backend, self.embedder)
            self.indexer.post_sync_hooks.append(lambda _changed: self.embed_missing())
        else:
            self.query_engine.retriever = LexicalRetriever(self.backend)

    def embed_missing(self) -> int:
        """Embed chunks that have no vector yet (hybrid mode only)."""
        from ai_db.embed.indexing import embed_missing

        if self.embedder is None:
            raise AiDbConfigError("embed_missing needs an embedding provider")
        batch = int(self.config.embedding.options.get("batch_size", 64))
        return embed_missing(self.backend, self.embedder, batch_size=batch)

    # Storage & DB management
    def status(self) -> Dict[str, Any]:
        return self.db.status()

    def optimize(self, prune_missing: bool = True, default_format: Optional[str] = None) -> Dict[str, Any]:
        return self.db.optimize(prune_missing=prune_missing, default_format=default_format)

    def get_session_state(self, key: str) -> Optional[Any]:
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
    def scan_directory(self, root_dir: str) -> List[str]:
        return self.indexer.scan_directory(root_dir)

    def sync(self, root_dir: str, project: Optional[str] = None, verbose: bool = True) -> Dict[str, int]:
        return self.indexer.sync(root_dir, project=project, verbose=verbose)

    def _index_file(self, filepath: str, file_hash: str, project: str = "global"):
        return self.indexer._index_file(filepath, file_hash, project=project)

    def prune_file(self, filepath: str):
        return self.indexer.prune_file(filepath)

    # Search & Code Query
    def query(self, search_text: str, top_k: int = 5, relative_to: Optional[str] = None,
              project: Optional[str] = None, allowed_projects: Optional[List[str]] = None,
              **kwargs: Any) -> List[Dict[str, Any]]:
        k = kwargs.get("top", top_k)
        t0 = time.perf_counter()
        results = self.query_engine.query(search_text, top_k=k, relative_to=relative_to,
                                          project=project, allowed_projects=allowed_projects,
                                          languages=kwargs.get("languages"),
                                          chunk_types=kwargs.get("chunk_types"),
                                          modified_since=kwargs.get("modified_since"))
        if getattr(self, "telemetry_tracker", None) is not None:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            backend_name = "sqlite_wal" if "sqlite" in getattr(self.backend, "__class__", type(self.backend)).__name__.lower() else "generic"
            self.telemetry_tracker.record_query(backend=backend_name, latency_ms=latency_ms, results_count=len(results))
        return results

    def query_symbol(self, name: str, relative_to: Optional[str] = None,
                     project: Optional[str] = None, allowed_projects: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        return self.query_engine.query_symbol(name, relative_to=relative_to,
                                              project=project, allowed_projects=allowed_projects)

    def check_syntax(self, target_path: Optional[str] = None, relative_to: Optional[str] = None,
                     project: Optional[str] = None, allowed_projects: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        return self.query_engine.check_syntax(target_path, relative_to=relative_to,
                                              project=project, allowed_projects=allowed_projects)

    # Skill discovery & routing
    def sync_skills(self, skill_dirs: Optional[List[str]] = None, project: str = "global", verbose: bool = True) -> Dict[str, int]:
        return self.skill_router.sync_skills(skill_dirs=skill_dirs, project=project, verbose=verbose)

    def route_skills(self, prompt: str, top_k: int = 3,
                     project: Optional[str] = None, allowed_projects: Optional[List[str]] = None,
                     min_confidence: Optional[float] = None) -> List[Dict[str, Any]]:
        return self.skill_router.route_skills(prompt, top_k=top_k, project=project,
                                              allowed_projects=allowed_projects,
                                              min_confidence=min_confidence)

    # Context Memory
    def save_context(self, session_id: str, summary: str, project: Optional[str] = None,
                     title: Optional[str] = None, active_files: Optional[List[str]] = None,
                     open_tasks: Optional[List[str]] = None, full_notes: Optional[str] = None) -> Dict[str, Any]:
        return self.context_memory.save_context(session_id, summary, project=project,
                                                title=title, active_files=active_files,
                                                open_tasks=open_tasks, full_notes=full_notes)

    def get_context(self, session_id: Optional[str] = None, project: Optional[str] = None,
                    allowed_projects: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        return self.context_memory.get_context(session_id=session_id, project=project, allowed_projects=allowed_projects)

    def list_contexts(self, project: Optional[str] = None, allowed_projects: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        return self.context_memory.list_contexts(project=project, allowed_projects=allowed_projects)

    def query_contexts(self, query_text: str, project: Optional[str] = None,
                       allowed_projects: Optional[List[str]] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        return self.context_memory.query_contexts(query_text, project=project, allowed_projects=allowed_projects, top_k=top_k)

    # Output Formatters
    @staticmethod
    def format_as_stub(data: Dict[str, Any]) -> str:
        return format_as_stub(data)

    @staticmethod
    def format_as_sexp(data: Dict[str, Any]) -> str:
        return format_as_sexp(data)

    # Analyzer Engine (RFC: tokenopt-analyzer v2)
    def _store_analysis_ref(self, filepath: str, name: str, start_line: int, end_line: int, kind: str, body_text: str) -> str:
        return self.analyzer_engine._store_analysis_ref(filepath, name, start_line, end_line, kind, body_text)

    def expand_ref(self, ref_id: str, depth: str = "full", span: Optional[Tuple[int, int]] = None) -> Optional[Dict[str, Any]]:
        return self.analyzer_engine.expand_ref(ref_id, depth=depth, span=span)

    def _diff_spans(self, filepath: str, current_content: str, since: Optional[str]) -> Dict[str, Any]:
        return self.analyzer_engine._diff_spans(filepath, current_content, since=since)

    def analyze_file(self, filepath: str, depth: str = "structure",
                     span: Optional[Tuple[int, int]] = None,
                     focus: Optional[str] = None,
                     q: Optional[str] = None,
                     since: Optional[str] = None,
                     ctx_lines: int = 10,
                     bypass_cache: bool = False,
                     no_cache: bool = False) -> Dict[str, Any]:
        result = self.analyzer_engine.analyze_file(filepath, depth=depth, span=span, focus=focus, q=q,
                                                   since=since, ctx_lines=ctx_lines,
                                                   bypass_cache=bypass_cache, no_cache=no_cache)
        if getattr(self, "telemetry_tracker", None) is not None:
            meta = result.get("meta", {})
            is_hit = bool(meta.get("cached", False))
            tokens_in = meta.get("tokens_in", 0)
            tokens_out = meta.get("tokens_out", 0)
            tokens_saved = max(0, tokens_in - tokens_out) if is_hit else 0
            self.telemetry_tracker.record_cache_access(hit=is_hit, tokens_saved=tokens_saved)
        return result

    def analyze_batch(self, targets: List[str], depth: str = "structure",
                      q: Optional[str] = None, focus: Optional[str] = None,
                      span: Optional[Tuple[int, int]] = None,
                      since: Optional[str] = None,
                      max_out: Optional[int] = None,
                      cursor: Optional[str] = None,
                      ctx_lines: int = 10) -> Dict[str, Any]:
        return self.analyzer_engine.analyze_batch(targets, depth=depth, q=q, focus=focus, span=span,
                                                 since=since, max_out=max_out, cursor=cursor, ctx_lines=ctx_lines)

    def locate_targets(self, q: str, scope: str = ".", k: int = 5) -> List[Dict[str, Any]]:
        return self.analyzer_engine.locate_targets(q=q, scope=scope, k=k)

    # F2: Cross-reference / callers
    def query_callers(self, symbol_name: str, relative_to: Optional[str] = None,
                      project: Optional[str] = None,
                      allowed_projects: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        return self.query_engine.query_callers(symbol_name, relative_to=relative_to,
                                               project=project, allowed_projects=allowed_projects)

    # F5: File diff against stored snapshot or git ref
    def diff_file(self, filepath: str, since: Optional[str] = "last") -> Dict[str, Any]:
        import os as _os
        abs_path = _os.path.abspath(_os.path.expanduser(filepath))
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        return self.analyzer_engine.ref_store._diff_spans(abs_path, content, since=since)

    # F10: Annotations / TODOs
    def query_annotations(self, kind: Optional[str] = None,
                          filepath: Optional[str] = None,
                          project: Optional[str] = None,
                          allowed_projects: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        return self.query_engine.query_annotations(kind=kind, filepath=filepath,
                                                   project=project, allowed_projects=allowed_projects)


def run_watch(db_path: str, target_dir: str, interval: float = 2.0):
    return _run_watch(VectorDB, db_path, target_dir, interval)
