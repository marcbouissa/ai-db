"""The ``VectorDB`` facade: one object wiring storage, indexing, retrieval and analysis.

Lives here rather than in ``ai_db/__init__.py`` so that importing any ai_db submodule
(``ai_db.constants``, ``ai_db.cli``) does not import the whole stack; the package
re-exports it lazily.
"""

import os
import time
from typing import Any

from ai_db.analysis.analyze import AnalyzerEngine
from ai_db.analysis.formatters import Formatters, format_as_sexp, format_as_stub
from ai_db.config import AppConfig, load_config
from ai_db.constants import (
    ANALYZE_CTX_LINES,
    DEFAULT_DB_FILE,
    INDEX_LANGUAGES_STATE_KEY,
)
from ai_db.errors import AiDbConfigError
from ai_db.logger import _logger
from ai_db.memory.context import ContextMemory
from ai_db.search.indexer import Indexer
from ai_db.search.query import QueryEngine
from ai_db.search.skills import SkillRouter
from ai_db.storage.backend import StorageBackend, VectorCapable
from ai_db.storage.factory import StorageBackendFactory
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
        self.indexer.set_languages(self.config.index.languages)
        # index.doc_weight applies with or without a reranker (it used to be dropped,
        # 0.5 hardcoded, whenever rerank.provider was 'none').
        self.query_engine = QueryEngine(db=self.backend, doc_weight=self.config.index.doc_weight,
                                        rerank_top_n=self.config.rerank.top_n)
        self.skill_router = SkillRouter(db=self.backend, db_path=self.db_path)
        self.analyzer_engine = AnalyzerEngine(db=self.backend, query_engine=self.query_engine)
        self.analyzer_engine.languages = self.config.index.languages
        self.analyzer_engine.ref_store.languages = self.config.index.languages
        self.formatters = Formatters
        for component in (self.context_memory, self.query_engine, self.skill_router):
            component.cross_project = self.config.cross_project
        self._configure_retrieval()
        self._apply_language_selection()
        # Graph centrality is derived from symbol_refs, so it must be recomputed
        # whenever a sync changed them. Backends without the "graph" capability
        # have no centrality table and are left untouched. Only the projects the
        # sync actually touched are rebuilt (TODO 13.4).
        if "graph" in self.backend.capabilities():
            self.indexer.post_sync_hooks.append(
                lambda changed, projects: self.backend.rebuild_symbol_centrality(
                    sorted(projects) or None) if changed else None)
        # Last hook: any change to the index (chunks, vectors, graph) invalidates cached results.
        self.indexer.post_sync_hooks.append(
            lambda changed, _projects: self.backend.bump_index_generation() if changed else None)
        # Recall extensions (recall.providers; built-in "vectors" by default when an
        # embedder is configured). Providers are lazy: models load on first use.
        from ai_db.ext.recall import build_recall_providers
        self.recall = build_recall_providers(self.config.recall_providers, self.backend, self.embedder)
        self.skill_router.recall = self.recall
        self.context_memory.recall = self.recall
        from ai_db.telemetry.tracker import TelemetryTracker
        self.telemetry_tracker = TelemetryTracker(self.backend, db_path=self.db_path)

    last_cache_hit: bool = False

    def _configure_retrieval(self) -> None:
        """Pick the single retriever for ``retrieval.mode`` (no runtime switching).

        Embedder and reranker are lazy: nothing is loaded here (TODO B1).
        """
        from ai_db.lazy import LazyEmbedder, LazyReranker
        from ai_db.search.ranking import Ranker
        from ai_db.search.retriever import HybridRetriever, LexicalRetriever

        self.reranker: LazyReranker | None = (
            None if self.config.rerank.provider == "none" else LazyReranker(self.config.rerank))
        if self.reranker is not None:
            self.query_engine.ranker = Ranker(
                self.backend, self.reranker, doc_weight=self.config.index.doc_weight,
                rerank_top_n=self.config.rerank.top_n)

        hybrid = self.config.retrieval_mode == "hybrid"

        def _on_embedder_load(inner: Any) -> None:
            if hybrid:
                assert isinstance(self.backend, VectorCapable)
                self.backend.ensure_vector_index(
                    inner.dim, inner.model_id,
                    vector_index=self.config.storage.options.get("vector_index", "exact"))

        self.embedder: LazyEmbedder | None = (
            None if self.config.embedding.provider == "none"
            else LazyEmbedder(self.config.embedding, on_load=_on_embedder_load))

        if hybrid:
            if self.embedder is None:
                raise AiDbConfigError("retrieval.mode 'hybrid' requires an embedding provider")
            if "vector" not in self.backend.capabilities():
                raise AiDbConfigError(
                    f"retrieval.mode 'hybrid' needs a backend with the 'vector' capability; "
                    f"{self.backend.backend_name} has {sorted(self.backend.capabilities())}")
            if not isinstance(self.backend, VectorCapable):
                raise AiDbConfigError(
                    f"{self.backend.backend_name} declares 'vector' but does not implement VectorCapable")
            self.query_engine.retriever = HybridRetriever(self.backend, self.embedder)
            self.indexer.post_sync_hooks.append(
                lambda _changed, _projects: self.embed_missing())
        else:
            self.query_engine.retriever = LexicalRetriever(self.backend)

    def _apply_language_selection(self) -> None:
        """Re-parse files whose grammar was turned on or off in ``index.languages``.

        Their stored hash is reset, so the next ``sync`` (or a read path, via
        ``search.freshness``) parses them again under the new selection; until then
        they stay searchable as they were. An index from before this setting existed
        was built with every grammar.
        """
        from ai_db.parser.languages import SUPPORTED_LANGUAGES, extensions_of
        from ai_db.storage.models import FileRecord

        current = sorted(self.config.index.languages)
        stored = self.backend.get_state(INDEX_LANGUAGES_STATE_KEY)
        previous = set(stored) if stored is not None else set(SUPPORTED_LANGUAGES)
        changed = previous ^ set(current)
        if changed:
            exts = extensions_of(changed)
            with self.backend.transaction():
                for path in self.backend.get_all_filepaths():
                    if os.path.splitext(path)[1].lower() not in exts:
                        continue
                    rec = self.backend.get_file(path)
                    if rec is not None:
                        self.backend.upsert_file(FileRecord(
                            path, "", 0.0, rec.chunk_count, rec.project))
            _logger.info("index.languages changed (%s): affected files re-parse on next sync",
                         ", ".join(sorted(changed)))
        if stored != current:
            self.backend.set_state(INDEX_LANGUAGES_STATE_KEY, current)

    def _ensure_embedder_loaded(self):
        """Force the embedder to load now (raises AiDbConfigError if unavailable)."""
        if self.embedder is not None:
            self.embedder.load()
        return self.embedder

    def _ensure_reranker_loaded(self):
        if self.reranker is not None:
            self.reranker.load()
        return self.reranker

    def unload_models(self) -> None:
        """Drop this instance's model references (models are shared process-wide;
        ``ai_db.lazy.release_models`` frees them once nothing uses them)."""
        if self.embedder is not None:
            self.embedder.unload()
        if self.reranker is not None:
            self.reranker.unload()

    def retrieval_signature(self) -> dict[str, Any]:
        """Everything that changes what a query would return, for cache isolation.

        A cached pack is only valid for the stack that produced it. Two configs
        pointed at the same DB file -- different vector index, different
        cross-project policy, different weights -- would otherwise read each
        other's results as hits, and the caller has no way to tell. (TODO 14.3)
        """
        from ai_db.lazy import configured_model_id
        return {
            "mode": self.config.retrieval_mode,
            "embedding_model": configured_model_id(self.config.embedding),
            "rerank_model": configured_model_id(self.config.rerank),
            "vector_index": self.config.storage.options.get("vector_index", "exact"),
            "cross_project": self.config.cross_project,
            "embedder_device": self.config.embedding.options.get("device"),
            # The Ranker's own inputs: they reorder results without changing any model.
            "doc_weight": self.config.index.doc_weight,
            "rerank_top_n": self.config.rerank.top_n,
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
        elif tool == "analyze":
            top = [[s["name"], s["span"]] for s in result.get("symbols", [])[:10]]
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

        embedder = self._ensure_embedder_loaded()
        if embedder is None:
            raise AiDbConfigError("embed_missing needs an embedding provider")
        batch = int(self.config.embedding.options.get("batch_size", 64))
        count = embed_missing(self.backend, embedder, batch_size=batch)
        for provider in self.recall:  # skill/context vectors follow the same model
            count += provider.sync()
        return count

    # Storage & DB management
    def status(self) -> dict[str, Any]:
        return self.db.status()

    def optimize(self, prune_missing: bool = True, default_format: str | None = None) -> dict[str, Any]:
        return self.db.optimize(prune_missing=prune_missing, default_format=default_format)

    def get_session_state(self, key: str) -> Any | None:
        return self.backend.get_state(key)

    def set_session_state(self, key: str, value: Any) -> None:
        self.backend.set_state(key, value)

    def close(self):
        self.db.close()

    # Indexing
    def scan_directory(self, root_dir: str) -> list[str]:
        return self.indexer.scan_directory(root_dir)

    def sync(self, root_dir: str, project: str | None = None, verbose: bool = True) -> dict[str, int]:
        return self.indexer.sync(root_dir, project=project, verbose=verbose)

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
        params = {"top_k": k, "relative_to": relative_to, "project": project,
                  "allowed_projects": allowed_projects, "languages": kwargs.get("languages"),
                  "chunk_types": kwargs.get("chunk_types"),
                  "modified_since": kwargs.get("modified_since")}
        results = self._cached("query", search_text, params, lambda: self.query_engine.query(
            search_text, top_k=k, relative_to=relative_to, project=project,
            allowed_projects=allowed_projects, languages=params["languages"],
            chunk_types=params["chunk_types"], modified_since=params["modified_since"]))
        hits: list[dict[str, Any]] = results  # logged (latency, cache hit) by _cached
        return hits

    def query_symbol(self, name: str, relative_to: str | None = None,
                     project: str | None = None, allowed_projects: list[str] | None = None,
                     prefix: bool = False) -> list[dict[str, Any]]:
        return self.query_engine.query_symbol(name, relative_to=relative_to,
                                              project=project, allowed_projects=allowed_projects,
                                              prefix=prefix)

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

    def delete_context(self, session_id: str, project: str | None = None) -> bool:
        return self.context_memory.delete_context(session_id=session_id, project=project)

    # Output Formatters
    @staticmethod
    def format_as_stub(data: dict[str, Any]) -> str:
        return format_as_stub(data)

    @staticmethod
    def format_as_sexp(data: dict[str, Any]) -> str:
        return format_as_sexp(data)

    # Analyzer Engine (RFC: tokenopt-analyzer v2)
    def expand_ref(self, ref_id: str, depth: str = "full", span: tuple[int, int] | None = None) -> dict[str, Any] | None:
        return self.analyzer_engine.expand_ref(ref_id, depth=depth, span=span)

    def analyze_file(self, filepath: str, depth: str = "structure",
                     span: tuple[int, int] | None = None,
                     focus: str | None = None,
                     q: str | None = None,
                     since: str | None = None,
                     ctx_lines: int = ANALYZE_CTX_LINES,
                     bypass_cache: bool = False,
                     no_cache: bool = False) -> dict[str, Any]:
        t0 = time.perf_counter()
        result = self.analyzer_engine.analyze_file(filepath, depth=depth, span=span, focus=focus, q=q,
                                                   since=since, ctx_lines=ctx_lines,
                                                   bypass_cache=bypass_cache, no_cache=no_cache)
        self._log_query("analyze", filepath, depth, (time.perf_counter() - t0) * 1000,
                        bool(result.get("meta", {}).get("cached", False)), {}, result)
        return result

    def analyze_batch(self, targets: list[str], depth: str = "structure",
                      q: str | None = None, focus: str | None = None,
                      span: tuple[int, int] | None = None,
                      since: str | None = None,
                      max_out: int | None = None,
                      cursor: str | None = None,
                      ctx_lines: int = ANALYZE_CTX_LINES) -> dict[str, Any]:
        return self.analyzer_engine.analyze_batch(targets, depth=depth, q=q, focus=focus, span=span,
                                                 since=since, max_out=max_out, cursor=cursor, ctx_lines=ctx_lines)

    def locate_targets(self, q: str, scope: str = ".", k: int = 5,
                       path_prefix: str | None = None, project: str | None = None,
                       allowed_projects: list[str] | None = None) -> list[dict[str, Any]]:
        return self.analyzer_engine.locate_targets(q=q, scope=scope, k=k, path_prefix=path_prefix,
                                                   project=project, allowed_projects=allowed_projects)

    # Investigation (replaces the agent's analysis loop)
    def investigate(self, query: str, budget_tokens: int = 8000, mode: str = "explain",
                    **kwargs: Any) -> dict[str, Any]:
        from ai_db.analysis.investigate import Investigator
        params = {"budget_tokens": budget_tokens, "mode": mode, **kwargs}
        if mode == "diff":
            # The answer depends on git state, which the index generation does not track.
            direct: dict[str, Any] = Investigator(self).investigate(
                query, budget_tokens=budget_tokens, mode=mode, **kwargs).to_dict()
            return direct
        pack: dict[str, Any] = self._cached("investigate", query, params, lambda: Investigator(self).investigate(
            query, budget_tokens=budget_tokens, mode=mode, **kwargs).to_dict())
        return pack

    # F2: Cross-reference / callers
    def query_callers(self, symbol_name: str, relative_to: str | None = None,
                      project: str | None = None,
                      allowed_projects: list[str] | None = None,
                      top_k: int = 100) -> dict[str, Any]:
        return self.query_engine.query_callers(symbol_name, relative_to=relative_to,
                                               project=project, allowed_projects=allowed_projects,
                                               top_k=top_k)

    # F5: File diff against stored snapshot or git ref
    def diff_file(self, filepath: str, since: str | None = "last") -> dict[str, Any]:
        import os as _os
        abs_path = _os.path.abspath(_os.path.expanduser(filepath))
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        from ai_db.analysis.diff import file_diff
        return file_diff(abs_path, since, self.backend, self.config.index.languages, content)

    # F10: Annotations / TODOs
    def query_annotations(self, kind: str | None = None,
                          filepath: str | None = None,
                          project: str | None = None,
                          allowed_projects: list[str] | None = None) -> list[dict[str, Any]]:
        return self.query_engine.query_annotations(kind=kind, filepath=filepath,
                                                   project=project, allowed_projects=allowed_projects)


def run_watch(db_path: str | None, target_dir: str, debounce_ms: int = 300):
    return _run_watch(VectorDB, db_path, target_dir, debounce_ms)
