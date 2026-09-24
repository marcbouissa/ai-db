"""StorageBackend Abstract Base Class for the ai-db storage layer.

Defines the pluggable storage contract isolating core code intelligence,
parsing, and search algorithms from database engine implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, ContextManager

from ai_db.storage.models import (
    AnalysisRefRecord,
    AnnotationRecord,
    ChunkRecord,
    ContextRecord,
    FileRecord,
    SearchResult,
    SkillRecord,
    SymbolRecord,
    SymbolRefRecord,
    SyntaxErrorRecord,
)


class StorageBackend(ABC):
    """Abstract Base Class defining the pluggable storage contract for ai-db."""

    # =========================================================================
    # Properties & Lifecycle
    # =========================================================================

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return canonical identifier for this storage backend (e.g. 'sqlite', 'mysql')."""

    def capabilities(self) -> frozenset:
        """Feature set this backend supports: subset of {'fts', 'vector', 'graph'}."""
        return frozenset({"fts", "graph"})

    @abstractmethod
    def initialize(self) -> None:
        """Create tables, indexes, virtual FTS tables, and configure engine settings."""

    @abstractmethod
    def close(self) -> None:
        """Gracefully close connections and release storage resources. Idempotent."""

    @abstractmethod
    def transaction(self) -> ContextManager[None]:
        """Context manager yielding an atomic transaction (commit on exit, rollback on exception)."""

    # =========================================================================
    # File Operations
    # =========================================================================

    @abstractmethod
    def upsert_file(self, record: FileRecord) -> None:
        """Insert or update a tracked file record."""

    @abstractmethod
    def get_file(self, filepath: str) -> FileRecord | None:
        """Retrieve a file record by exact filepath, or return None."""

    @abstractmethod
    def delete_file(self, filepath: str) -> None:
        """Delete a file and cascade deletion across chunks, symbols, and references."""

    @abstractmethod
    def get_files_by_prefix(self, prefix: str) -> dict[str, str]:
        """Return a mapping of {filepath: sha256} for all files with the given directory prefix."""

    @abstractmethod
    def get_all_filepaths(self) -> list[str]:
        """Return a list of all tracked file paths across all projects."""

    # =========================================================================
    # Chunk Operations & Code Search
    # =========================================================================

    @abstractmethod
    def insert_chunks(self, chunks: list[ChunkRecord]) -> None:
        """Persist chunk records and synchronize full-text search index."""

    def replace_file_chunks(self, filepath: str, chunks: list[ChunkRecord]) -> dict[str, int]:
        """Make ``chunks`` the stored chunks of ``filepath``, keeping ids of unchanged
        chunks (matched by ``(content_hash, name)``) and resolving ``parent_index``.
        Returns ``{"kept", "inserted", "deleted"}`` counts."""
        raise NotImplementedError(f"{type(self).__name__} must implement replace_file_chunks")

    def get_chunks_by_ids(self, ids: list[int]) -> list[ChunkRecord]:
        """Chunks (with content) for ``ids``, in the given order; unknown ids are skipped."""
        raise NotImplementedError(f"{type(self).__name__} must implement get_chunks_by_ids")

    def rebuild_symbol_centrality(self) -> None:
        """('graph' capability) Recompute per-symbol normalized call in-degree."""
        raise NotImplementedError(f"{type(self).__name__} must implement rebuild_symbol_centrality")

    def get_symbol_centrality(self, names: list[str],
                              allowed_projects: list[str] | None = None) -> dict[str, float]:
        """('graph' capability) ``{bare_symbol_name: score in [0, 1]}``."""
        raise NotImplementedError(f"{type(self).__name__} must implement get_symbol_centrality")

    def get_refs_from(self, filepath: str, caller_scope: str | None,
                      ref_types: tuple[str, ...] = ("call",)) -> list[SymbolRefRecord]:
        """('graph') References made inside ``caller_scope`` (``module.Class.method``);
        None means every scope of the file."""
        raise NotImplementedError(f"{type(self).__name__} must implement get_refs_from")

    def find_chunks_by_symbol(self, names: list[str], allowed_projects: list[str] | None = None,
                              limit_per_name: int = 8) -> dict[str, list[ChunkRecord]]:
        """('graph') Definition chunks whose last qualified-name component is in ``names``."""
        raise NotImplementedError(f"{type(self).__name__} must implement find_chunks_by_symbol")

    def get_chunk_by_qualified_name(self, filepath: str, qualified_name: str) -> ChunkRecord | None:
        """First chunk of ``qualified_name`` in ``filepath`` or None."""
        raise NotImplementedError(f"{type(self).__name__} must implement get_chunk_by_qualified_name")

    def get_index_generation(self) -> int:
        """Counter incremented by every sync that changed the index."""
        raise NotImplementedError(f"{type(self).__name__} must implement get_index_generation")

    def bump_index_generation(self) -> int:
        """Increment the generation, delete cache rows of older generations, return it."""
        raise NotImplementedError(f"{type(self).__name__} must implement bump_index_generation")

    def get_query_cache(self, cache_key: str, index_gen: int) -> Any | None:
        """Cached JSON result for ``cache_key`` at ``index_gen`` or None (miss)."""
        raise NotImplementedError(f"{type(self).__name__} must implement get_query_cache")

    def set_query_cache(self, cache_key: str, index_gen: int, result: Any) -> None:
        raise NotImplementedError(f"{type(self).__name__} must implement set_query_cache")

    def log_query(self, entry: dict[str, Any]) -> None:
        """Append a query-log entry: timestamp, tool, query, mode, total_ms, cache_hit,
        stages {name: ms}, providers {stage: model_id}, top [[chunk_id, score], ...]."""
        raise NotImplementedError(f"{type(self).__name__} must implement log_query")

    def get_query_log(self, min_total_ms: float = 0.0, limit: int = 50) -> list[dict[str, Any]]:
        """Newest-first log entries with ``total_ms >= min_total_ms``."""
        raise NotImplementedError(f"{type(self).__name__} must implement get_query_log")

    def clear_file_metadata(self, filepath: str) -> None:
        """Delete symbols, refs, annotations, syntax errors and analysis refs of a file
        (but not its file row or chunks)."""
        raise NotImplementedError(f"{type(self).__name__} must implement clear_file_metadata")

    @abstractmethod
    def get_chunks_for_file(self, filepath: str) -> list[ChunkRecord]:
        """Retrieve all chunks for a file, ordered by start_line, uncompressed."""

    @abstractmethod
    def search_chunks(
        self,
        query_tokens: list[str],
        allowed_projects: list[str] | None = None,
        top_k: int = 5,
        path_prefix: str | None = None,
    ) -> list[SearchResult]:
        """Execute full-text relevance search over code chunks within project scopes."""

    # =========================================================================
    # Symbol Operations & Cross-References
    # =========================================================================

    @abstractmethod
    def insert_symbols(self, symbols: list[SymbolRecord]) -> None:
        """Insert symbol records."""

    @abstractmethod
    def query_symbols(
        self,
        name: str,
        allowed_projects: list[str] | None = None,
        limit: int = 50,
    ) -> list[SymbolRecord]:
        """Query symbols matching name (exact first, then substring) within project scopes."""

    @abstractmethod
    def insert_symbol_refs(self, refs: list[SymbolRefRecord]) -> None:
        """Insert symbol cross-references (callers, imports, inheritance)."""

    @abstractmethod
    def query_symbol_callers(
        self,
        callee_name: str,
        allowed_projects: list[str] | None = None,
        limit: int = 100,
    ) -> list[SymbolRefRecord]:
        """Query all call sites referencing callee_name within project scopes."""

    # =========================================================================
    # Diagnostics & Annotations
    # =========================================================================

    @abstractmethod
    def upsert_syntax_error(self, error: SyntaxErrorRecord) -> None:
        """Insert or replace a syntax error diagnostic."""

    @abstractmethod
    def delete_syntax_error(self, filepath: str) -> None:
        """Delete syntax errors for a file."""

    @abstractmethod
    def get_syntax_errors(
        self,
        target_path: str | None = None,
        allowed_projects: list[str] | None = None,
    ) -> list[SyntaxErrorRecord]:
        """Retrieve syntax errors optionally filtered by target path prefix."""

    @abstractmethod
    def insert_annotations(self, annotations: list[AnnotationRecord]) -> None:
        """Insert code annotations (TODO/FIXME/docstrings)."""

    @abstractmethod
    def query_annotations(
        self,
        kind: str | None = None,
        filepath: str | None = None,
        allowed_projects: list[str] | None = None,
        limit: int = 200,
    ) -> list[AnnotationRecord]:
        """Query code annotations filtered by kind and/or filepath."""

    # =========================================================================
    # Skills
    # =========================================================================

    @abstractmethod
    def get_skills(
        self,
        allowed_projects: list[str] | None = None,
    ) -> list[SkillRecord]:
        """Retrieve all skills within allowed project scopes."""

    @abstractmethod
    def get_skills_by_project(self, project: str) -> dict[str, tuple[str, str]]:
        """Return {filepath: (name, sha256)} mapping for skills in a project."""

    @abstractmethod
    def upsert_skill(self, skill: SkillRecord) -> None:
        """Insert or update a skill and synchronize its FTS index."""

    @abstractmethod
    def delete_skill(self, filepath: str, project: str, name: str | None = None) -> None:
        """Delete a skill and remove its FTS entry."""

    @abstractmethod
    def search_skills(
        self,
        query_tokens: list[str],
        allowed_projects: list[str] | None = None,
        limit: int = 20,
    ) -> list[tuple[str, float]]:
        """Search skills via FTS BM25, returning list of (skill_name, score)."""

    # =========================================================================
    # Context Memory
    # =========================================================================

    @abstractmethod
    def save_context(self, context: ContextRecord) -> None:
        """Persist a conversation session context snapshot."""

    @abstractmethod
    def get_context(
        self,
        session_id: str | None = None,
        allowed_projects: list[str] | None = None,
    ) -> ContextRecord | None:
        """Retrieve latest or specified session context snapshot."""

    @abstractmethod
    def list_contexts(
        self,
        allowed_projects: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """List summary metadata for saved session contexts."""

    @abstractmethod
    def search_contexts(
        self,
        query_tokens: list[str],
        allowed_projects: list[str] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Search saved contexts via full-text search."""

    # =========================================================================
    # Analysis References
    # =========================================================================

    @abstractmethod
    def store_analysis_ref(self, ref: AnalysisRefRecord) -> None:
        """Store an opaque analysis progressive disclosure ref."""

    @abstractmethod
    def get_analysis_ref(self, ref_id: str) -> AnalysisRefRecord | None:
        """Retrieve an analysis progressive disclosure ref by ID."""

    @abstractmethod
    def evict_stale_analysis_refs(self, older_than_seconds: float) -> int:
        """Evict analysis references older than the specified age in seconds."""

    # =========================================================================
    # Session State & Semantic Cache
    # =========================================================================

    @abstractmethod
    def get_state(self, key: str) -> Any | None:
        """Retrieve deserialized session state value for key."""

    @abstractmethod
    def set_state(self, key: str, value: Any) -> None:
        """Store JSON-serializable session state value for key."""

    @abstractmethod
    def get_semantic_cache(self, cache_key: str, file_hash: str) -> dict[str, Any] | None:
        """Retrieve cached AST analysis result if file hash matches."""

    @abstractmethod
    def set_semantic_cache(self, cache_key: str, file_hash: str, result: dict[str, Any]) -> None:
        """Save AST analysis result in semantic cache."""

    @abstractmethod
    def clear_semantic_cache(self) -> None:
        """Purge all entries in the semantic cache without affecting session state."""

    # =========================================================================
    # Maintenance & Health
    # =========================================================================

    @abstractmethod
    def status(self) -> dict[str, Any]:
        """Return storage health statistics (file/chunk/symbol counts, size in KB)."""

    @abstractmethod
    def optimize(
        self,
        prune_missing: bool = True,
        default_format: str | None = None,
    ) -> dict[str, Any]:
        """Optimize full-text indexes, prune missing files, vacuum storage, update format."""

    def get_table_counts(self) -> dict[str, int]:
        """Return row counts for all core tables without leaking direct SQL queries."""
        st = self.status()
        return {
            "files": st.get("files", 0),
            "chunks": st.get("chunks", 0),
            "symbols": st.get("symbols", 0),
            "syntax_errors": st.get("syntax_errors", 0),
            "skills": st.get("skills", 0),
            "contexts": st.get("contexts", 0),
        }

    def get_weak_points(self) -> dict[str, Any]:
        """Return codebase diagnostics (syntax error density, complexity hotspots, unindexed/stale files)."""
        return {
            "syntax_error_density_pct": 0.0,
            "complexity_hotspots": [],
            "unindexed_or_stale_files": [],
        }

    # =========================================================================
    # Compatibility & Convenience Aliases (Non-Abstract Defaults)
    # =========================================================================

    def search_chunks_bm25(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Convenience alias for search_chunks supporting single-string queries."""
        tokens = [t for t in query.split() if t.strip()]
        return self.search_chunks(tokens, top_k=limit)

    def upsert_symbols(self, symbols: list[SymbolRecord]) -> None:
        """Convenience alias for insert_symbols."""
        return self.insert_symbols(symbols)

    def search_symbols(self, query: str, limit: int = 10) -> list[SymbolRecord]:
        """Convenience alias for query_symbols."""
        return self.query_symbols(query, limit=limit)


class VectorCapable(ABC):
    """Mixin contract for backends that declare the ``'vector'`` capability.

    Vectors are L2-normalized ``list[float]``; distances are cosine distances
    (0 = identical). ``filters`` has the same keys as ``search_chunks`` filters.
    """

    @abstractmethod
    def ensure_vector_index(self, dim: int, model_id: str) -> None:
        """Create the vector index for ``dim`` and record ``model_id``/``dim`` in embed meta."""

    @abstractmethod
    def upsert_embeddings(self, items: list[tuple[int, list[float]]]) -> None:
        """Store ``(chunk_id, vector)`` pairs, replacing existing vectors."""

    @abstractmethod
    def search_vectors(self, vector: list[float], k: int,
                       filters: dict[str, Any] | None = None) -> list[tuple[int, float]]:
        """Return up to ``k`` ``(chunk_id, distance)`` pairs, nearest first."""

    @abstractmethod
    def chunks_missing_embeddings(self, limit: int) -> list[ChunkRecord]:
        """Chunks that have no stored vector yet (with content)."""

    @abstractmethod
    def get_embed_meta(self) -> dict[str, Any] | None:
        """``{"model_id": str, "dim": int}`` of the stored index, or None if never built."""

    @abstractmethod
    def drop_vector_index(self) -> None:
        """Delete all vectors and embed meta (used by ``ai-db reindex --embeddings``)."""

