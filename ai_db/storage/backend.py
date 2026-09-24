"""StorageBackend Abstract Base Class for the ai-db storage layer.

Defines the pluggable storage contract isolating core code intelligence,
parsing, and search algorithms from database engine implementations.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Tuple, ContextManager

from ai_db.storage.models import (
    FileRecord,
    ChunkRecord,
    SymbolRecord,
    SymbolRefRecord,
    AnnotationRecord,
    SyntaxErrorRecord,
    SkillRecord,
    ContextRecord,
    AnalysisRefRecord,
    SearchResult,
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
        pass

    def capabilities(self) -> frozenset:
        """Feature set this backend supports: subset of {'fts', 'vector', 'graph'}."""
        return frozenset({"fts", "graph"})

    @abstractmethod
    def initialize(self) -> None:
        """Create tables, indexes, virtual FTS tables, and configure engine settings."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Gracefully close connections and release storage resources. Idempotent."""
        pass

    @abstractmethod
    def transaction(self) -> ContextManager[None]:
        """Context manager yielding an atomic transaction (commit on exit, rollback on exception)."""
        pass

    # =========================================================================
    # File Operations
    # =========================================================================

    @abstractmethod
    def upsert_file(self, record: FileRecord) -> None:
        """Insert or update a tracked file record."""
        pass

    @abstractmethod
    def get_file(self, filepath: str) -> Optional[FileRecord]:
        """Retrieve a file record by exact filepath, or return None."""
        pass

    @abstractmethod
    def delete_file(self, filepath: str) -> None:
        """Delete a file and cascade deletion across chunks, symbols, and references."""
        pass

    @abstractmethod
    def get_files_by_prefix(self, prefix: str) -> Dict[str, str]:
        """Return a mapping of {filepath: sha256} for all files with the given directory prefix."""
        pass

    @abstractmethod
    def get_all_filepaths(self) -> List[str]:
        """Return a list of all tracked file paths across all projects."""
        pass

    # =========================================================================
    # Chunk Operations & Code Search
    # =========================================================================

    @abstractmethod
    def insert_chunks(self, chunks: List[ChunkRecord]) -> None:
        """Persist chunk records and synchronize full-text search index."""
        pass

    def replace_file_chunks(self, filepath: str, chunks: List[ChunkRecord]) -> Dict[str, int]:
        """Make ``chunks`` the stored chunks of ``filepath``, keeping ids of unchanged
        chunks (matched by ``(content_hash, name)``) and resolving ``parent_index``.
        Returns ``{"kept", "inserted", "deleted"}`` counts."""
        raise NotImplementedError(f"{type(self).__name__} must implement replace_file_chunks")

    def get_chunks_by_ids(self, ids: List[int]) -> List[ChunkRecord]:
        """Chunks (with content) for ``ids``, in the given order; unknown ids are skipped."""
        raise NotImplementedError(f"{type(self).__name__} must implement get_chunks_by_ids")

    def rebuild_symbol_centrality(self) -> None:
        """('graph' capability) Recompute per-symbol normalized call in-degree."""
        raise NotImplementedError(f"{type(self).__name__} must implement rebuild_symbol_centrality")

    def get_symbol_centrality(self, names: List[str],
                              allowed_projects: Optional[List[str]] = None) -> Dict[str, float]:
        """('graph' capability) ``{bare_symbol_name: score in [0, 1]}``."""
        raise NotImplementedError(f"{type(self).__name__} must implement get_symbol_centrality")

    def get_refs_from(self, filepath: str, caller_scope: Optional[str],
                      ref_types: Tuple[str, ...] = ("call",)) -> List[SymbolRefRecord]:
        """('graph') References made inside ``caller_scope`` (``module.Class.method``);
        None means every scope of the file."""
        raise NotImplementedError(f"{type(self).__name__} must implement get_refs_from")

    def find_chunks_by_symbol(self, names: List[str], allowed_projects: Optional[List[str]] = None,
                              limit_per_name: int = 8) -> Dict[str, List[ChunkRecord]]:
        """('graph') Definition chunks whose last qualified-name component is in ``names``."""
        raise NotImplementedError(f"{type(self).__name__} must implement find_chunks_by_symbol")

    def get_chunk_by_qualified_name(self, filepath: str, qualified_name: str) -> Optional[ChunkRecord]:
        """First chunk of ``qualified_name`` in ``filepath`` or None."""
        raise NotImplementedError(f"{type(self).__name__} must implement get_chunk_by_qualified_name")

    def get_index_generation(self) -> int:
        """Counter incremented by every sync that changed the index."""
        raise NotImplementedError(f"{type(self).__name__} must implement get_index_generation")

    def bump_index_generation(self) -> int:
        """Increment the generation, delete cache rows of older generations, return it."""
        raise NotImplementedError(f"{type(self).__name__} must implement bump_index_generation")

    def get_query_cache(self, cache_key: str, index_gen: int) -> Optional[Any]:
        """Cached JSON result for ``cache_key`` at ``index_gen`` or None (miss)."""
        raise NotImplementedError(f"{type(self).__name__} must implement get_query_cache")

    def set_query_cache(self, cache_key: str, index_gen: int, result: Any) -> None:
        raise NotImplementedError(f"{type(self).__name__} must implement set_query_cache")

    def log_query(self, entry: Dict[str, Any]) -> None:
        """Append a query-log entry: timestamp, tool, query, mode, total_ms, cache_hit,
        stages {name: ms}, providers {stage: model_id}, top [[chunk_id, score], ...]."""
        raise NotImplementedError(f"{type(self).__name__} must implement log_query")

    def get_query_log(self, min_total_ms: float = 0.0, limit: int = 50) -> List[Dict[str, Any]]:
        """Newest-first log entries with ``total_ms >= min_total_ms``."""
        raise NotImplementedError(f"{type(self).__name__} must implement get_query_log")

    def clear_file_metadata(self, filepath: str) -> None:
        """Delete symbols, refs, annotations, syntax errors and analysis refs of a file
        (but not its file row or chunks)."""
        raise NotImplementedError(f"{type(self).__name__} must implement clear_file_metadata")

    @abstractmethod
    def get_chunks_for_file(self, filepath: str) -> List[ChunkRecord]:
        """Retrieve all chunks for a file, ordered by start_line, uncompressed."""
        pass

    @abstractmethod
    def search_chunks(
        self,
        query_tokens: List[str],
        allowed_projects: Optional[List[str]] = None,
        top_k: int = 5,
        path_prefix: Optional[str] = None,
    ) -> List[SearchResult]:
        """Execute full-text relevance search over code chunks within project scopes."""
        pass

    # =========================================================================
    # Symbol Operations & Cross-References
    # =========================================================================

    @abstractmethod
    def insert_symbols(self, symbols: List[SymbolRecord]) -> None:
        """Insert symbol records."""
        pass

    @abstractmethod
    def query_symbols(
        self,
        name: str,
        allowed_projects: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[SymbolRecord]:
        """Query symbols matching name (exact first, then substring) within project scopes."""
        pass

    @abstractmethod
    def insert_symbol_refs(self, refs: List[SymbolRefRecord]) -> None:
        """Insert symbol cross-references (callers, imports, inheritance)."""
        pass

    @abstractmethod
    def query_symbol_callers(
        self,
        callee_name: str,
        allowed_projects: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[SymbolRefRecord]:
        """Query all call sites referencing callee_name within project scopes."""
        pass

    # =========================================================================
    # Diagnostics & Annotations
    # =========================================================================

    @abstractmethod
    def upsert_syntax_error(self, error: SyntaxErrorRecord) -> None:
        """Insert or replace a syntax error diagnostic."""
        pass

    @abstractmethod
    def delete_syntax_error(self, filepath: str) -> None:
        """Delete syntax errors for a file."""
        pass

    @abstractmethod
    def get_syntax_errors(
        self,
        target_path: Optional[str] = None,
        allowed_projects: Optional[List[str]] = None,
    ) -> List[SyntaxErrorRecord]:
        """Retrieve syntax errors optionally filtered by target path prefix."""
        pass

    @abstractmethod
    def insert_annotations(self, annotations: List[AnnotationRecord]) -> None:
        """Insert code annotations (TODO/FIXME/docstrings)."""
        pass

    @abstractmethod
    def query_annotations(
        self,
        kind: Optional[str] = None,
        filepath: Optional[str] = None,
        allowed_projects: Optional[List[str]] = None,
        limit: int = 200,
    ) -> List[AnnotationRecord]:
        """Query code annotations filtered by kind and/or filepath."""
        pass

    # =========================================================================
    # Skills
    # =========================================================================

    @abstractmethod
    def get_skills(
        self,
        allowed_projects: Optional[List[str]] = None,
    ) -> List[SkillRecord]:
        """Retrieve all skills within allowed project scopes."""
        pass

    @abstractmethod
    def get_skills_by_project(self, project: str) -> Dict[str, Tuple[str, str]]:
        """Return {filepath: (name, sha256)} mapping for skills in a project."""
        pass

    @abstractmethod
    def upsert_skill(self, skill: SkillRecord) -> None:
        """Insert or update a skill and synchronize its FTS index."""
        pass

    @abstractmethod
    def delete_skill(self, filepath: str, project: str, name: Optional[str] = None) -> None:
        """Delete a skill and remove its FTS entry."""
        pass

    @abstractmethod
    def search_skills(
        self,
        query_tokens: List[str],
        allowed_projects: Optional[List[str]] = None,
        limit: int = 20,
    ) -> List[Tuple[str, float]]:
        """Search skills via FTS BM25, returning list of (skill_name, score)."""
        pass

    # =========================================================================
    # Context Memory
    # =========================================================================

    @abstractmethod
    def save_context(self, context: ContextRecord) -> None:
        """Persist a conversation session context snapshot."""
        pass

    @abstractmethod
    def get_context(
        self,
        session_id: Optional[str] = None,
        allowed_projects: Optional[List[str]] = None,
    ) -> Optional[ContextRecord]:
        """Retrieve latest or specified session context snapshot."""
        pass

    @abstractmethod
    def list_contexts(
        self,
        allowed_projects: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """List summary metadata for saved session contexts."""
        pass

    @abstractmethod
    def search_contexts(
        self,
        query_tokens: List[str],
        allowed_projects: Optional[List[str]] = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Search saved contexts via full-text search."""
        pass

    # =========================================================================
    # Analysis References
    # =========================================================================

    @abstractmethod
    def store_analysis_ref(self, ref: AnalysisRefRecord) -> None:
        """Store an opaque analysis progressive disclosure ref."""
        pass

    @abstractmethod
    def get_analysis_ref(self, ref_id: str) -> Optional[AnalysisRefRecord]:
        """Retrieve an analysis progressive disclosure ref by ID."""
        pass

    @abstractmethod
    def evict_stale_analysis_refs(self, older_than_seconds: float) -> int:
        """Evict analysis references older than the specified age in seconds."""
        pass

    # =========================================================================
    # Session State & Semantic Cache
    # =========================================================================

    @abstractmethod
    def get_state(self, key: str) -> Optional[Any]:
        """Retrieve deserialized session state value for key."""
        pass

    @abstractmethod
    def set_state(self, key: str, value: Any) -> None:
        """Store JSON-serializable session state value for key."""
        pass

    @abstractmethod
    def get_semantic_cache(self, cache_key: str, file_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached AST analysis result if file hash matches."""
        pass

    @abstractmethod
    def set_semantic_cache(self, cache_key: str, file_hash: str, result: Dict[str, Any]) -> None:
        """Save AST analysis result in semantic cache."""
        pass

    @abstractmethod
    def clear_semantic_cache(self) -> None:
        """Purge all entries in the semantic cache without affecting session state."""
        pass

    # =========================================================================
    # Maintenance & Health
    # =========================================================================

    @abstractmethod
    def status(self) -> Dict[str, Any]:
        """Return storage health statistics (file/chunk/symbol counts, size in KB)."""
        pass

    @abstractmethod
    def optimize(
        self,
        prune_missing: bool = True,
        default_format: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Optimize full-text indexes, prune missing files, vacuum storage, update format."""
        pass

    def get_table_counts(self) -> Dict[str, int]:
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

    def get_weak_points(self) -> Dict[str, Any]:
        """Return codebase diagnostics (syntax error density, complexity hotspots, unindexed/stale files)."""
        return {
            "syntax_error_density_pct": 0.0,
            "complexity_hotspots": [],
            "unindexed_or_stale_files": [],
        }

    # =========================================================================
    # Compatibility & Convenience Aliases (Non-Abstract Defaults)
    # =========================================================================

    def search_chunks_bm25(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Convenience alias for search_chunks supporting single-string queries."""
        tokens = [t for t in query.split() if t.strip()]
        return self.search_chunks(tokens, top_k=limit)

    def upsert_symbols(self, symbols: List[SymbolRecord]) -> None:
        """Convenience alias for insert_symbols."""
        return self.insert_symbols(symbols)

    def search_symbols(self, query: str, limit: int = 10) -> List[SymbolRecord]:
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
    def upsert_embeddings(self, items: List[Tuple[int, List[float]]]) -> None:
        """Store ``(chunk_id, vector)`` pairs, replacing existing vectors."""

    @abstractmethod
    def search_vectors(self, vector: List[float], k: int,
                       filters: Optional[Dict[str, Any]] = None) -> List[Tuple[int, float]]:
        """Return up to ``k`` ``(chunk_id, distance)`` pairs, nearest first."""

    @abstractmethod
    def chunks_missing_embeddings(self, limit: int) -> List[ChunkRecord]:
        """Chunks that have no stored vector yet (with content)."""

    @abstractmethod
    def get_embed_meta(self) -> Optional[Dict[str, Any]]:
        """``{"model_id": str, "dim": int}`` of the stored index, or None if never built."""

    @abstractmethod
    def drop_vector_index(self) -> None:
        """Delete all vectors and embed meta (used by ``ai-db reindex --embeddings``)."""

