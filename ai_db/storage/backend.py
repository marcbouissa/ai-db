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
