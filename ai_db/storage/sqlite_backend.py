"""High-performance SQLite storage backend for ai-db (Milestone 2, Feature 8).

Implements StorageBackend ABC with WAL mode, FTS5 porter unicode61 full-text search,
BM25 scoring, transparent zlib (level 6) compression, savepoint-based nested
transactions, and one connection per thread (WAL allows concurrent readers).

``SQLiteBackend`` owns the connection, schema, transactions and maintenance; the data
access lives in one repository per concern (``ai_db/storage/sqlite_repos/``: chunks,
vectors, graph, memory, cache, state), which it composes over that connection.
"""

import json
import os
import sqlite3
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from ai_db.constants import (
    ANALYSIS_REF_TTL_S,
    DEFAULT_DB_FILE,
)
from ai_db.errors import AiDbConfigError
from ai_db.storage.backend import StorageBackend, VectorCapable
from ai_db.storage.sqlite_repos._base import (  # noqa: F401 - re-exported
    _NULL_LOCK,
    _WRITE_LOCK,
    _decompress,
    _loads,
    _writes,
    cosine_distance,
    pack_vector,
    unpack_vector,
)
from ai_db.storage.sqlite_repos.cache import CacheRepository
from ai_db.storage.sqlite_repos.chunks import ChunkRepository
from ai_db.storage.sqlite_repos.graph import GraphRepository
from ai_db.storage.sqlite_repos.memory import MemoryRepository
from ai_db.storage.sqlite_repos.state import StateRepository
from ai_db.storage.sqlite_repos.vectors import VectorRepository

SCHEMA_VERSION = "13"  # 13: module_summaries dropped; re-parse with the G1/G5 naming and query fixes

# Tables rebuilt (not migrated) when SCHEMA_VERSION changes; the next sync re-indexes.
INDEX_TABLES = ("fts_index", "chunks", "symbols", "symbol_refs", "annotations",
                "syntax_errors", "analysis_refs", "files", "semantic_cache")
# Derived tables dropped wholesale on a schema bump; repopulated by sync_skills /
# save_context in hybrid mode.
DERIVED_TABLES = ("skill_vectors", "context_vectors")
MIN_SQLITE_VERSION = (3, 35, 0)  # INSERT ... RETURNING


class SQLiteBackend(ChunkRepository, VectorRepository, GraphRepository, MemoryRepository,
                    CacheRepository, StateRepository, StorageBackend, VectorCapable):
    """SQLite implementation of the StorageBackend interface."""

    OPTION_KEYS = frozenset({"path", "vector_index"})

    @classmethod
    def from_options(cls, options: dict[str, Any]) -> "SQLiteBackend":
        """Entry-point constructor: ``options = {"path", "vector_index"}``."""
        unknown = set(options) - cls.OPTION_KEYS
        if unknown:
            raise AiDbConfigError(f"unknown sqlite storage option(s): {sorted(unknown)}")
        path = options.get("path")
        if path is not None and not isinstance(path, str):
            raise AiDbConfigError("storage.options.path must be a string or null")
        vector_index = options.get("vector_index", "exact")
        if vector_index not in cls.VECTOR_INDEX_MODES:
            raise AiDbConfigError(
                f"storage.options.vector_index must be one of "
                f"{list(cls.VECTOR_INDEX_MODES)}, got {vector_index!r}")
        return cls(path, vector_index=vector_index)  # None -> $AI_DB_PATH / XDG default

    def __init__(self, db_path: str | None = None, vector_index: str = "exact"):
        if sqlite3.sqlite_version_info < MIN_SQLITE_VERSION:
            raise AiDbConfigError(
                f"SQLite >= {'.'.join(map(str, MIN_SQLITE_VERSION))} required, "
                f"found {sqlite3.sqlite_version}"
            )
        if vector_index not in self.VECTOR_INDEX_MODES:
            raise AiDbConfigError(
                f"vector_index must be one of {list(self.VECTOR_INDEX_MODES)}, "
                f"got {vector_index!r}")
        self.vector_index = vector_index
        if db_path is None:
            db_path = os.environ.get("AI_DB_PATH", DEFAULT_DB_FILE)

        # Handle sqlite:/// URI prefixes
        if isinstance(db_path, str) and db_path.lower().startswith("sqlite:///"):
            sub = db_path[len("sqlite:///"):]
            if sub.lower() in (":memory:", "/:memory:"):
                db_path = ":memory:"
            else:
                db_path = sub

        self._closed = False
        self._local = threading.local()
        self._conns: list[sqlite3.Connection] = []
        self._conns_lock = threading.Lock()

        if isinstance(db_path, str) and db_path.lower() in (":memory:", "/:memory:"):
            self.db_path = ":memory:"
        else:
            self.db_path = os.path.abspath(db_path)
            parent = os.path.dirname(self.db_path)
            if parent:
                os.makedirs(parent, exist_ok=True)

        # An in-memory database exists only inside one connection, so it is shared.
        self._shared: sqlite3.Connection | None = (
            self._open_connection() if self.db_path == ":memory:" else None
        )
        if self._shared is None:
            self._local.conn = self._open_connection()

    def _open_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA temp_store = MEMORY;")
        conn.execute("PRAGMA cache_size = -64000;")
        self._configure_connection(conn)
        with self._conns_lock:
            self._conns.append(conn)
        return conn

    def _configure_connection(self, conn: sqlite3.Connection) -> None:
        """Load sqlite-vec on every connection (provides vec_distance_cosine)."""
        import sqlite_vec

        if not hasattr(conn, "enable_load_extension"):
            raise AiDbConfigError("this Python's sqlite3 cannot load extensions (needed for sqlite-vec)")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

    def capabilities(self) -> frozenset:
        return frozenset({"fts", "vector", "graph"})

    @property
    def conn(self) -> sqlite3.Connection:
        """The calling thread's connection (created on first use)."""
        if self._closed:
            raise RuntimeError("Storage backend is closed")
        if self._shared is not None:
            return self._shared
        existing = getattr(self._local, "conn", None)
        if existing is None:
            existing = self._open_connection()
            self._local.conn = existing
        return existing

    @property
    def _tx_depth(self) -> int:
        return getattr(self._local, "tx_depth", 0)

    @_tx_depth.setter
    def _tx_depth(self, value: int) -> None:
        self._local.tx_depth = value

    @property
    def backend_name(self) -> str:
        return "sqlite"

    def _check_closed(self) -> None:
        if self._closed:
            raise RuntimeError("Storage backend is closed")

    # =========================================================================
    # Lifecycle & Transactions
    # =========================================================================

    def initialize(self) -> None:
        self._check_closed()
        cur = self.conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS session_state (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                updated REAL NOT NULL
            )
        """)
        cur.execute("SELECT value_json FROM session_state WHERE key = 'schema_version'")
        row = cur.fetchone()
        stored_ver = json.loads(row["value_json"]) if row else None
        if stored_ver is not None and stored_ver != SCHEMA_VERSION:
            for table in INDEX_TABLES:
                cur.execute(f"DROP TABLE IF EXISTS {table}")
            for table in DERIVED_TABLES:
                cur.execute(f"DROP TABLE IF EXISTS {table}")
            self._drop_extra_index_tables(cur)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS files (
                filepath TEXT PRIMARY KEY,
                sha256 TEXT NOT NULL,
                last_modified REAL NOT NULL,
                chunk_count INTEGER NOT NULL,
                project TEXT DEFAULT 'global'
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filepath TEXT NOT NULL,
                chunk_type TEXT NOT NULL,
                name TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                zcontent BLOB NOT NULL,
                project TEXT DEFAULT 'global',
                qualified_name TEXT NOT NULL DEFAULT '',
                language TEXT NOT NULL DEFAULT '',
                token_count INTEGER NOT NULL DEFAULT 0,
                content_hash TEXT NOT NULL DEFAULT '',
                parent_id INTEGER NULL REFERENCES chunks(id) ON DELETE SET NULL,
                symbol_name TEXT NOT NULL DEFAULT '',
                FOREIGN KEY (filepath) REFERENCES files(filepath) ON DELETE CASCADE
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_filepath ON chunks(filepath)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_file_hash ON chunks(filepath, content_hash)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_symbol ON chunks(symbol_name)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_qualified ON chunks(filepath, qualified_name)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_project ON chunks(project)")

        cur.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_index USING fts5(
                name,
                qualified_name,
                filepath,
                content,
                idents,
                tokenize = 'porter unicode61'
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS syntax_errors (
                filepath TEXT PRIMARY KEY,
                line INTEGER,
                col INTEGER,
                message TEXT,
                timestamp REAL,
                project TEXT DEFAULT 'global'
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_syntax_errors_project ON syntax_errors(project)")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS symbols (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                symbol_type TEXT NOT NULL,
                filepath TEXT NOT NULL,
                line INTEGER NOT NULL,
                signature TEXT,
                project TEXT DEFAULT 'global',
                FOREIGN KEY (filepath) REFERENCES files(filepath) ON DELETE CASCADE
            )
        """)
        # Safety net against the same def being indexed twice (insert uses OR IGNORE).
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_symbol_site ON symbols(filepath, line, name)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_symbol_name ON symbols(name)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_symbol_filepath ON symbols(filepath)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_symbol_project ON symbols(project)")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS skills (
                name TEXT,
                description TEXT,
                filepath TEXT NOT NULL,
                triggers TEXT,
                sha256 TEXT NOT NULL,
                last_modified REAL NOT NULL,
                project TEXT DEFAULT 'global',
                PRIMARY KEY (name, project)
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_skills_project ON skills(project)")

        cur.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_skills USING fts5(
                name,
                description,
                triggers,
                content,
                project,
                tokenize = 'porter unicode61'
            )
        """)

        # Hybrid retrieval for skills: BLOB float32 vectors, brute-force cosine.
        # Plain BLOB (not vec0) so the dimension follows the configured embedder.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS skill_vectors (
                name TEXT NOT NULL,
                project TEXT NOT NULL,
                embedding BLOB NOT NULL,
                PRIMARY KEY (name, project)
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS contexts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                project TEXT NOT NULL,
                title TEXT,
                summary TEXT NOT NULL,
                active_files TEXT,
                open_tasks TEXT,
                timestamp REAL NOT NULL,
                zcontent BLOB NOT NULL,
                UNIQUE(session_id, project)
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_contexts_proj_sess ON contexts(project, session_id)")

        cur.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_contexts USING fts5(
                session_id,
                project,
                title,
                summary,
                content,
                tokenize = 'porter unicode61'
            )
        """)

        # Hybrid retrieval for contexts: cascades away with the parent row.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS context_vectors (
                context_id INTEGER PRIMARY KEY
                    REFERENCES contexts(id) ON DELETE CASCADE,
                embedding BLOB NOT NULL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS semantic_cache (
                cache_key TEXT PRIMARY KEY,
                file_hash TEXT NOT NULL,
                result_json TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_semantic_cache_hash ON semantic_cache(file_hash)")

        cur.execute("DROP TABLE IF EXISTS module_summaries")  # never written or read (K2)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS analysis_refs (
                ref_id TEXT PRIMARY KEY,
                filepath TEXT NOT NULL,
                name TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                kind TEXT NOT NULL,
                zbody BLOB NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_analysis_refs_file ON analysis_refs(filepath)")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS symbol_refs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                caller_filepath TEXT NOT NULL,
                caller_name TEXT NOT NULL,
                caller_line INTEGER NOT NULL,
                callee_name TEXT NOT NULL,
                ref_type TEXT NOT NULL,
                project TEXT DEFAULT 'global',
                call_line INTEGER NOT NULL DEFAULT 0,
                call_col INTEGER NOT NULL DEFAULT 0,
                callee_qualifier TEXT,
                callee_resolved TEXT,
                seq INTEGER,
                await_kind TEXT,
                guard TEXT,
                receiver TEXT
            )
        """)
        # One row per call site; INSERT OR IGNORE in insert_symbol_refs relies on this.
        cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_symrefs_site
            ON symbol_refs(caller_filepath, call_line, call_col, callee_name, ref_type)
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_symrefs_resolved ON symbol_refs(callee_resolved)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_symrefs_callee ON symbol_refs(callee_name)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_symrefs_caller ON symbol_refs(caller_filepath, caller_name)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_symrefs_project ON symbol_refs(project)")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filepath TEXT NOT NULL,
                line INTEGER NOT NULL,
                kind TEXT NOT NULL,
                symbol TEXT,
                content TEXT NOT NULL,
                project TEXT DEFAULT 'global'
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_annotations_filepath ON annotations(filepath)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_annotations_kind ON annotations(kind, project)")

        self._create_extra_tables(cur)

        if stored_ver != SCHEMA_VERSION:
            cur.execute("DROP TABLE IF EXISTS vec_chunks")
            cur.execute(
                "INSERT OR REPLACE INTO session_state (key, value_json, updated) VALUES (?, ?, ?)",
                ("schema_version", json.dumps(SCHEMA_VERSION), time.time())
            )

        self.conn.commit()

    def _drop_extra_index_tables(self, cur: sqlite3.Cursor) -> None:
        cur.execute("DROP TABLE IF EXISTS chunk_vectors")
        cur.execute("DROP TABLE IF EXISTS symbol_centrality")
        cur.execute("DROP TABLE IF EXISTS query_cache")
        cur.execute("DROP TABLE IF EXISTS query_log")
        cur.execute("DELETE FROM session_state WHERE key = 'index_gen'")
        cur.execute("DELETE FROM session_state WHERE key = 'embed_meta'")

    def _create_extra_tables(self, cur: sqlite3.Cursor) -> None:
        # Exact filtered KNN: vectors live in a plain table (FK-cascaded with chunks) and are
        # scored with sqlite-vec's vec_distance_cosine under the same filters as FTS.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS query_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                tool TEXT NOT NULL,
                query TEXT NOT NULL,
                mode TEXT,
                total_ms REAL NOT NULL,
                cache_hit INTEGER NOT NULL,
                stages_json TEXT NOT NULL,
                providers_json TEXT NOT NULL,
                top_json TEXT NOT NULL
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_query_log_total ON query_log(total_ms)")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS query_cache (
                cache_key TEXT PRIMARY KEY,
                index_gen INTEGER NOT NULL,
                result_json TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS symbol_centrality (
                project TEXT NOT NULL,
                name TEXT NOT NULL,
                in_degree INTEGER NOT NULL,
                score REAL NOT NULL,
                PRIMARY KEY (project, name)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chunk_vectors (
                chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
                embedding BLOB NOT NULL
            )
        """)

    def _vector_tables(self, cur: sqlite3.Cursor) -> list[str]:
        """Vector tables that actually exist (vec_chunks only exists in vec0 mode)."""
        present = {r[0] for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') "
            "AND name IN ('chunk_vectors','vec_chunks')").fetchall()}
        return [t for t in ("chunk_vectors", "vec_chunks") if t in present]

    def _before_delete_chunk_ids(self, cur: sqlite3.Cursor, ids: list[int]) -> None:
        """Hook: remove vector rows keyed by chunk id before chunks are deleted."""
        if not ids:
            return
        for table in self._vector_tables(cur):
            for i in range(0, len(ids), 900):
                part = ids[i:i + 900]
                cur.execute(
                    f"DELETE FROM {table} WHERE chunk_id IN ({','.join('?' * len(part))})",
                    part)

    def _before_delete_file_chunks(self, cur: sqlite3.Cursor, filepath: str) -> None:
        """Hook: remove vector rows for a file's chunks before they are deleted."""
        rows = cur.execute(
            "SELECT id FROM chunks WHERE filepath = ?", (filepath,)).fetchall()
        self._before_delete_chunk_ids(cur, [r["id"] for r in rows])

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._conns_lock:
            conns, self._conns = self._conns, []
        for c in conns:
            c.close()
        self._shared = None

    @contextmanager
    def transaction(self) -> Generator[None, None, None]:
        self._check_closed()
        # Only the outermost transaction serialises writers; nested savepoints on
        # the same thread already hold the RLock (it is reentrant).
        outermost = self._tx_depth == 0
        lock = _WRITE_LOCK if outermost else _NULL_LOCK
        with lock:
            self._tx_depth += 1
            sp_name = f"sp_level_{self._tx_depth}"
            try:
                self.conn.execute(f"SAVEPOINT {sp_name}")
                yield
                self.conn.execute(f"RELEASE SAVEPOINT {sp_name}")
                if self._tx_depth == 1:
                    self.conn.commit()
            except Exception:
                try:
                    self.conn.execute(f"ROLLBACK TO SAVEPOINT {sp_name}")
                    self.conn.execute(f"RELEASE SAVEPOINT {sp_name}")
                    if self._tx_depth == 1:
                        self.conn.rollback()
                except sqlite3.Error:
                    # The rollback itself failed (connection already rolled back by
                    # SQLite); the original exception is the one worth surfacing.
                    pass
                raise
            finally:
                self._tx_depth -= 1

    # =========================================================================
    # Status & Optimization
    # =========================================================================

    def status(self) -> dict[str, Any]:
        self._check_closed()
        counts = self.get_table_counts()

        size_bytes = os.path.getsize(self.db_path) if self.db_path != ":memory:" and os.path.exists(self.db_path) else 0

        return {
            "backend": "sqlite",
            "db": self.db_path,
            "files": counts.get("files", 0),
            "chunks": counts.get("chunks", 0),
            "symbols": counts.get("symbols", 0),
            "syntax_errors": counts.get("syntax_errors", 0),
            "skills": counts.get("skills", 0),
            "contexts": counts.get("contexts", 0),
            "kb": round(size_bytes / 1024, 1),
            "format": "zlib-compressed binary blob (token-dense)"
        }

    def optimize(self, prune_missing: bool = True, default_format: str | None = None) -> dict[str, Any]:
        self._check_closed()
        initial_size = os.path.getsize(self.db_path) if self.db_path != ":memory:" and os.path.exists(self.db_path) else 0
        pruned_files = 0

        cur = self.conn.cursor()
        if prune_missing and self.db_path != ":memory:":
            cur.execute("SELECT filepath FROM files")
            for row in cur.fetchall():
                fp = row["filepath"]
                if not os.path.exists(fp):
                    self.delete_file(fp)
                    pruned_files += 1

        for fts in ["fts_index", "fts_skills", "fts_contexts"]:
            cur.execute(f"INSERT INTO {fts}({fts}) VALUES('optimize')")
        # The old telemetry blob; telemetry is derived from query_log now (K5).
        cur.execute("DELETE FROM session_state WHERE key = 'telemetry_metrics_v1'")
        self.conn.commit()
        cur.execute("PRAGMA optimize")
        cur.execute("VACUUM")

        # Evict stale analysis refs older than 7 days
        self.evict_stale_analysis_refs(ANALYSIS_REF_TTL_S)

        if default_format:
            norm_fmt = default_format.strip().lower()
            if norm_fmt in ("stub", "sexp", "json", "outline", "prose"):
                self.set_state("default_format", norm_fmt)

        final_size = os.path.getsize(self.db_path) if self.db_path != ":memory:" and os.path.exists(self.db_path) else 0
        reclaimed_kb = round(max(0, initial_size - final_size) / 1024, 1)
        active_fmt = self.get_state("default_format") or "stub"

        return {
            "initial_kb": round(initial_size / 1024, 1),
            "final_kb": round(final_size / 1024, 1),
            "reclaimed_kb": reclaimed_kb,
            "pruned_files": pruned_files,
            "default_format": active_fmt
        }

    def wal_checkpoint(self, mode: str = "TRUNCATE") -> None:
        """Checkpoint the WAL file, optionally truncating it."""
        self._check_closed()
        valid_modes = {"PASSIVE", "FULL", "RESTART", "TRUNCATE"}
        if mode.upper() not in valid_modes:
            raise ValueError(f"Invalid WAL checkpoint mode: {mode}. Valid: {valid_modes}")
        self.conn.execute(f"PRAGMA wal_checkpoint({mode.upper()})")
