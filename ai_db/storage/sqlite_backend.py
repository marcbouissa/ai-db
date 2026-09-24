"""High-performance SQLite storage backend for ai-db (Milestone 2, Feature 8).

Implements StorageBackend ABC with WAL mode, FTS5 porter unicode61 full-text search,
BM25 scoring, transparent zlib (level 6) compression, savepoint-based nested
transactions, and one connection per thread (WAL allows concurrent readers).
"""

import json
import os
import sqlite3
import threading
import time
import zlib
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from ai_db.constants import DEFAULT_DB_FILE
from ai_db.errors import AiDbConfigError, AiDbQueryError, AiDbStorageError
from ai_db.search.query_builder import build_fts, identifier_words, split_identifier
from ai_db.storage.backend import StorageBackend, VectorCapable
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

ZLIB_LEVEL = 6
SCHEMA_VERSION = "8"
# Tables rebuilt (not migrated) when SCHEMA_VERSION changes; the next sync re-indexes.
INDEX_TABLES = ("fts_index", "chunks", "symbols", "symbol_refs", "annotations",
                "syntax_errors", "analysis_refs", "files", "semantic_cache")
MIN_SQLITE_VERSION = (3, 35, 0)  # INSERT ... RETURNING


def _decompress(blob: bytes, what: str) -> str:
    try:
        return zlib.decompress(blob).decode("utf-8", errors="replace")
    except zlib.error as exc:
        raise AiDbStorageError(f"corrupt compressed payload for {what}: {exc}") from exc


def _loads(raw: str, what: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AiDbStorageError(f"corrupt JSON in {what}: {exc}") from exc


class SQLiteBackend(StorageBackend, VectorCapable):
    """SQLite implementation of the StorageBackend interface."""

    OPTION_KEYS = frozenset({"path"})

    @classmethod
    def from_options(cls, options: dict[str, Any]) -> "SQLiteBackend":
        """Entry-point constructor: ``options = {"path": str | None}``."""
        unknown = set(options) - cls.OPTION_KEYS
        if unknown:
            raise AiDbConfigError(f"unknown sqlite storage option(s): {sorted(unknown)}")
        path = options.get("path")
        if path is not None and not isinstance(path, str):
            raise AiDbConfigError("storage.options.path must be a string or null")
        return cls(path)  # None -> $AI_DB_PATH or the XDG default location

    def __init__(self, db_path: str | None = None):
        if sqlite3.sqlite_version_info < MIN_SQLITE_VERSION:
            raise AiDbConfigError(
                f"SQLite >= {'.'.join(map(str, MIN_SQLITE_VERSION))} required, "
                f"found {sqlite3.sqlite_version}"
            )
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

    def _auto_commit(self) -> None:
        if self._tx_depth == 0:
            self.conn.commit()

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

        cur.execute("""
            CREATE TABLE IF NOT EXISTS semantic_cache (
                cache_key TEXT PRIMARY KEY,
                file_hash TEXT NOT NULL,
                result_json TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_semantic_cache_hash ON semantic_cache(file_hash)")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS module_summaries (
                filepath TEXT PRIMARY KEY,
                sha256 TEXT NOT NULL,
                summary TEXT NOT NULL,
                updated REAL NOT NULL
            )
        """)

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
                project TEXT DEFAULT 'global'
            )
        """)
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

    def _before_delete_chunk_ids(self, cur: sqlite3.Cursor, ids: list[int]) -> None:
        """Hook: remove rows keyed by chunk id before chunks are deleted."""

    def _before_delete_file_chunks(self, cur: sqlite3.Cursor, filepath: str) -> None:
        """Hook: remove rows keyed by chunk id before a file's chunks are deleted."""

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
        cur = self.conn.cursor()
        counts = {}
        for tbl in ["files", "chunks", "symbols", "syntax_errors", "skills", "contexts"]:
            cur.execute(f"SELECT COUNT(*) as c FROM {tbl}")
            counts[tbl] = cur.fetchone()["c"]

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
        self.conn.commit()
        cur.execute("PRAGMA optimize")
        cur.execute("VACUUM")

        # Evict stale analysis refs older than 7 days
        self.evict_stale_analysis_refs(86400 * 7)

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

    # =========================================================================
    # Files
    # =========================================================================

    def get_file(self, filepath: str) -> FileRecord | None:
        self._check_closed()
        cur = self.conn.cursor()
        cur.execute(
            "SELECT filepath, sha256, last_modified, chunk_count, project FROM files WHERE filepath = ?",
            (filepath,)
        )
        row = cur.fetchone()
        if not row:
            return None
        return FileRecord(
            filepath=row["filepath"],
            sha256=row["sha256"],
            last_modified=float(row["last_modified"]),
            chunk_count=int(row["chunk_count"]),
            project=row["project"]
        )

    def get_files_by_prefix(self, prefix: str) -> dict[str, str]:
        self._check_closed()
        prefix = prefix.replace("\x00", "") if prefix else ""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT filepath, sha256 FROM files WHERE filepath = ? OR filepath LIKE ?",
            (prefix, f"{prefix.rstrip('/')}/%")
        )
        return {r["filepath"]: r["sha256"] for r in cur.fetchall()}

    def get_all_filepaths(self) -> list[str]:
        self._check_closed()
        cur = self.conn.cursor()
        cur.execute("SELECT filepath FROM files")
        return [r["filepath"] for r in cur.fetchall()]

    def upsert_file(self, record: FileRecord) -> None:
        self._check_closed()
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO files (filepath, sha256, last_modified, chunk_count, project)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(filepath) DO UPDATE SET
                sha256 = excluded.sha256,
                last_modified = excluded.last_modified,
                chunk_count = excluded.chunk_count,
                project = excluded.project
            """,
            (record.filepath, record.sha256, record.last_modified, record.chunk_count, record.project)
        )
        self._auto_commit()

    def delete_file(self, filepath: str) -> None:
        self._check_closed()
        cur = self.conn.cursor()
        cur.execute("DELETE FROM fts_index WHERE rowid IN (SELECT id FROM chunks WHERE filepath = ?)",
                    (filepath,))
        self._before_delete_file_chunks(cur, filepath)
        cur.execute("DELETE FROM chunks WHERE filepath = ?", (filepath,))
        cur.execute("DELETE FROM symbols WHERE filepath = ?", (filepath,))
        cur.execute("DELETE FROM symbol_refs WHERE caller_filepath = ?", (filepath,))
        cur.execute("DELETE FROM annotations WHERE filepath = ?", (filepath,))
        cur.execute("DELETE FROM syntax_errors WHERE filepath = ?", (filepath,))
        cur.execute("DELETE FROM analysis_refs WHERE filepath = ?", (filepath,))
        cur.execute("DELETE FROM files WHERE filepath = ?", (filepath,))
        self._auto_commit()

    # =========================================================================
    # Chunks & Full-Text Search (BM25)
    # =========================================================================

    _CHUNK_COLS = ("filepath, chunk_type, name, start_line, end_line, zcontent, project, "
                   "qualified_name, language, token_count, content_hash, symbol_name")
    SYMBOL_CHUNK_TYPES = ("code", "class_header")

    def _insert_chunk_rows(self, chunks: list[ChunkRecord]) -> None:
        """Insert chunks + FTS rows and assign ``c.id``. Does not resolve parents."""
        if not chunks:
            return
        cur = self.conn.cursor()
        rows = [
            (c.filepath, c.chunk_type, c.name, c.start_line, c.end_line,
             zlib.compress(c.content.encode("utf-8"), level=ZLIB_LEVEL), c.project,
             c.qualified_name, c.language, c.token_count, c.content_hash,
             c.qualified_name.rsplit(".", 1)[-1] if c.chunk_type in self.SYMBOL_CHUNK_TYPES else "")
            for c in chunks
        ]
        width = len(rows[0])
        batch = 32000 // width  # SQLite host-parameter limit is 32766
        ids: list[int] = []
        for i in range(0, len(rows), batch):
            part = rows[i:i + batch]
            placeholders = ",".join(["(" + ",".join("?" * width) + ")"] * len(part))
            cur.execute(
                f"INSERT INTO chunks ({self._CHUNK_COLS}) VALUES {placeholders} RETURNING id",
                [v for row in part for v in row],
            )
            # RETURNING order is unspecified; AUTOINCREMENT ids ascend in insertion order.
            ids.extend(sorted(r[0] for r in cur.fetchall()))
        if len(ids) != len(chunks):
            raise AiDbStorageError(f"inserted {len(chunks)} chunks but got {len(ids)} ids")
        for c, chunk_id in zip(chunks, ids):
            c.id = chunk_id
        self._index_chunks_fts(cur, chunks)

    def _index_chunks_fts(self, cur: sqlite3.Cursor, chunks: list[ChunkRecord]) -> None:
        cur.executemany(
            "INSERT INTO fts_index (rowid, name, qualified_name, filepath, content, idents) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [(c.id, c.name, " ".join(split_identifier(c.qualified_name.replace(".", "_"))) + " "
              + c.qualified_name, c.filepath, c.content,
              identifier_words(c.name + " " + c.content)) for c in chunks],
        )

    def _resolve_parents(self, chunks: list[ChunkRecord]) -> None:
        updates = []
        for c in chunks:
            if c.parent_index is not None:
                if not 0 <= c.parent_index < len(chunks):
                    raise AiDbStorageError(f"chunk {c.name}: parent_index {c.parent_index} out of range")
                c.parent_id = chunks[c.parent_index].id
            updates.append((c.parent_id, c.id))
        self.conn.executemany("UPDATE chunks SET parent_id = ? WHERE id = ?", updates)

    def _delete_chunk_ids(self, ids: list[int]) -> None:
        if not ids:
            return
        cur = self.conn.cursor()
        cur.executemany("DELETE FROM fts_index WHERE rowid = ?", [(i,) for i in ids])
        self._before_delete_chunk_ids(cur, ids)
        cur.executemany("DELETE FROM chunks WHERE id = ?", [(i,) for i in ids])

    def insert_chunks(self, chunks: list[ChunkRecord]) -> None:
        self._check_closed()
        self._insert_chunk_rows(chunks)
        self._resolve_parents(chunks)
        self._auto_commit()

    def replace_file_chunks(self, filepath: str, chunks: list[ChunkRecord]) -> dict[str, int]:
        """Diff ``chunks`` against the stored chunks of ``filepath`` by (content_hash, name).

        Unchanged chunks keep their id (and therefore their embedding); their line span
        and parent are updated. Returns counts of kept/inserted/deleted chunks.
        """
        self._check_closed()
        cur = self.conn.cursor()
        cur.execute("SELECT id, content_hash, name FROM chunks WHERE filepath = ? ORDER BY id", (filepath,))
        pool: dict[tuple[str, str], list[int]] = {}
        for r in cur.fetchall():
            pool.setdefault((r["content_hash"], r["name"]), []).append(r["id"])
        kept: list[ChunkRecord] = []
        new: list[ChunkRecord] = []
        for c in chunks:
            ids = pool.get((c.content_hash, c.name))
            if ids:
                c.id = ids.pop(0)
                kept.append(c)
            else:
                new.append(c)
        stale = [i for ids in pool.values() for i in ids]
        self._delete_chunk_ids(stale)
        cur.executemany(
            "UPDATE chunks SET start_line = ?, end_line = ?, chunk_type = ?, project = ?, "
            "qualified_name = ?, language = ? WHERE id = ?",
            [(c.start_line, c.end_line, c.chunk_type, c.project, c.qualified_name, c.language, c.id)
             for c in kept],
        )
        self._insert_chunk_rows(new)
        self._resolve_parents(chunks)
        self._auto_commit()
        return {"kept": len(kept), "inserted": len(new), "deleted": len(stale)}

    def clear_file_metadata(self, filepath: str) -> None:
        """Delete everything derived from ``filepath`` except its file row and chunks."""
        self._check_closed()
        cur = self.conn.cursor()
        cur.execute("DELETE FROM symbols WHERE filepath = ?", (filepath,))
        cur.execute("DELETE FROM symbol_refs WHERE caller_filepath = ?", (filepath,))
        cur.execute("DELETE FROM annotations WHERE filepath = ?", (filepath,))
        cur.execute("DELETE FROM syntax_errors WHERE filepath = ?", (filepath,))
        cur.execute("DELETE FROM analysis_refs WHERE filepath = ?", (filepath,))
        self._auto_commit()

    def get_chunks_for_file(self, filepath: str) -> list[ChunkRecord]:
        self._check_closed()
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT id, filepath, chunk_type, name, start_line, end_line, zcontent, project,
                   qualified_name, language, token_count, content_hash, parent_id
            FROM chunks WHERE filepath = ? ORDER BY start_line ASC, id ASC
            """,
            (filepath,)
        )
        result = []
        for r in cur.fetchall():
            content = _decompress(r["zcontent"], f"chunk {r['id']}")
            result.append(ChunkRecord(
                id=r["id"],
                filepath=r["filepath"],
                chunk_type=r["chunk_type"],
                name=r["name"],
                start_line=r["start_line"],
                end_line=r["end_line"],
                content=content,
                project=r["project"],
                qualified_name=r["qualified_name"],
                language=r["language"],
                token_count=r["token_count"],
                content_hash=r["content_hash"],
                parent_id=r["parent_id"],
            ))
        return result

    # bm25 column weights: name, qualified_name, filepath, content, idents
    BM25_WEIGHTS = (8.0, 6.0, 2.0, 1.0, 1.0)

    def search_chunks(
        self,
        query_tokens: list[str],
        allowed_projects: list[str] | None = None,
        top_k: int = 5,
        path_prefix: str | None = None,
        core_terms: list[str] | None = None,
        languages: list[str] | None = None,
        chunk_types: list[str] | None = None,
        modified_since: float | None = None,
    ) -> list[SearchResult]:
        """BM25 search. ``query_tokens`` are all (expanded) terms; ``core_terms`` the
        user's base words used for the AND/NEAR groups (defaults to all terms)."""
        self._check_closed()
        if allowed_projects is not None and len(allowed_projects) == 0:
            return []
        clean_tokens = [t.replace("\x00", "").strip() for t in query_tokens]
        clean_tokens = [t for t in clean_tokens if t]
        if not clean_tokens:
            return []
        fts_query = build_fts(clean_tokens, core_terms)

        where_clauses = ["fts_index MATCH ?"]
        params: list[Any] = [fts_query]
        self._append_filters(where_clauses, params, allowed_projects, path_prefix,
                             languages, chunk_types, modified_since)
        params.append(top_k)
        w = ", ".join(str(x) for x in self.BM25_WEIGHTS)
        sql = f"""
            SELECT chunks.id AS chunk_id, chunks.filepath, chunks.name,
                   bm25(fts_index, {w}) AS bm25_rank,
                   snippet(fts_index, 3, '«', '»', '…', 32) AS snip,
                   chunks.start_line, chunks.end_line, chunks.chunk_type, chunks.project,
                   chunks.qualified_name, chunks.language, chunks.parent_id
            FROM fts_index
            JOIN chunks ON fts_index.rowid = chunks.id
            JOIN files ON files.filepath = chunks.filepath
            WHERE {" AND ".join(where_clauses)}
            ORDER BY bm25_rank
            LIMIT ?
        """
        cur = self.conn.cursor()
        try:
            cur.execute(sql, params)
        except sqlite3.OperationalError as exc:
            raise AiDbQueryError(fts_query, exc) from exc

        results = []
        for r in cur.fetchall():
            raw_score = abs(float(r["bm25_rank"]))
            results.append(SearchResult(
                chunk_id=r["chunk_id"],
                filepath=r["filepath"],
                name=r["name"],
                chunk_type=r["chunk_type"],
                project=r["project"],
                start_line=r["start_line"],
                end_line=r["end_line"],
                score=round(raw_score, 8) if raw_score else 0.001,
                snippet=(r["snip"] or "").strip(),
                qualified_name=r["qualified_name"],
                language=r["language"],
                parent_id=r["parent_id"],
            ))
        return results

    @staticmethod
    def _append_filters(where: list[str], params: list[Any],
                        allowed_projects: list[str] | None, path_prefix: str | None,
                        languages: list[str] | None, chunk_types: list[str] | None,
                        modified_since: float | None) -> None:
        """Shared chunk filters (expects ``chunks`` and ``files`` in the FROM clause)."""
        if allowed_projects is not None:
            where.append(f"chunks.project IN ({','.join('?' * len(allowed_projects))})")
            params.extend(allowed_projects)
        if path_prefix:
            where.append("chunks.filepath LIKE ?")
            params.append(f"{path_prefix.replace(chr(0), '').rstrip('/')}/%")
        if languages:
            where.append(f"chunks.language IN ({','.join('?' * len(languages))})")
            params.extend(languages)
        if chunk_types:
            where.append(f"chunks.chunk_type IN ({','.join('?' * len(chunk_types))})")
            params.extend(chunk_types)
        if modified_since is not None:
            where.append("files.last_modified >= ?")
            params.append(float(modified_since))

    # =========================================================================
    # Vectors (VectorCapable)
    # =========================================================================

    def get_embed_meta(self) -> dict[str, Any] | None:
        self._check_closed()
        return self.get_state("embed_meta")

    def ensure_vector_index(self, dim: int, model_id: str) -> None:
        self._check_closed()
        meta = self.get_embed_meta()
        wanted = {"model_id": model_id, "dim": int(dim)}
        if meta is None:
            self.set_state("embed_meta", wanted)
        elif meta != wanted:
            raise AiDbConfigError(
                f"vector index was built with {meta}, config wants {wanted}; "
                "run: ai-db reindex --embeddings"
            )

    def upsert_embeddings(self, items: list[tuple[int, list[float]]]) -> None:
        self._check_closed()
        import sqlite_vec

        meta = self.get_embed_meta()
        if meta is None:
            raise AiDbStorageError("upsert_embeddings called before ensure_vector_index")
        for chunk_id, vec in items:
            if len(vec) != meta["dim"]:
                raise AiDbStorageError(f"chunk {chunk_id}: vector dim {len(vec)} != index dim {meta['dim']}")
        self.conn.executemany(
            "INSERT OR REPLACE INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
            [(cid, sqlite_vec.serialize_float32(vec)) for cid, vec in items],
        )
        self._auto_commit()

    def search_vectors(self, vector: list[float], k: int,
                       filters: dict[str, Any] | None = None) -> list[tuple[int, float]]:
        self._check_closed()
        import sqlite_vec

        f = dict(filters or {})
        allowed = f.pop("allowed_projects", None)
        if allowed is not None and len(allowed) == 0:
            return []
        where: list[str] = []
        params: list[Any] = [sqlite_vec.serialize_float32(vector)]
        self._append_filters(where, params, allowed, f.pop("path_prefix", None),
                             f.pop("languages", None), f.pop("chunk_types", None),
                             f.pop("modified_since", None))
        if f:
            raise ValueError(f"unknown vector filter(s): {sorted(f)}")
        params.append(int(k))
        where_sql = ("WHERE " + " AND ".join(where)) if where else ""
        cur = self.conn.execute(
            f"""
            SELECT chunks.id AS chunk_id, vec_distance_cosine(cv.embedding, ?) AS distance
            FROM chunk_vectors cv
            JOIN chunks ON chunks.id = cv.chunk_id
            JOIN files ON files.filepath = chunks.filepath
            {where_sql}
            ORDER BY distance ASC
            LIMIT ?
            """,
            params,
        )
        return [(r["chunk_id"], float(r["distance"])) for r in cur.fetchall()]

    def chunks_missing_embeddings(self, limit: int) -> list[ChunkRecord]:
        self._check_closed()
        cur = self.conn.execute(
            """
            SELECT c.id FROM chunks c
            LEFT JOIN chunk_vectors cv ON cv.chunk_id = c.id
            WHERE cv.chunk_id IS NULL
            ORDER BY c.id LIMIT ?
            """,
            (int(limit),),
        )
        return self.get_chunks_by_ids([r["id"] for r in cur.fetchall()])

    def drop_vector_index(self) -> None:
        self._check_closed()
        self.conn.execute("DELETE FROM chunk_vectors")
        self.conn.execute("DELETE FROM session_state WHERE key = 'embed_meta'")
        self._auto_commit()

    def get_chunks_by_ids(self, ids: list[int]) -> list[ChunkRecord]:
        """Chunks for ``ids`` in the same order (missing ids are skipped)."""
        self._check_closed()
        if not ids:
            return []
        rows: dict[int, Any] = {}
        for i in range(0, len(ids), 900):
            part = ids[i:i + 900]
            cur = self.conn.execute(
                f"""SELECT id, filepath, chunk_type, name, start_line, end_line, zcontent, project,
                           qualified_name, language, token_count, content_hash, parent_id
                    FROM chunks WHERE id IN ({",".join("?" * len(part))})""",
                part,
            )
            for r in cur.fetchall():
                rows[r["id"]] = r
        return [self._row_to_chunk(rows[i]) for i in ids if i in rows]

    @staticmethod
    def _row_to_chunk(r: Any) -> ChunkRecord:
        return ChunkRecord(
            id=r["id"], filepath=r["filepath"], chunk_type=r["chunk_type"], name=r["name"],
            start_line=r["start_line"], end_line=r["end_line"],
            content=_decompress(r["zcontent"], f"chunk {r['id']}"), project=r["project"],
            qualified_name=r["qualified_name"], language=r["language"],
            token_count=r["token_count"], content_hash=r["content_hash"], parent_id=r["parent_id"],
        )

    # =========================================================================
    # Graph signals
    # =========================================================================

    def rebuild_symbol_centrality(self) -> None:
        """Recompute normalized call/inherit in-degree per (project, symbol name)."""
        self._check_closed()
        cur = self.conn.cursor()
        import math

        cur.execute("DELETE FROM symbol_centrality")
        cur.execute("""
            SELECT project, callee_name, COUNT(*) AS n FROM symbol_refs
            WHERE ref_type IN ('call', 'inherit')
            GROUP BY project, callee_name
        """)
        rows = cur.fetchall()
        max_by_project: dict[str, int] = {}
        for r in rows:
            max_by_project[r["project"]] = max(max_by_project.get(r["project"], 0), r["n"])
        cur.executemany(
            "INSERT INTO symbol_centrality (project, name, in_degree, score) VALUES (?, ?, ?, ?)",
            [(r["project"], r["callee_name"], r["n"],
              math.log1p(r["n"]) / math.log1p(max_by_project[r["project"]])) for r in rows],
        )
        self._auto_commit()

    def get_symbol_centrality(self, names: list[str],
                              allowed_projects: list[str] | None = None) -> dict[str, float]:
        """Max normalized in-degree score per bare symbol name (0..1)."""
        self._check_closed()
        names = sorted(set(names))
        if not names or (allowed_projects is not None and not allowed_projects):
            return {}
        params: list[Any] = list(names)
        where = f"name IN ({','.join('?' * len(names))})"
        if allowed_projects is not None:
            where += f" AND project IN ({','.join('?' * len(allowed_projects))})"
            params.extend(allowed_projects)
        cur = self.conn.execute(
            f"SELECT name, MAX(score) AS s FROM symbol_centrality WHERE {where} GROUP BY name", params)
        return {r["name"]: float(r["s"]) for r in cur.fetchall()}

    def get_refs_from(self, filepath: str, caller_scope: str | None,
                      ref_types: tuple[str, ...] = ("call",)) -> list[SymbolRefRecord]:
        """References made inside ``caller_scope`` (e.g. ``module.Class.method``) of a file;
        ``caller_scope=None`` returns references from every scope of the file."""
        self._check_closed()
        scope_sql = "" if caller_scope is None else " AND caller_name = ?"
        scope_params = [] if caller_scope is None else [caller_scope]
        cur = self.conn.execute(
            f"""SELECT id, caller_filepath, caller_name, caller_line, callee_name, ref_type, project
                FROM symbol_refs WHERE caller_filepath = ?{scope_sql}
                AND ref_type IN ({",".join("?" * len(ref_types))})
                ORDER BY caller_line""",
            [filepath, *scope_params, *ref_types],
        )
        return [SymbolRefRecord(id=r["id"], caller_filepath=r["caller_filepath"],
                                caller_name=r["caller_name"], caller_line=r["caller_line"],
                                callee_name=r["callee_name"], ref_type=r["ref_type"],
                                project=r["project"]) for r in cur.fetchall()]

    def find_chunks_by_symbol(self, names: list[str], allowed_projects: list[str] | None = None,
                              limit_per_name: int = 8) -> dict[str, list[ChunkRecord]]:
        """Definition chunks whose last qualified-name component is in ``names``."""
        self._check_closed()
        names = sorted(set(names))
        if not names or (allowed_projects is not None and not allowed_projects):
            return {}
        params: list[Any] = list(names)
        where = f"symbol_name IN ({','.join('?' * len(names))})"
        if allowed_projects is not None:
            where += f" AND project IN ({','.join('?' * len(allowed_projects))})"
            params.extend(allowed_projects)
        cur = self.conn.execute(
            f"SELECT id, symbol_name FROM chunks WHERE {where} ORDER BY symbol_name, id", params)
        ids_by_name: dict[str, list[int]] = {}
        for r in cur.fetchall():
            bucket = ids_by_name.setdefault(r["symbol_name"], [])
            if len(bucket) < limit_per_name:
                bucket.append(r["id"])
        all_ids = [i for ids in ids_by_name.values() for i in ids]
        by_id = {c.id: c for c in self.get_chunks_by_ids(all_ids)}
        return {n: [by_id[i] for i in ids if i in by_id] for n, ids in ids_by_name.items()}

    def get_chunk_by_qualified_name(self, filepath: str, qualified_name: str) -> ChunkRecord | None:
        """First chunk (lowest start line) of ``qualified_name`` in ``filepath``."""
        self._check_closed()
        cur = self.conn.execute(
            "SELECT id FROM chunks WHERE filepath = ? AND qualified_name = ? ORDER BY start_line LIMIT 1",
            (filepath, qualified_name))
        row = cur.fetchone()
        if row is None:
            return None
        return self.get_chunks_by_ids([row["id"]])[0]

    # =========================================================================
    # Query-result cache (invalidated by index generation)
    # =========================================================================

    def get_index_generation(self) -> int:
        self._check_closed()
        return int(self.get_state("index_gen") or 0)

    def bump_index_generation(self) -> int:
        """Increment the generation and drop cache rows from older generations."""
        self._check_closed()
        gen = self.get_index_generation() + 1
        self.set_state("index_gen", gen)
        self.conn.execute("DELETE FROM query_cache WHERE index_gen != ?", (gen,))
        self._auto_commit()
        return gen

    def get_query_cache(self, cache_key: str, index_gen: int) -> Any | None:
        self._check_closed()
        row = self.conn.execute(
            "SELECT result_json FROM query_cache WHERE cache_key = ? AND index_gen = ?",
            (cache_key, index_gen)).fetchone()
        return json.loads(row["result_json"]) if row else None

    def set_query_cache(self, cache_key: str, index_gen: int, result: Any) -> None:
        self._check_closed()
        self.conn.execute(
            "INSERT OR REPLACE INTO query_cache (cache_key, index_gen, result_json, timestamp) "
            "VALUES (?, ?, ?, ?)",
            (cache_key, index_gen, json.dumps(result, ensure_ascii=False), time.time()))
        self._auto_commit()

    # =========================================================================
    # Query log (observability)
    # =========================================================================

    QUERY_LOG_KEEP = 10000

    def log_query(self, entry: dict[str, Any]) -> None:
        """Append one query-log row; keeps the newest QUERY_LOG_KEEP rows."""
        self._check_closed()
        cur = self.conn.execute(
            "INSERT INTO query_log (timestamp, tool, query, mode, total_ms, cache_hit, "
            "stages_json, providers_json, top_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (entry["timestamp"], entry["tool"], entry["query"], entry.get("mode"),
             float(entry["total_ms"]), int(bool(entry["cache_hit"])),
             json.dumps(entry.get("stages", {})), json.dumps(entry.get("providers", {})),
             json.dumps(entry.get("top", []))))
        self.conn.execute("DELETE FROM query_log WHERE id <= ?",
                          ((cur.lastrowid or 0) - self.QUERY_LOG_KEEP,))
        self._auto_commit()

    def get_query_log(self, min_total_ms: float = 0.0, limit: int = 50) -> list[dict[str, Any]]:
        self._check_closed()
        cur = self.conn.execute(
            "SELECT * FROM query_log WHERE total_ms >= ? ORDER BY id DESC LIMIT ?",
            (float(min_total_ms), int(limit)))
        return [{
            "timestamp": r["timestamp"], "tool": r["tool"], "query": r["query"], "mode": r["mode"],
            "total_ms": r["total_ms"], "cache_hit": bool(r["cache_hit"]),
            "stages": json.loads(r["stages_json"]), "providers": json.loads(r["providers_json"]),
            "top": json.loads(r["top_json"]),
        } for r in cur.fetchall()]

    # =========================================================================
    # Symbols & Cross-References
    # =========================================================================

    def insert_symbols(self, symbols: list[SymbolRecord]) -> None:
        self._check_closed()
        cur = self.conn.cursor()
        cur.executemany(
            """
            INSERT INTO symbols (name, symbol_type, filepath, line, signature, project)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [(s.name, s.symbol_type, s.filepath, s.line, s.signature, s.project) for s in symbols]
        )
        self._auto_commit()

    def query_symbols(
        self,
        name: str,
        allowed_projects: list[str] | None = None,
        limit: int = 50,
    ) -> list[SymbolRecord]:
        self._check_closed()
        if allowed_projects is not None and len(allowed_projects) == 0:
            return []
        cur = self.conn.cursor()
        clauses = ["(name = ? OR name LIKE ?)"]
        params: list[Any] = [name, f"{name}%"]

        if allowed_projects is not None:
            placeholders = ",".join("?" for _ in allowed_projects)
            clauses.append(f"project IN ({placeholders})")
            params.extend(allowed_projects)

        where = " AND ".join(clauses)
        params.extend([name, limit])

        cur.execute(
            f"""
            SELECT id, name, symbol_type, filepath, line, signature, project
            FROM symbols
            WHERE {where}
            ORDER BY CASE WHEN name = ? THEN 0 ELSE 1 END, line ASC
            LIMIT ?
            """,
            params
        )
        return [
            SymbolRecord(
                id=r["id"],
                name=r["name"],
                symbol_type=r["symbol_type"],
                filepath=r["filepath"],
                line=r["line"],
                signature=r["signature"],
                project=r["project"]
            )
            for r in cur.fetchall()
        ]

    def insert_symbol_refs(self, refs: list[SymbolRefRecord]) -> None:
        self._check_closed()
        cur = self.conn.cursor()
        cur.executemany(
            """
            INSERT INTO symbol_refs (caller_filepath, caller_name, caller_line, callee_name, ref_type, project)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [(r.caller_filepath, r.caller_name, r.caller_line, r.callee_name, r.ref_type, r.project) for r in refs]
        )
        self._auto_commit()

    def query_symbol_callers(
        self,
        callee_name: str,
        allowed_projects: list[str] | None = None,
        limit: int = 100,
    ) -> list[SymbolRefRecord]:
        self._check_closed()
        if allowed_projects is not None and len(allowed_projects) == 0:
            return []
        cur = self.conn.cursor()
        clauses = ["callee_name = ?"]
        params: list[Any] = [callee_name]

        if allowed_projects is not None:
            placeholders = ",".join("?" for _ in allowed_projects)
            clauses.append(f"project IN ({placeholders})")
            params.extend(allowed_projects)

        where = " AND ".join(clauses)
        params.append(limit)

        cur.execute(
            f"""
            SELECT id, caller_filepath, caller_name, caller_line, callee_name, ref_type, project
            FROM symbol_refs
            WHERE {where}
            ORDER BY caller_filepath ASC, caller_line ASC
            LIMIT ?
            """,
            params
        )
        return [
            SymbolRefRecord(
                id=r["id"],
                caller_filepath=r["caller_filepath"],
                caller_name=r["caller_name"],
                caller_line=r["caller_line"],
                callee_name=r["callee_name"],
                ref_type=r["ref_type"],
                project=r["project"]
            )
            for r in cur.fetchall()
        ]

    # =========================================================================
    # Diagnostics & Annotations
    # =========================================================================

    def upsert_syntax_error(self, error: SyntaxErrorRecord) -> None:
        self._check_closed()
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO syntax_errors (filepath, line, col, message, timestamp, project)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (error.filepath, error.line, error.col, error.message, error.timestamp, error.project)
        )
        self._auto_commit()

    def delete_syntax_error(self, filepath: str) -> None:
        self._check_closed()
        cur = self.conn.cursor()
        cur.execute("DELETE FROM syntax_errors WHERE filepath = ?", (filepath,))
        self._auto_commit()

    def get_syntax_errors(
        self,
        target_path: str | None = None,
        allowed_projects: list[str] | None = None,
    ) -> list[SyntaxErrorRecord]:
        self._check_closed()
        if allowed_projects is not None and len(allowed_projects) == 0:
            return []
        cur = self.conn.cursor()
        clauses = []
        params: list[Any] = []

        if target_path:
            clauses.append("filepath = ?")
            params.append(target_path)

        if allowed_projects is not None:
            placeholders = ",".join("?" for _ in allowed_projects)
            clauses.append(f"project IN ({placeholders})")
            params.extend(allowed_projects)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        cur.execute(
            f"SELECT filepath, line, col, message, timestamp, project FROM syntax_errors {where} ORDER BY filepath ASC, line ASC",
            params
        )
        return [
            SyntaxErrorRecord(
                filepath=r["filepath"],
                line=r["line"],
                col=r["col"],
                message=r["message"],
                timestamp=r["timestamp"],
                project=r["project"]
            )
            for r in cur.fetchall()
        ]

    def insert_annotations(self, annotations: list[AnnotationRecord]) -> None:
        self._check_closed()
        cur = self.conn.cursor()
        cur.executemany(
            """
            INSERT INTO annotations (filepath, line, kind, symbol, content, project)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [(a.filepath, a.line, a.kind, a.symbol, a.content, a.project) for a in annotations]
        )
        self._auto_commit()

    def query_annotations(
        self,
        kind: str | None = None,
        filepath: str | None = None,
        allowed_projects: list[str] | None = None,
        limit: int = 200,
    ) -> list[AnnotationRecord]:
        self._check_closed()
        if allowed_projects is not None and len(allowed_projects) == 0:
            return []
        cur = self.conn.cursor()
        clauses = []
        params: list[Any] = []

        if allowed_projects is not None:
            placeholders = ",".join("?" for _ in allowed_projects)
            clauses.append(f"project IN ({placeholders})")
            params.extend(allowed_projects)

        if kind:
            clauses.append("kind = ?")
            params.append(kind)

        if filepath:
            clauses.append("filepath = ?")
            params.append(filepath)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        cur.execute(
            f"SELECT id, filepath, line, kind, symbol, content, project FROM annotations {where} LIMIT ?",
            params
        )
        return [
            AnnotationRecord(
                id=r["id"],
                filepath=r["filepath"],
                line=r["line"],
                kind=r["kind"],
                symbol=r["symbol"],
                content=r["content"],
                project=r["project"]
            )
            for r in cur.fetchall()
        ]

    # =========================================================================
    # Skills
    # =========================================================================

    def get_skills(self, allowed_projects: list[str] | None = None) -> list[SkillRecord]:
        self._check_closed()
        if allowed_projects is not None and len(allowed_projects) == 0:
            return []
        cur = self.conn.cursor()
        if allowed_projects is not None:
            placeholders = ",".join("?" for _ in allowed_projects)
            cur.execute(
                f"SELECT name, description, filepath, triggers, sha256, last_modified, project FROM skills WHERE project IN ({placeholders})",
                list(allowed_projects)
            )
        else:
            cur.execute("SELECT name, description, filepath, triggers, sha256, last_modified, project FROM skills")
        return [
            SkillRecord(
                name=r["name"],
                description=r["description"],
                filepath=r["filepath"],
                triggers=r["triggers"],
                sha256=r["sha256"],
                last_modified=r["last_modified"],
                project=r["project"]
            )
            for r in cur.fetchall()
        ]

    def get_skills_by_project(self, project: str) -> dict[str, tuple[str, str]]:
        self._check_closed()
        cur = self.conn.cursor()
        cur.execute("SELECT filepath, name, sha256 FROM skills WHERE project = ?", (project,))
        return {r["filepath"]: (r["name"], r["sha256"]) for r in cur.fetchall()}

    def upsert_skill(self, skill: SkillRecord) -> None:
        self._check_closed()
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO skills (name, description, filepath, triggers, sha256, last_modified, project)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name, project) DO UPDATE SET
                description = excluded.description,
                filepath = excluded.filepath,
                triggers = excluded.triggers,
                sha256 = excluded.sha256,
                last_modified = excluded.last_modified
            """,
            (skill.name, skill.description, skill.filepath, skill.triggers, skill.sha256, skill.last_modified, skill.project)
        )
        cur.execute("DELETE FROM fts_skills WHERE name = ? AND project = ?", (skill.name, skill.project))
        cur.execute(
            "INSERT INTO fts_skills (name, description, triggers, content, project) VALUES (?, ?, ?, ?, ?)",
            (skill.name, skill.description, skill.triggers, skill.content[:4000], skill.project)
        )
        self._auto_commit()

    def delete_skill(self, filepath: str, project: str, name: str | None = None) -> None:
        self._check_closed()
        cur = self.conn.cursor()
        if name:
            cur.execute("DELETE FROM skills WHERE name = ? AND project = ?", (name, project))
            cur.execute("DELETE FROM fts_skills WHERE name = ? AND project = ?", (name, project))
        else:
            cur.execute("DELETE FROM skills WHERE filepath = ? AND project = ?", (filepath, project))
        self._auto_commit()

    def search_skills(
        self,
        query_tokens: list[str],
        allowed_projects: list[str] | None = None,
        limit: int = 20,
    ) -> list[tuple[str, float]]:
        self._check_closed()
        if allowed_projects is not None and len(allowed_projects) == 0:
            return []
        clean = [t.strip() for t in query_tokens if t.strip()]
        if not clean:
            return []
        cur = self.conn.cursor()
        fts_query = " OR ".join(f'"{t.replace(chr(34), chr(34)+chr(34))}"' for t in clean)

        clauses = ["fts_skills MATCH ?"]
        params: list[Any] = [fts_query]

        if allowed_projects is not None:
            placeholders = ",".join("?" for _ in allowed_projects)
            clauses.append(f"project IN ({placeholders})")
            params.extend(allowed_projects)

        where = " AND ".join(clauses)
        params.append(limit)

        try:
            cur.execute(
                f"""
                SELECT name, project, bm25(fts_skills) as rank
                FROM fts_skills
                WHERE {where}
                ORDER BY rank
                LIMIT ?
                """,
                params
            )
        except sqlite3.OperationalError as exc:
            raise AiDbQueryError(fts_query, exc) from exc
        return [(r["name"], round(abs(float(r["rank"])), 3)) for r in cur.fetchall()]

    # =========================================================================
    # Contexts (Session Memory)
    # =========================================================================

    def save_context(self, context: ContextRecord) -> None:
        self._check_closed()
        cur = self.conn.cursor()
        zcontent = zlib.compress(context.full_notes.encode("utf-8"), level=ZLIB_LEVEL)
        active_json = json.dumps(context.active_files)
        tasks_json = json.dumps(context.open_tasks)

        cur.execute(
            """
            INSERT INTO contexts (session_id, project, title, summary, active_files, open_tasks, timestamp, zcontent)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, project) DO UPDATE SET
                title = excluded.title,
                summary = excluded.summary,
                active_files = excluded.active_files,
                open_tasks = excluded.open_tasks,
                timestamp = excluded.timestamp,
                zcontent = excluded.zcontent
            """,
            (context.session_id, context.project, context.title, context.summary, active_json, tasks_json, context.timestamp, zcontent)
        )

        cur.execute("DELETE FROM fts_contexts WHERE session_id = ? AND project = ?", (context.session_id, context.project))
        cur.execute(
            "INSERT INTO fts_contexts (session_id, project, title, summary, content) VALUES (?, ?, ?, ?, ?)",
            (context.session_id, context.project, context.title or "", context.summary, context.full_notes[:4000])
        )
        self._auto_commit()

    def get_context(
        self,
        session_id: str | None = None,
        allowed_projects: list[str] | None = None,
    ) -> ContextRecord | None:
        self._check_closed()
        if allowed_projects is not None and len(allowed_projects) == 0:
            return None
        cur = self.conn.cursor()
        clauses = []
        params: list[Any] = []

        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)

        if allowed_projects is not None:
            placeholders = ",".join("?" for _ in allowed_projects)
            clauses.append(f"project IN ({placeholders})")
            params.extend(allowed_projects)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        order_limit = "" if session_id else "ORDER BY timestamp DESC LIMIT 1"

        cur.execute(
            f"SELECT session_id, project, title, summary, active_files, open_tasks, timestamp, zcontent FROM contexts {where} {order_limit}",
            params
        )
        row = cur.fetchone()
        if not row:
            return None
        full_notes = _decompress(row["zcontent"], f"context {row['session_id']}")
        active_files = _loads(row["active_files"], "context.active_files") if row["active_files"] else []
        open_tasks = _loads(row["open_tasks"], "context.open_tasks") if row["open_tasks"] else []

        return ContextRecord(
            session_id=row["session_id"],
            project=row["project"],
            title=row["title"] or "",
            summary=row["summary"],
            active_files=active_files,
            open_tasks=open_tasks,
            timestamp=float(row["timestamp"]),
            full_notes=full_notes
        )

    def list_contexts(self, allowed_projects: list[str] | None = None) -> list[dict[str, Any]]:
        self._check_closed()
        if allowed_projects is not None and len(allowed_projects) == 0:
            return []
        cur = self.conn.cursor()
        clauses = []
        params: list[Any] = []

        if allowed_projects is not None:
            placeholders = ",".join("?" for _ in allowed_projects)
            clauses.append(f"project IN ({placeholders})")
            params.extend(allowed_projects)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        cur.execute(
            f"SELECT session_id, project, title, summary, timestamp, active_files, open_tasks FROM contexts {where} ORDER BY timestamp DESC",
            params
        )
        results = []
        for r in cur.fetchall():
            af = _loads(r["active_files"], "context.active_files") if r["active_files"] else []
            ot = _loads(r["open_tasks"], "context.open_tasks") if r["open_tasks"] else []
            results.append({
                "session_id": r["session_id"],
                "project": r["project"],
                "title": r["title"] or "",
                "summary": r["summary"],
                "timestamp": r["timestamp"],
                "active_files": af,
                "open_tasks": ot
            })
        return results

    def search_contexts(
        self,
        query_tokens: list[str],
        allowed_projects: list[str] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        self._check_closed()
        if allowed_projects is not None and len(allowed_projects) == 0:
            return []
        clean = [t.strip() for t in query_tokens if t.strip()]
        if not clean:
            return []
        cur = self.conn.cursor()
        fts_query = " OR ".join(f'"{t.replace(chr(34), chr(34)+chr(34))}"' for t in clean)

        clauses = ["fts_contexts MATCH ?"]
        params: list[Any] = [fts_query]

        if allowed_projects is not None:
            placeholders = ",".join("?" for _ in allowed_projects)
            clauses.append(f"fts_contexts.project IN ({placeholders})")
            params.extend(allowed_projects)

        where = " AND ".join(clauses)
        params.append(top_k)

        try:
            cur.execute(
                f"""
                SELECT fts_contexts.session_id, fts_contexts.project, fts_contexts.title,
                       fts_contexts.summary, bm25(fts_contexts) as rank, contexts.timestamp
                FROM fts_contexts
                JOIN contexts ON fts_contexts.session_id = contexts.session_id AND fts_contexts.project = contexts.project
                WHERE {where}
                ORDER BY rank
                LIMIT ?
                """,
                params
            )
        except sqlite3.OperationalError as exc:
            raise AiDbQueryError(fts_query, exc) from exc
        return [
            {
                "session_id": r["session_id"],
                "project": r["project"],
                "title": r["title"],
                "summary": r["summary"],
                "score": round(abs(float(r["rank"])), 3),
                "timestamp": r["timestamp"]
            }
            for r in cur.fetchall()
        ]

    # =========================================================================
    # Analysis References
    # =========================================================================

    def store_analysis_ref(self, ref: AnalysisRefRecord) -> None:
        self._check_closed()
        cur = self.conn.cursor()
        zbody = zlib.compress(ref.body_text.encode("utf-8"), level=ZLIB_LEVEL)
        cur.execute(
            """
            INSERT OR REPLACE INTO analysis_refs (ref_id, filepath, name, start_line, end_line, kind, zbody, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ref.ref_id, ref.filepath, ref.name, ref.start_line, ref.end_line, ref.kind, zbody, ref.timestamp)
        )
        self._auto_commit()

    def get_analysis_ref(self, ref_id: str) -> AnalysisRefRecord | None:
        self._check_closed()
        cur = self.conn.cursor()
        cur.execute(
            "SELECT ref_id, filepath, name, start_line, end_line, kind, zbody, timestamp FROM analysis_refs WHERE ref_id = ?",
            (ref_id,)
        )
        row = cur.fetchone()
        if not row:
            return None
        body = _decompress(row["zbody"], f"analysis ref {row['ref_id']}")
        return AnalysisRefRecord(
            ref_id=row["ref_id"],
            filepath=row["filepath"],
            name=row["name"],
            start_line=row["start_line"],
            end_line=row["end_line"],
            kind=row["kind"],
            body_text=body,
            timestamp=row["timestamp"]
        )

    def evict_stale_analysis_refs(self, older_than_seconds: float) -> int:
        self._check_closed()
        cur = self.conn.cursor()
        cutoff = time.time() - older_than_seconds
        cur.execute("DELETE FROM analysis_refs WHERE timestamp < ?", (cutoff,))
        count = cur.rowcount
        self._auto_commit()
        return count

    # =========================================================================
    # Semantic Cache & State
    # =========================================================================

    def get_semantic_cache(self, cache_key: str, file_hash: str) -> dict[str, Any] | None:
        self._check_closed()
        cur = self.conn.cursor()
        cur.execute(
            "SELECT result_json FROM semantic_cache WHERE cache_key = ? AND file_hash = ?",
            (cache_key, file_hash)
        )
        row = cur.fetchone()
        if row:
            cached: dict[str, Any] = _loads(row["result_json"], "semantic_cache.result_json")
            return cached
        return None

    def set_semantic_cache(self, cache_key: str, file_hash: str, result: dict[str, Any]) -> None:
        self._check_closed()
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO semantic_cache (cache_key, file_hash, result_json, timestamp)
            VALUES (?, ?, ?, ?)
            """,
            (cache_key, file_hash, json.dumps(result), time.time())
        )
        self._auto_commit()

    def clear_semantic_cache(self) -> None:
        self._check_closed()
        cur = self.conn.cursor()
        cur.execute("DELETE FROM semantic_cache")
        self._auto_commit()

    def get_state(self, key: str) -> Any | None:
        self._check_closed()
        cur = self.conn.cursor()
        cur.execute("SELECT value_json FROM session_state WHERE key = ?", (key,))
        row = cur.fetchone()
        if row:
            return _loads(row["value_json"], "session_state.value_json")
        return None

    def set_state(self, key: str, value: Any) -> None:
        self._check_closed()
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO session_state (key, value_json, updated)
            VALUES (?, ?, ?)
            """,
            (key, json.dumps(value), time.time())
        )
        self._auto_commit()

    def get_table_counts(self) -> dict[str, int]:
        self._check_closed()
        cur = self.conn.cursor()
        counts: dict[str, int] = {}
        for tbl in ["files", "chunks", "symbols", "skills", "syntax_errors", "contexts"]:
            cur.execute(f"SELECT COUNT(*) as c FROM {tbl}")
            counts[tbl] = int(cur.fetchone()["c"])
        return counts

    def get_weak_points(self) -> dict[str, Any]:
        self._check_closed()
        cur = self.conn.cursor()
        result: dict[str, Any] = {
            "syntax_error_density_pct": 0.0,
            "complexity_hotspots": [],
            "unindexed_or_stale_files": [],
        }
        cur.execute("SELECT COUNT(*) as c FROM files")
        row = cur.fetchone()
        total_files = int(row["c"]) if row else 0

        cur.execute("SELECT COUNT(DISTINCT filepath) as c FROM syntax_errors")
        row_err = cur.fetchone()
        files_with_errors = int(row_err["c"]) if row_err else 0

        if total_files > 0:
            result["syntax_error_density_pct"] = round((files_with_errors / total_files) * 100.0, 2)

        cur.execute(
            """
            SELECT f.filepath, COUNT(s.id) as sym_count
            FROM files f
            JOIN symbols s ON f.filepath = s.filepath
            GROUP BY f.filepath
            ORDER BY sym_count DESC
            LIMIT 10
            """
        )
        hotspots = []
        for r in cur.fetchall():
            hotspots.append({
                "filepath": r["filepath"],
                "symbols_count": int(r["sym_count"]),
            })
        result["complexity_hotspots"] = hotspots

        cur.execute(
            """
            SELECT filepath FROM files
            WHERE last_modified IS NULL OR last_modified = 0
            LIMIT 10
            """
        )
        result["unindexed_or_stale_files"] = [r["filepath"] for r in cur.fetchall()]
        return result
