import os
import sqlite3
from typing import Dict, Any, Optional, Callable
from ai_db.constants import DEFAULT_DB_FILE
from ai_db.storage.state import get_session_state, set_session_state

class Database:
    def __init__(self, db_path: str = DEFAULT_DB_FILE):
        self.db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        cur = self.conn.cursor()
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
                FOREIGN KEY (filepath) REFERENCES files(filepath) ON DELETE CASCADE
            )
        """)

        cur.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_index USING fts5(
                content,
                filepath UNINDEXED,
                name UNINDEXED,
                chunk_id UNINDEXED,
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

        # Dynamic Schema Migrations for existing SQLite tables
        def add_column_if_missing(table: str, col: str, col_type: str):
            cur.execute(f"PRAGMA table_info({table})")
            existing_cols = [r[1] for r in cur.fetchall()]
            if col not in existing_cols:
                try:
                    cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
                except Exception:
                    pass

        add_column_if_missing("files", "project", "TEXT DEFAULT 'global'")
        add_column_if_missing("chunks", "project", "TEXT DEFAULT 'global'")
        add_column_if_missing("symbols", "project", "TEXT DEFAULT 'global'")
        add_column_if_missing("syntax_errors", "project", "TEXT DEFAULT 'global'")
        add_column_if_missing("skills", "project", "TEXT DEFAULT 'global'")

        # Ensure fts_skills virtual table has project column
        cur.execute("PRAGMA table_info(fts_skills)")
        fts_cols = [r[1] for r in cur.fetchall()]
        if "project" not in fts_cols:
            try:
                cur.execute("DROP TABLE IF EXISTS fts_skills")
                cur.execute("""
                    CREATE VIRTUAL TABLE fts_skills USING fts5(
                        name,
                        description,
                        triggers,
                        content,
                        project,
                        tokenize = 'porter unicode61'
                    )
                """)
            except Exception:
                pass

        # RFC: tokenopt-analyzer v2 tables
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
            CREATE TABLE IF NOT EXISTS session_state (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
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

        self.conn.commit()


    def status(self) -> Dict[str, Any]:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) as c FROM files")
        file_count = cur.fetchone()["c"]
        cur.execute("SELECT COUNT(*) as c FROM chunks")
        chunk_count = cur.fetchone()["c"]
        cur.execute("SELECT COUNT(*) as c FROM symbols")
        symbol_count = cur.fetchone()["c"]
        cur.execute("SELECT COUNT(*) as c FROM syntax_errors")
        syntax_err_count = cur.fetchone()["c"]
        cur.execute("SELECT COUNT(*) as c FROM skills")
        skills_count = cur.fetchone()["c"]
        cur.execute("SELECT COUNT(*) as c FROM contexts")
        contexts_count = cur.fetchone()["c"]
        size_bytes = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0

        return {
            "db": self.db_path,
            "files": file_count,
            "chunks": chunk_count,
            "symbols": symbol_count,
            "syntax_errors": syntax_err_count,
            "skills": skills_count,
            "contexts": contexts_count,
            "kb": round(size_bytes / 1024, 1),
            "format": "zlib-compressed binary blob (token-dense)"
        }

    def optimize(self, prune_missing: bool = True, default_format: Optional[str] = None) -> Dict[str, Any]:
        """Runs comprehensive database optimizations:
        1. Prunes references to deleted files (optional).
        2. Merges and optimizes FTS5 inverted index b-trees (fts_index, fts_skills, fts_contexts).
        3. Runs PRAGMA optimize for query planner statistics.
        4. Reclaims fragmented disk space via VACUUM.
        5. Optionally configures and persists the default AI output format (e.g. 'stub', 'sexp', 'json').
        """
        initial_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        pruned_files = 0

        cur = self.conn.cursor()
        if prune_missing:
            cur.execute("SELECT filepath FROM files")
            for row in cur.fetchall():
                fp = row["filepath"]
                if not os.path.exists(fp):
                    cur.execute("DELETE FROM files WHERE filepath = ?", (fp,)); cur.execute("DELETE FROM chunks WHERE filepath = ?", (fp,)); cur.execute("DELETE FROM symbols WHERE filepath = ?", (fp,))
                    pruned_files += 1

        # 2. Merge FTS5 indexes
        fts_tables = ["fts_index", "fts_skills", "fts_contexts"]
        for fts in fts_tables:
            try:
                cur.execute(f"INSERT INTO {fts}({fts}) VALUES('optimize')")
            except Exception:
                pass
        self.conn.commit()

        # 3. Optimize query planner stats
        try:
            cur.execute("PRAGMA optimize")
        except Exception:
            pass

        # 4. Defragment and reclaim pages via VACUUM
        try:
            cur.execute("VACUUM")
            self.conn.commit()
        except Exception:
            pass

        # 5. Persist default AI output format if provided
        if default_format:
            norm_fmt = default_format.strip().lower()
            if norm_fmt in ("stub", "sexp", "json", "outline", "prose"):
                self._set_session_state("default_format", norm_fmt)

        final_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        reclaimed_kb = round(max(0, initial_size - final_size) / 1024, 1)

        active_fmt = self._get_session_state("default_format") or "stub"

        return {
            "initial_kb": round(initial_size / 1024, 1),
            "final_kb": round(final_size / 1024, 1),
            "reclaimed_kb": reclaimed_kb,
            "pruned_files": pruned_files,
            "default_format": active_fmt
        }


    def close(self):
        self.conn.close()

    def _get_session_state(self, key: str) -> Optional[Any]:
        return get_session_state(self.conn, key)

    def _set_session_state(self, key: str, value: Any):
        return set_session_state(self.conn, key, value)
