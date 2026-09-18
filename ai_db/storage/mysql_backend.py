"""MySQL 8.0+ / MariaDB relational storage adapter for ai-db (Milestone 2, Feature 10).

Implements StorageBackend ABC using InnoDB VARCHAR(768) primary keys, LONGBLOBs,
FULLTEXT indexing (MATCH ... AGAINST), ON DUPLICATE KEY UPDATE upserts,
and credential-masked error reporting.
"""

import sys
import os
import json
import time
import zlib
from contextlib import contextmanager
from typing import Optional, List, Dict, Any, Tuple, Generator
from urllib.parse import urlparse, unquote

from ai_db.storage.backend import StorageBackend
from ai_db.storage.models import (
    FileRecord, ChunkRecord, SymbolRecord, SymbolRefRecord,
    AnnotationRecord, SyntaxErrorRecord, SkillRecord,
    ContextRecord, AnalysisRefRecord, SearchResult
)


class MySQLBackend(StorageBackend):
    """MySQL 8.0+ relational adapter implementing StorageBackend interface."""

    def __init__(self, connection_string: str):
        if sys.modules.get("pymysql") is None:
            raise ImportError(
                "MySQL backend requires 'pymysql'. "
                "Install with: pip install 'ai-db[mysql]' or pip install pymysql"
            )

        try:
            import pymysql
            import pymysql.cursors
        except (ImportError, ModuleNotFoundError) as e:
            raise ImportError(
                "MySQL backend requires 'pymysql'. "
                "Install with: pip install 'ai-db[mysql]' or pip install pymysql"
            ) from e

        self.uri = connection_string
        self._closed = False
        self._tx_depth = 0

        parsed = urlparse(connection_string)
        self.user = unquote(parsed.username) if parsed.username else "root"
        self.password = unquote(parsed.password) if parsed.password else ""
        self.host = parsed.hostname or "localhost"
        self.port = parsed.port or 3306
        self.database = parsed.path.lstrip("/")
        if not self.database:
            raise ValueError(f"Malformed MySQL URI: missing database name in '{self._mask(connection_string)}'")

        try:
            cursor_cls = getattr(pymysql.cursors, "DictCursor", None)
            self.conn = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                charset="utf8mb4",
                cursorclass=cursor_cls,
                autocommit=False,
            )
        except Exception as e:
            masked_err = self._mask(str(e))
            raise RuntimeError(f"Failed to connect to MySQL ({self._mask(connection_string)}): {masked_err}") from None

    @property
    def backend_name(self) -> str:
        return "mysql"

    def _mask(self, text: str) -> str:
        if hasattr(self, "password") and self.password and self.password in text:
            text = text.replace(self.password, "******")
        if hasattr(self, "uri") and self.uri and hasattr(self, "password") and self.password:
            masked_uri = self.uri.replace(self.password, "******")
            text = text.replace(self.uri, masked_uri)
        return text

    def _check_closed(self) -> None:
        if self._closed or self.conn is None:
            raise RuntimeError("Storage backend is closed")

    def _auto_commit(self) -> None:
        if self._tx_depth == 0 and self.conn is not None:
            try:
                self.conn.commit()
            except Exception as e:
                raise RuntimeError(self._mask(str(e))) from None

    def _execute(self, sql: str, params: Any = None) -> Any:
        self._check_closed()
        try:
            cur = self.conn.cursor()
            cur.execute(sql, params)
            return cur
        except Exception as e:
            raise RuntimeError(self._mask(str(e))) from None

    def _executemany(self, sql: str, params_seq: Any) -> Any:
        self._check_closed()
        try:
            cur = self.conn.cursor()
            cur.executemany(sql, params_seq)
            return cur
        except Exception as e:
            raise RuntimeError(self._mask(str(e))) from None

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            if self.conn is not None:
                try:
                    self.conn.close()
                except Exception:
                    pass
                self.conn = None

    @contextmanager
    def transaction(self) -> Generator[None, None, None]:
        self._check_closed()
        self._tx_depth += 1
        sp_name = f"sp_level_{self._tx_depth}"
        try:
            self._execute(f"SAVEPOINT {sp_name}")
            yield
            self._execute(f"RELEASE SAVEPOINT {sp_name}")
            if self._tx_depth == 1 and self.conn is not None:
                self.conn.commit()
        except Exception as e:
            try:
                self._execute(f"ROLLBACK TO SAVEPOINT {sp_name}")
                self._execute(f"RELEASE SAVEPOINT {sp_name}")
            except Exception:
                pass
            if self._tx_depth == 1 and self.conn is not None:
                try:
                    self.conn.rollback()
                except Exception:
                    pass
            masked_msg = self._mask(str(e))
            if masked_msg != str(e):
                raise RuntimeError(masked_msg) from None
            raise
        finally:
            self._tx_depth -= 1

    def initialize(self) -> None:
        self._check_closed()
        ddl_statements = [
            """
            CREATE TABLE IF NOT EXISTS files (
                filepath VARCHAR(768) PRIMARY KEY,
                sha256 VARCHAR(64) NOT NULL,
                last_modified DOUBLE NOT NULL,
                chunk_count INT NOT NULL,
                project VARCHAR(255) DEFAULT 'global'
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """,
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                filepath VARCHAR(768) NOT NULL,
                chunk_type VARCHAR(64) NOT NULL,
                name VARCHAR(255) NOT NULL,
                start_line INT NOT NULL,
                end_line INT NOT NULL,
                zcontent LONGBLOB NOT NULL,
                content MEDIUMTEXT,
                project VARCHAR(255) DEFAULT 'global',
                FULLTEXT INDEX idx_fts_chunks (content),
                INDEX idx_chunks_filepath (filepath),
                INDEX idx_chunks_project (project)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """,
            """
            CREATE TABLE IF NOT EXISTS symbols (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                symbol_type VARCHAR(64) NOT NULL,
                filepath VARCHAR(768) NOT NULL,
                line INT NOT NULL,
                signature TEXT,
                project VARCHAR(255) DEFAULT 'global',
                INDEX idx_symbols_name (name),
                INDEX idx_symbols_filepath (filepath),
                INDEX idx_symbols_project (project)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """,
            """
            CREATE TABLE IF NOT EXISTS syntax_errors (
                filepath VARCHAR(768) PRIMARY KEY,
                line INT,
                col INT,
                message TEXT,
                timestamp DOUBLE,
                project VARCHAR(255) DEFAULT 'global'
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """,
            """
            CREATE TABLE IF NOT EXISTS skills (
                name VARCHAR(255) NOT NULL,
                description TEXT,
                filepath VARCHAR(768) NOT NULL,
                triggers TEXT,
                sha256 VARCHAR(64) NOT NULL,
                last_modified DOUBLE NOT NULL,
                content MEDIUMTEXT,
                project VARCHAR(255) DEFAULT 'global',
                PRIMARY KEY (name, project),
                FULLTEXT INDEX idx_fts_skills (name, description, triggers, content),
                INDEX idx_skills_project (project)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """,
            """
            CREATE TABLE IF NOT EXISTS contexts (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(255) NOT NULL,
                project VARCHAR(255) NOT NULL,
                title VARCHAR(255),
                summary TEXT NOT NULL,
                active_files JSON,
                open_tasks JSON,
                timestamp DOUBLE NOT NULL,
                zcontent LONGBLOB NOT NULL,
                content MEDIUMTEXT,
                UNIQUE KEY uk_session_project (session_id, project),
                FULLTEXT INDEX idx_fts_contexts (session_id, title, summary, content)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """,
            """
            CREATE TABLE IF NOT EXISTS semantic_cache (
                cache_key VARCHAR(255) PRIMARY KEY,
                file_hash VARCHAR(64) NOT NULL,
                result_json JSON NOT NULL,
                timestamp DOUBLE NOT NULL,
                INDEX idx_semantic_cache_hash (file_hash)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """,
            """
            CREATE TABLE IF NOT EXISTS module_summaries (
                filepath VARCHAR(768) PRIMARY KEY,
                sha256 VARCHAR(64) NOT NULL,
                summary TEXT NOT NULL,
                updated DOUBLE NOT NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """,
            """
            CREATE TABLE IF NOT EXISTS session_state (
                `key` VARCHAR(255) PRIMARY KEY,
                value_json JSON NOT NULL,
                updated DOUBLE NOT NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """,
            """
            CREATE TABLE IF NOT EXISTS analysis_refs (
                ref_id VARCHAR(64) PRIMARY KEY,
                filepath VARCHAR(768) NOT NULL,
                name VARCHAR(255) NOT NULL,
                start_line INT NOT NULL,
                end_line INT NOT NULL,
                kind VARCHAR(64) NOT NULL,
                zbody LONGBLOB NOT NULL,
                timestamp DOUBLE NOT NULL,
                INDEX idx_analysis_refs_file (filepath)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """,
            """
            CREATE TABLE IF NOT EXISTS symbol_refs (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                caller_filepath VARCHAR(768) NOT NULL,
                caller_name VARCHAR(255) NOT NULL,
                caller_line INT NOT NULL,
                callee_name VARCHAR(255) NOT NULL,
                ref_type VARCHAR(64) NOT NULL,
                project VARCHAR(255) DEFAULT 'global',
                INDEX idx_symrefs_callee (callee_name),
                INDEX idx_symrefs_caller (caller_filepath, caller_name),
                INDEX idx_symrefs_project (project)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """,
            """
            CREATE TABLE IF NOT EXISTS annotations (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                filepath VARCHAR(768) NOT NULL,
                line INT NOT NULL,
                kind VARCHAR(64) NOT NULL,
                symbol VARCHAR(255),
                content TEXT NOT NULL,
                project VARCHAR(255) DEFAULT 'global',
                INDEX idx_annotations_filepath (filepath),
                INDEX idx_annotations_kind (kind, project)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        ]
        for ddl in ddl_statements:
            self._execute(ddl)
        self._auto_commit()

    def status(self) -> Dict[str, Any]:
        self._check_closed()
        counts = {}
        for tbl in ["files", "chunks", "symbols", "syntax_errors", "skills", "contexts"]:
            try:
                cur = self._execute(f"SELECT COUNT(*) as c FROM {tbl}")
                row = cur.fetchone()
                if row:
                    counts[tbl] = row["c"] if isinstance(row, dict) else row[0]
                else:
                    counts[tbl] = 0
            except Exception:
                counts[tbl] = 0

        return {
            "backend": "mysql",
            "db": self.database,
            "host": self.host,
            "port": self.port,
            "files": counts.get("files", 0),
            "chunks": counts.get("chunks", 0),
            "symbols": counts.get("symbols", 0),
            "syntax_errors": counts.get("syntax_errors", 0),
            "skills": counts.get("skills", 0),
            "contexts": counts.get("contexts", 0),
            "format": "InnoDB compressed BLOB / JSON"
        }

    def optimize(self, prune_missing: bool = True, default_format: Optional[str] = None) -> Dict[str, Any]:
        self._check_closed()
        pruned_files = 0
        if prune_missing:
            cur = self._execute("SELECT filepath FROM files")
            for row in cur.fetchall():
                fp = row["filepath"] if isinstance(row, dict) else row[0]
                if not os.path.exists(fp):
                    self.delete_file(fp)
                    pruned_files += 1

        for tbl in ["chunks", "skills", "contexts"]:
            try:
                self._execute(f"OPTIMIZE TABLE {tbl}")
            except Exception:
                pass

        self.evict_stale_analysis_refs(86400 * 7)

        if default_format:
            norm = default_format.strip().lower()
            if norm in ("stub", "sexp", "json", "outline", "prose"):
                self.set_state("default_format", norm)

        active_fmt = self.get_state("default_format") or "stub"
        return {
            "backend": "mysql",
            "pruned_files": pruned_files,
            "default_format": active_fmt
        }

    # =========================================================================
    # Files
    # =========================================================================

    def get_file(self, filepath: str) -> Optional[FileRecord]:
        cur = self._execute(
            "SELECT filepath, sha256, last_modified, chunk_count, project FROM files WHERE filepath = %s",
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

    def get_files_by_prefix(self, prefix: str) -> Dict[str, str]:
        prefix = prefix.replace("\x00", "") if prefix else ""
        cur = self._execute(
            "SELECT filepath, sha256 FROM files WHERE filepath = %s OR filepath LIKE %s",
            (prefix, f"{prefix.rstrip('/')}/%")
        )
        return {r["filepath"]: r["sha256"] for r in cur.fetchall()}

    def get_all_filepaths(self) -> List[str]:
        cur = self._execute("SELECT filepath FROM files")
        rows = cur.fetchall()
        return [r["filepath"] if isinstance(r, dict) else r[0] for r in rows]

    def upsert_file(self, record: FileRecord) -> None:
        sql = """
            INSERT INTO files (filepath, sha256, last_modified, chunk_count, project)
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                sha256 = VALUES(sha256),
                last_modified = VALUES(last_modified),
                chunk_count = VALUES(chunk_count),
                project = VALUES(project)
        """
        self._execute(sql, (record.filepath, record.sha256, record.last_modified, record.chunk_count, record.project))
        self._auto_commit()

    def delete_file(self, filepath: str) -> None:
        self._execute("DELETE FROM chunks WHERE filepath = %s", (filepath,))
        self._execute("DELETE FROM symbols WHERE filepath = %s", (filepath,))
        self._execute("DELETE FROM symbol_refs WHERE caller_filepath = %s", (filepath,))
        self._execute("DELETE FROM annotations WHERE filepath = %s", (filepath,))
        self._execute("DELETE FROM syntax_errors WHERE filepath = %s", (filepath,))
        self._execute("DELETE FROM analysis_refs WHERE filepath = %s", (filepath,))
        self._execute("DELETE FROM files WHERE filepath = %s", (filepath,))
        self._auto_commit()

    # =========================================================================
    # Chunks & Code Search
    # =========================================================================

    def insert_chunks(self, chunks: List[ChunkRecord]) -> None:
        for c in chunks:
            zcontent = zlib.compress(c.content.encode("utf-8"), level=9)
            cur = self._execute(
                """
                INSERT INTO chunks (filepath, chunk_type, name, start_line, end_line, zcontent, content, project)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (c.filepath, c.chunk_type, c.name, c.start_line, c.end_line, zcontent, f"{c.name} {c.content}", c.project)
            )
            c.id = getattr(cur, "lastrowid", None)
        self._auto_commit()

    def get_chunks_for_file(self, filepath: str) -> List[ChunkRecord]:
        cur = self._execute(
            "SELECT id, filepath, chunk_type, name, start_line, end_line, zcontent, project FROM chunks WHERE filepath = %s ORDER BY start_line ASC",
            (filepath,)
        )
        results = []
        for r in cur.fetchall():
            try:
                content = zlib.decompress(r["zcontent"]).decode("utf-8", errors="replace")
            except Exception:
                content = ""
            results.append(ChunkRecord(
                id=r["id"],
                filepath=r["filepath"],
                chunk_type=r["chunk_type"],
                name=r["name"],
                start_line=r["start_line"],
                end_line=r["end_line"],
                content=content,
                project=r["project"]
            ))
        return results

    def search_chunks(
        self,
        query_tokens: List[str],
        allowed_projects: Optional[List[str]] = None,
        top_k: int = 5,
        path_prefix: Optional[str] = None,
    ) -> List[SearchResult]:
        if allowed_projects is not None and len(allowed_projects) == 0:
            return []
        clean = [t.replace("\x00", "").strip() for t in query_tokens if t.replace("\x00", "").strip()]
        if not clean:
            return []

        query_str = " ".join(clean)
        where_clauses = ["MATCH(content) AGAINST(%s IN NATURAL LANGUAGE MODE)"]
        params: List[Any] = [query_str, query_str]

        if allowed_projects is not None:
            placeholders = ", ".join(["%s"] * len(allowed_projects))
            where_clauses.append(f"project IN ({placeholders})")
            params.extend(allowed_projects)

        if path_prefix:
            clean_prefix = path_prefix.replace("\x00", "")
            where_clauses.append("filepath LIKE %s")
            params.append(f"{clean_prefix.rstrip('/')}/%")

        where_sql = " AND ".join(where_clauses)
        params.append(top_k)

        sql = f"""
            SELECT id, filepath, name, chunk_type, project, start_line, end_line, zcontent,
                   MATCH(content) AGAINST(%s IN NATURAL LANGUAGE MODE) AS score
            FROM chunks
            WHERE {where_sql}
            ORDER BY score DESC
            LIMIT %s
        """

        cur = self._execute(sql, tuple(params))
        results = []
        for r in cur.fetchall():
            try:
                content = zlib.decompress(r["zcontent"]).decode("utf-8", errors="replace")
            except Exception:
                content = ""
            snippet = content[:300].strip() if content else ""
            score = float(r.get("score", 1.0)) if isinstance(r, dict) else 1.0

            results.append(SearchResult(
                chunk_id=r["id"],
                filepath=r["filepath"],
                name=r["name"],
                chunk_type=r["chunk_type"],
                project=r["project"],
                start_line=r["start_line"],
                end_line=r["end_line"],
                score=round(score, 3),
                snippet=snippet
            ))
        return results

    # =========================================================================
    # Symbols & References
    # =========================================================================

    def insert_symbols(self, symbols: List[SymbolRecord]) -> None:
        sql = """
            INSERT INTO symbols (name, symbol_type, filepath, line, signature, project)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        self._executemany(sql, [(s.name, s.symbol_type, s.filepath, s.line, s.signature, s.project) for s in symbols])
        self._auto_commit()

    def query_symbols(
        self,
        name: str,
        allowed_projects: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[SymbolRecord]:
        if allowed_projects is not None and len(allowed_projects) == 0:
            return []
        where_clauses = ["(name = %s OR name LIKE %s)"]
        params: List[Any] = [name, f"{name}%"]

        if allowed_projects is not None:
            placeholders = ", ".join(["%s"] * len(allowed_projects))
            where_clauses.append(f"project IN ({placeholders})")
            params.extend(allowed_projects)

        where_sql = " AND ".join(where_clauses)
        params.extend([name, limit])

        sql = f"""
            SELECT id, name, symbol_type, filepath, line, signature, project
            FROM symbols
            WHERE {where_sql}
            ORDER BY CASE WHEN name = %s THEN 0 ELSE 1 END, line ASC
            LIMIT %s
        """
        cur = self._execute(sql, tuple(params))
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

    def insert_symbol_refs(self, refs: List[SymbolRefRecord]) -> None:
        sql = """
            INSERT INTO symbol_refs (caller_filepath, caller_name, caller_line, callee_name, ref_type, project)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        self._executemany(sql, [(r.caller_filepath, r.caller_name, r.caller_line, r.callee_name, r.ref_type, r.project) for r in refs])
        self._auto_commit()

    def query_symbol_callers(
        self,
        callee_name: str,
        allowed_projects: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[SymbolRefRecord]:
        if allowed_projects is not None and len(allowed_projects) == 0:
            return []
        where_clauses = ["callee_name = %s"]
        params: List[Any] = [callee_name]

        if allowed_projects is not None:
            placeholders = ", ".join(["%s"] * len(allowed_projects))
            where_clauses.append(f"project IN ({placeholders})")
            params.extend(allowed_projects)

        where_sql = " AND ".join(where_clauses)
        params.append(limit)

        sql = f"""
            SELECT id, caller_filepath, caller_name, caller_line, callee_name, ref_type, project
            FROM symbol_refs
            WHERE {where_sql}
            ORDER BY caller_filepath ASC, caller_line ASC
            LIMIT %s
        """
        cur = self._execute(sql, tuple(params))
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
    # Syntax Errors & Annotations
    # =========================================================================

    def upsert_syntax_error(self, error: SyntaxErrorRecord) -> None:
        sql = """
            INSERT INTO syntax_errors (filepath, line, col, message, timestamp, project)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                line = VALUES(line),
                col = VALUES(col),
                message = VALUES(message),
                timestamp = VALUES(timestamp),
                project = VALUES(project)
        """
        self._execute(sql, (error.filepath, error.line, error.col, error.message, error.timestamp, error.project))
        self._auto_commit()

    def delete_syntax_error(self, filepath: str) -> None:
        self._execute("DELETE FROM syntax_errors WHERE filepath = %s", (filepath,))
        self._auto_commit()

    def get_syntax_errors(
        self,
        target_path: Optional[str] = None,
        allowed_projects: Optional[List[str]] = None,
    ) -> List[SyntaxErrorRecord]:
        if allowed_projects is not None and len(allowed_projects) == 0:
            return []
        where_clauses = []
        params: List[Any] = []

        if target_path:
            where_clauses.append("filepath = %s")
            params.append(target_path)

        if allowed_projects is not None:
            placeholders = ", ".join(["%s"] * len(allowed_projects))
            where_clauses.append(f"project IN ({placeholders})")
            params.extend(allowed_projects)

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        sql = f"SELECT filepath, line, col, message, timestamp, project FROM syntax_errors {where_sql}"
        cur = self._execute(sql, tuple(params))
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

    def insert_annotations(self, annotations: List[AnnotationRecord]) -> None:
        sql = """
            INSERT INTO annotations (filepath, line, kind, symbol, content, project)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        self._executemany(sql, [(a.filepath, a.line, a.kind, a.symbol, a.content, a.project) for a in annotations] if annotations else [])
        self._auto_commit()

    def query_annotations(
        self,
        kind: Optional[str] = None,
        filepath: Optional[str] = None,
        allowed_projects: Optional[List[str]] = None,
        limit: int = 200,
    ) -> List[AnnotationRecord]:
        if allowed_projects is not None and len(allowed_projects) == 0:
            return []
        where_clauses = []
        params: List[Any] = []

        if allowed_projects is not None:
            placeholders = ", ".join(["%s"] * len(allowed_projects))
            where_clauses.append(f"project IN ({placeholders})")
            params.extend(allowed_projects)

        if kind:
            where_clauses.append("kind = %s")
            params.append(kind)

        if filepath:
            where_clauses.append("filepath = %s")
            params.append(filepath)

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        params.append(limit)
        sql = f"SELECT id, filepath, line, kind, symbol, content, project FROM annotations {where_sql} LIMIT %s"
        cur = self._execute(sql, tuple(params))
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

    def get_skills(self, allowed_projects: Optional[List[str]] = None) -> List[SkillRecord]:
        if allowed_projects is not None:
            placeholders = ", ".join(["%s"] * len(allowed_projects))
            sql = f"SELECT name, description, filepath, triggers, sha256, last_modified, project FROM skills WHERE project IN ({placeholders})"
            params = tuple(allowed_projects)
        else:
            sql = "SELECT name, description, filepath, triggers, sha256, last_modified, project FROM skills"
            params = ()
        cur = self._execute(sql, params)
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

    def get_skills_by_project(self, project: str) -> Dict[str, Tuple[str, str]]:
        cur = self._execute("SELECT filepath, name, sha256 FROM skills WHERE project = %s", (project,))
        return {r["filepath"]: (r["name"], r["sha256"]) for r in cur.fetchall()}

    def upsert_skill(self, skill: SkillRecord) -> None:
        sql = """
            INSERT INTO skills (name, description, filepath, triggers, sha256, last_modified, content, project)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                description = VALUES(description),
                filepath = VALUES(filepath),
                triggers = VALUES(triggers),
                sha256 = VALUES(sha256),
                last_modified = VALUES(last_modified),
                content = VALUES(content)
        """
        self._execute(
            sql,
            (skill.name, skill.description, skill.filepath, skill.triggers, skill.sha256, skill.last_modified, skill.content[:4000], skill.project)
        )
        self._auto_commit()

    def delete_skill(self, filepath: str, project: str, name: Optional[str] = None) -> None:
        if name:
            self._execute("DELETE FROM skills WHERE name = %s AND project = %s", (name, project))
        else:
            self._execute("DELETE FROM skills WHERE filepath = %s AND project = %s", (filepath, project))
        self._auto_commit()

    def search_skills(
        self,
        query_tokens: List[str],
        allowed_projects: Optional[List[str]] = None,
        limit: int = 20,
    ) -> List[Tuple[str, float]]:
        clean = [t.strip() for t in query_tokens if t.strip()]
        if not clean:
            return []

        query_str = " ".join(clean)
        where_clauses = ["MATCH(name, description, triggers, content) AGAINST(%s IN NATURAL LANGUAGE MODE)"]
        params: List[Any] = [query_str, query_str]

        if allowed_projects is not None:
            placeholders = ", ".join(["%s"] * len(allowed_projects))
            where_clauses.append(f"project IN ({placeholders})")
            params.extend(allowed_projects)

        where_sql = " AND ".join(where_clauses)
        params.append(limit)

        sql = f"""
            SELECT name, MATCH(name, description, triggers, content) AGAINST(%s IN NATURAL LANGUAGE MODE) AS score
            FROM skills
            WHERE {where_sql}
            ORDER BY score DESC
            LIMIT %s
        """
        cur = self._execute(sql, tuple(params))
        return [(r["name"], round(float(r["score"]), 3) if isinstance(r, dict) else 1.0) for r in cur.fetchall()]

    # =========================================================================
    # Contexts (Session Memory)
    # =========================================================================

    def save_context(self, context: ContextRecord) -> None:
        zcontent = zlib.compress(context.full_notes.encode("utf-8"), level=9)
        active_json = json.dumps(context.active_files)
        tasks_json = json.dumps(context.open_tasks)

        sql = """
            INSERT INTO contexts (session_id, project, title, summary, active_files, open_tasks, timestamp, zcontent, content)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                title = VALUES(title),
                summary = VALUES(summary),
                active_files = VALUES(active_files),
                open_tasks = VALUES(open_tasks),
                timestamp = VALUES(timestamp),
                zcontent = VALUES(zcontent),
                content = VALUES(content)
        """
        self._execute(
            sql,
            (context.session_id, context.project, context.title, context.summary, active_json, tasks_json, context.timestamp, zcontent, context.full_notes[:4000])
        )
        self._auto_commit()

    def get_context(
        self,
        session_id: Optional[str] = None,
        allowed_projects: Optional[List[str]] = None,
    ) -> Optional[ContextRecord]:
        where_clauses = []
        params: List[Any] = []

        if session_id:
            where_clauses.append("session_id = %s")
            params.append(session_id)

        if allowed_projects is not None:
            placeholders = ", ".join(["%s"] * len(allowed_projects))
            where_clauses.append(f"project IN ({placeholders})")
            params.extend(allowed_projects)

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        order_limit = "" if session_id else "ORDER BY timestamp DESC LIMIT 1"

        sql = f"SELECT session_id, project, title, summary, active_files, open_tasks, timestamp, zcontent FROM contexts {where_sql} {order_limit}"
        cur = self._execute(sql, tuple(params))
        row = cur.fetchone()
        if not row:
            return None

        try:
            full_notes = zlib.decompress(row["zcontent"]).decode("utf-8", errors="replace")
        except Exception:
            full_notes = ""

        try:
            af = json.loads(row["active_files"]) if isinstance(row["active_files"], str) else (row["active_files"] or [])
        except Exception:
            af = []

        try:
            ot = json.loads(row["open_tasks"]) if isinstance(row["open_tasks"], str) else (row["open_tasks"] or [])
        except Exception:
            ot = []

        return ContextRecord(
            session_id=row["session_id"],
            project=row["project"],
            title=row["title"] or "",
            summary=row["summary"],
            active_files=af,
            open_tasks=ot,
            timestamp=float(row["timestamp"]),
            full_notes=full_notes
        )

    def list_contexts(self, allowed_projects: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        where_sql = ""
        params: List[Any] = []
        if allowed_projects is not None:
            placeholders = ", ".join(["%s"] * len(allowed_projects))
            where_sql = f"WHERE project IN ({placeholders})"
            params = list(allowed_projects)

        sql = f"SELECT session_id, project, title, summary, timestamp, active_files, open_tasks FROM contexts {where_sql} ORDER BY timestamp DESC"
        cur = self._execute(sql, tuple(params))
        results = []
        for r in cur.fetchall():
            try:
                af = json.loads(r["active_files"]) if isinstance(r["active_files"], str) else (r["active_files"] or [])
            except Exception:
                af = []
            try:
                ot = json.loads(r["open_tasks"]) if isinstance(r["open_tasks"], str) else (r["open_tasks"] or [])
            except Exception:
                ot = []
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
        query_tokens: List[str],
        allowed_projects: Optional[List[str]] = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        clean = [t.strip() for t in query_tokens if t.strip()]
        if not clean:
            return []

        query_str = " ".join(clean)
        where_clauses = ["MATCH(session_id, title, summary, content) AGAINST(%s IN NATURAL LANGUAGE MODE)"]
        params: List[Any] = [query_str]

        if allowed_projects is not None:
            placeholders = ", ".join(["%s"] * len(allowed_projects))
            where_clauses.append(f"project IN ({placeholders})")
            params.extend(allowed_projects)

        where_sql = " AND ".join(where_clauses)
        params.append(top_k)

        sql = f"""
            SELECT session_id, project, title, summary, timestamp,
                   MATCH(session_id, title, summary, content) AGAINST(%s IN NATURAL LANGUAGE MODE) AS score
            FROM contexts
            WHERE {where_sql}
            ORDER BY score DESC
            LIMIT %s
        """
        cur = self._execute(sql, tuple([query_str] + params))
        return [
            {
                "session_id": r["session_id"],
                "project": r["project"],
                "title": r["title"] or "",
                "summary": r["summary"],
                "score": round(float(r["score"]), 3) if isinstance(r, dict) else 1.0,
                "timestamp": r["timestamp"]
            }
            for r in cur.fetchall()
        ]

    # =========================================================================
    # Analysis References
    # =========================================================================

    def store_analysis_ref(self, ref: AnalysisRefRecord) -> None:
        zbody = zlib.compress(ref.body_text.encode("utf-8"), level=9)
        sql = """
            INSERT INTO analysis_refs (ref_id, filepath, name, start_line, end_line, kind, zbody, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                filepath = VALUES(filepath),
                name = VALUES(name),
                start_line = VALUES(start_line),
                end_line = VALUES(end_line),
                kind = VALUES(kind),
                zbody = VALUES(zbody),
                timestamp = VALUES(timestamp)
        """
        self._execute(sql, (ref.ref_id, ref.filepath, ref.name, ref.start_line, ref.end_line, ref.kind, zbody, ref.timestamp))
        self._auto_commit()

    def get_analysis_ref(self, ref_id: str) -> Optional[AnalysisRefRecord]:
        cur = self._execute(
            "SELECT ref_id, filepath, name, start_line, end_line, kind, zbody, timestamp FROM analysis_refs WHERE ref_id = %s",
            (ref_id,)
        )
        row = cur.fetchone()
        if not row:
            return None
        try:
            body = zlib.decompress(row["zbody"]).decode("utf-8", errors="replace")
        except Exception:
            body = ""
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
        cutoff = time.time() - older_than_seconds
        cur = self._execute("DELETE FROM analysis_refs WHERE timestamp < %s", (cutoff,))
        count = getattr(cur, "rowcount", 0)
        self._auto_commit()
        return count

    # =========================================================================
    # Semantic Cache & State
    # =========================================================================

    def get_semantic_cache(self, cache_key: str, file_hash: str) -> Optional[Dict[str, Any]]:
        cur = self._execute(
            "SELECT result_json FROM semantic_cache WHERE cache_key = %s AND file_hash = %s",
            (cache_key, file_hash)
        )
        row = cur.fetchone()
        if row:
            val = row["result_json"]
            if isinstance(val, dict):
                return val
            try:
                return json.loads(val)
            except Exception:
                return None
        return None

    def set_semantic_cache(self, cache_key: str, file_hash: str, result: Dict[str, Any]) -> None:
        sql = """
            INSERT INTO semantic_cache (cache_key, file_hash, result_json, timestamp)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                file_hash = VALUES(file_hash),
                result_json = VALUES(result_json),
                timestamp = VALUES(timestamp)
        """
        self._execute(sql, (cache_key, file_hash, json.dumps(result), time.time()))
        self._auto_commit()

    def clear_semantic_cache(self) -> None:
        self._execute("DELETE FROM semantic_cache")
        self._auto_commit()

    def get_state(self, key: str) -> Optional[Any]:
        cur = self._execute("SELECT value_json FROM session_state WHERE `key` = %s", (key,))
        row = cur.fetchone()
        if row:
            val = row["value_json"]
            if isinstance(val, (dict, list)):
                return val
            try:
                return json.loads(val)
            except Exception:
                return None
        return None

    def set_state(self, key: str, value: Any) -> None:
        sql = """
            INSERT INTO session_state (`key`, value_json, updated)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE
                value_json = VALUES(value_json),
                updated = VALUES(updated)
        """
        self._execute(sql, (key, json.dumps(value), time.time()))
        self._auto_commit()
