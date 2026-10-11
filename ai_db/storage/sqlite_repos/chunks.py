"""Files, chunks with their FTS rows, search, diagnostics and annotations.

A mixin of :class:`ai_db.storage.sqlite_backend.SQLiteBackend`: every repository shares
the backend's connection and ``transaction()`` (see ``_base.SQLiteRepository``)."""

from __future__ import annotations

import sqlite3
import zlib
from typing import Any

from ai_db.constants import (
    BM25_WEIGHTS,
    ZLIB_LEVEL,
)
from ai_db.errors import AiDbQueryError, AiDbStorageError
from ai_db.search.query_builder import (
    build_fts,
    identifier_words,
    split_identifier,
)
from ai_db.storage.models import (
    AnnotationRecord,
    ChunkRecord,
    FileRecord,
    SearchResult,
    SyntaxErrorRecord,
)
from ai_db.storage.sqlite_repos._base import (
    SQLiteRepository,
    _decompress,
    _writes,
)


class ChunkRepository(SQLiteRepository):
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
        clause, params = self._path_prefix_clause("filepath", prefix)
        cur = self.conn.cursor()
        cur.execute(f"SELECT filepath, sha256 FROM files WHERE {clause}", params)
        return {r["filepath"]: r["sha256"] for r in cur.fetchall()}

    def get_all_filepaths(self) -> list[str]:
        self._check_closed()
        cur = self.conn.cursor()
        cur.execute("SELECT filepath FROM files")
        return [r["filepath"] for r in cur.fetchall()]

    @_writes
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

    @_writes
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

    _CHUNK_COLS = ("filepath, chunk_type, name, start_line, end_line, zcontent, project, "
                   "qualified_name, language, token_count, content_hash, symbol_name")

    SYMBOL_CHUNK_TYPES = ("code", "class_header")

    @classmethod
    def _symbol_name(cls, c: ChunkRecord) -> str:
        """Lookup key for ``find_chunks_by_symbol``: last qualified-name component."""
        return c.qualified_name.rsplit(".", 1)[-1] if c.chunk_type in cls.SYMBOL_CHUNK_TYPES else ""

    def _insert_chunk_rows(self, chunks: list[ChunkRecord]) -> None:
        """Insert chunks + FTS rows and assign ``c.id``. Does not resolve parents."""
        if not chunks:
            return
        cur = self.conn.cursor()
        rows = [
            (c.filepath, c.chunk_type, c.name, c.start_line, c.end_line,
             zlib.compress(c.content.encode("utf-8"), level=ZLIB_LEVEL), c.project,
             c.qualified_name, c.language, c.token_count, c.content_hash, self._symbol_name(c))
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

    @_writes
    def insert_chunks(self, chunks: list[ChunkRecord]) -> None:
        self._check_closed()
        self._insert_chunk_rows(chunks)
        self._resolve_parents(chunks)

    @_writes
    def replace_file_chunks(self, filepath: str, chunks: list[ChunkRecord]) -> dict[str, int]:
        """Diff ``chunks`` against the stored chunks of ``filepath`` by (content_hash, name).

        Unchanged chunks keep their id (and therefore their embedding); their line span
        and parent are updated. Returns counts of kept/inserted/deleted chunks.
        """
        self._check_closed()
        cur = self.conn.cursor()
        cur.execute("SELECT id, content_hash, name, qualified_name, project, language, chunk_type "
                    "FROM chunks WHERE filepath = ? ORDER BY id", (filepath,))
        pool: dict[tuple[str, str], list[int]] = {}
        old: dict[int | None, sqlite3.Row] = {}  # keyed by id; kept chunks always have one
        for r in cur.fetchall():
            pool.setdefault((r["content_hash"], r["name"]), []).append(r["id"])
            old[r["id"]] = r
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
            "qualified_name = ?, language = ?, symbol_name = ? WHERE id = ?",
            [(c.start_line, c.end_line, c.chunk_type, c.project, c.qualified_name, c.language,
              self._symbol_name(c), c.id) for c in kept],
        )
        # A kept chunk's body is unchanged, but what is derived from its name or metadata
        # is not: re-derive the FTS row (indexes qualified_name) and the vec0 metadata
        # copy (filters by project), or a rename / project move leaves them stale.
        renamed = [c for c in kept if old[c.id]["qualified_name"] != c.qualified_name]
        if renamed:
            cur.executemany("DELETE FROM fts_index WHERE rowid = ?", [(c.id,) for c in renamed])
            self._index_chunks_fts(cur, renamed)
        moved = [c for c in kept
                 if (old[c.id]["project"], old[c.id]["language"], old[c.id]["chunk_type"])
                 != (c.project, c.language, c.chunk_type)]
        if moved and "vec_chunks" in self._vector_tables(cur):
            self._move_vec0_rows(cur, moved)
        self._insert_chunk_rows(new)
        self._resolve_parents(chunks)
        return {"kept": len(kept), "inserted": len(new), "deleted": len(stale)}

    @_writes
    def clear_file_metadata(self, filepath: str) -> None:
        """Delete everything derived from ``filepath`` except its file row and chunks."""
        self._check_closed()
        cur = self.conn.cursor()
        cur.execute("DELETE FROM symbols WHERE filepath = ?", (filepath,))
        cur.execute("DELETE FROM symbol_refs WHERE caller_filepath = ?", (filepath,))
        cur.execute("DELETE FROM annotations WHERE filepath = ?", (filepath,))
        cur.execute("DELETE FROM syntax_errors WHERE filepath = ?", (filepath,))
        cur.execute("DELETE FROM analysis_refs WHERE filepath = ?", (filepath,))

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
        w = ", ".join(str(x) for x in BM25_WEIGHTS)
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
    def _path_prefix_clause(column: str, prefix: str | None) -> tuple[str, list[Any]]:
        """Match ``prefix`` itself or anything under the ``prefix/`` directory.

        Compares bytes with ``substr`` instead of ``LIKE``: ``LIKE`` treats ``_``/``%`` as
        wildcards and ignores ASCII case, so ``/src/a_b`` would also match ``/src/A-B/``
        and a sync of one project would prune the other.
        """
        exact = (prefix or "").replace("\x00", "")
        under = exact.rstrip("/") + "/"
        return (f"({column} = ? OR substr({column}, 1, ?) = ?)", [exact, len(under), under])

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
            clause, prefix_params = ChunkRepository._path_prefix_clause("chunks.filepath", path_prefix)
            where.append(clause)
            params.extend(prefix_params)
        if languages:
            where.append(f"chunks.language IN ({','.join('?' * len(languages))})")
            params.extend(languages)
        if chunk_types:
            where.append(f"chunks.chunk_type IN ({','.join('?' * len(chunk_types))})")
            params.extend(chunk_types)
        if modified_since is not None:
            where.append("files.last_modified >= ?")
            params.append(float(modified_since))

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

    @_writes
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

    @_writes
    def delete_syntax_error(self, filepath: str) -> None:
        self._check_closed()
        cur = self.conn.cursor()
        cur.execute("DELETE FROM syntax_errors WHERE filepath = ?", (filepath,))

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

    @_writes
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
