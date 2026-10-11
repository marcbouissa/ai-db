"""Chunk embeddings: the exact table or the vec0 ANN index (VectorCapable).

A mixin of :class:`ai_db.storage.sqlite_backend.SQLiteBackend`: every repository shares
the backend's connection and ``transaction()`` (see ``_base.SQLiteRepository``)."""

from __future__ import annotations

import sqlite3
from typing import Any

from ai_db.constants import (
    VEC0_OVERFETCH,
)
from ai_db.errors import AiDbConfigError, AiDbStorageError
from ai_db.storage.models import (
    ChunkRecord,
)
from ai_db.storage.sqlite_repos._base import (
    SQLiteRepository,
    _writes,
)


class VectorRepository(SQLiteRepository):
    VECTOR_INDEX_MODES = ("exact", "vec0")

    def get_embed_meta(self) -> dict[str, Any] | None:
        self._check_closed()
        return self.get_state("embed_meta")

    def ensure_vector_index(self, dim: int, model_id: str,
                            vector_index: str | None = None) -> None:
        self._check_closed()
        mode = vector_index or self.vector_index
        if mode not in self.VECTOR_INDEX_MODES:
            raise AiDbConfigError(
                f"vector_index must be one of {list(self.VECTOR_INDEX_MODES)}, got {mode!r}")
        meta = self.get_embed_meta()
        wanted = {"model_id": model_id, "dim": int(dim), "index": mode}
        if meta is None:
            self.set_state("embed_meta", wanted)
        elif meta != wanted:
            raise AiDbConfigError(
                f"vector index was built with {meta}, config wants {wanted}; "
                "run: ai-db reindex --embeddings"
            )
        if mode == "vec0":
            self._create_vec0_table(int(dim))
        self.vector_index = mode

    def _create_vec0_table(self, dim: int) -> None:
        """ANN table. project is a partition key so it can be constrained in-query."""
        self.conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                chunk_id INTEGER PRIMARY KEY,
                embedding float[{dim}] distance_metric=cosine,
                project TEXT partition key,
                language TEXT,
                chunk_type TEXT
            )
        """)

    def _vec_table(self) -> str:
        return "vec_chunks" if self.vector_index == "vec0" else "chunk_vectors"

    @_writes
    def upsert_embeddings(self, items: list[tuple[int, list[float]]]) -> None:
        self._check_closed()
        import sqlite_vec

        meta = self.get_embed_meta()
        if meta is None:
            raise AiDbStorageError("upsert_embeddings called before ensure_vector_index")
        for chunk_id, vec in items:
            if len(vec) != meta["dim"]:
                raise AiDbStorageError(f"chunk {chunk_id}: vector dim {len(vec)} != index dim {meta['dim']}")

        if self.vector_index == "vec0":
            # vec0 carries the metadata columns so KNN can constrain without a join.
            rows = []
            for chunk_id, vec in items:
                info = self.conn.execute(
                    "SELECT project, language, chunk_type FROM chunks WHERE id = ?",
                    (chunk_id,)).fetchone()
                if info is None:
                    continue
                rows.append((chunk_id, sqlite_vec.serialize_float32(vec),
                             info["project"], info["language"], info["chunk_type"]))
            self.conn.executemany(
                "INSERT OR REPLACE INTO vec_chunks "
                "(chunk_id, embedding, project, language, chunk_type) VALUES (?, ?, ?, ?, ?)",
                rows,
            )
        else:
            self.conn.executemany(
                "INSERT OR REPLACE INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
                [(cid, sqlite_vec.serialize_float32(vec)) for cid, vec in items],
            )

    def search_vectors(self, vector: list[float], k: int,
                       filters: dict[str, Any] | None = None) -> list[tuple[int, float]]:
        self._check_closed()
        import sqlite_vec

        f = dict(filters or {})
        allowed = f.pop("allowed_projects", None)
        if allowed is not None and len(allowed) == 0:
            return []
        path_prefix = f.pop("path_prefix", None)
        languages = f.pop("languages", None)
        chunk_types = f.pop("chunk_types", None)
        modified_since = f.pop("modified_since", None)
        if f:
            raise ValueError(f"unknown vector filter(s): {sorted(f)}")

        if self.vector_index == "vec0":
            return self._search_vectors_vec0(
                vector, k, allowed, languages, chunk_types, path_prefix, modified_since)

        where: list[str] = []
        params: list[Any] = [sqlite_vec.serialize_float32(vector)]
        self._append_filters(where, params, allowed, path_prefix,
                             languages, chunk_types, modified_since)
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

    def _search_vectors_vec0(self, vector: list[float], k: int,
                             allowed: list[str] | None, languages: list[str] | None,
                             chunk_types: list[str] | None, path_prefix: str | None,
                             modified_since: float | None) -> list[tuple[int, float]]:
        """ANN KNN. Metadata that vec0 carries is constrained in-query; filters that
        need the chunks/files join over-fetch k*VEC0_OVERFETCH and truncate afterwards."""
        import sqlite_vec

        conds = ["embedding MATCH ?"]
        params: list[Any] = [sqlite_vec.serialize_float32(vector)]
        if allowed is not None:
            conds.append("project IN (" + ",".join("?" * len(allowed)) + ")")
            params.extend(allowed)
        if languages:
            conds.append("language IN (" + ",".join("?" * len(languages)) + ")")
            params.extend(languages)
        if chunk_types:
            conds.append("chunk_type IN (" + ",".join("?" * len(chunk_types)) + ")")
            params.extend(chunk_types)

        # These need a join, so KNN must over-fetch and we filter afterwards.
        needs_join = path_prefix is not None or modified_since is not None
        fetch = int(k) * VEC0_OVERFETCH if needs_join else int(k)
        conds.append("k = ?")
        params.append(fetch)

        rows = self.conn.execute(
            f"""
            SELECT chunk_id, distance
            FROM vec_chunks
            WHERE {" AND ".join(conds)}
            ORDER BY distance
            """,
            params,
        ).fetchall()

        if not needs_join:
            return [(r["chunk_id"], float(r["distance"])) for r in rows[:int(k)]]

        ids = [r["chunk_id"] for r in rows]
        if not ids:
            return []
        meta_where: list[str] = ["chunks.id IN (" + ",".join("?" * len(ids)) + ")"]
        meta_params: list[Any] = list(ids)
        self._append_filters(meta_where, meta_params, None, path_prefix,
                             None, None, modified_since)
        kept = {
            r["id"]: None for r in self.conn.execute(
                f"SELECT chunks.id FROM chunks "
                f"JOIN files ON files.filepath = chunks.filepath "
                f"WHERE {' AND '.join(meta_where)}", meta_params).fetchall()
        }
        return [(r["chunk_id"], float(r["distance"])) for r in rows
                if r["chunk_id"] in kept][:int(k)]

    def chunks_missing_embeddings(self, limit: int) -> list[ChunkRecord]:
        self._check_closed()
        cur = self.conn.execute(
            f"""
            SELECT c.id FROM chunks c
            LEFT JOIN {self._vec_table()} cv ON cv.chunk_id = c.id
            WHERE cv.chunk_id IS NULL
            ORDER BY c.id LIMIT ?
            """,
            (int(limit),),
        )
        return self.get_chunks_by_ids([r["id"] for r in cur.fetchall()])

    @_writes
    def drop_vector_index(self) -> None:
        self._check_closed()
        for table in self._vector_tables(self.conn.cursor()):
            # vec_chunks fixes its dimension at CREATE time: drop it so a model with a
            # different dimension can recreate it (ensure_vector_index does).
            self.conn.execute("DROP TABLE vec_chunks" if table == "vec_chunks" else f"DELETE FROM {table}")
        self.conn.execute("DELETE FROM session_state WHERE key = 'embed_meta'")

    @staticmethod
    def _move_vec0_rows(cur: sqlite3.Cursor, chunks: list[ChunkRecord]) -> None:
        """Rewrite vec0 metadata for ``chunks``. ``project`` is a vec0 partition key, which
        cannot be UPDATEd in place, so each row is re-inserted with its stored vector."""
        for c in chunks:
            row = cur.execute("SELECT embedding FROM vec_chunks WHERE chunk_id = ?", (c.id,)).fetchone()
            if row is None:
                continue  # not embedded yet: embed_missing writes it with current metadata
            cur.execute("DELETE FROM vec_chunks WHERE chunk_id = ?", (c.id,))
            cur.execute("INSERT INTO vec_chunks (chunk_id, embedding, project, language, chunk_type) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (c.id, row["embedding"], c.project, c.language, c.chunk_type))
