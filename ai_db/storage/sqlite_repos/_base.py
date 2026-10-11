"""What every SQLite repository shares: the write lock, the ``@_writes`` single write
path, payload codecs, and the attributes the composing backend provides."""

from __future__ import annotations

import contextlib
import functools
import json
import sqlite3
import struct
import threading
import zlib
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, TypeVar, cast

from ai_db.errors import AiDbStorageError

if TYPE_CHECKING:
    from ai_db.storage.models import ChunkRecord

# Process-wide write mutex (TODO 13.6).
#
# WAL allows many readers but exactly one writer, and a *deferred* transaction
# that starts as a reader and later tries to write gets SQLITE_BUSY immediately
# -- it does not honour busy_timeout -- which surfaces as "database is locked".
# Serialising writers inside the process removes that upgrade race entirely.
# transaction() holds it for the whole write; every public write method goes
# through transaction() via @_writes, so no write statement runs outside it.
# It is an RLock so nested transaction() calls on one thread re-enter freely,
# and it is module-level so every backend instance on the same file contends.
# Readers never take it, so query latency is unaffected.
_WRITE_LOCK = threading.RLock()

# No-op context manager for nested savepoints, which already hold the lock.
_NULL_LOCK = contextlib.nullcontext()

_F = TypeVar("_F", bound=Callable[..., Any])


def _writes(method: _F) -> _F:
    """Run a public write method as one ``transaction()``.

    The single write path: it takes ``_WRITE_LOCK`` for the whole write (not just the
    COMMIT), commits on success and rolls back on any exception. Without it Python's
    implicit BEGIN left a failed write's partial rows pending, and the next unrelated
    write committed them. Nested calls (a write inside ``transaction()`` or another
    write) become savepoints and commit with the outermost one.
    """
    @functools.wraps(method)
    def wrapper(self: SQLiteRepository, *args: Any, **kwargs: Any) -> Any:
        with self.transaction():
            return method(self, *args, **kwargs)
    return cast(_F, wrapper)


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


def pack_vector(vec: list[float]) -> bytes:
    """float32 little-endian blob (stdlib ``struct``; no numpy needed)."""
    return struct.pack(f"<{len(vec)}f", *vec)


def unpack_vector(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob[: n * 4]))


def cosine_distance(a: list[float], b: list[float]) -> float:
    """1 - cosine similarity; 1.0 for a zero vector.

    A length mismatch raises: it means vectors from two embedding models are being
    compared, which the recall extension prevents by re-embedding on a model change.
    """
    if len(a) != len(b):
        raise AiDbStorageError(f"vector dimension mismatch: {len(a)} vs {len(b)}")
    if not a:
        return 1.0
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 1.0
    similarity: float = dot / ((na ** 0.5) * (nb ** 0.5))
    return 1.0 - similarity


class SQLiteRepository:
    """Base of the repository mixins. ``SQLiteBackend`` composes them and provides the
    connection, ``transaction()`` and the settings below; each repository may call the
    others' methods through ``self`` (declared here for the type checker)."""

    if TYPE_CHECKING:
        db_path: str
        vector_index: str

        @property
        def conn(self) -> sqlite3.Connection: ...
        def _check_closed(self) -> None: ...
        def transaction(self) -> AbstractContextManager[None]: ...
        def _vector_tables(self, cur: sqlite3.Cursor) -> list[str]: ...
        def _before_delete_chunk_ids(self, cur: sqlite3.Cursor, ids: list[int]) -> None: ...
        def _before_delete_file_chunks(self, cur: sqlite3.Cursor, filepath: str) -> None: ...

        # provided by the other repositories
        def get_state(self, key: str) -> Any | None: ...
        def set_state(self, key: str, value: Any) -> None: ...
        def get_chunks_by_ids(self, ids: list[int]) -> list[ChunkRecord]: ...
        @staticmethod
        def _append_filters(where: list[str], params: list[Any],
                            allowed_projects: list[str] | None, path_prefix: str | None,
                            languages: list[str] | None, chunk_types: list[str] | None,
                            modified_since: float | None) -> None: ...
        @staticmethod
        def _move_vec0_rows(cur: sqlite3.Cursor, chunks: list[ChunkRecord]) -> None: ...
