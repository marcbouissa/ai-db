"""Reusable conformance tests for StorageBackend implementations.

Third-party connectors subclass ``BackendConformance`` in their own test suite and
provide a ``backend`` fixture returning a fresh, initialized backend::

    import pytest
    from ai_db.storage.conformance import BackendConformance

    class TestMyBackend(BackendConformance):
        @pytest.fixture
        def backend(self, tmp_path):
            b = MyBackend({"dsn": "..."})
            b.initialize()
            yield b
            b.close()

Vector tests run only when the backend declares the ``'vector'`` capability.
"""

from __future__ import annotations

import math

import pytest

from ai_db.storage.backend import StorageBackend, VectorCapable
from ai_db.storage.models import ChunkRecord, FileRecord, SymbolRecord


def _unit(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec]


class BackendConformance:
    """Contract tests. Requires a ``backend`` fixture."""

    @pytest.fixture
    def backend(self) -> StorageBackend:  # pragma: no cover - overridden
        raise NotImplementedError("subclass must provide a 'backend' fixture")

    def _seed(self, backend: StorageBackend, path: str = "/r/a.py", project: str = "p") -> None:
        backend.upsert_file(FileRecord(path, "sha", 1.0, 2, project))
        backend.insert_chunks([
            ChunkRecord(path, "code", "def alpha", 1, 3, "def alpha(): return 'needle'", project),
            ChunkRecord(path, "code", "def beta", 4, 6, "def beta(): return 'hay'", project),
        ])

    def test_is_storage_backend(self, backend):
        assert isinstance(backend, StorageBackend)
        assert {"fts"} <= set(backend.capabilities())

    def test_file_crud(self, backend):
        backend.upsert_file(FileRecord("/r/x.py", "h1", 1.0, 0, "p"))
        assert backend.get_file("/r/x.py").sha256 == "h1"
        backend.upsert_file(FileRecord("/r/x.py", "h2", 2.0, 0, "p"))
        assert backend.get_files_by_prefix("/r") == {"/r/x.py": "h2"}
        backend.delete_file("/r/x.py")
        assert backend.get_file("/r/x.py") is None

    def test_insert_chunks_assigns_ids_and_roundtrips(self, backend):
        self._seed(backend)
        chunks = backend.get_chunks_for_file("/r/a.py")
        assert [c.name for c in chunks] == ["def alpha", "def beta"]
        assert all(c.id is not None for c in chunks)
        assert chunks[0].content == "def alpha(): return 'needle'"

    def test_fts_ranks_matching_chunk_first(self, backend):
        self._seed(backend)
        hits = backend.search_chunks(["needle"], allowed_projects=["p"], top_k=5)
        assert hits and hits[0].name == "def alpha"

    def test_project_filter(self, backend):
        self._seed(backend, project="p")
        assert backend.search_chunks(["needle"], allowed_projects=["other"], top_k=5) == []
        assert backend.search_chunks(["needle"], allowed_projects=[], top_k=5) == []

    def test_path_prefix_filter(self, backend):
        self._seed(backend, path="/r/a.py")
        self._seed(backend, path="/s/a.py")
        hits = backend.search_chunks(["needle"], allowed_projects=["p"], top_k=5, path_prefix="/s")
        assert {h.filepath for h in hits} == {"/s/a.py"}

    def test_cascade_delete(self, backend):
        self._seed(backend)
        backend.insert_symbols([SymbolRecord("alpha", "function", "/r/a.py", 1, "def alpha()", "p")])
        backend.delete_file("/r/a.py")
        assert backend.get_chunks_for_file("/r/a.py") == []
        assert backend.query_symbols(name="alpha", allowed_projects=["p"]) == []
        assert backend.search_chunks(["needle"], allowed_projects=["p"], top_k=5) == []

    def test_transaction_rollback(self, backend):
        with pytest.raises(RuntimeError), backend.transaction():
            backend.upsert_file(FileRecord("/r/t.py", "h", 1.0, 0, "p"))
            raise RuntimeError("boom")
        assert backend.get_file("/r/t.py") is None

    def test_nested_transaction_commit(self, backend):
        with backend.transaction(), backend.transaction():
            backend.upsert_file(FileRecord("/r/n.py", "h", 1.0, 0, "p"))
        assert backend.get_file("/r/n.py") is not None

    # --- vector capability -------------------------------------------------

    def _vector_backend(self, backend):
        if "vector" not in backend.capabilities():
            pytest.skip("backend does not declare the 'vector' capability")
        assert isinstance(backend, VectorCapable)
        return backend

    def test_vector_roundtrip_and_nearest_first(self, backend):
        vb = self._vector_backend(backend)
        self._seed(vb)
        vb.ensure_vector_index(3, "test:model")
        assert vb.get_embed_meta() == {"model_id": "test:model", "dim": 3}
        missing = vb.chunks_missing_embeddings(limit=10)
        assert len(missing) == 2
        a, b = sorted(missing, key=lambda c: c.start_line)
        vb.upsert_embeddings([(a.id, _unit([1, 0, 0])), (b.id, _unit([0, 1, 0]))])
        assert vb.chunks_missing_embeddings(limit=10) == []
        hits = vb.search_vectors(_unit([0.9, 0.1, 0]), k=2, filters={"allowed_projects": ["p"]})
        assert [cid for cid, _ in hits] == [a.id, b.id]
        assert hits[0][1] <= hits[1][1]

    def test_vector_filters_and_cascade(self, backend):
        vb = self._vector_backend(backend)
        self._seed(vb)
        vb.ensure_vector_index(3, "test:model")
        items = [(c.id, _unit([1, 1, 0])) for c in vb.chunks_missing_embeddings(limit=10)]
        vb.upsert_embeddings(items)
        assert vb.search_vectors(_unit([1, 1, 0]), k=5, filters={"allowed_projects": ["x"]}) == []
        vb.delete_file("/r/a.py")
        assert vb.search_vectors(_unit([1, 1, 0]), k=5, filters={"allowed_projects": ["p"]}) == []

    def test_drop_vector_index(self, backend):
        vb = self._vector_backend(backend)
        vb.ensure_vector_index(3, "test:model")
        vb.drop_vector_index()
        assert vb.get_embed_meta() is None

    # --- incremental chunk replacement ---------------------------------------

    def test_replace_file_chunks_keeps_unchanged_ids(self, backend):
        path = "/r/inc.py"
        backend.upsert_file(FileRecord(path, "s1", 1.0, 2, "p"))
        v1 = [
            ChunkRecord(path, "class_header", "class A", 1, 2, "class A:", "p"),
            ChunkRecord(path, "code", "def A.f", 3, 4, "def f(): return 1", "p", parent_index=0),
        ]
        assert backend.replace_file_chunks(path, v1) == {"kept": 0, "inserted": 2, "deleted": 0}
        header_id, f_id = v1[0].id, v1[1].id
        v2 = [
            ChunkRecord(path, "class_header", "class A", 1, 2, "class A:", "p"),
            ChunkRecord(path, "code", "def A.g", 3, 4, "def g(): return 2", "p", parent_index=0),
        ]
        assert backend.replace_file_chunks(path, v2) == {"kept": 1, "inserted": 1, "deleted": 1}
        assert v2[0].id == header_id and v2[1].id != f_id
        stored = {c.name: c for c in backend.get_chunks_for_file(path)}
        assert set(stored) == {"class A", "def A.g"}
        assert stored["def A.g"].parent_id == header_id
        assert backend.search_chunks(["return"], allowed_projects=["p"], top_k=5)[0].name == "def A.g"

    def test_clear_file_metadata_keeps_chunks(self, backend):
        self._seed(backend)
        backend.insert_symbols([SymbolRecord("alpha", "function", "/r/a.py", 1, "def alpha()", "p")])
        backend.clear_file_metadata("/r/a.py")
        assert backend.query_symbols(name="alpha", allowed_projects=["p"]) == []
        assert len(backend.get_chunks_for_file("/r/a.py")) == 2
