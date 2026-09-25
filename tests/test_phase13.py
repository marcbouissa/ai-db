"""Phase 13 performance/scale tests: write lock, incremental centrality, vec0 mode."""
from __future__ import annotations

import threading
import time

import pytest

from ai_db.storage.sqlite_backend import SQLiteBackend

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ------------------------------------------------------------- 13.6 write lock
def test_eight_threads_sync_and_query_no_lock_error(tmp_path):
    """8 threads writing + reading one DB must not raise 'database is locked'."""
    db = SQLiteBackend(str(tmp_path / "c.db"))
    db.initialize()
    db.conn.execute("PRAGMA busy_timeout = 50")  # fail fast if the lock is not working
    db.conn.commit()

    src = tmp_path / "src"
    src.mkdir()
    for i in range(12):
        (src / f"m{i}.py").write_text(f"def f{i}():\n    return {i}\n")

    errors: list[BaseException] = []
    start = threading.Barrier(8)

    def writer(n: int) -> None:
        try:
            start.wait(timeout=30)
            for i in range(3):
                db.set_state(f"w{n}", {"i": i})
        except BaseException as exc:  # noqa: BLE001 - collecting for assert
            errors.append(exc)

    def reader(n: int) -> None:
        try:
            start.wait(timeout=30)
            for _ in range(6):
                db.search_chunks(["f0"], top_k=3)
                db.get_table_counts()
        except BaseException as exc:  # noqa: BLE001 - collecting for assert
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
    threads += [threading.Thread(target=reader, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert not [t for t in threads if t.is_alive()], "thread deadlock"
    locked = [e for e in errors if "locked" in str(e).lower()]
    assert not locked, f"database is locked under concurrency: {locked[:3]}"
    assert not errors, f"unexpected errors: {errors[:3]}"
    db.close()


def test_write_lock_is_reentrant_for_nested_transactions(tmp_path):
    """Nested savepoints must not deadlock against the process write lock."""
    db = SQLiteBackend(str(tmp_path / "n.db"))
    db.initialize()
    before = db.conn.execute("SELECT COUNT(*) FROM session_state").fetchone()[0]
    # Nested on purpose: this is the re-entrancy check for the process write lock.
    with db.transaction():  # noqa: SIM117
        with db.transaction():
            db.conn.execute("INSERT INTO session_state (key, value_json, updated) "
                            "VALUES ('k', '1', 1.0)")
    after = db.conn.execute("SELECT COUNT(*) FROM session_state").fetchone()[0]
    assert after == before + 1
    db.close()


def test_write_lock_released_after_failure(tmp_path):
    """A failing transaction must not leave the lock held."""
    db = SQLiteBackend(str(tmp_path / "f.db"))
    db.initialize()
    with pytest.raises(ValueError):  # noqa: SIM117
        with db.transaction():
            raise ValueError("boom")
    # If the lock leaked, the probe below would block forever.
    done = threading.Event()

    def probe() -> None:
        with db.transaction():
            pass
        done.set()

    t = threading.Thread(target=probe, daemon=True)
    t.start()
    assert done.wait(timeout=10), "write lock leaked after a failed transaction"
    db.close()


# -------------------------------------------------- 13.4 incremental centrality
def test_rebuild_symbol_centrality_subset(tmp_path):
    """Only the named projects are recomputed; others keep their rows."""
    db = SQLiteBackend(str(tmp_path / "s.db"))
    db.initialize()
    for project, callee, n in (("a", "shared", 3), ("b", "only_b", 2)):
        for _ in range(n):
            db.conn.execute(
                "INSERT INTO symbol_refs (caller_filepath, caller_name, caller_line, "
                "callee_name, ref_type, project) VALUES (?,?,?,?,?,?)",
                (f"/{project}/c.py", f"module.caller{_}", 1, callee, "call", project))
    db.conn.commit()

    db.rebuild_symbol_centrality()  # all projects
    assert {r["project"] for r in db.conn.execute(
        "SELECT project FROM symbol_centrality").fetchall()} == {"a", "b"}

    # Recompute only "a" and prove "b" is untouched.
    db.conn.execute("UPDATE symbol_refs SET callee_name='renamed' WHERE project='a'")
    db.conn.commit()
    db.rebuild_symbol_centrality(projects=["a"])

    a_names = {r[0] for r in db.conn.execute(
        "SELECT name FROM symbol_centrality WHERE project='a'").fetchall()}
    b_names = {r[0] for r in db.conn.execute(
        "SELECT name FROM symbol_centrality WHERE project='b'").fetchall()}
    assert a_names == {"renamed"}, a_names
    assert b_names == {"only_b"}, b_names
    db.close()


# --------------------------------------------------------------- 13.2 vec0 option
def test_vector_index_option_key_required_and_validated(tmp_path):
    from ai_db.errors import AiDbConfigError
    from ai_db.storage.sqlite_backend import SQLiteBackend as B

    assert "vector_index" in B.OPTION_KEYS
    db = B(str(tmp_path / "o.db"))
    db.initialize()
    try:
        with pytest.raises(AiDbConfigError):
            db.ensure_vector_index(4, "m", vector_index="nonsense")
    finally:
        db.close()


@pytest.mark.parametrize("mode", ["exact", "vec0"])
def test_conformance_vector_roundtrip_both_modes(mode, tmp_path):
    """VectorCapable conformance must hold for both vector_index modes."""
    from ai_db.storage.models import ChunkRecord, FileRecord
    from ai_db.storage.sqlite_backend import SQLiteBackend as B

    db = B(str(tmp_path / f"{mode}.db"), vector_index=mode)
    db.initialize()
    for i in range(3):
        db.upsert_file(FileRecord(filepath=f"/r/f{i}.py", sha256=f"h{i}",
                                  last_modified=time.time(), chunk_count=1, project="p"))
        db.insert_chunks([ChunkRecord(
            filepath=f"/r/f{i}.py", name=f"n{i}", qualified_name=f"module.n{i}",
            chunk_type="code", content=f"def n{i}(): pass", start_line=1, end_line=1,
            language="python", project="p")])
    db.ensure_vector_index(4, "test-model", vector_index=mode)
    missing = db.chunks_missing_embeddings(10)
    assert len(missing) == 3
    db.upsert_embeddings([(c.id, [1.0, 0.0, 0.0, 0.0]) for c in missing])
    assert db.chunks_missing_embeddings(10) == []

    hits = db.search_vectors([1.0, 0.0, 0.0, 0.0], 3, {"allowed_projects": ["p"]})
    assert len(hits) == 3
    # project filter must exclude everything
    assert db.search_vectors([1.0, 0.0, 0.0, 0.0], 3, {"allowed_projects": ["other"]}) == []
    # delete hook removes the vectors
    first = missing[0].id
    db.delete_file("/r/f0.py")
    assert all(cid != first for cid, _ in
               db.search_vectors([1.0, 0.0, 0.0, 0.0], 3, {"allowed_projects": None}))
    db.close()


def test_vec0_query_uses_match_and_k(tmp_path):
    """The vec0 path must use the `embedding MATCH ? AND k = ?` form."""
    db = SQLiteBackend(str(tmp_path / "v.db"), vector_index="vec0")
    db.initialize()
    db.ensure_vector_index(4, "m", vector_index="vec0")
    seen: list[str] = []
    db.conn.set_trace_callback(lambda s: seen.append(" ".join(s.split())))
    try:
        db.search_vectors([1.0, 0.0, 0.0, 0.0], 5, {"allowed_projects": ["p"]})
    finally:
        db.conn.set_trace_callback(None)
    joined = " | ".join(seen)
    # trace callbacks report interpolated SQL, so match the shape not the placeholders
    assert "embedding MATCH" in joined, joined[:300]
    assert "k = " in joined, joined[:300]
    assert "vec_chunks" in joined, joined[:300]
    assert "project IN" in joined, joined[:300]
    db.close()


def test_vec0_path_prefix_overfetch_and_truncate(tmp_path):
    """path_prefix needs a join, so KNN over-fetches k*4 and truncates to k."""
    from ai_db.storage.models import ChunkRecord, FileRecord
    from ai_db.storage.sqlite_backend import SQLiteBackend as B

    db = B(str(tmp_path / "pp.db"), vector_index="vec0")
    db.initialize()
    for i in range(6):
        fp = f"/r{i}/f.py"
        db.upsert_file(FileRecord(filepath=fp, sha256=f"h{i}", last_modified=time.time(),
                                  chunk_count=1, project="p"))
        db.insert_chunks([ChunkRecord(filepath=fp, name="n", qualified_name="module.n",
                                      chunk_type="code", content="x", start_line=1, end_line=1,
                                      language="python", project="p")])
    db.ensure_vector_index(4, "m", vector_index="vec0")
    ids = [c.id for c in db.chunks_missing_embeddings(10)]
    db.upsert_embeddings([(i, [1.0, 0.0, 0.0, 0.0]) for i in ids])

    hits = db.search_vectors([1.0, 0.0, 0.0, 0.0], 2, {"path_prefix": "/r1"})
    assert 1 <= len(hits) <= 2
    paths = {r["filepath"] for r in db.conn.execute(
        f"SELECT filepath FROM chunks WHERE id IN ({','.join('?' * len(hits))})",
        [h[0] for h in hits]).fetchall()}
    assert paths == {"/r1/f.py"}
    db.close()
