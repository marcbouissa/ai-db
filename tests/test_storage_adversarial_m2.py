"""Empirical adversarial stress test suite for Milestone 2: Pluggable Storage Layer.

Focus areas:
1. WAL mode concurrency (readers not blocked by uncommitted writes, snapshot isolation, thread/process concurrency).
2. Atomic transaction rollback (multi-table atomicity, zero orphaned records across all tables, FK enforcement).
3. Nested transactions (multi-level savepoint rollback and commit, sibling savepoints, depth tracking).
4. Cascading deletions (delete_file cleanly purges files, chunks, symbols, refs, annotations, syntax_errors, analysis_refs, and FTS index).
"""

import os
import sqlite3
import subprocess
import sys
import threading
import time

import pytest

from ai_db.storage.models import (
    AnalysisRefRecord,
    AnnotationRecord,
    ChunkRecord,
    ContextRecord,
    FileRecord,
    SkillRecord,
    SymbolRecord,
    SymbolRefRecord,
    SyntaxErrorRecord,
)
from ai_db.storage.sqlite_backend import SQLiteBackend

# ==============================================================================
# 1. WAL Mode Concurrency Stress Tests
# ==============================================================================

@pytest.mark.storage
class TestWALModeConcurrency:
    """Empirical verification of SQLite WAL mode concurrency and non-blocking reads."""

    def test_wal_mode_active_and_pragmas(self, temp_db_path):
        """Verify journal_mode is WAL and synchronous is NORMAL on real disk file."""
        backend = SQLiteBackend(temp_db_path)
        backend.initialize()

        # Check pragmas on backend's internal connection
        cur = backend.conn.cursor()
        cur.execute("PRAGMA journal_mode")
        journal_mode = cur.fetchone()[0].lower()
        cur.execute("PRAGMA synchronous")
        sync_mode = cur.fetchone()[0]
        cur.execute("PRAGMA foreign_keys")
        fk_mode = cur.fetchone()[0]

        # Also check journal mode from a separate connection
        conn = sqlite3.connect(temp_db_path)
        ext_cur = conn.cursor()
        ext_cur.execute("PRAGMA journal_mode")
        ext_journal = ext_cur.fetchone()[0].lower()
        conn.close()
        backend.close()

        assert journal_mode == "wal", f"Expected WAL journal mode on backend.conn, got {journal_mode}"
        assert ext_journal == "wal", f"Expected WAL journal mode on external conn, got {ext_journal}"
        assert sync_mode in (1, "1", "NORMAL", "normal"), f"Expected NORMAL synchronous (1), got {sync_mode}"
        assert fk_mode in (1, "1", "ON", "on"), f"Expected foreign_keys ON (1), got {fk_mode}"

    def test_wal_shm_files_created_on_disk(self, temp_db_path):
        """Verify that writing to SQLite backend in WAL mode generates WAL and SHM files."""
        backend = SQLiteBackend(temp_db_path)
        backend.initialize()

        backend.upsert_file(FileRecord("src/wal_test.py", "wal_hash", 1000.0, 1))

        # Check on disk for db file
        assert os.path.exists(temp_db_path)
        wal_path = temp_db_path + "-wal"
        shm_path = temp_db_path + "-shm"

        # Under WAL mode with active transactions or dirty cache, wal or shm exists
        assert os.path.exists(wal_path) or os.path.exists(shm_path)
        backend.close()

    def test_concurrent_readers_not_blocked_by_uncommitted_write_threads(self, temp_db_path):
        """Verify multiple reader threads are NOT blocked by an uncommitted write transaction."""
        backend = SQLiteBackend(temp_db_path)
        backend.initialize()

        # Seed initial data
        backend.upsert_file(FileRecord("src/stable.py", "hash_stable", 1000.0, 1))
        backend.insert_chunks([
            ChunkRecord("src/stable.py", "func", "stable_func", 1, 10, "def stable_func(): return 'stable'")
        ])
        backend.set_state("status", {"ready": True, "version": 1})

        reader_results = []
        reader_errors = []
        barrier = threading.Barrier(5)  # 1 writer + 4 readers

        def reader_task(reader_id: int):
            try:
                # Wait for writer to enter uncommitted transaction
                barrier.wait(timeout=5.0)
                # Reader opens independent connection to same DB file
                r_backend = SQLiteBackend(temp_db_path)
                r_backend.initialize()

                # Attempt multiple reads during writer's uncommitted transaction
                t0 = time.time()
                file_rec = r_backend.get_file("src/stable.py")
                chunks = r_backend.get_chunks_for_file("src/stable.py")
                state = r_backend.get_state("status")
                # Attempt to read uncommitted file
                dirty_file = r_backend.get_file("src/uncommitted.py")
                read_elapsed = time.time() - t0

                r_backend.close()
                reader_results.append({
                    "reader_id": reader_id,
                    "file_present": file_rec is not None,
                    "chunk_count": len(chunks),
                    "version": state.get("version") if state else None,
                    "dirty_read": dirty_file is not None,
                    "elapsed": read_elapsed,
                })
            except Exception as e:
                reader_errors.append((reader_id, str(e)))

        threads = [threading.Thread(target=reader_task, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()

        # Writer opens long uncommitted transaction
        with backend.transaction():
            backend.upsert_file(FileRecord("src/uncommitted.py", "hash_dirty", 2000.0, 1))
            backend.set_state("status", {"ready": False, "version": 2})
            # Signal readers to execute while write lock / transaction is open
            barrier.wait(timeout=5.0)
            # Hold lock to allow readers to complete while uncommitted
            time.sleep(0.5)

        for t in threads:
            t.join(timeout=5.0)
        backend.close()

        assert len(reader_errors) == 0, f"Reader threads encountered errors: {reader_errors}"
        assert len(reader_results) == 4, f"Expected 4 reader results, got {len(reader_results)}"
        for res in reader_results:
            assert res["file_present"] is True, f"Reader {res['reader_id']} could not read stable file"
            assert res["chunk_count"] == 1, f"Reader {res['reader_id']} read wrong chunk count"
            # Snapshot isolation: reader saw version 1, NOT version 2
            assert res["version"] == 1, f"Dirty read detected: reader saw version {res['version']} instead of 1"
            # Snapshot isolation: reader did NOT see uncommitted file
            assert res["dirty_read"] is False, "Dirty read detected: reader saw uncommitted file"
            # Read should be fast (not blocked/waiting for 0.5s write transaction)
            assert res["elapsed"] < 0.4, f"Reader {res['reader_id']} was delayed ({res['elapsed']}s), possibly blocked"

    def test_concurrent_reader_subprocess_during_uncommitted_write(self, temp_db_path):
        """Verify an OS subprocess reader is NOT blocked by an uncommitted write in main process."""
        backend = SQLiteBackend(temp_db_path)
        backend.initialize()

        backend.set_state("counter", 100)

        # Worker script to run in a separate subprocess
        worker_code = f"""
import sys
import sqlite3
try:
    conn = sqlite3.connect(r"{temp_db_path}", timeout=1.0)
    cur = conn.cursor()
    cur.execute("SELECT value_json FROM session_state WHERE key = 'counter'")
    row = cur.fetchone()
    conn.close()
    if row and "100" in row[0]:
        sys.exit(0)
    else:
        sys.exit(2)
except Exception as exc:
    print(exc, file=sys.stderr)
    sys.exit(1)
"""
        with backend.transaction():
            backend.set_state("counter", 200)

            # While transaction is uncommitted, launch external subprocess
            proc = subprocess.run(
                [sys.executable, "-c", worker_code],
                capture_output=True,
                text=True,
                timeout=3.0, check=False
            )

        backend.close()

        assert proc.returncode == 0, (
            f"Subprocess reader failed or blocked! code={proc.returncode}, "
            f"stdout={proc.stdout}, stderr={proc.stderr}"
        )

    def test_concurrent_read_bm25_search_under_continuous_writes(self, temp_db_path):
        """Verify high-concurrency FTS5 BM25 queries execute reliably during background writes."""
        backend = SQLiteBackend(temp_db_path)
        backend.initialize()

        # Seed initial searchable files
        for i in range(10):
            backend.upsert_file(FileRecord(f"src/file_{i}.py", f"hash_{i}", 1000.0, 1))
            backend.insert_chunks([
                ChunkRecord(f"src/file_{i}.py", "func", f"compute_{i}", 1, 10,
                            f"def compute_{i}(): return 'searchable_token_{i % 3}'")
            ])

        stop_event = threading.Event()
        writer_errors = []
        reader_errors = []
        read_counts = [0, 0, 0]

        def writer_loop():
            w_backend = SQLiteBackend(temp_db_path)
            w_backend.initialize()
            count = 100
            try:
                while not stop_event.is_set():
                    with w_backend.transaction():
                        fn = f"src/dyn_{count}.py"
                        w_backend.upsert_file(FileRecord(fn, f"hash_{count}", 1000.0, 1))
                        w_backend.insert_chunks([
                            ChunkRecord(fn, "func", f"dyn_{count}", 1, 5, f"def dyn_{count}(): return 'token'")
                        ])
                    count += 1
                    time.sleep(0.01)
            except Exception as e:
                writer_errors.append(str(e))
            finally:
                w_backend.close()

        def reader_loop(idx: int):
            r_backend = SQLiteBackend(temp_db_path)
            r_backend.initialize()
            try:
                while not stop_event.is_set():
                    hits = r_backend.search_chunks(["searchable_token_0"], top_k=5)
                    assert len(hits) > 0
                    read_counts[idx] += 1
                    time.sleep(0.005)
            except Exception as e:
                reader_errors.append(f"Reader {idx}: {e}")
            finally:
                r_backend.close()

        writer_thread = threading.Thread(target=writer_loop)
        reader_threads = [threading.Thread(target=reader_loop, args=(i,)) for i in range(3)]

        writer_thread.start()
        for rt in reader_threads:
            rt.start()

        # Run under concurrent load for 0.5s
        time.sleep(0.5)
        stop_event.set()

        writer_thread.join(timeout=3.0)
        for rt in reader_threads:
            rt.join(timeout=3.0)

        backend.close()

        assert len(writer_errors) == 0, f"Writer encountered errors: {writer_errors}"
        assert len(reader_errors) == 0, f"Readers encountered errors: {reader_errors}"
        assert all(c > 10 for c in read_counts), f"Readers did not achieve expected throughput: {read_counts}"

    def test_concurrent_writers_serialize_via_busy_timeout(self, temp_db_path):
        """Verify two writer connections serialize cleanly without failing when lock is held briefly."""
        backend1 = SQLiteBackend(temp_db_path)
        backend1.initialize()

        writer2_started = threading.Event()
        writer2_done = threading.Event()
        writer2_errors = []

        def second_writer():
            try:
                b2 = SQLiteBackend(temp_db_path)
                b2.initialize()
                writer2_started.set()
                with b2.transaction():
                    b2.upsert_file(FileRecord("src/w2.py", "h2", 1000.0, 1))
                b2.close()
                writer2_done.set()
            except Exception as e:
                writer2_errors.append(str(e))

        t2 = threading.Thread(target=second_writer)

        with backend1.transaction():
            backend1.upsert_file(FileRecord("src/w1.py", "h1", 1000.0, 1))
            t2.start()
            writer2_started.wait(timeout=2.0)
            # Hold write lock for 200ms while Writer 2 is attempting to write
            time.sleep(0.2)

        # After backend1 commits, Writer 2 should acquire lock and complete
        t2.join(timeout=5.0)

        assert len(writer2_errors) == 0, f"Writer 2 failed during serialization: {writer2_errors}"
        assert writer2_done.is_set()

        # Both files should exist in database
        assert backend1.get_file("src/w1.py") is not None
        assert backend1.get_file("src/w2.py") is not None
        backend1.close()


# ==============================================================================
# 2. Atomic Transaction Rollback & Data Integrity Stress Tests
# ==============================================================================

@pytest.mark.storage
class TestAtomicTransactionRollback:
    """Empirical verification of atomic multi-table rollback with zero orphaned records."""

    def test_multi_table_write_failure_leaves_zero_orphaned_records(self, temp_db_path):
        """Verify midway exception rolls back all tables: files, chunks, fts, symbols, refs, annotations, errors, analysis."""
        backend = SQLiteBackend(temp_db_path)
        backend.initialize()

        # Verify initial state is empty
        assert backend.status()["files"] == 0

        with pytest.raises(RuntimeError) as exc_info, backend.transaction():
            # Step 1: Write file
            backend.upsert_file(FileRecord("src/atomic.py", "hash_atomic", 1000.0, 2))

            # Step 2: Write chunks (updates both chunks table and fts_index)
            backend.insert_chunks([
                ChunkRecord("src/atomic.py", "func", "alpha", 1, 10, "def alpha(): return 'secret_needle_alpha'"),
                ChunkRecord("src/atomic.py", "func", "beta", 11, 20, "def beta(): return 'secret_needle_beta'")
            ])

            # Step 3: Write symbols
            backend.insert_symbols([
                SymbolRecord("alpha", "func", "src/atomic.py", 1, signature="def alpha()"),
                SymbolRecord("beta", "func", "src/atomic.py", 11, signature="def beta()")
            ])

            # Step 4: Write symbol references
            backend.insert_symbol_refs([
                SymbolRefRecord("src/atomic.py", "alpha", 5, "external_api", "call")
            ])

            # Step 5: Write annotations
            backend.insert_annotations([
                AnnotationRecord("src/atomic.py", 2, "todo", "alpha", "TODO: implement")
            ])

            # Step 6: Write syntax error
            backend.upsert_syntax_error(
                SyntaxErrorRecord("src/atomic.py", 15, 4, "unexpected token", 1000.0)
            )

            # Step 7: Write analysis ref
            backend.store_analysis_ref(
                AnalysisRefRecord("ref_atomic_1", "src/atomic.py", "alpha", 1, 10, "func", "body text", 1000.0)
            )

            # Step 8: Catastrophic mid-transaction failure
            raise RuntimeError("Catastrophic failure before commit")

        assert "Catastrophic failure" in str(exc_info.value)

        # Verify high-level API sees ZERO records
        assert backend.get_file("src/atomic.py") is None
        assert len(backend.get_chunks_for_file("src/atomic.py")) == 0
        assert len(backend.search_chunks(["secret_needle_alpha"])) == 0
        assert len(backend.query_symbols("alpha")) == 0
        assert len(backend.query_symbol_callers("external_api")) == 0
        assert len(backend.query_annotations(filepath="src/atomic.py")) == 0
        assert len(backend.get_syntax_errors("src/atomic.py")) == 0
        assert backend.get_analysis_ref("ref_atomic_1") is None

        # Verify directly in SQLite tables (zero orphaned records in raw SQL)
        conn = sqlite3.connect(temp_db_path)
        cur = conn.cursor()
        tables = [
            "files", "chunks", "symbols", "symbol_refs", "annotations",
            "syntax_errors", "analysis_refs"
        ]
        for tbl in tables:
            cur.execute(f"SELECT COUNT(*) FROM {tbl}")
            count = cur.fetchone()[0]
            assert count == 0, f"Table '{tbl}' has {count} orphaned records after rollback!"

        # Explicitly check FTS5 virtual table
        cur.execute("SELECT COUNT(*) FROM fts_index WHERE content MATCH 'secret_needle_alpha'")
        fts_count = cur.fetchone()[0]
        assert fts_count == 0, f"fts_index has {fts_count} orphaned full-text entries after rollback!"

        conn.close()
        backend.close()

    def test_update_rollback_restores_original_data_intact(self, temp_db_path):
        """Verify rollback of an update preserves the original pre-transaction rows completely."""
        backend = SQLiteBackend(temp_db_path)
        backend.initialize()

        # 1. Baseline insert
        orig_file = FileRecord("src/orig.py", "orig_hash", 1000.0, 1, project="core")
        backend.upsert_file(orig_file)
        backend.insert_chunks([
            ChunkRecord("src/orig.py", "func", "orig_func", 1, 10, "def orig_func(): return 'original'", project="core")
        ])
        backend.insert_symbols([
            SymbolRecord("orig_func", "func", "src/orig.py", 1, signature="def orig_func()", project="core")
        ])
        backend.set_state("app_config", {"setting": "orig_val"})

        # 2. Mutating transaction that aborts
        try:
            with backend.transaction():
                backend.upsert_file(FileRecord("src/orig.py", "modified_hash", 2000.0, 5, project="core"))
                backend.set_state("app_config", {"setting": "mutated_val"})
                # Delete old file and chunks
                backend.delete_file("src/orig.py")
                raise ValueError("Abort update transaction")
        except ValueError:
            pass

        # 3. Verify original records are 100% intact
        f = backend.get_file("src/orig.py")
        assert f is not None
        assert f.sha256 == "orig_hash"
        assert f.chunk_count == 1
        assert f.last_modified == 1000.0

        chunks = backend.get_chunks_for_file("src/orig.py")
        assert len(chunks) == 1
        assert chunks[0].name == "orig_func"
        assert "return 'original'" in chunks[0].content

        syms = backend.query_symbols("orig_func")
        assert len(syms) == 1
        assert syms[0].signature == "def orig_func()"

        cfg = backend.get_state("app_config")
        assert cfg == {"setting": "orig_val"}

        backend.close()

    def test_context_and_skill_atomic_rollback(self, temp_db_path):
        """Verify that failed transaction rolls back both context/skill relational table and FTS virtual table."""
        backend = SQLiteBackend(temp_db_path)
        backend.initialize()

        ctx = ContextRecord(
            session_id="tx_sess", project="global", title="Atomic Title",
            summary="Atomic Summary", active_files=["a.py"], open_tasks=["t1"],
            timestamp=1000.0, full_notes="Special atomic notes keyword"
        )
        sk = SkillRecord(
            name="atomic_skill", description="Atomic desc", filepath="skills/atomic.md",
            triggers="atom", sha256="sk_hash", last_modified=1000.0,
            content="Atomic skill instruction content"
        )

        try:
            with backend.transaction():
                backend.save_context(ctx)
                backend.upsert_skill(sk)
                raise RuntimeError("Abort skill and context save")
        except RuntimeError:
            pass

        # Verify relational tables are empty
        assert backend.get_context("tx_sess") is None
        assert len(backend.get_skills()) == 0

        # Verify FTS tables are empty
        assert len(backend.search_contexts(["Atomic"])) == 0
        assert len(backend.search_skills(["Atomic"])) == 0

        backend.close()

    def test_foreign_key_violation_triggers_rollback(self, temp_db_path):
        """Verify foreign key constraint error rolls back all preceding writes in transaction."""
        backend = SQLiteBackend(temp_db_path)
        backend.initialize()

        with pytest.raises(sqlite3.IntegrityError), backend.transaction():
            # Write a valid file
            backend.upsert_file(FileRecord("src/valid.py", "vhash", 1000.0, 1))

            # Direct FK violation: insert chunk referencing non-existent file
            backend.conn.execute(
                """
                INSERT INTO chunks (filepath, chunk_type, name, start_line, end_line, zcontent, project)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                ("src/nonexistent_parent.py", "func", "orphan", 1, 5, b"dummy", "global")
            )

        # Preceding write must be rolled back
        assert backend.get_file("src/valid.py") is None
        backend.close()


# ==============================================================================
# 3. Nested Transactions & Multi-level Savepoints
# ==============================================================================

@pytest.mark.storage
class TestNestedTransactions:
    """Empirical verification of multi-level savepoint rollbacks and commits."""

    def test_three_level_nested_innermost_rollback(self, temp_db_path):
        """Level 3 rolls back; Level 2 catches it; Levels 1 and 2 commit successfully."""
        backend = SQLiteBackend(temp_db_path)
        backend.initialize()

        with backend.transaction():  # Level 1
            backend.set_state("l1", "committed_l1")

            with backend.transaction():  # Level 2
                backend.set_state("l2", "committed_l2")

                try:
                    with backend.transaction():  # Level 3
                        backend.set_state("l3", "aborted_l3")
                        raise RuntimeError("Fail Level 3")
                except RuntimeError:
                    pass  # Handled at Level 2

                backend.set_state("l2_after", "committed_l2_after")

        # After outer exit, L1 and L2 writes are committed; L3 write is rolled back
        assert backend.get_state("l1") == "committed_l1"
        assert backend.get_state("l2") == "committed_l2"
        assert backend.get_state("l2_after") == "committed_l2_after"
        assert backend.get_state("l3") is None, "Innermost savepoint rollback leaked into outer transaction!"

        backend.close()

    def test_three_level_nested_middle_rollback(self, temp_db_path):
        """Level 2 rolls back (discarding L2 and L3); Level 1 catches and commits L1."""
        backend = SQLiteBackend(temp_db_path)
        backend.initialize()

        with backend.transaction():  # Level 1
            backend.set_state("l1", "committed_l1")

            try:
                with backend.transaction():  # Level 2
                    backend.set_state("l2", "aborted_l2")

                    with backend.transaction():  # Level 3
                        backend.set_state("l3", "aborted_l3")

                    raise RuntimeError("Fail Level 2 after Level 3 committed to Level 2 savepoint")
            except RuntimeError:
                pass  # Handled at Level 1

            backend.set_state("l1_after", "committed_l1_after")

        assert backend.get_state("l1") == "committed_l1"
        assert backend.get_state("l1_after") == "committed_l1_after"
        assert backend.get_state("l2") is None, "Level 2 rollback leaked into outer transaction!"
        assert backend.get_state("l3") is None, "Level 3 write persisted despite Level 2 rollback!"

        backend.close()

    def test_outer_rollback_discards_all_nested_levels(self, temp_db_path):
        """Outer Level 1 failure rolls back everything, even successfully exited inner levels."""
        backend = SQLiteBackend(temp_db_path)
        backend.initialize()

        try:
            with backend.transaction():  # Level 1
                backend.set_state("l1", "v1")

                with backend.transaction():  # Level 2
                    backend.set_state("l2", "v2")

                    with backend.transaction():  # Level 3
                        backend.set_state("l3", "v3")

                raise RuntimeError("Catastrophic outer failure")
        except RuntimeError:
            pass

        assert backend.get_state("l1") is None
        assert backend.get_state("l2") is None
        assert backend.get_state("l3") is None
        backend.close()

    def test_five_level_deep_nesting(self, temp_db_path):
        """Stress test 5-level deep savepoint nesting with rollback at level 4."""
        backend = SQLiteBackend(temp_db_path)
        backend.initialize()

        with backend.transaction():  # 1
            backend.set_state("s1", 1)
            with backend.transaction():  # 2
                backend.set_state("s2", 2)
                with backend.transaction():  # 3
                    backend.set_state("s3", 3)
                    try:
                        with backend.transaction():  # 4
                            backend.set_state("s4", 4)
                            with backend.transaction():  # 5
                                backend.set_state("s5", 5)
                            raise RuntimeError("Rollback 4 and 5")
                    except RuntimeError:
                        pass
                    backend.set_state("s3_after", 33)

        assert backend.get_state("s1") == 1
        assert backend.get_state("s2") == 2
        assert backend.get_state("s3") == 3
        assert backend.get_state("s3_after") == 33
        assert backend.get_state("s4") is None
        assert backend.get_state("s5") is None

        backend.close()

    def test_sibling_savepoints_within_single_outer_transaction(self, temp_db_path):
        """Multiple sequential inner savepoints inside one outer transaction: alternating pass and fail."""
        backend = SQLiteBackend(temp_db_path)
        backend.initialize()

        with backend.transaction():
            backend.set_state("root", "root_val")

            # Sibling 1: aborts
            try:
                with backend.transaction():
                    backend.set_state("sub1", "fail1")
                    raise ValueError("Fail sub1")
            except ValueError:
                pass

            # Sibling 2: succeeds
            with backend.transaction():
                backend.set_state("sub2", "pass2")

            # Sibling 3: aborts
            try:
                with backend.transaction():
                    backend.set_state("sub3", "fail3")
                    raise ValueError("Fail sub3")
            except ValueError:
                pass

            # Sibling 4: succeeds
            with backend.transaction():
                backend.set_state("sub4", "pass4")

        assert backend.get_state("root") == "root_val"
        assert backend.get_state("sub1") is None
        assert backend.get_state("sub2") == "pass2"
        assert backend.get_state("sub3") is None
        assert backend.get_state("sub4") == "pass4"

        backend.close()

    def test_transaction_depth_integrity_under_exceptions(self, temp_db_path):
        """Verify _tx_depth is strictly restored to 0 after arbitrary exception chains."""
        backend = SQLiteBackend(temp_db_path)
        backend.initialize()

        assert backend._tx_depth == 0

        # Sequence of failing transactions
        for i in range(5):
            try:
                with backend.transaction(), backend.transaction(), backend.transaction():
                    raise ValueError(f"Crash {i}")
            except ValueError:
                pass
            assert backend._tx_depth == 0, f"Depth leaked on iteration {i}: depth={backend._tx_depth}"

        # Normal transaction afterwards must still work
        with backend.transaction():
            backend.set_state("post_depth", "ok")
        assert backend.get_state("post_depth") == "ok"
        assert backend._tx_depth == 0

        backend.close()


# ==============================================================================
# 4. Cascading Deletions & Data Cleanliness
# ==============================================================================

@pytest.mark.storage
class TestCascadingDeletions:
    """Empirical verification that delete_file thoroughly cleans all dependent entities."""

    def test_delete_file_cleans_all_entities_and_fts_index(self, temp_db_path):
        """delete_file must remove file, chunks, symbols, refs, annotations, syntax errors, analysis refs, and FTS entries."""
        backend = SQLiteBackend(temp_db_path)
        backend.initialize()

        target_file = "src/cascade_target.py"
        other_file = "src/cascade_other.py"

        # 1. Populate target file with comprehensive dependencies
        backend.upsert_file(FileRecord(target_file, "target_hash", 1000.0, 2))
        backend.insert_chunks([
            ChunkRecord(target_file, "func", "target_func1", 1, 10, "def target_func1(): return 'cascade_needle_target'"),
            ChunkRecord(target_file, "func", "target_func2", 11, 20, "def target_func2(): return 'other_target_needle'")
        ])
        backend.insert_symbols([
            SymbolRecord("target_func1", "func", target_file, 1),
            SymbolRecord("target_func2", "func", target_file, 11)
        ])
        backend.insert_symbol_refs([
            SymbolRefRecord(target_file, "target_func1", 5, "dep_func", "call")
        ])
        backend.insert_annotations([
            AnnotationRecord(target_file, 2, "note", "target_func1", "Note on target")
        ])
        backend.upsert_syntax_error(
            SyntaxErrorRecord(target_file, 15, 2, "syntax issue", 1000.0)
        )
        backend.store_analysis_ref(
            AnalysisRefRecord("ref_tgt_1", target_file, "target_func1", 1, 10, "func", "body target", 1000.0)
        )

        # 2. Populate other file that MUST NOT be touched
        backend.upsert_file(FileRecord(other_file, "other_hash", 1000.0, 1))
        backend.insert_chunks([
            ChunkRecord(other_file, "func", "other_func", 1, 10, "def other_func(): return 'other_file_needle'")
        ])
        backend.insert_symbols([
            SymbolRecord("other_func", "func", other_file, 1)
        ])
        backend.insert_symbol_refs([
            SymbolRefRecord(other_file, "other_func", 5, "dep_func", "call")
        ])
        backend.insert_annotations([
            AnnotationRecord(other_file, 2, "note", "other_func", "Note on other")
        ])
        backend.upsert_syntax_error(
            SyntaxErrorRecord(other_file, 8, 1, "other syntax issue", 1000.0)
        )
        backend.store_analysis_ref(
            AnalysisRefRecord("ref_oth_1", other_file, "other_func", 1, 10, "func", "body other", 1000.0)
        )

        # Verify both exist before deletion
        assert backend.get_file(target_file) is not None
        assert backend.get_file(other_file) is not None
        assert len(backend.search_chunks(["cascade_needle_target"])) == 1
        assert len(backend.search_chunks(["other_file_needle"])) == 1

        # 3. Perform cascading delete on target file
        backend.delete_file(target_file)

        # 4. Verify target file and all dependent entities are wiped
        assert backend.get_file(target_file) is None
        assert len(backend.get_chunks_for_file(target_file)) == 0
        assert len(backend.query_symbols("target_func1")) == 0
        assert len(backend.query_symbols("target_func2")) == 0
        assert len(backend.query_annotations(filepath=target_file)) == 0
        assert len(backend.get_syntax_errors(target_file)) == 0
        assert backend.get_analysis_ref("ref_tgt_1") is None

        # Verify FTS5 search finds nothing for target needles
        assert len(backend.search_chunks(["cascade_needle_target"])) == 0
        assert len(backend.search_chunks(["other_target_needle"])) == 0

        # Verify caller refs from target file are deleted
        conn = sqlite3.connect(temp_db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM symbol_refs WHERE caller_filepath = ?", (target_file,))
        assert cur.fetchone()[0] == 0, "Symbol refs from target file were not deleted"

        # Verify FTS index table directly has 0 rows for target file
        cur.execute("SELECT COUNT(*) FROM fts_index WHERE filepath = ?", (target_file,))
        assert cur.fetchone()[0] == 0, "FTS5 index retains orphaned rows for deleted file!"

        # 5. Verify other file is completely unaffected
        assert backend.get_file(other_file) is not None
        assert len(backend.get_chunks_for_file(other_file)) == 1
        assert len(backend.query_symbols("other_func")) == 1
        assert len(backend.query_annotations(filepath=other_file)) == 1
        assert len(backend.get_syntax_errors(other_file)) == 1
        assert backend.get_analysis_ref("ref_oth_1") is not None
        assert len(backend.search_chunks(["other_file_needle"])) == 1

        conn.close()
        backend.close()

    def test_delete_nonexistent_file_is_clean_noop(self, temp_db_path):
        """Calling delete_file on a non-existent file path executes cleanly without error."""
        backend = SQLiteBackend(temp_db_path)
        backend.initialize()
        backend.delete_file("nonexistent/file/path.py")
        backend.close()

    def test_cascading_delete_rollback_restores_all_records(self, temp_db_path):
        """If delete_file is executed inside a transaction that fails, all deleted entities are restored."""
        backend = SQLiteBackend(temp_db_path)
        backend.initialize()

        path = "src/restore_me.py"
        backend.upsert_file(FileRecord(path, "rh", 1000.0, 1))
        backend.insert_chunks([ChunkRecord(path, "func", "f", 1, 5, "def f(): return 'restore_token'")])
        backend.insert_symbols([SymbolRecord("f", "func", path, 1)])

        try:
            with backend.transaction():
                backend.delete_file(path)
                assert backend.get_file(path) is None
                raise RuntimeError("Abort deletion")
        except RuntimeError:
            pass

        # Deletion must be completely undone
        assert backend.get_file(path) is not None
        assert len(backend.get_chunks_for_file(path)) == 1
        assert len(backend.query_symbols("f")) == 1
        assert len(backend.search_chunks(["restore_token"])) == 1

        backend.close()
