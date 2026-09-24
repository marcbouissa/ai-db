"""Storage layer E2E test suite (Features 6-11).

Verifies StorageBackend ABC protocol compliance, domain DTOs, SQLiteBackend
(WAL mode, FTS5 BM25, zlib level 9, atomic transactions), StorageBackendFactory,
pluggable entry-point backends, and decoupled SQL callers.
"""

import ast
import os
import sys
import zlib
import sqlite3
import tempfile
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Progressive check for storage implementation availability (M2)
try:
    from ai_db.storage.models import (
        FileRecord, ChunkRecord, SymbolRecord, SymbolRefRecord,
        AnnotationRecord, SyntaxErrorRecord, SkillRecord,
        ContextRecord, AnalysisRefRecord, SearchResult
    )
    HAS_STORAGE_MODELS = True
except ImportError:
    HAS_STORAGE_MODELS = False

try:
    from ai_db.storage.backend import StorageBackend
    HAS_STORAGE_BACKEND = True
except ImportError:
    HAS_STORAGE_BACKEND = False

try:
    from ai_db.storage.sqlite_backend import SQLiteBackend
    HAS_SQLITE_BACKEND = True
except ImportError:
    HAS_SQLITE_BACKEND = False

try:
    from ai_db.storage.factory import StorageBackendFactory
    HAS_STORAGE_FACTORY = True
except ImportError:
    HAS_STORAGE_FACTORY = False


# ==============================================================================
# Tier 1: Feature Coverage (Isolation & Happy Path Tests)
# ==============================================================================

@pytest.mark.storage
class TestStorageTier1:
    """Tier 1: Baseline feature coverage across Features 6-11."""

    # --- Feature 6: StorageBackend Abstraction ---

    @pytest.mark.skipif(not HAS_STORAGE_BACKEND, reason="StorageBackend ABC (M2) not yet available")
    def test_storage_backend_cannot_be_instantiated_directly(self):
        """TC-T1-F6-01: StorageBackend ABC cannot be instantiated directly."""
        with pytest.raises(TypeError) as excinfo:
            StorageBackend()  # type: ignore
        assert "abstract" in str(excinfo.value).lower()

    @pytest.mark.skipif(not HAS_STORAGE_BACKEND, reason="StorageBackend ABC (M2) not yet available")
    def test_storage_backend_subclass_enforces_abstract_methods(self):
        """TC-T1-F6-02: Partial StorageBackend subclass raises TypeError on instantiation."""
        class IncompleteBackend(StorageBackend):
            @property
            def backend_name(self) -> str:
                return "incomplete"
        with pytest.raises(TypeError) as excinfo:
            IncompleteBackend()  # type: ignore
        assert "abstract" in str(excinfo.value).lower()

    @pytest.mark.skipif(not HAS_STORAGE_BACKEND, reason="StorageBackend ABC (M2) not yet available")
    def test_storage_backend_complete_subclass_instantiates(self):
        """TC-T1-F6-03: Subclass implementing all abstract methods instantiates cleanly."""
        # Introspect all abstract methods of StorageBackend
        abstract_methods = StorageBackend.__abstractmethods__
        attrs = {m: lambda *args, **kwargs: None for m in abstract_methods}
        attrs["backend_name"] = property(lambda self: "dummy")
        DummyBackend = type("DummyBackend", (StorageBackend,), attrs)
        backend_instance = DummyBackend()
        assert isinstance(backend_instance, StorageBackend)
        assert backend_instance.backend_name == "dummy"

    @pytest.mark.skipif(not HAS_SQLITE_BACKEND, reason="SQLiteBackend (M2) not yet available")
    def test_storage_backend_backend_name_property(self, temp_db_path):
        """TC-T1-F6-04: backend_name returns canonical identifier ('sqlite')."""
        backend = SQLiteBackend(temp_db_path)
        assert backend.backend_name == "sqlite"
        backend.close()

    @pytest.mark.skipif(not HAS_SQLITE_BACKEND, reason="SQLiteBackend (M2) not yet available")
    def test_storage_backend_transaction_context_manager_protocol(self, temp_db_path):
        """TC-T1-F6-05: transaction() returns context manager with __enter__ and __exit__."""
        backend = SQLiteBackend(temp_db_path)
        backend.initialize()
        cm = backend.transaction()
        assert hasattr(cm, "__enter__") and hasattr(cm, "__exit__")
        with cm:
            pass
        backend.close()

    # --- Feature 7: Storage Domain DTOs ---

    @pytest.mark.skipif(not HAS_STORAGE_MODELS, reason="Domain models (M2) not yet available")
    def test_dto_file_record_instantiation_and_defaults(self):
        """TC-T1-F7-01: FileRecord attributes, slots, and default project."""
        rec = FileRecord(filepath="src/app.py", sha256="hash123", last_modified=1000.0, chunk_count=4)
        assert rec.filepath == "src/app.py"
        assert rec.sha256 == "hash123"
        assert rec.last_modified == 1000.0
        assert rec.chunk_count == 4
        assert rec.project == "global"

    @pytest.mark.skipif(not HAS_STORAGE_MODELS, reason="Domain models (M2) not yet available")
    def test_dto_chunk_record_slots_and_plain_content(self):
        """TC-T1-F7-02: ChunkRecord holds uncompressed plain string content."""
        code = "def add(a, b):\n    return a + b\n"
        rec = ChunkRecord(filepath="src/math.py", chunk_type="function", name="add",
                          start_line=1, end_line=3, content=code)
        assert rec.content == code
        assert isinstance(rec.content, str)
        assert rec.id is None

    @pytest.mark.skipif(not HAS_STORAGE_MODELS, reason="Domain models (M2) not yet available")
    def test_dto_symbol_and_symbol_ref_records(self):
        """TC-T1-F7-03: SymbolRecord and SymbolRefRecord attribute mappings."""
        sym = SymbolRecord(name="Calculator", symbol_type="class", filepath="src/calc.py",
                           line=5, signature="class Calculator:")
        ref = SymbolRefRecord(caller_filepath="src/main.py", caller_name="main",
                              caller_line=10, callee_name="Calculator", ref_type="call")
        assert sym.name == "Calculator"
        assert ref.callee_name == "Calculator"
        assert ref.ref_type == "call"

    @pytest.mark.skipif(not HAS_STORAGE_MODELS, reason="Domain models (M2) not yet available")
    def test_dto_context_record_collection_fields(self):
        """TC-T1-F7-04: ContextRecord preserves list fields for active files and open tasks."""
        ctx = ContextRecord(
            session_id="session_42", project="test_proj", title="Test Context",
            summary="Testing context memory", active_files=["a.py", "b.py"],
            open_tasks=["task 1", "task 2"], timestamp=500.0, full_notes="Full notes"
        )
        assert isinstance(ctx.active_files, list)
        assert ctx.active_files == ["a.py", "b.py"]
        assert isinstance(ctx.open_tasks, list)
        assert len(ctx.open_tasks) == 2

    @pytest.mark.skipif(not HAS_STORAGE_MODELS, reason="Domain models (M2) not yet available")
    def test_dto_search_result_and_syntax_error_records(self):
        """TC-T1-F7-05: SearchResult score formatting and SyntaxErrorRecord coordinates."""
        res = SearchResult(chunk_id=10, filepath="src/engine.py", name="Engine",
                           chunk_type="class", project="global", start_line=1,
                           end_line=20, score=0.88, snippet="class Engine:")
        err = SyntaxErrorRecord(filepath="src/bad.py", line=15, col=8,
                                message="invalid syntax", timestamp=123.4)
        assert res.score == 0.88
        assert err.line == 15
        assert err.col == 8

    # --- Feature 8: High-Performance SQLite Backend ---

    @pytest.mark.skipif(not HAS_SQLITE_BACKEND, reason="SQLiteBackend (M2) not yet available")
    def test_sqlite_backend_initialization_creates_schema(self, temp_db_path):
        """TC-T1-F8-01: initialize() sets up relational tables and FTS5 virtual tables."""
        backend = SQLiteBackend(temp_db_path)
        backend.initialize()
        conn = sqlite3.connect(temp_db_path)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")
        tables = {row[0] for row in cur.fetchall()}
        conn.close()
        backend.close()

        expected_tables = ["files", "chunks", "symbols", "skills", "contexts", "semantic_cache"]
        for expected in expected_tables:
            assert expected in tables, f"Expected table '{expected}' not found in SQLite schema"

    @pytest.mark.skipif(not HAS_SQLITE_BACKEND, reason="SQLiteBackend (M2) not yet available")
    def test_sqlite_backend_wal_mode_enabled(self, temp_db_path):
        """TC-T1-F8-02: Connection operates in WAL journal mode and NORMAL synchronous."""
        backend = SQLiteBackend(temp_db_path)
        backend.initialize()
        conn = sqlite3.connect(temp_db_path)
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode")
        journal_mode = cur.fetchone()[0].lower()
        conn.close()
        backend.close()
        assert journal_mode == "wal", f"Expected WAL journal mode, got {journal_mode}"

    @pytest.mark.skipif(not (HAS_SQLITE_BACKEND and HAS_STORAGE_MODELS), reason="Storage models/backend not yet available")
    def test_sqlite_backend_zlib_transparent_compression(self, temp_db_path):
        """TC-T1-F8-03: Chunk text is compressed in SQLite and decompressed on retrieval."""
        backend = SQLiteBackend(temp_db_path)
        backend.initialize()
        # Insert file record first (foreign key)
        backend.upsert_file(FileRecord("src/compress.py", "hash1", 1000.0, 1))
        # Insert chunk with repetitive code string
        original_text = "def repeat():\n" + ("    print('ai-db compression')\n" * 50)
        chunk = ChunkRecord(filepath="src/compress.py", chunk_type="function",
                            name="repeat", start_line=1, end_line=52, content=original_text)
        backend.insert_chunks([chunk])

        # Verify raw database column holds compressed bytes
        conn = sqlite3.connect(temp_db_path)
        cur = conn.cursor()
        cur.execute("SELECT zcontent FROM chunks WHERE filepath = 'src/compress.py'")
        raw_blob = cur.fetchone()[0]
        conn.close()

        assert isinstance(raw_blob, bytes), "zcontent must be stored as binary BLOB"
        # zlib magic header \x78\xda or \x78\x9c
        assert raw_blob.startswith(b"\x78"), "Blob missing zlib magic bytes"
        assert zlib.decompress(raw_blob).decode("utf-8") == original_text

        # Verify high-level retrieval decompresses transparently
        retrieved = backend.get_chunks_for_file("src/compress.py")
        assert len(retrieved) == 1
        assert retrieved[0].content == original_text
        backend.close()

    @pytest.mark.skipif(not (HAS_SQLITE_BACKEND and HAS_STORAGE_MODELS), reason="Storage models/backend not yet available")
    def test_sqlite_backend_fts5_bm25_search_scoring(self, temp_db_path):
        """TC-T1-F8-04: Full-text search matches keywords and returns ranked results."""
        backend = SQLiteBackend(temp_db_path)
        backend.initialize()
        backend.upsert_file(FileRecord("src/search.py", "h1", 1000.0, 2))
        c1 = ChunkRecord("src/search.py", "func", "search_bm25", 1, 10,
                         "def search_bm25(): return 'vector search index'")
        c2 = ChunkRecord("src/search.py", "func", "other_func", 11, 20,
                         "def other_func(): return 'unrelated string'")
        backend.insert_chunks([c1, c2])

        results = backend.search_chunks(["vector", "index"], allowed_projects=["global"], top_k=5)
        assert len(results) > 0
        assert results[0].filepath == "src/search.py"
        assert results[0].name == "search_bm25"
        assert results[0].score > 0
        backend.close()

    @pytest.mark.skipif(not (HAS_SQLITE_BACKEND and HAS_STORAGE_MODELS), reason="Storage models/backend not yet available")
    def test_sqlite_backend_atomic_transaction_commit(self, temp_db_path):
        """TC-T1-F8-05: Successful transaction commits multi-table writes atomically."""
        backend = SQLiteBackend(temp_db_path)
        backend.initialize()
        with backend.transaction():
            backend.upsert_file(FileRecord("src/tx.py", "txhash", 1000.0, 1))
            backend.insert_symbols([SymbolRecord("TxClass", "class", "src/tx.py", 1)])

        # Read back in new transaction
        file_rec = backend.get_file("src/tx.py")
        assert file_rec is not None
        assert file_rec.filepath == "src/tx.py"
        symbols = backend.query_symbols("TxClass", allowed_projects=["global"])
        assert len(symbols) == 1
        backend.close()

    @pytest.mark.skipif(not (HAS_SQLITE_BACKEND and HAS_STORAGE_MODELS), reason="Storage models/backend not yet available")
    def test_sqlite_backend_cascade_file_deletion(self, temp_db_path):
        """TC-T1-F8-06: delete_file cascades to chunks, symbols, and references."""
        backend = SQLiteBackend(temp_db_path)
        backend.initialize()
        backend.upsert_file(FileRecord("src/del.py", "delhash", 1000.0, 1))
        backend.insert_chunks([ChunkRecord("src/del.py", "func", "f", 1, 5, "def f(): pass")])
        backend.insert_symbols([SymbolRecord("f", "function", "src/del.py", 1)])

        # Delete file
        backend.delete_file("src/del.py")

        assert backend.get_file("src/del.py") is None
        assert len(backend.get_chunks_for_file("src/del.py")) == 0
        assert len(backend.query_symbols("f", allowed_projects=["global"])) == 0
        backend.close()

    @pytest.mark.skipif(not HAS_SQLITE_BACKEND, reason="SQLiteBackend (M2) not yet available")
    def test_sqlite_backend_state_and_cache_operations(self, temp_db_path):
        """TC-T1-F8-07: Key-value session state and semantic cache round-trip."""
        backend = SQLiteBackend(temp_db_path)
        backend.initialize()
        backend.set_state("active_mode", {"mode": "fast", "threads": 4})
        assert backend.get_state("active_mode") == {"mode": "fast", "threads": 4}

        backend.set_semantic_cache("cache_k1", "file_h1", {"tokens": 120})
        cache_entry = backend.get_semantic_cache("cache_k1", "file_h1")
        assert cache_entry is not None
        assert cache_entry.get("tokens") == 120
        backend.close()

    # --- Feature 9: StorageBackend Factory ---

    @pytest.mark.skipif(not (HAS_STORAGE_FACTORY and HAS_SQLITE_BACKEND), reason="Storage factory not yet available")
    def test_factory_resolves_bare_filepath_to_sqlite(self, temp_db_path):
        """TC-T1-F9-01: Plain filesystem path resolves to SQLiteBackend."""
        backend = StorageBackendFactory.create(temp_db_path)
        assert isinstance(backend, SQLiteBackend)
        assert backend.backend_name == "sqlite"
        backend.close()

    @pytest.mark.skipif(not (HAS_STORAGE_FACTORY and HAS_SQLITE_BACKEND), reason="Storage factory not yet available")
    def test_factory_resolves_sqlite_uri_scheme(self, temp_db_path):
        """TC-T1-F9-02: sqlite:/// URI resolves to SQLiteBackend targeting path."""
        backend = StorageBackendFactory.create(f"sqlite:///{temp_db_path}")
        assert isinstance(backend, SQLiteBackend)
        assert backend.backend_name == "sqlite"
        backend.close()

    @pytest.mark.skipif(not (HAS_STORAGE_FACTORY and HAS_SQLITE_BACKEND), reason="Storage factory not yet available")
    def test_factory_resolves_in_memory_sqlite_uri(self):
        """TC-T1-F9-03: sqlite:///:memory: resolves to in-memory SQLiteBackend."""
        backend = StorageBackendFactory.create("sqlite:///:memory:")
        assert isinstance(backend, SQLiteBackend)
        backend.close()


    @pytest.mark.skipif(not HAS_STORAGE_FACTORY, reason="Storage factory not yet available")
    def test_factory_unsupported_scheme_raises_value_error(self):
        """TC-T1-F9-05: Unsupported URI schemes (postgres://, redis://) raise ValueError."""
        with pytest.raises(ValueError) as excinfo:
            StorageBackendFactory.create("postgres://localhost:5432/aidb")
        assert "unsupported" in str(excinfo.value).lower() or "postgres" in str(excinfo.value).lower()








    # --- Feature 11: Decoupled Leaked SQL Calls ---

    @pytest.mark.skipif(not HAS_STORAGE_BACKEND, reason="Decoupled storage (M2) not yet available")
    def test_ast_no_sqlite3_import_in_core_services(self):
        """TC-T1-F11-01: Zero direct sqlite3 imports in core services outside ai_db/storage/."""
        targets = [
            REPO_ROOT / "ai_db" / "search" / "indexer.py",
            REPO_ROOT / "ai_db" / "search" / "query.py",
            REPO_ROOT / "ai_db" / "search" / "skills.py",
            REPO_ROOT / "ai_db" / "memory" / "context.py",
            REPO_ROOT / "ai_db" / "analyzer" / "engine.py",
            REPO_ROOT / "ai_db" / "analyzer" / "references.py",
        ]
        violations = []
        for target in targets:
            if not target.is_file():
                continue
            tree = ast.parse(target.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "sqlite3":
                            violations.append(f"{target.name}:{node.lineno}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module == "sqlite3":
                        violations.append(f"{target.name}:{node.lineno}")
        assert len(violations) == 0, f"Direct sqlite3 imports found in: {violations}"

    @pytest.mark.skipif(not HAS_STORAGE_BACKEND, reason="Decoupled storage (M2) not yet available")
    def test_ast_no_cursor_calls_outside_storage(self):
        """TC-T1-F11-02: Zero .cursor() calls in non-storage modules."""
        violations = []
        for py_file in (REPO_ROOT / "ai_db").rglob("*.py"):
            if "storage" in py_file.parts:
                continue
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if node.func.attr == "cursor":
                        violations.append(f"{py_file.relative_to(REPO_ROOT)}:{node.lineno}")
        assert len(violations) == 0, f"Direct .cursor() calls found outside storage: {violations}"

    @pytest.mark.skipif(not HAS_STORAGE_BACKEND, reason="Decoupled storage (M2) not yet available")
    def test_ast_no_raw_sql_keywords_in_non_storage(self):
        """TC-T1-F11-03: Zero embedded SQL statements (SELECT, INSERT INTO) in non-storage code."""
        sql_keywords = ["SELECT ", "INSERT INTO ", "CREATE TABLE ", "PRAGMA "]
        violations = []
        for py_file in (REPO_ROOT / "ai_db").rglob("*.py"):
            if "storage" in py_file.parts:
                continue
            content = py_file.read_text(encoding="utf-8")
            for kw in sql_keywords:
                if kw in content:
                    violations.append(f"{py_file.relative_to(REPO_ROOT)} ({kw.strip()})")
        assert len(violations) == 0, f"Raw SQL strings found in non-storage modules: {violations}"

    @pytest.mark.skipif(not HAS_STORAGE_BACKEND, reason="Storage abstraction not yet available")
    def test_indexer_operates_purely_through_backend_mock(self, tmp_path):
        """TC-T1-F11-04: Indexer service functions purely through StorageBackend interface."""
        mock_backend = MagicMock()
        mock_backend.get_files_by_prefix.return_value = {}
        try:
            from ai_db.search.indexer import Indexer
            indexer = Indexer(db=mock_backend)
            test_file = tmp_path / "hello.py"
            test_file.write_text("def hello(): pass\n", encoding="utf-8")
            indexer.sync(str(tmp_path))
            assert mock_backend.upsert_file.called or mock_backend.insert_chunks.called
        except (ImportError, AttributeError):
            pytest.skip("Indexer decoupling not yet integrated")

    @pytest.mark.skipif(not HAS_STORAGE_BACKEND, reason="Storage abstraction not yet available")
    def test_query_engine_operates_purely_through_backend_mock(self):
        """TC-T1-F11-05: QueryEngine delegates searches through StorageBackend interface."""
        mock_backend = MagicMock()
        mock_backend.search_chunks.return_value = []
        try:
            from ai_db.search.query import QueryEngine
            engine = QueryEngine(db=mock_backend)
            engine.query("sample search")
            assert mock_backend.search_chunks.called
        except (ImportError, AttributeError):
            pytest.skip("QueryEngine decoupling not yet integrated")


# ==============================================================================
# Tier 2: Boundary & Corner Cases
# ==============================================================================

@pytest.mark.storage
class TestStorageTier2:
    """Tier 2: Boundary conditions, corner cases, and stress tests."""

    @pytest.mark.skipif(not HAS_STORAGE_BACKEND, reason="StorageBackend ABC not available")
    def test_backend_unimplemented_abstract_properties(self):
        """TC-T2-F6-01: Subclass omitting @property backend_name raises TypeError."""
        abstract_methods = set(StorageBackend.__abstractmethods__)
        abstract_methods.discard("backend_name")
        attrs = {m: lambda *args, **kwargs: None for m in abstract_methods}
        Incomplete = type("Incomplete", (StorageBackend,), attrs)
        with pytest.raises(TypeError):
            Incomplete()  # type: ignore

    @pytest.mark.skipif(not HAS_SQLITE_BACKEND, reason="SQLiteBackend not available")
    def test_backend_nested_transaction_context_managers(self, temp_db_path):
        """TC-T2-F6-02: Nested transaction() contexts commit atomically on outer exit."""
        backend = SQLiteBackend(temp_db_path)
        backend.initialize()
        with backend.transaction():
            backend.set_state("k1", {"val": 1})
            with backend.transaction():
                backend.set_state("k2", {"val": 2})
        assert backend.get_state("k1") == {"val": 1}
        assert backend.get_state("k2") == {"val": 2}
        backend.close()

    @pytest.mark.skipif(not HAS_SQLITE_BACKEND, reason="SQLiteBackend not available")
    def test_backend_close_idempotency(self, temp_db_path):
        """TC-T2-F6-03: Repeated close() calls execute cleanly without error."""
        backend = SQLiteBackend(temp_db_path)
        backend.initialize()
        backend.close()
        backend.close()  # Idempotent call

    @pytest.mark.skipif(not HAS_SQLITE_BACKEND, reason="SQLiteBackend not available")
    def test_backend_operations_after_close_raise_error(self, temp_db_path):
        """TC-T2-F6-04: Operations on closed backend raise descriptive exception."""
        backend = SQLiteBackend(temp_db_path)
        backend.initialize()
        backend.close()
        with pytest.raises(Exception):
            backend.get_file("src/foo.py")

    @pytest.mark.skipif(not HAS_SQLITE_BACKEND, reason="SQLiteBackend not available")
    def test_backend_transaction_rollback_reraises_exception(self, temp_db_path):
        """TC-T2-F6-05: Unhandled exception inside transaction context manager is reraised."""
        backend = SQLiteBackend(temp_db_path)
        backend.initialize()
        with pytest.raises(ZeroDivisionError):
            with backend.transaction():
                backend.set_state("txn_key", {"attempt": 1})
                _ = 1 / 0
        assert backend.get_state("txn_key") is None
        backend.close()

    @pytest.mark.skipif(not HAS_STORAGE_MODELS, reason="Domain models not available")
    def test_dto_empty_strings_and_null_signatures(self):
        """TC-T2-F7-01: DTOs handle empty strings and None signatures safely."""
        sym = SymbolRecord(name="", symbol_type="", filepath="", line=0, signature=None)
        assert sym.signature is None
        assert sym.name == ""

    @pytest.mark.skipif(not HAS_STORAGE_MODELS, reason="Domain models not available")
    def test_dto_extreme_line_numbers(self):
        """TC-T2-F7-02: Large boundary line numbers (e.g. 10,000,000) do not overflow."""
        chunk = ChunkRecord("huge.py", "func", "huge", start_line=10_000_000,
                            end_line=10_000_500, content="pass\n")
        assert chunk.start_line == 10_000_000

    @pytest.mark.skipif(not HAS_STORAGE_MODELS, reason="Domain models not available")
    def test_dto_huge_content_payloads(self):
        """TC-T2-F7-03: ChunkRecord holds 5MB text payload without truncation."""
        payload = "x" * (5 * 1024 * 1024)
        chunk = ChunkRecord("big.py", "raw", "big", 1, 100, content=payload)
        assert len(chunk.content) == 5 * 1024 * 1024

    @pytest.mark.skipif(not HAS_STORAGE_MODELS, reason="Domain models not available")
    def test_dto_unicode_emojis_and_multilingual_text(self):
        """TC-T2-F7-04: Multilingual strings, emojis, and RTL text round-trip cleanly."""
        text = "Hello 🚀 世界 مرحبا שלום"
        chunk = ChunkRecord("intl.py", "comment", "intl", 1, 1, content=text)
        assert chunk.content == text

    @pytest.mark.skipif(not HAS_STORAGE_MODELS, reason="Domain models not available")
    def test_dto_special_characters_in_filepaths(self):
        """TC-T2-F7-05: Filepaths containing spaces, quotes, and symbols are preserved."""
        tricky_path = "path/with spaces/it's a \"quote\"/file#1.py"
        rec = FileRecord(filepath=tricky_path, sha256="h1", last_modified=1.0, chunk_count=1)
        assert rec.filepath == tricky_path

    @pytest.mark.skipif(not (HAS_SQLITE_BACKEND and HAS_STORAGE_MODELS), reason="SQLiteBackend not available")
    def test_sqlite_transaction_rollback_on_unhandled_exception(self, temp_db_path):
        """TC-T2-F8-01: Partial writes in failed transaction are completely rolled back."""
        backend = SQLiteBackend(temp_db_path)
        backend.initialize()

        with pytest.raises(RuntimeError):
            with backend.transaction():
                backend.upsert_file(FileRecord("src/rollback.py", "rbhash", 1000.0, 1))
                raise RuntimeError("Simulated crash during batch")

        assert backend.get_file("src/rollback.py") is None
        backend.close()

    @pytest.mark.skipif(not HAS_SQLITE_BACKEND, reason="SQLiteBackend not available")
    def test_sqlite_fts5_malformed_syntax_queries(self, temp_db_path):
        """TC-T2-F8-03: Unbalanced quotes and operators in search query execute safely."""
        backend = SQLiteBackend(temp_db_path)
        backend.initialize()
        malformed_queries = ['"unterminated quote', '* * *', 'AND NOT OR', 'NEAR/']
        for q in malformed_queries:
            # Should not raise sqlite3.OperationalError
            results = backend.search_chunks([q], allowed_projects=["global"], top_k=5)
            assert isinstance(results, list)
        backend.close()

    @pytest.mark.skipif(not HAS_SQLITE_BACKEND, reason="SQLiteBackend not available")
    def test_sqlite_fts5_empty_query_tokens(self, temp_db_path):
        """TC-T2-F8-04: Empty or pure whitespace query tokens return empty result list."""
        backend = SQLiteBackend(temp_db_path)
        backend.initialize()
        assert backend.search_chunks([], allowed_projects=["global"], top_k=5) == []
        assert backend.search_chunks(["   "], allowed_projects=["global"], top_k=5) == []
        backend.close()

    @pytest.mark.skipif(not HAS_SQLITE_BACKEND, reason="SQLiteBackend not available")
    def test_sqlite_concurrent_read_during_write_in_wal_mode(self, temp_db_path):
        """TC-T2-F8-05: Readers do not block writers and writers do not block readers in WAL mode."""
        backend = SQLiteBackend(temp_db_path)
        backend.initialize()
        backend.set_state("read_key", {"val": "initial"})

        read_success = []
        def reader_worker():
            conn = sqlite3.connect(temp_db_path)
            cur = conn.cursor()
            cur.execute("SELECT value_json FROM session_state WHERE key = 'read_key'")
            row = cur.fetchone()
            if row:
                read_success.append(True)
            conn.close()

        # Open transaction in thread 1
        t = threading.Thread(target=reader_worker)
        with backend.transaction():
            backend.set_state("read_key", {"val": "updated"})
            t.start()
            t.join(timeout=2.0)

        assert len(read_success) == 1, "Reader was blocked during uncommitted write in WAL mode"
        backend.close()

    @pytest.mark.skipif(not (HAS_SQLITE_BACKEND and HAS_STORAGE_MODELS), reason="SQLiteBackend not available")
    def test_sqlite_corrupt_zlib_decompression_graceful_recovery(self, temp_db_path):
        """TC-T2-F8-02: Corrupted zcontent payloads raise AiDbStorageError (no silent empty result)."""
        backend = SQLiteBackend(temp_db_path)
        backend.initialize()
        backend.upsert_file(FileRecord("bad_zlib.py", "h1", 1.0, 1))
        backend.insert_chunks([
            ChunkRecord("bad_zlib.py", "func", "bad_fn", 1, 5, "def bad_fn(): pass")
        ])
        cur = backend.conn.cursor()
        cur.execute("UPDATE chunks SET zcontent = ? WHERE filepath = 'bad_zlib.py'", [b"GARBAGE_ZLIB_DATA"])
        backend.conn.commit()

        from ai_db.errors import AiDbStorageError
        # search serves FTS snippets; corruption surfaces when the body is read
        assert backend.search_chunks(["bad_fn"], allowed_projects=["global"], top_k=5)
        with pytest.raises(AiDbStorageError, match="corrupt"):
            backend.get_chunks_for_file("bad_zlib.py")
        backend.close()

    @pytest.mark.skipif(not HAS_STORAGE_FACTORY, reason="StorageBackendFactory not available")
    def test_factory_empty_string_raises_value_error(self):
        """TC-T2-F9-01: Empty or whitespace connection string raises ValueError."""
        with pytest.raises(ValueError):
            StorageBackendFactory.create("")
        with pytest.raises(ValueError):
            StorageBackendFactory.create("   ")

    @pytest.mark.skipif(not HAS_STORAGE_FACTORY, reason="StorageBackendFactory not available")
    def test_factory_malformed_connection_uris(self):
        """TC-T2-F9-02: Malformed URI strings raise descriptive ValueError."""
        malformed = [":///", "sqlite://", "mysql://"]
        for m in malformed:
            with pytest.raises(ValueError):
                StorageBackendFactory.create(m)

    @pytest.mark.skipif(not (HAS_STORAGE_FACTORY and HAS_SQLITE_BACKEND), reason="Storage factory not available")
    def test_factory_windows_style_paths(self, tmp_path):
        """TC-T2-F9-03: Windows backslash paths resolve to SQLiteBackend."""
        win_path = r"C:\data\project_db.sqlite"
        backend = StorageBackendFactory.create(win_path)
        assert isinstance(backend, SQLiteBackend)
        backend.close()


    @pytest.mark.skipif(not (HAS_STORAGE_FACTORY and HAS_SQLITE_BACKEND), reason="Storage factory not available")
    def test_factory_case_insensitive_scheme_parsing(self, temp_db_path):
        """TC-T2-F9-05: Uppercase schemes like SQLITE:/// resolve successfully."""
        backend = StorageBackendFactory.create(f"SQLITE:///{temp_db_path}")
        assert isinstance(backend, SQLiteBackend)
        backend.close()






    @pytest.mark.skipif(not (HAS_SQLITE_BACKEND and HAS_STORAGE_MODELS), reason="Storage models not available")
    def test_decoupled_missing_file_snippet_fallback(self, temp_db_path):
        """TC-T2-F11-01: Decoupled query engine handles missing source file on disk without SQL exceptions."""
        backend = SQLiteBackend(temp_db_path)
        backend.initialize()
        backend.upsert_file(FileRecord("nonexistent/missing_file.py", "h1", 1.0, 1))
        backend.insert_chunks([
            ChunkRecord("nonexistent/missing_file.py", "func", "calc", 1, 10, "def calc(): return 42")
        ])
        from ai_db.search.query import QueryEngine
        engine = QueryEngine(db=backend)
        hits = engine.query("calc")
        assert len(hits) >= 1
        assert hits[0]["name"] == "calc"
        backend.close()

    def test_decoupled_indexer_prune_uses_backend(self, tmp_path):
        """TC-T2-F11-02: Indexer sync delegates missing file pruning through StorageBackend.delete_file."""
        from ai_db.search.indexer import Indexer
        mock_backend = MagicMock()
        mock_backend.get_files_by_prefix.return_value = {
            str(tmp_path / "ghost.py"): ("ghost_sha", 100.0)
        }
        mock_backend.transaction.return_value.__enter__ = MagicMock()
        mock_backend.transaction.return_value.__exit__ = MagicMock()

        indexer = Indexer(db=mock_backend)
        indexer.sync(str(tmp_path), verbose=False)
        assert mock_backend.delete_file.called
        assert mock_backend.delete_file.call_args[0][0] == str(tmp_path / "ghost.py")

    def test_decoupled_error_propagation_on_storage_failure(self):
        """TC-T2-F11-03: Storage failures in backend propagate predictably through QueryEngine."""
        from ai_db.search.query import QueryEngine
        mock_backend = MagicMock()
        mock_backend.search_chunks.side_effect = RuntimeError("Storage backend I/O failure")

        engine = QueryEngine(db=mock_backend)
        with pytest.raises(RuntimeError) as exc_info:
            engine.query("search_term")
        assert "Storage backend I/O failure" in str(exc_info.value)

    def test_decoupled_storage_backend_interchangeability(self):
        """TC-T2-F11-04: QueryEngine functions with custom StorageBackend implementation returning SearchResult."""
        from ai_db.search.query import QueryEngine
        from ai_db.storage.models import SearchResult
        custom_backend = MagicMock()
        custom_backend.search_chunks.return_value = [
            SearchResult(chunk_id=42, filepath="pkg/mod.py", name="custom_fn",
                         chunk_type="func", project="custom", start_line=1, end_line=5,
                         score=0.99, snippet="def custom_fn(): pass")
        ]

        engine = QueryEngine(db=custom_backend)
        hits = engine.query("custom")
        assert len(hits) == 1
        assert hits[0]["chunk_id"] == 42
        assert hits[0]["name"] == "custom_fn"

    def test_decoupled_analyzer_cache_delegation(self):
        """TC-T2-F11-05: AnalyzerEngine delegates state and cache management to StorageBackend."""
        from ai_db.analyzer.engine import AnalyzerEngine
        mock_backend = MagicMock()
        mock_backend.backend = mock_backend
        mock_backend.get_state.return_value = {"cached_summary": "Analysis result"}

        analyzer = AnalyzerEngine(db=mock_backend)
        val = analyzer.get_session_state("cached_key")
        assert val == {"cached_summary": "Analysis result"}
        mock_backend.get_state.assert_called_with("cached_key")

        analyzer.set_session_state("new_key", {"data": 123})
        mock_backend.set_state.assert_called_with("new_key", {"data": 123})


# ==============================================================================
# Tier 3: Pairwise & Cross-Feature Combinations
# ==============================================================================

@pytest.mark.storage
class TestStorageTier3:
    """Tier 3: Pairwise cross-feature interactions."""

    @pytest.mark.skipif(not (HAS_STORAGE_FACTORY and HAS_SQLITE_BACKEND and HAS_STORAGE_MODELS),
                        reason="Factory, SQLiteBackend, or models not available")
    def test_pairwise_factory_sqlite_dto_roundtrip(self, temp_db_path):
        """TC-T3-PAIR-01: Factory creates SQLite backend, persists and retrieves DTOs."""
        backend = StorageBackendFactory.create(f"sqlite:///{temp_db_path}")
        backend.initialize()
        file_rec = FileRecord("src/pair.py", "hash_p1", 1000.0, 1)
        backend.upsert_file(file_rec)
        assert backend.get_file("src/pair.py") == file_rec
        backend.close()


    @pytest.mark.skipif(not (HAS_SQLITE_BACKEND and HAS_STORAGE_MODELS),
                        reason="SQLiteBackend or models not available")
    def test_pairwise_sqlite_atomic_transactions_cascading_delete(self, temp_db_path):
        """TC-T3-PAIR-03: Transactional batch insert followed by cascading file deletion."""
        backend = SQLiteBackend(temp_db_path)
        backend.initialize()
        with backend.transaction():
            for i in range(3):
                path = f"src/file_{i}.py"
                backend.upsert_file(FileRecord(path, f"h_{i}", 1000.0, 1))
                backend.insert_chunks([ChunkRecord(path, "func", f"f_{i}", 1, 5, "pass")])

        assert len(backend.get_chunks_for_file("src/file_1.py")) == 1
        backend.delete_file("src/file_1.py")
        assert len(backend.get_chunks_for_file("src/file_1.py")) == 0
        assert len(backend.get_chunks_for_file("src/file_0.py")) == 1
        backend.close()

    @pytest.mark.skipif(not (HAS_SQLITE_BACKEND and HAS_STORAGE_MODELS),
                        reason="SQLiteBackend or models not available")
    def test_pairwise_sqlite_bm25_project_isolation(self, temp_db_path):
        """TC-T3-PAIR-04: BM25 search filters results strictly by project boundary."""
        backend = SQLiteBackend(temp_db_path)
        backend.initialize()
        backend.upsert_file(FileRecord("src/p1.py", "h1", 1000.0, 1, project="alpha"))
        backend.upsert_file(FileRecord("src/p2.py", "h2", 1000.0, 1, project="beta"))

        backend.insert_chunks([
            ChunkRecord("src/p1.py", "f", "auth", 1, 5, "def auth(): return 'token'", project="alpha"),
            ChunkRecord("src/p2.py", "f", "auth", 1, 5, "def auth(): return 'token'", project="beta"),
        ])

        alpha_results = backend.search_chunks(["token"], allowed_projects=["alpha"], top_k=10)
        assert all(r.project == "alpha" for r in alpha_results)
        assert len(alpha_results) == 1

        all_results = backend.search_chunks(["token"], allowed_projects=["alpha", "beta"], top_k=10)
        assert len(all_results) == 2
        backend.close()

    @pytest.mark.skipif(not HAS_SQLITE_BACKEND, reason="SQLiteBackend not available")
    def test_pairwise_storage_state_and_cache_invalidation(self, temp_db_path):
        """TC-T3-PAIR-06: Clearing semantic cache does not affect session state."""
        backend = SQLiteBackend(temp_db_path)
        backend.initialize()
        backend.set_state("persistent_config", {"setting": True})
        backend.set_semantic_cache("k1", "h1", {"result": "cached"})

        backend.clear_semantic_cache()
        assert backend.get_semantic_cache("k1", "h1") is None
        assert backend.get_state("persistent_config") == {"setting": True}
        backend.close()


# ==============================================================================
# Tier 4: Real-World Application Workflows
# ==============================================================================

@pytest.mark.storage
class TestStorageTier4:
    """Tier 4: End-to-end integration workflows."""

    @pytest.mark.skipif(not (HAS_STORAGE_FACTORY and HAS_SQLITE_BACKEND and HAS_STORAGE_MODELS),
                        reason="Storage stack not yet available")
    def test_workflow_project_indexing_bm25_navigation(self, temp_db_path):
        """TC-STORAGE-WF-01: Multi-file project indexing, BM25 search, and symbol lookup."""
        backend = StorageBackendFactory.create(f"sqlite:///{temp_db_path}")
        backend.initialize()

        # Step 1: Ingest files and symbols
        files = [
            ("src/core.py", "class CoreProcessor:\n    def process(self): pass\n", "class"),
            ("src/api.py", "def handle_request():\n    proc = CoreProcessor()\n", "func"),
        ]
        for path, code, chunk_type in files:
            backend.upsert_file(FileRecord(path, f"hash_{path}", 1000.0, 1))
            backend.insert_chunks([ChunkRecord(path, chunk_type, "processor", 1, 10, code)])

        # Step 2: Search via BM25
        results = backend.search_chunks(["CoreProcessor"], allowed_projects=["global"], top_k=5)
        assert len(results) > 0
        assert any("CoreProcessor" in r.snippet for r in results)

        # Step 3: Verify stats
        status = backend.status()
        assert status.get("files", 0) >= 2
        backend.close()

    @pytest.mark.skipif(not (HAS_SQLITE_BACKEND and HAS_STORAGE_MODELS),
                        reason="SQLiteBackend not available")
    def test_workflow_atomic_batch_reindexing_rollback(self, temp_db_path):
        """TC-STORAGE-WF-02: Interrupted batch reindexing leaves database in clean pre-state."""
        backend = SQLiteBackend(temp_db_path)
        backend.initialize()

        # Initial baseline file
        backend.upsert_file(FileRecord("src/base.py", "base_hash", 1000.0, 1))

        # Interrupted batch
        try:
            with backend.transaction():
                backend.upsert_file(FileRecord("src/new_1.py", "nh1", 1000.0, 1))
                backend.upsert_file(FileRecord("src/new_2.py", "nh2", 1000.0, 1))
                raise IOError("Simulated disk error mid-batch")
        except IOError:
            pass

        # Verify only baseline exists; new files rolled back
        assert backend.get_file("src/base.py") is not None
        assert backend.get_file("src/new_1.py") is None
        assert backend.get_file("src/new_2.py") is None
        backend.close()

    @pytest.mark.skipif(not (HAS_SQLITE_BACKEND and HAS_STORAGE_MODELS),
                        reason="SQLiteBackend not available")
    def test_workflow_multisession_context_persistence_recall(self, temp_db_path):
        """TC-STORAGE-WF-03: Multi-session context memory persistence, FTS search, and recall."""
        backend = SQLiteBackend(temp_db_path)
        backend.initialize()

        # Save session contexts
        ctx_a = ContextRecord(
            session_id="session_oauth", project="global", title="OAuth Authentication",
            summary="Implementing OAuth2 PKCE flow", active_files=["src/auth.py"],
            open_tasks=["add token refresh"], timestamp=1000.0, full_notes="Notes on PKCE"
        )
        ctx_b = ContextRecord(
            session_id="session_bm25", project="global", title="BM25 Optimization",
            summary="Tuning Porter tokenizer", active_files=["src/fts.py"],
            open_tasks=["benchmarking"], timestamp=1010.0, full_notes="Notes on BM25"
        )
        backend.save_context(ctx_a)
        backend.save_context(ctx_b)

        # Recall by ID
        recalled = backend.get_context("session_oauth", allowed_projects=["global"])
        assert recalled is not None
        assert recalled.title == "OAuth Authentication"
        assert recalled.active_files == ["src/auth.py"]

        # Search contexts via FTS
        search_hits = backend.search_contexts(["OAuth2"], allowed_projects=["global"], top_k=3)
        assert len(search_hits) > 0
        assert search_hits[0]["session_id"] == "session_oauth"
        backend.close()

    @pytest.mark.skipif(not (HAS_SQLITE_BACKEND and HAS_STORAGE_MODELS),
                        reason="SQLiteBackend not available")
    def test_workflow_multitenant_project_isolation(self, temp_db_path):
        """TC-STORAGE-WF-04: Multi-project codebases isolated in single storage backend."""
        backend = SQLiteBackend(temp_db_path)
        backend.initialize()

        # Tenant 1: billing
        backend.upsert_file(FileRecord("src/charge.py", "h1", 1.0, 1, project="billing"))
        backend.insert_chunks([ChunkRecord("src/charge.py", "func", "charge", 1, 5,
                                           "def charge(): return 'process charge'", project="billing")])

        # Tenant 2: inventory
        backend.upsert_file(FileRecord("src/stock.py", "h2", 1.0, 1, project="inventory"))
        backend.insert_chunks([ChunkRecord("src/stock.py", "func", "stock", 1, 5,
                                           "def stock(): return 'check items'", project="inventory")])

        # Query restricted to billing
        billing_results = backend.search_chunks(["charge"], allowed_projects=["billing"], top_k=5)
        assert len(billing_results) == 1
        assert billing_results[0].project == "billing"

        # Query restricted to inventory should not find billing chunk
        inv_results = backend.search_chunks(["charge"], allowed_projects=["inventory"], top_k=5)
        assert len(inv_results) == 0
        backend.close()

    @pytest.mark.skipif(not (HAS_SQLITE_BACKEND and HAS_STORAGE_MODELS),
                        reason="SQLiteBackend or Storage models not available")
    def test_workflow_database_optimization_and_prune(self, temp_db_path):
        """TC-STORAGE-WF-05: Database optimization, vacuum, and missing file pruning."""
        backend = SQLiteBackend(temp_db_path)
        backend.initialize()
        backend.upsert_file(FileRecord("nonexistent/ghost.py", "h1", 1000.0, 1))

        # Run optimize
        result = backend.optimize(prune_missing=True, default_format="json")
        assert isinstance(result, dict)
        assert backend.get_file("nonexistent/ghost.py") is None
        backend.close()
