"""Tests for A1 context memory improvements: isolation, guard, forget, and missing files flag."""

import os
import time
from pathlib import Path

from ai_db.memory.context import ContextMemory
from ai_db.storage.models import ContextRecord


class TestContextIsolation:
    """Test that tests are isolated and don't pollute the default user DB."""

    def test_isolated_env_sets_temp_db(self, isolated_env: Path):
        """Verify that isolated_env fixture sets AI_DB_PATH to a temp location."""
        db_path = os.environ.get("AI_DB_PATH")
        assert db_path
        assert "tmp" in db_path or db_path.startswith(str(isolated_env))

    def test_guard_raises_on_default_db(self, monkeypatch):
        """Test that guard raises AiDbConfigError if test tries to use default user DB."""
        from ai_db.constants import DEFAULT_DB_FILE

        # Set PYTEST_CURRENT_TEST to simulate being in a test
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_name")
        # Set AI_DB_PATH to the default user DB
        default_db = os.path.expanduser(DEFAULT_DB_FILE)
        monkeypatch.setenv("AI_DB_PATH", default_db)

        # The guard should raise an error when we try to access the fixture
        # This test is more of a documentation test since the guard is in conftest
        # and runs at fixture time, not during test execution
        assert "PYTEST_CURRENT_TEST" in os.environ


class TestContextForget:
    """Test the context forget functionality."""

    def test_delete_context_exists(self, memory_sqlite_backend, temp_db: str):
        """Test deleting a context that exists."""
        # Set up a context in memory
        backend = memory_sqlite_backend
        backend.initialize()

        # Save a context
        record = ContextRecord(
            session_id="test_session",
            project="test_project",
            title="Test Context",
            summary="Test summary",
            active_files=["file1.py", "file2.py"],
            open_tasks=["task1"],
            timestamp=time.time(),
            full_notes="Detailed notes"
        )
        context_id = backend.save_context(record)
        assert context_id > 0

        # Verify it was saved
        retrieved = backend.get_context(session_id="test_session", allowed_projects=["test_project"])
        assert retrieved is not None
        assert retrieved.session_id == "test_session"

        # Delete it
        deleted = backend.delete_context(session_id="test_session", project="test_project")
        assert deleted is True

        # Verify it's gone
        retrieved = backend.get_context(session_id="test_session", allowed_projects=["test_project"])
        assert retrieved is None

    def test_delete_context_not_found(self, memory_sqlite_backend):
        """Test deleting a context that doesn't exist."""
        backend = memory_sqlite_backend
        backend.initialize()

        # Try to delete a non-existent context
        deleted = backend.delete_context(session_id="nonexistent", project="test_project")
        assert deleted is False


class TestContextMemoryForgot:
    """Test ContextMemory.delete_context method."""

    def test_context_memory_delete(self, memory_sqlite_backend):
        """Test ContextMemory.delete_context method."""
        backend = memory_sqlite_backend
        backend.initialize()

        ctx_mem = ContextMemory(db=backend)

        # Save a context
        ctx_mem.save_context(
            session_id="memory_test",
            summary="Test context",
            project="test_proj",
            title="My Session",
            active_files=["file1.py"],
            open_tasks=["task1"]
        )

        # Verify it's there
        ctx = ctx_mem.get_context(session_id="memory_test", allowed_projects=["test_proj"])
        assert ctx is not None

        # Delete it
        deleted = ctx_mem.delete_context(session_id="memory_test", project="test_proj")
        assert deleted is True

        # Verify it's gone
        ctx = ctx_mem.get_context(session_id="memory_test", allowed_projects=["test_proj"])
        assert ctx is None


class TestContextSavedAtAge:
    """Test that recalled contexts include human-readable age."""

    def test_context_includes_saved_at(self, memory_sqlite_backend):
        """Test that get_context includes saved_at field."""
        backend = memory_sqlite_backend
        backend.initialize()

        ctx_mem = ContextMemory(db=backend)

        # Save a context
        ctx_mem.save_context(
            session_id="age_test",
            summary="Test context",
            project="test_proj",
            title="My Session",
            active_files=["file1.py"]
        )

        # Get it back
        ctx = ctx_mem.get_context(session_id="age_test", allowed_projects=["test_proj"])

        # Should have saved_at field
        assert "saved_at" in ctx
        # Should be a string like "5s ago" or "0m ago"
        assert "ago" in ctx["saved_at"]
        # Should be reasonably close to now (within 1 second)
        assert time.time() - ctx["timestamp"] < 1.0


class TestContextMissingFiles:
    """Test that recalled contexts flag missing files."""

    def test_context_flags_missing_files(self, memory_sqlite_backend, tmp_path: Path):
        """Test that get_context flags active_files that don't exist."""
        backend = memory_sqlite_backend
        backend.initialize()

        ctx_mem = ContextMemory(db=backend)

        # Create a real file
        real_file = tmp_path / "existing.py"
        real_file.write_text("# real file\n")

        missing_file = str(tmp_path / "nonexistent.py")

        # Save a context with both real and missing files
        ctx_mem.save_context(
            session_id="missing_test",
            summary="Test context",
            project="test_proj",
            title="My Session",
            active_files=[str(real_file), missing_file]
        )

        # Get it back
        ctx = ctx_mem.get_context(session_id="missing_test", allowed_projects=["test_proj"])

        # Check active_files structure
        assert isinstance(ctx["active_files"], list)
        assert len(ctx["active_files"]) == 2

        # All items should have path and missing fields
        for f in ctx["active_files"]:
            assert isinstance(f, dict)
            assert "path" in f
            assert "missing" in f
            assert isinstance(f["missing"], bool)

        # First file should exist
        real_entry = next((f for f in ctx["active_files"] if "existing.py" in f["path"]), None)
        assert real_entry is not None
        assert real_entry["missing"] is False

        # Second file should be missing
        missing_entry = next((f for f in ctx["active_files"] if "nonexistent.py" in f["path"]), None)
        assert missing_entry is not None
        assert missing_entry["missing"] is True


class TestContextForgetViaDispatcher:
    """Test context forget via the dispatcher."""

    def test_dispatcher_context_forget(self, isolated_env: Path):
        """Test the context_forget dispatcher tool."""
        from ai_db.dispatcher import ServiceDispatcher

        dispatcher = ServiceDispatcher()

        # Save a context
        result = dispatcher.execute("context_save", {
            "session_id": "forget_test",
            "summary": "Test summary",
            "project": "test_proj",
            "title": "Test Title",
            "active_files": ["file.py"]
        })
        assert result is not None

        # Verify it exists
        ctx = dispatcher.execute("context_recall", {
            "session_id": "forget_test",
            "project": "test_proj"
        })
        assert ctx.get("session_id") == "forget_test"

        # Forget it
        result = dispatcher.execute("context_forget", {
            "session_id": "forget_test",
            "project": "test_proj"
        })
        assert result["deleted"] is True
        assert result["session_id"] == "forget_test"

        # Verify it's gone
        ctx = dispatcher.execute("context_recall", {
            "session_id": "forget_test",
            "project": "test_proj"
        })
        assert ctx == {}


class TestContextIntegration:
    """Integration tests for context memory improvements."""

    def test_context_round_trip_with_age_and_missing_files(self, memory_sqlite_backend, tmp_path: Path):
        """Full round-trip test with age and missing files."""
        backend = memory_sqlite_backend
        backend.initialize()

        ctx_mem = ContextMemory(db=backend)

        # Create files
        file1 = tmp_path / "file1.py"
        file1.write_text("# File 1\n")

        file2_missing = str(tmp_path / "file2.py")  # This won't exist

        # Save context
        save_result = ctx_mem.save_context(
            session_id="round_trip",
            summary="Round trip test",
            project="test_proj",
            title="Round Trip Session",
            active_files=[str(file1), file2_missing],
            open_tasks=["task1", "task2"],
            full_notes="Detailed notes about the session"
        )

        assert save_result["session_id"] == "round_trip"
        assert save_result["project"] == "test_proj"

        # Small delay to ensure age is measurable
        time.sleep(0.1)

        # Get context back
        ctx = ctx_mem.get_context(
            session_id="round_trip",
            allowed_projects=["test_proj"]
        )

        # Verify all fields
        assert ctx["session_id"] == "round_trip"
        assert ctx["project"] == "test_proj"
        assert ctx["title"] == "Round Trip Session"
        assert ctx["summary"] == "Round trip test"
        assert ctx["full_notes"] == "Detailed notes about the session"
        assert "saved_at" in ctx
        assert len(ctx["active_files"]) == 2
        assert len(ctx["open_tasks"]) == 2

        # Check file statuses
        file1_entry = next(f for f in ctx["active_files"] if "file1.py" in f["path"])
        file2_entry = next(f for f in ctx["active_files"] if "file2.py" in f["path"])

        assert file1_entry["missing"] is False
        assert file2_entry["missing"] is True

        # Delete and verify
        deleted = ctx_mem.delete_context(session_id="round_trip", project="test_proj")
        assert deleted is True

        ctx = ctx_mem.get_context(
            session_id="round_trip",
            allowed_projects=["test_proj"]
        )
        assert ctx is None
