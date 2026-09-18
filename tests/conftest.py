"""Hermetic test fixtures and execution harness for ai-db test suite."""

import io
import os
import sys
import shutil
import sqlite3
import tempfile
import subprocess
from pathlib import Path
from typing import Generator, Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Session-level protection: Guarantee host database is never accessed during test imports
_SESSION_GUARD_DIR = tempfile.mkdtemp(prefix="ai_db_pytest_guard_")
_SESSION_GUARD_DB = os.path.join(_SESSION_GUARD_DIR, "session_guard.db")
os.environ.setdefault("AI_DB_PATH", _SESSION_GUARD_DB)
os.environ.setdefault("AI_DB_CONFIG", os.path.join(_SESSION_GUARD_DIR, "config.json"))
os.environ.setdefault("AI_DB_SKILL_DIRS", os.path.join(_SESSION_GUARD_DIR, "skills"))


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Virtualize all ai-db environment variables to an isolated temporary sandbox.
    
    Guarantees that test operations never access ~/.gemini, host databases,
    or user configuration files.
    """
    data_dir = tmp_path / "data"
    config_dir = tmp_path / "config"
    skills_dir = tmp_path / "skills"
    data_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    skills_dir.mkdir(parents=True, exist_ok=True)

    db_file = str(data_dir / "test_knowledge.db")
    config_file = str(config_dir / "config.json")

    monkeypatch.setenv("AI_DB_PATH", db_file)
    monkeypatch.setenv("AI_DB_CONFIG", config_file)
    monkeypatch.setenv("AI_DB_SKILL_DIRS", str(skills_dir))
    monkeypatch.setenv("AI_DB_CONNECTION_STRING", f"sqlite:///{db_file}")
    monkeypatch.setenv("XDG_DATA_HOME", str(data_dir))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_dir))

    # Patch already loaded modules
    for mod_name in ("ai_db", "ai_db.cli", "ai_db.constants"):
        if mod_name in sys.modules:
            mod = sys.modules[mod_name]
            if hasattr(mod, "DEFAULT_DB_FILE"):
                monkeypatch.setattr(mod, "DEFAULT_DB_FILE", db_file, raising=False)
            if hasattr(mod, "DEFAULT_CONFIG_FILE"):
                monkeypatch.setattr(mod, "DEFAULT_CONFIG_FILE", config_file, raising=False)
            if hasattr(mod, "DEFAULT_SKILL_DIRS"):
                monkeypatch.setattr(mod, "DEFAULT_SKILL_DIRS", [str(skills_dir)], raising=False)

    # Patch VectorDB.__init__ default parameter
    if "ai_db" in sys.modules and hasattr(sys.modules["ai_db"], "VectorDB"):
        vdb_cls = sys.modules["ai_db"].VectorDB
        monkeypatch.setattr(vdb_cls.__init__, "__defaults__", (db_file,), raising=False)

    return tmp_path


@pytest.fixture
def temp_db(isolated_env: Path) -> Generator[str, None, None]:
    """Provide a path to an isolated SQLite database file within isolated_env."""
    db_path = os.environ["AI_DB_PATH"]
    yield db_path
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except OSError:
            pass


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Generator[str, None, None]:
    """Provide a distinct clean SQLite database path within tmp_path."""
    db_file = str(tmp_path / "custom_temp.db")
    yield db_file
    for ext in ("", "-wal", "-shm"):
        target = db_file + ext
        if os.path.exists(target):
            try:
                os.remove(target)
            except OSError:
                pass


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Path:
    """Create a temporary multi-language workspace for indexing and parser testing."""
    ws = tmp_path / "workspace"
    ws.mkdir(parents=True, exist_ok=True)

    src_dir = ws / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    (src_dir / "service.py").write_text(
        '"""Core service module for testing."""\n\n'
        'class PaymentProcessor:\n'
        '    """Processes credit card and crypto transactions."""\n\n'
        '    def __init__(self, api_key: str = "mock-key") -> None:\n'
        '        self.api_key = api_key\n\n'
        '    def process(self, amount: float) -> bool:\n'
        '        """Process a payment amount."""\n'
        '        if amount <= 0:\n'
        '            raise ValueError("Amount must be positive")\n'
        '        return True\n\n'
        'def calculate_tax(subtotal: float, rate: float = 0.08) -> float:\n'
        '    """Calculate tax for transaction."""\n'
        '    return subtotal * rate\n',
        encoding="utf-8"
    )

    (src_dir / "utils.py").write_text(
        '"""Utility functions for tests."""\n\n'
        'def format_currency(amount: float) -> str:\n'
        '    return f"${amount:.2f}"\n\n'
        'def validate_account(account_id: str) -> bool:\n'
        '    return bool(account_id and account_id.isalnum())\n',
        encoding="utf-8"
    )

    (src_dir / "broken_syntax.py").write_text(
        'def invalid_syntax_function(\n'
        '    print("missing closing paren"\n',
        encoding="utf-8"
    )

    frontend_dir = ws / "frontend"
    frontend_dir.mkdir(parents=True, exist_ok=True)
    (frontend_dir / "index.js").write_text(
        'function renderHeader(title) {\n'
        '    console.log("Header: " + title);\n'
        '}\n',
        encoding="utf-8"
    )

    (ws / "README.md").write_text(
        '# Sample Workspace\n'
        'This is an isolated test workspace.\n',
        encoding="utf-8"
    )

    return ws


@pytest.fixture
def sample_code_dir(temp_workspace: Path) -> Path:
    """Convenience alias providing the source directory of temp_workspace."""
    return temp_workspace / "src"


@pytest.fixture
def cli_runner(isolated_env: Path):
    """Helper to execute CLI commands with mandatory database isolation."""
    class CLIRunner:
        def run(
            self,
            *args: str,
            use_subprocess: bool = False,
            env_overrides: Optional[Dict[str, str]] = None
        ) -> tuple[int, str, str]:
            effective_args = list(args)
            db_target = os.environ.get("AI_DB_PATH")

            # Defense-in-depth: inject --db if command supports it and caller omitted it
            subcommands_with_db = {
                "sync", "query", "check", "lint", "symbol", "status",
                "watch", "sync-all", "route-skill", "sync-skills",
                "context", "remember", "analyze", "expand", "locate",
                "mcp", "prune", "optimize", "telemetry"
            }
            if (
                db_target
                and not any(str(a) == "--db" or str(a).startswith("--db=") for a in effective_args)
                and any(cmd in effective_args for cmd in subcommands_with_db)
            ):
                effective_args.extend(["--db", db_target])

            if use_subprocess:
                cmd = [sys.executable, "-m", "ai_db.cli"] + effective_args
                env = os.environ.copy()
                if env_overrides:
                    env.update(env_overrides)
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    env=env,
                    cwd=str(REPO_ROOT)
                )
                return proc.returncode, proc.stdout, proc.stderr
            else:
                from ai_db.cli import main
                old_stdout, old_stderr = sys.stdout, sys.stderr
                old_argv = sys.argv
                sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
                sys.argv = ["ai-db"] + effective_args
                retcode = 0
                try:
                    try:
                        main(effective_args)
                    except TypeError:
                        main()
                except SystemExit as e:
                    retcode = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
                except Exception as exc:
                    retcode = 1
                    sys.stderr.write(str(exc))
                finally:
                    out = sys.stdout.getvalue()
                    err = sys.stderr.getvalue()
                    sys.stdout, sys.stderr = old_stdout, old_stderr
                    sys.argv = old_argv
                return retcode, out, err

    return CLIRunner()


@pytest.fixture
def sqlite_backend(temp_db_path: str):
    """Yield an initialized SQLiteBackend if M2 storage module is available."""
    try:
        from ai_db.storage.sqlite_backend import SQLiteBackend
        backend = SQLiteBackend(temp_db_path)
        backend.initialize()
        yield backend
        backend.close()
    except ImportError:
        pytest.skip("SQLiteBackend (M2) not yet available")


@pytest.fixture
def memory_sqlite_backend():
    """Yield an initialized in-memory SQLiteBackend if M2 storage module is available."""
    try:
        from ai_db.storage.sqlite_backend import SQLiteBackend
        backend = SQLiteBackend("sqlite:///:memory:")
        backend.initialize()
        yield backend
        backend.close()
    except ImportError:
        pytest.skip("SQLiteBackend (M2) not yet available")


@pytest.fixture
def mock_mysql_connection(monkeypatch: pytest.MonkeyPatch):
    """Mock pymysql connection and cursor for testing MySQL adapter without live server."""
    mock_cursor = MagicMock()
    mock_cursor.execute.return_value = 0
    mock_cursor.fetchall.return_value = []
    mock_cursor.fetchone.return_value = None
    mock_cursor.lastrowid = 1
    mock_cursor.rowcount = 1

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_conn.commit.return_value = None
    mock_conn.rollback.return_value = None
    mock_conn.close.return_value = None

    mock_pymysql = MagicMock()
    mock_pymysql.connect.return_value = mock_conn

    monkeypatch.setitem(sys.modules, "pymysql", mock_pymysql)
    monkeypatch.setitem(sys.modules, "pymysql.cursors", MagicMock())

    return {
        "module": mock_pymysql,
        "connection": mock_conn,
        "cursor": mock_cursor,
    }


@pytest.fixture
def sample_records():
    """Factory returning standardized domain DTO objects or compatible fallbacks."""
    try:
        from ai_db.storage.models import (
            FileRecord, ChunkRecord, SymbolRecord, SymbolRefRecord,
            AnnotationRecord, SyntaxErrorRecord, SkillRecord,
            ContextRecord, AnalysisRefRecord, SearchResult
        )
        return {
            "file": FileRecord(filepath="src/service.py", sha256="abc123hash", last_modified=1000.0, chunk_count=2),
            "chunk": ChunkRecord(
                filepath="src/service.py", chunk_type="class", name="PaymentProcessor",
                start_line=3, end_line=12, content="class PaymentProcessor:\n    pass\n"
            ),
            "symbol": SymbolRecord(
                name="PaymentProcessor", symbol_type="class", filepath="src/service.py",
                line=3, signature="class PaymentProcessor:"
            ),
            "symbol_ref": SymbolRefRecord(
                caller_filepath="src/main.py", caller_name="main", caller_line=10,
                callee_name="PaymentProcessor", ref_type="call"
            ),
            "annotation": AnnotationRecord(
                filepath="src/service.py", line=4, kind="docstring",
                content="Processes payments"
            ),
            "syntax_error": SyntaxErrorRecord(
                filepath="src/broken.py", line=2, col=5,
                message="invalid syntax", timestamp=1000.0
            ),
            "skill": SkillRecord(
                name="test_skill", description="Testing skill",
                filepath="skills/test.md", triggers="test,run", sha256="sk123",
                last_modified=1000.0, content="# Test Skill"
            ),
            "context": ContextRecord(
                session_id="sess_test_1", project="global", title="Test Session",
                summary="Summary of test session", active_files=["src/service.py"],
                open_tasks=["task1"], timestamp=1000.0, full_notes="Detailed notes"
            ),
            "analysis_ref": AnalysisRefRecord(
                ref_id="ref_1", filepath="src/service.py", name="process",
                start_line=8, end_line=12, kind="method", body_text="def process():",
                timestamp=1000.0
            ),
            "search_result": SearchResult(
                chunk_id=1, filepath="src/service.py", name="PaymentProcessor",
                chunk_type="class", project="global", start_line=3, end_line=12,
                score=0.95, snippet="class PaymentProcessor:"
            ),
        }
    except ImportError:
        return {}
