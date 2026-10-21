"""`ai-db context` from the CLI: one task per --tasks, large notes intact, daemon-less path."""

import json
import socket

import pytest

from ai_db import cli
from ai_db.daemon import server
from ai_db.errors import AiDbDaemonUnavailableError

MARC_TASK = ("Record v9: place p1 then p2 (guard on, watchdog off TEMP, "
             "holding_threshold 0.65 TEMP)")


def _notes() -> str:
    """~13 kB of markdown with non-ASCII text, emoji and a NUL-free mix of line endings."""
    line = "- Entscheidung: Greifer schliesst bei 0,65 N; Δt=12 ms; 注意: 再試行; 🚀 ok\n"
    notes = "# Session notes — Grösse 13 kB\n\n" + line * 170 + "\n```py\nx = {'a': 1}\n```\nEND"
    assert len(notes.encode("utf-8")) > 13_000
    return notes


def _run(argv: list[str]) -> int:
    try:
        return cli.main(argv)
    except SystemExit as e:
        return int(e.code or 0)


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "ctx.db")


def test_each_tasks_flag_is_one_task(db, capsys):
    assert _run(["context", "save", "s1", "--summary", "sum", "--project", "p", "--db", db,
                 "--tasks", MARC_TASK, "--tasks", "second, with a comma"]) == 0
    capsys.readouterr()
    _run(["context", "get", "s1", "--project", "p", "--db", db, "--format", "json"])
    ctx = json.loads(capsys.readouterr().out)
    assert ctx["open_tasks"] == [MARC_TASK, "second, with a comma"]


def test_large_unicode_notes_round_trip(db, capsys):
    notes = _notes()
    assert _run(["context", "save", "big", "--summary", "s", "--project", "p", "--db", db,
                 "--notes", notes]) == 0
    capsys.readouterr()
    _run(["context", "get", "big", "--project", "p", "--db", db, "--format", "json"])
    assert json.loads(capsys.readouterr().out)["full_notes"] == notes
    _run(["context", "get", "big", "--project", "p", "--db", db])
    assert notes in capsys.readouterr().out
    _run(["context", "list", "--project", "p", "--db", db])
    out = capsys.readouterr().out
    assert f"0 files, 0 tasks, notes {len(notes)} chars: # Session notes — Grösse 13 kB" in out


def test_large_notes_round_trip_through_the_daemon(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from tests.test_daemon_server import _call, _text, _write_config

    monkeypatch.setenv("AI_DB_CONFIG", str(_write_config(tmp_path)))
    server.rate_limiter._buckets.clear()
    notes = _notes()
    with TestClient(server.app) as daemon:
        is_error, text = _text(_call(daemon, "context_save", {
            "session_id": "big", "summary": "s", "notes": notes, "open_tasks": [MARC_TASK]}))
        assert not is_error, text
        is_error, text = _text(_call(daemon, "context_recall", {"session_id": "big"}))
        assert not is_error, text
        ctx = json.loads(text)
        assert ctx["full_notes"] == notes and ctx["open_tasks"] == [MARC_TASK]
        is_error, text = _text(_call(daemon, "context_recall", {"list": True}))
        [row] = json.loads(text)
        assert row["notes_chars"] == len(notes)


def _closed_port_url() -> str:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{s.getsockname()[1]}"


def test_context_and_sync_run_locally_when_the_daemon_cannot_start(db, tmp_path, monkeypatch,
                                                                   capsys):
    """AI_DB_DAEMON_AUTOSTART=0 (conftest) stands in for an auto-start that failed."""
    monkeypatch.setenv("AI_DB_DAEMON_URL", _closed_port_url())
    assert _run(["context", "save", "s1", "--summary", "sum", "--project", "p", "--db", db,
                 "--tasks", MARC_TASK]) == 0
    captured = capsys.readouterr()
    assert "without the daemon, directly on the local database" in captured.err
    assert "ai-db daemon start" in captured.err
    _run(["context", "get", "s1", "--project", "p", "--db", db, "--format", "json"])
    assert json.loads(capsys.readouterr().out)["open_tasks"] == [MARC_TASK]

    notes = tmp_path / "notes"  # not a project: no .git, no marker file
    notes.mkdir()
    (notes / "a.md").write_text("# hello\n")
    assert _run(["sync", str(notes), "--db", db]) == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == "+1 ~0 -0 =0"


def test_other_commands_fail_with_how_to_start(db, monkeypatch):
    monkeypatch.setenv("AI_DB_DAEMON_URL", _closed_port_url())
    with pytest.raises(AiDbDaemonUnavailableError, match="ai-db daemon start"):
        cli._main(["query", "x", "--db", db])
