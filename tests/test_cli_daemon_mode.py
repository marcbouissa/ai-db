"""Regression: CLI commands must work when the dispatcher is a DaemonClient
(which has no `_get_db`) by going through the tool surface only."""

from typing import Any

import pytest

from ai_db import cli
from ai_db.dispatcher import ServiceDispatcher


class FakeDaemonClient:
    """Mimics DaemonClient: only execute()/close(), no _get_db."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def execute(self, tool_name: str, arguments: dict[str, Any] | None = None, progress: Any = None) -> Any:
        args = arguments or {}
        self.calls.append((tool_name, args))
        if tool_name == "status":
            return {"db": "d", "files": 1, "chunks": 2, "symbols": 3, "syntax_errors": 0, "kb": 4}
        if tool_name == "session_state":
            return {"key": args["key"], "value": "sexp"}
        if tool_name == "query_log":
            return []
        if tool_name == "trace":
            return "TRACE_TREE"
        raise AssertionError(tool_name)

    def close(self) -> None:
        pass


@pytest.fixture
def fake(monkeypatch):
    client = FakeDaemonClient()
    monkeypatch.setattr(cli, "create_daemon_client", lambda *a, **k: client)
    return client


def _run(argv):
    try:
        return cli.main(argv)
    except SystemExit as e:
        return e.code


def test_status_in_daemon_mode(fake, capsys):
    _run(["--daemon-url", "http://127.0.0.1:1", "--project", "p", "status"])
    out = capsys.readouterr().out
    assert "default_format:sexp" in out
    assert ("session_state", {"key": "default_format"}) in fake.calls


def test_trace_in_daemon_mode(fake, capsys):
    _run(["--daemon-url", "http://127.0.0.1:1", "--project", "p", "trace", "foo"])
    assert "TRACE_TREE" in capsys.readouterr().out


def test_session_state_and_query_log_tools(tmp_path):
    d = ServiceDispatcher(db_path=str(tmp_path / "t.db"))
    try:
        assert d.execute("session_state", {"key": "nope"}) == {"key": "nope", "value": None}
        assert d.execute("query_log", {"limit": 5}) == [] or isinstance(d.execute("query_log", {}), list)
    finally:
        d.close()


def test_json_boundary_helpers_reject_wrong_shape():
    from ai_db.daemon._json import as_dict, as_list
    from ai_db.errors import AiDbProtocolError

    assert as_dict({"a": 1}) == {"a": 1}
    assert as_list([{"a": 1}]) == [{"a": 1}]
    with pytest.raises(AiDbProtocolError):
        as_dict([])
    with pytest.raises(AiDbProtocolError):
        as_list("text")
