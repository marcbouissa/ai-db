"""DaemonClient error reporting and timeouts (16/21 Oct: `ai-db sync` failed with
"daemon at http://127.0.0.1:8080 unreachable:" and no reason, while the daemon was up).

The daemon log showed the two syncs finish after 108 s and 72 s with HTTP 200: the client
gave up after DAEMON_CLIENT_TIMEOUT_S (30 s) and httpx.ReadTimeout has an empty str().
"""

import http.server
import json
import threading
import time
from collections.abc import Iterator

import httpx
import pytest

from ai_db.daemon.client import DaemonClient, describe_http_error
from ai_db.errors import AiDbDaemonError

_OK = json.dumps({"isError": False, "content": [{"type": "text", "text": "{\"ok\": true}"}]})


class _Handler(http.server.BaseHTTPRequestHandler):
    delay = 0.0
    status = 200
    body = _OK

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        time.sleep(self.delay)
        data = self.body.encode()
        self.send_response(self.status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def server() -> Iterator[tuple[str, type[_Handler]]]:
    handler = type("H", (_Handler,), {})
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_port}", handler
    srv.shutdown()
    srv.server_close()


def test_a_tool_call_outlives_the_management_timeout(server):
    url, handler = server
    handler.delay = 0.6
    client = DaemonClient(url, default_project="p", timeout=0.2, tool_timeout=5.0)
    assert client.execute("sync", {"path": "."}) == {"ok": True}


def test_a_timed_out_call_names_the_exception_and_the_limit(server):
    url, handler = server
    handler.delay = 0.6
    client = DaemonClient(url, default_project="p", timeout=0.2, tool_timeout=0.2)
    with pytest.raises(AiDbDaemonError) as exc:
        client.execute("sync", {"path": "."})
    msg = str(exc.value)
    assert "ReadTimeout" in msg and "limit 0.2s" in msg and "ai-db daemon status" in msg
    assert not msg.rstrip().endswith(":")


def test_an_http_error_shows_status_and_body(server):
    url, handler = server
    handler.status, handler.body = 500, ""
    with pytest.raises(AiDbDaemonError, match=r"HTTP 500: \(empty response body\)"):
        DaemonClient(url, default_project="p").execute("sync", {})


def test_an_unknown_project_says_how_to_register(server):
    url, handler = server
    handler.status = 404
    handler.body = json.dumps({"detail": "project 'notes' is not registered (daemon.auto_discover "
                                         "is off): register it with `ai-db daemon register notes`"})
    with pytest.raises(AiDbDaemonError, match="HTTP 404: project 'notes'.*ai-db daemon register"):
        DaemonClient(url, default_project="notes").execute("sync", {})


def test_describe_never_returns_an_empty_reason():
    assert describe_http_error(httpx.ReadTimeout("")) == "ReadTimeout (no message)"
    assert describe_http_error(httpx.ConnectError("refused")) == "ConnectError: refused"
