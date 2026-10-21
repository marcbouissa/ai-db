"""On-demand daemon start (16 Oct: the daemon was down, nothing restarted it, the MCP
server exited with CONNECTION_CLOSED and every CLI call failed "unreachable")."""

import http.server
import json
import socket
import sys
import threading
from pathlib import Path

import pytest

from ai_db.daemon import autostart
from ai_db.daemon.client import DaemonClient
from ai_db.errors import AiDbConfigError, AiDbDaemonUnavailableError

_FAKE_DAEMON = """
import http.server, sys
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.send_header("Content-Length", "2"); self.end_headers()
        self.wfile.write(b"{}")
    def log_message(self, *a): pass
http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def local_state(tmp_path, monkeypatch):
    monkeypatch.setattr(autostart, "RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.setattr(autostart, "DAEMON_LOG_FILE", str(tmp_path / "daemon.log"))
    monkeypatch.setenv(autostart.AUTOSTART_ENV, "1")
    return tmp_path


def test_a_remote_daemon_is_never_started(local_state):
    with pytest.raises(AiDbDaemonUnavailableError, match="not on this machine"):
        autostart.ensure_daemon("http://10.1.2.3:8080", "ConnectError: refused")


def test_disabled_autostart_says_how_to_start(local_state, monkeypatch):
    monkeypatch.setenv(autostart.AUTOSTART_ENV, "0")
    with pytest.raises(AiDbDaemonUnavailableError, match="ConnectError: refused.*ai-db daemon start"):
        autostart.ensure_daemon("http://127.0.0.1:1", "ConnectError: refused")


def test_a_bad_autostart_value_is_a_config_error(monkeypatch):
    monkeypatch.setenv(autostart.AUTOSTART_ENV, "maybe")
    with pytest.raises(AiDbConfigError):
        autostart.autostart_enabled()


def test_starts_the_daemon_and_waits_for_health(local_state, monkeypatch):
    port = _free_port()
    monkeypatch.setattr(autostart, "daemon_command",
                        lambda host, p: [sys.executable, "-c", _FAKE_DAEMON, str(p)])
    procs = []
    real_popen = autostart.subprocess.Popen

    def popen(*a, **kw):
        assert kw["start_new_session"] is True  # outlives the CLI call / MCP server
        procs.append(real_popen(*a, **kw))
        return procs[-1]

    monkeypatch.setattr(autostart.subprocess, "Popen", popen)
    try:
        autostart.ensure_daemon(f"http://127.0.0.1:{port}", "ConnectError: refused")
        assert autostart.health_ok(f"http://127.0.0.1:{port}")
        # Already up: a second call starts nothing.
        autostart.ensure_daemon(f"http://127.0.0.1:{port}", "ConnectError: refused")
        assert len(procs) == 1
    finally:
        for p in procs:
            p.kill()
            p.wait()


def test_a_daemon_that_exits_reports_its_exit_code_and_log(local_state, monkeypatch):
    monkeypatch.setattr(autostart, "daemon_command",
                        lambda host, p: [sys.executable, "-c", "raise SystemExit(3)"])
    with pytest.raises(AiDbDaemonUnavailableError, match=r"exited with 3, see .*daemon\.log"):
        autostart.ensure_daemon(f"http://127.0.0.1:{_free_port()}", "ConnectError: refused")


def test_the_client_starts_the_daemon_then_retries(monkeypatch):
    """One code path for CLI and MCP: DaemonClient itself recovers from a refused connect."""
    port = _free_port()
    body = json.dumps({"isError": False, "content": [{"type": "text", "text": "[]"}]}).encode()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    servers = []

    def fake_ensure(base_url, reason):
        assert "ConnectError" in reason
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)

    monkeypatch.setattr(autostart, "ensure_daemon", fake_ensure)
    try:
        assert DaemonClient(f"http://127.0.0.1:{port}", default_project="p").execute("query", {}) == []
        assert len(servers) == 1
    finally:
        for srv in servers:
            srv.shutdown()
            srv.server_close()


def test_install_service_writes_the_unit_and_never_enables_it(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("AI_DB_CONFIG", str(tmp_path / "cfg.json"))
    units = tmp_path / "systemd" / "user"
    units.mkdir(parents=True)
    (units / "ai-db.service").write_text("ExecStart=/home/x/bin/ai-db daemon start --foreground\n")

    assert autostart.print_service_install("127.0.0.1", 8080, force=False) == 0
    unit = (units / autostart.SERVICE_NAME).read_text()
    assert "Restart=on-failure" in unit
    assert f"ExecStart={sys.executable} -m ai_db.cli --config {tmp_path / 'cfg.json'} daemon start" in unit
    out = capsys.readouterr().out
    assert f"systemctl --user enable --now {autostart.SERVICE_NAME}" in out
    assert "ai-db.service also runs an ai-db daemon" in out
    with pytest.raises(AiDbConfigError, match="--force"):
        autostart.print_service_install("127.0.0.1", 8080, force=False)
    assert autostart.print_service_install("127.0.0.1", 8081, force=True) == 0
    assert "--port 8081" in Path(units / autostart.SERVICE_NAME).read_text()
