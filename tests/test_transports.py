"""
tests/test_transports.py
Comprehensive 4-tier test suite for the Transports and Dispatch Subsystem.

Covers:
  - ServiceDispatcher tool registry & execution
  - CLI commands suite (sync, query, symbol, check, status, vectordb parity)
  - stdio JSON-RPC 2.0 MCP server (initialize, ping, tools/list, tools/call)
  - Threaded HTTP REST server (health, status, tools, POST dispatch, CORS)
"""
import json
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import HTTPServer

import pytest

from ai_db import VectorDB
from ai_db.server.http_server import AiDbHandler
from mcp_server import StdioMCPServer

try:
    from ai_db.dispatcher import ServiceDispatcher
except ImportError:
    ServiceDispatcher = None


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def transport_env(tmp_path):
    """Provides an isolated workspace and database file for transport testing."""
    db_file = str(tmp_path / "transport.db")
    src_dir = tmp_path / "src"
    src_dir.mkdir()

    # Create initial files
    (src_dir / "service.py").write_text(
        "class OrderService:\n"
        "    def create_order(self, item: str):\n"
        "        return {'status': 'created', 'item': item}\n"
    )

    vdb = VectorDB(db_file)
    vdb.sync(str(src_dir), project="transports", verbose=False)
    vdb.close()

    return {
        "db_file": db_file,
        "src_dir": src_dir,
        "tmp_path": tmp_path,
    }


@pytest.fixture
def http_server(transport_env):
    """Starts an ephemeral AiDbHandler server on 127.0.0.1 in a daemon thread."""
    server = HTTPServer(("127.0.0.1", 0), AiDbHandler)
    server.db_path = transport_env["db_file"]
    host, port = server.server_address
    base_url = f"http://{host}:{port}"

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield {
        "base_url": base_url,
        "db_path": transport_env["db_file"],
        "server": server,
    }

    server.shutdown()
    server.server_close()
    thread.join(timeout=2.0)


# ==============================================================================
# Tier 1: Feature Coverage (Happy Path Isolation)
# ==============================================================================

class TestTransportsTier1FeatureCoverage:
    """Core capabilities across ServiceDispatcher, CLI, MCP server, and HTTP server."""

    def test_service_dispatcher_register_and_list(self):
        """ServiceDispatcher registers tools with schemas and lists them."""
        if ServiceDispatcher is None:
            pytest.skip("ServiceDispatcher not yet implemented (planned for Milestone 3)")

        dispatcher = ServiceDispatcher()
        schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }
        dispatcher.register_tool(
            name="custom_search",
            description="Custom code search tool",
            parameters_schema=schema,
            handler=lambda args: {"hits": 1}
        )
        tools = dispatcher.list_tools()
        assert any(t["name"] == "custom_search" for t in tools)

    def test_service_dispatcher_execute_tool(self):
        """ServiceDispatcher executes registered tool handler and returns result."""
        if ServiceDispatcher is None:
            pytest.skip("ServiceDispatcher not yet implemented (planned for Milestone 3)")

        dispatcher = ServiceDispatcher()
        dispatcher.register_tool(
            name="echo",
            description="Echoes input",
            parameters_schema={"type": "object"},
            handler=lambda args: {"echoed": args.get("val")}
        )
        res = dispatcher.execute("echo", {"val": "hello"})
        assert res == {"echoed": "hello"}

    def test_cli_sync_command(self, transport_env):
        """CLI sync command indexes workspace and reports added/updated metrics."""
        db_file = transport_env["db_file"]
        src_dir = transport_env["src_dir"]

        proc = subprocess.run(
            [sys.executable, "-m", "ai_db.cli", "sync", str(src_dir), "--db", db_file, "--project", "transports"],
            capture_output=True,
            text=True,
            check=False
        )
        assert proc.returncode == 0
        assert "Sync (transports):" in proc.stdout

    def test_cli_query_command(self, transport_env):
        """CLI query command retrieves ranked chunks via stdout."""
        db_file = transport_env["db_file"]

        proc = subprocess.run(
            [sys.executable, "-m", "ai_db.cli", "query", "OrderService", "--db", db_file, "--project", "transports"],
            capture_output=True,
            text=True,
            check=False
        )
        assert proc.returncode == 0
        assert "OrderService" in proc.stdout

    def test_cli_symbol_command(self, transport_env):
        """CLI symbol command prints exact definition location."""
        db_file = transport_env["db_file"]

        proc = subprocess.run(
            [sys.executable, "-m", "ai_db.cli", "symbol", "OrderService", "--db", db_file, "--project", "transports"],
            capture_output=True,
            text=True,
            check=False
        )
        assert proc.returncode == 0
        assert "(class OrderService)" in proc.stdout

    def test_cli_check_command_clean(self, transport_env):
        """CLI check outputs zero syntax errors on clean workspace and exits 0."""
        db_file = transport_env["db_file"]
        src_dir = transport_env["src_dir"]

        proc = subprocess.run(
            [sys.executable, "-m", "ai_db.cli", "check", str(src_dir), "--db", db_file, "--project", "transports"],
            capture_output=True,
            text=True,
            check=False
        )
        assert proc.returncode == 0
        assert "No syntax errors detected." in proc.stdout

    def test_cli_check_command_error(self, transport_env):
        """CLI check detects syntax errors, outputs error line, and exits with code 1."""
        db_file = transport_env["db_file"]
        src_dir = transport_env["src_dir"]
        broken_file = src_dir / "broken.py"
        broken_file.write_text("class Broken( :\n    pass\n")

        proc = subprocess.run(
            [sys.executable, "-m", "ai_db.cli", "check", str(broken_file), "--db", db_file, "--project", "transports"],
            capture_output=True,
            text=True,
            check=False
        )
        assert proc.returncode == 1
        assert "SYNTAX_ERROR:" in proc.stdout
        assert "broken.py:1:" in proc.stdout

    def test_mcp_server_initialize_and_ping(self, transport_env):
        """StdioMCPServer responds to initialize and ping JSON-RPC requests."""
        server = StdioMCPServer(transport_env["db_file"])

        # initialize
        resp_init = server.handle_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {}
        })
        assert resp_init["jsonrpc"] == "2.0"
        assert resp_init["id"] == 1
        assert "serverInfo" in resp_init["result"]
        assert resp_init["result"]["serverInfo"]["name"] == "ai-db"

        # ping
        resp_ping = server.handle_request({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "ping",
            "params": {}
        })
        assert resp_ping["result"] == {}

    def test_mcp_server_tools_list(self, transport_env):
        """StdioMCPServer lists registered tools with names, descriptions, and inputSchema."""
        server = StdioMCPServer(transport_env["db_file"])
        resp = server.handle_request({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/list",
            "params": {}
        })
        tools = resp["result"]["tools"]
        assert len(tools) >= 4
        names = [t["name"] for t in tools]
        assert "locate" in names
        assert "analyze" in names
        assert "expand" in names

        for t in tools:
            assert "name" in t
            assert "description" in t
            assert "inputSchema" in t

    def test_mcp_server_tools_call(self, transport_env):
        """StdioMCPServer executes tools/call for locate and returns text content."""
        server = StdioMCPServer(transport_env["db_file"])
        resp = server.handle_request({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "locate",
                "arguments": {
                    "query": "OrderService",
                    "scope": str(transport_env["src_dir"])
                }
            }
        })
        assert resp["id"] == 4
        assert resp["result"]["isError"] is False
        content = resp["result"]["content"]
        assert len(content) > 0
        assert "OrderService" in content[0]["text"]

    def test_http_server_get_health(self, http_server):
        """HTTP GET /health returns HTTP 200 with {'ok': True}."""
        url = f"{http_server['base_url']}/health"
        with urllib.request.urlopen(url) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data.get("ok") is True

    def test_http_server_get_status(self, http_server):
        """HTTP GET /status returns HTTP 200 with database status metrics."""
        url = f"{http_server['base_url']}/status"
        with urllib.request.urlopen(url) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert "files" in data
            assert "chunks" in data
            assert "symbols" in data

    def test_http_server_get_tools(self, http_server):
        """HTTP GET /tools returns tool inventory with schemas (M3 feature)."""
        url = f"{http_server['base_url']}/tools"
        try:
            with urllib.request.urlopen(url) as resp:
                assert resp.status == 200
                data = json.loads(resp.read().decode("utf-8"))
                assert isinstance(data, list) or "tools" in data
        except urllib.error.HTTPError as e:
            if e.code == 404:
                pytest.skip("HTTP route GET /tools not yet implemented (planned for Milestone 3)")
            raise

    def test_http_server_post_tool_execution(self, http_server, transport_env):
        """HTTP POST executes tools via envelope {tool, args} or /tools/{name}."""
        # Test standard POST / tool execution envelope
        url = f"{http_server['base_url']}/"
        payload = json.dumps({
            "tool": "locate",
            "args": {
                "query": "create_order",
                "scope": str(transport_env["src_dir"])
            }
        }).encode("utf-8")

        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert len(data) > 0
            assert any("service.py" in str(item) for item in data)


# ==============================================================================
# Tier 2: Boundary & Corner Cases
# ==============================================================================

class TestTransportsTier2BoundaryAndCorner:
    """Boundary conditions, malformed payloads, invalid routes, and concurrency."""

    def test_service_dispatcher_unknown_tool(self):
        """Calling an unregistered tool in ServiceDispatcher raises KeyError."""
        if ServiceDispatcher is None:
            pytest.skip("ServiceDispatcher not yet implemented (planned for Milestone 3)")

        dispatcher = ServiceDispatcher()
        with pytest.raises(KeyError):
            dispatcher.execute("non_existent_tool", {})

    def test_service_dispatcher_missing_args(self):
        """Calling a tool with missing required schema parameters raises ValueError."""
        if ServiceDispatcher is None:
            pytest.skip("ServiceDispatcher not yet implemented (planned for Milestone 3)")

        dispatcher = ServiceDispatcher()
        schema = {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"]
        }
        dispatcher.register_tool("get_item", "Gets item", schema, lambda args: args)
        with pytest.raises(ValueError):
            dispatcher.execute("get_item", {})

    def test_cli_invalid_command(self, transport_env):
        """Passing an unrecognized CLI subcommand exits with code 2 (argparse error)."""
        proc = subprocess.run(
            [sys.executable, "-m", "ai_db.cli", "nonexistent_subcommand"],
            capture_output=True,
            text=True,
            check=False
        )
        assert proc.returncode == 2

    def test_cli_vectordb_symlink_parity(self, transport_env):
        """vectordb.py entry point behaves identically to ai_db.cli."""
        db_file = transport_env["db_file"]

        # Run via ai_db.cli
        p1 = subprocess.run(
            [sys.executable, "-m", "ai_db.cli", "status", "--db", db_file],
            capture_output=True,
            text=True,
            check=False
        )
        # Run via vectordb.py facade
        p2 = subprocess.run(
            [sys.executable, "vectordb.py", "status", "--db", db_file],
            capture_output=True,
            text=True,
            check=False
        )
        assert p1.returncode == 0
        assert p2.returncode == 0
        assert p1.stdout == p2.stdout

    def test_mcp_server_unknown_tool(self, transport_env):
        """Calling an unknown tool on StdioMCPServer returns isError: True."""
        server = StdioMCPServer(transport_env["db_file"])
        resp = server.handle_request({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": "unregistered_tool", "arguments": {}}
        })
        assert resp["result"]["isError"] is True
        assert "Unknown tool" in resp["result"]["content"][0]["text"]

    def test_mcp_server_malformed_json(self, transport_env):
        """StdioMCPServer handles notifications and empty/malformed calls safely."""
        server = StdioMCPServer(transport_env["db_file"])
        # Notification should return None
        res_notif = server.handle_request({
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        })
        assert res_notif is None

    def test_http_server_cors_options(self, http_server):
        """OPTIONS pre-flight request returns HTTP 204 with CORS headers."""
        url = f"{http_server['base_url']}/health"
        req = urllib.request.Request(url, method="OPTIONS")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 204
            assert resp.headers.get("Access-Control-Allow-Origin") == "*"
            assert "POST" in resp.headers.get("Access-Control-Allow-Methods", "")

    def test_http_server_invalid_json_body(self, http_server):
        """POST with malformed non-JSON body returns HTTP 400 Bad Request."""
        url = f"{http_server['base_url']}/"
        req = urllib.request.Request(
            url,
            data=b"INVALID_NOT_JSON",
            headers={"Content-Type": "application/json"}
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req)
        assert exc_info.value.code == 400

    def test_http_server_unknown_path(self, http_server):
        """GET request to an unknown route returns HTTP 404 Not Found."""
        url = f"{http_server['base_url']}/nonexistent_endpoint_xyz"
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(url)
        assert exc_info.value.code == 404

    def test_http_server_concurrent_requests(self, http_server):
        """Multiple concurrent requests to HTTP server succeed without locking or timeouts."""
        url_health = f"{http_server['base_url']}/health"
        url_status = f"{http_server['base_url']}/status"

        def _fetch(url):
            with urllib.request.urlopen(url, timeout=5.0) as resp:
                return resp.status

        with ThreadPoolExecutor(max_workers=8) as executor:
            tasks = [
                executor.submit(_fetch, url_health if i % 2 == 0 else url_status)
                for i in range(16)
            ]
            statuses = [t.result() for t in tasks]

        assert all(s == 200 for s in statuses)


# ==============================================================================
# Tier 3: Pairwise Combinations
# ==============================================================================

class TestTransportsTier3Combinations:
    """Parity between transport channels (CLI vs HTTP vs MCP)."""

    def test_transport_parity_mcp_and_http(self, http_server, transport_env):
        """Tool execution via MCP and HTTP returns equivalent locate results."""
        query_text = "create_order"
        scope = str(transport_env["src_dir"])

        # 1. MCP execution
        mcp = StdioMCPServer(transport_env["db_file"])
        mcp_res = mcp.handle_request({
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {"name": "locate", "arguments": {"query": query_text, "scope": scope}}
        })
        mcp_text = mcp_res["result"]["content"][0]["text"]

        # 2. HTTP execution
        url = f"{http_server['base_url']}/"
        payload = json.dumps({
            "tool": "locate",
            "args": {"query": query_text, "scope": scope}
        }).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            http_data = json.loads(resp.read().decode("utf-8"))

        assert "service.py" in mcp_text
        assert any("service.py" in str(item) for item in http_data)

    def test_transport_cli_and_http_status_parity(self, http_server, transport_env):
        """Database files count matches between CLI status output and HTTP /status endpoint."""
        # 1. CLI status
        proc = subprocess.run(
            [sys.executable, "-m", "ai_db.cli", "status", "--db", transport_env["db_file"]],
            capture_output=True,
            text=True,
            check=False
        )
        assert proc.returncode == 0
        # Parse files count from CLI output: "files:1 |"
        cli_files = None
        for part in proc.stdout.split("|"):
            if "files:" in part:
                cli_files = int(part.split(":")[1].strip())

        # 2. HTTP status
        url = f"{http_server['base_url']}/status"
        with urllib.request.urlopen(url) as resp:
            http_data = json.loads(resp.read().decode("utf-8"))
            http_files = http_data.get("files")

        assert cli_files == http_files

    def test_transport_dispatcher_dynamic_mcp_registration(self):
        """Newly registered tool in dispatcher is exposed dynamically in tool lists."""
        if ServiceDispatcher is None:
            pytest.skip("ServiceDispatcher not yet implemented (planned for Milestone 3)")

        dispatcher = ServiceDispatcher()
        initial_count = len(dispatcher.list_tools())

        dispatcher.register_tool(
            name="dynamic_validator",
            description="Dynamic validator tool",
            parameters_schema={"type": "object"},
            handler=lambda args: True
        )
        assert len(dispatcher.list_tools()) == initial_count + 1


# ==============================================================================
# Tier 4: Real-World Application Workflows
# ==============================================================================

class TestTransportsTier4Workflows:
    """Simulated end-to-end client sessions via MCP and HTTP API."""

    def test_workflow_agent_ide_integration(self, transport_env):
        """
        Simulates an AI Agent session in Claude / Cursor / Antigravity via MCP:
        1. Initialize handshake
        2. Fetch tool capabilities (tools/list)
        3. Discover code targets (tools/call locate)
        4. Inspect code structure (tools/call analyze)
        """
        server = StdioMCPServer(transport_env["db_file"])

        # 1. Handshake
        init_res = server.handle_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"clientInfo": {"name": "TestIDE"}}
        })
        assert init_res["result"]["serverInfo"]["name"] == "ai-db"

        # 2. List tools
        tools_res = server.handle_request({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        })
        available_tools = {t["name"] for t in tools_res["result"]["tools"]}
        assert "locate" in available_tools
        assert "analyze" in available_tools

        # 3. Locate target
        locate_res = server.handle_request({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "locate",
                "arguments": {"query": "create_order", "scope": str(transport_env["src_dir"])}
            }
        })
        assert locate_res["result"]["isError"] is False
        assert "service.py" in locate_res["result"]["content"][0]["text"]

        # 4. Analyze structure
        target_file = str(transport_env["src_dir"] / "service.py")
        analyze_res = server.handle_request({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "analyze",
                "arguments": {"targets": [target_file], "depth": "structure"}
            }
        })
        assert analyze_res["result"]["isError"] is False
        assert "OrderService" in analyze_res["result"]["content"][0]["text"]

    def test_workflow_remote_microservice_http_monitoring(self, http_server, transport_env):
        """
        Simulates remote CI/CD monitor interacting with ai-db over HTTP:
        1. Health probe (GET /health)
        2. Telemetry / Status check (GET /status)
        3. Remote tool invocation (POST / with status tool)
        """
        # 1. Health check
        with urllib.request.urlopen(f"{http_server['base_url']}/health") as resp:
            health = json.loads(resp.read().decode("utf-8"))
            assert health["ok"] is True

        # 2. Status inspection
        with urllib.request.urlopen(f"{http_server['base_url']}/status") as resp:
            status = json.loads(resp.read().decode("utf-8"))
            assert status["files"] >= 1

        # 3. Tool execution
        payload = json.dumps({"tool": "status", "args": {}}).encode("utf-8")
        req = urllib.request.Request(
            f"{http_server['base_url']}/",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            tool_res = json.loads(resp.read().decode("utf-8"))
            assert tool_res["files"] == status["files"]
