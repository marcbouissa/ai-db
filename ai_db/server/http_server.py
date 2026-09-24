"""
ai_db.server.http_server
Lightweight, multi-threaded HTTP/JSON REST API server for ai-db.
Zero external dependencies — standard library http.server only.

Usage:
    ai-db serve --port 8765

Endpoints:
    GET  /health            Liveness probe -> { "ok": true, "version": "0.1.0", "status": "healthy" }
    GET  /status            Database entity metrics -> { "files": ..., "chunks": ..., ... }
    GET  /tools             Tool registry inventory & schemas -> { "tools": [...] }
    GET  /telemetry         Performance & token metrics -> { "latency": ..., "cache": ..., ... }
    POST /tools/{name}      Execute named tool with JSON body as arguments
    POST /                  Execute tool via envelope { "tool": "<name>", "args": { ... } }
    OPTIONS *               CORS pre-flight headers
"""
import json
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast

from ai_db.constants import DEFAULT_DB_FILE


class ThreadedAiDbServer(ThreadingHTTPServer):
    """Multi-threaded HTTP server with shared DB facade and dispatcher."""
    daemon_threads = True

    def __init__(
        self,
        server_address,
        RequestHandlerClass,
        db_path: str | None = None,
        dispatcher: Any | None = None,
    ):
        super().__init__(server_address, RequestHandlerClass)
        self.db_path = db_path or DEFAULT_DB_FILE
        self.dispatcher = dispatcher
        self.start_time = time.time()


class _AiDbHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt: str, *args: Any):
        # Suppress routine request logging to prevent terminal pollution
        pass

    def _send_json(self, status: int, data: Any):
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()
        self.wfile.write(body)

    def _get_dispatcher(self) -> Any:
        """Returns the shared dispatcher, or lazily initializes and caches it on the server."""
        dispatcher = getattr(self.server, "dispatcher", None)
        if dispatcher is not None:
            return dispatcher

        from ai_db.dispatcher import ServiceDispatcher
        server = cast(ThreadedAiDbServer, self.server)
        server.dispatcher = ServiceDispatcher(db_path=server.db_path)
        return server.dispatcher

    def do_OPTIONS(self):
        """CORS pre-flight negotiation."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        clean_path = self.path.split("?")[0].rstrip("/")
        if not clean_path:
            clean_path = "/"

        if clean_path == "/health":
            self._send_json(200, {"ok": True, "version": "0.1.0", "status": "healthy"})

        elif clean_path in ("/status", "/tools", "/telemetry"):
            dispatcher = self._get_dispatcher()
            try:
                if clean_path == "/tools":
                    self._send_json(200, {"tools": dispatcher.list_tools()})
                else:
                    self._send_json(200, dispatcher.execute(clean_path[1:], {}))
            except Exception as e:  # noqa: BLE001 - transport boundary: report to client
                self._send_json(500, {"error": str(e)})

        else:
            self._send_json(404, {
                "error": "Endpoint not found",
                "path": self.path,
                "supported_endpoints": ["GET /health", "GET /status", "GET /tools", "GET /telemetry", "POST /tools/{name}", "POST /investigate", "POST /"]
            })

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length > 0 else b""
        if body:
            try:
                payload = json.loads(body.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                self._send_json(400, {"error": f"Invalid JSON body: {e}"})
                return
        else:
            payload = {}

        clean_path = self.path.split("?")[0].rstrip("/")
        if not clean_path:
            clean_path = "/"

        # Route matching
        if clean_path.startswith("/tools/"):
            tool_name = urllib.parse.unquote(clean_path[len("/tools/"):].strip("/"))
            if not tool_name:
                self._send_json(400, {"error": "Missing tool name in URL path"})
                return
            tool_args = payload if isinstance(payload, dict) else {}
        elif clean_path == "/investigate":
            tool_name = "investigate"
            tool_args = payload if isinstance(payload, dict) else {}
        elif clean_path in ("/", "/call"):
            tool_name = payload.get("tool") or payload.get("name", "")
            if not tool_name:
                self._send_json(400, {"error": "Missing 'tool' field in request body"})
                return
            tool_args = payload.get("args") if "args" in payload else payload.get("arguments", {})
        else:
            self._send_json(404, {"error": f"Path not found: '{self.path}'"})
            return

        dispatcher = self._get_dispatcher()
        if dispatcher is None:
            self._send_json(500, {"error": "Service dispatcher is not available"})
            return

        try:
            result = dispatcher.execute(tool_name, tool_args)
            self._send_json(200, result)
        except KeyError as e:
            self._send_json(404, {"error": f"Tool '{tool_name}' not found", "details": str(e)})
        except ValueError as e:
            self._send_json(400, {"error": f"Invalid arguments for tool '{tool_name}'", "details": str(e)})
        except Exception as e:  # noqa: BLE001 - transport boundary: report to client
            self._send_json(500, {"error": f"Error executing tool '{tool_name}'", "details": str(e)})


AiDbHandler = _AiDbHandler  # Public alias


def create_http_server(
    db_path: str | None = None,
    host: str = "127.0.0.1",
    port: int = 0,
    dispatcher: Any | None = None,
) -> ThreadedAiDbServer:
    """Public factory creating a ThreadedAiDbServer instance bound to the given database."""
    return ThreadedAiDbServer((host, port), _AiDbHandler, db_path=db_path, dispatcher=dispatcher)


def start_http_server(
    db_path: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    dispatcher: Any | None = None,
    config: Any | None = None,
):
    """Starts the threaded HTTP server (blocking). Call from CLI."""
    if dispatcher is None:
        from ai_db.config import load_config
        from ai_db.dispatcher import ServiceDispatcher
        cfg = config if config is not None else load_config()
        dispatcher = ServiceDispatcher(db_path=db_path, config=cfg)
        dispatcher._get_db()  # fail fast on invalid storage/provider config

    server = ThreadedAiDbServer((host, port), _AiDbHandler, db_path=db_path, dispatcher=dispatcher)
    print(f"[ai-db serve] Listening on http://{host}:{port} | DB: {db_path} (Threaded)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[ai-db serve] Stopped.")
    finally:
        server.server_close()
