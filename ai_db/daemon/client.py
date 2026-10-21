"""HTTP client for ai-db daemon, matching ServiceDispatcher interface."""

import asyncio
import contextlib
import json
import os
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx

from ai_db.constants import DAEMON_CLIENT_TIMEOUT_S, DAEMON_TOOL_CALL_TIMEOUT_S
from ai_db.daemon._json import as_dict, as_list
from ai_db.errors import AiDbDaemonError

try:
    from websockets.asyncio.client import connect as ws_connect
    from websockets.exceptions import WebSocketException
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

# How long to wait for the progress stream to deliver "done" after the call returned.
_PROGRESS_DRAIN_S = 5.0


def describe_http_error(exc: httpx.HTTPError) -> str:
    """``Type: message``. httpx timeouts often carry an empty message (``ReadTimeout('')``)."""
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else f"{type(exc).__name__} (no message)"


def _error_detail(resp: httpx.Response) -> str:
    """The ``detail`` of a JSON error body, else the raw body, never an empty string."""
    try:
        body = resp.json()
    except ValueError:
        body = None
    detail = body.get("detail") if isinstance(body, dict) else None
    text = str(detail) if detail else resp.text
    return text.strip() or "(empty response body)"


class DaemonClient:
    """Drop-in replacement for ServiceDispatcher that routes to HTTP daemon.

    Every public method is synchronous (``asyncio.run``), so each call opens and closes
    its own ``httpx.AsyncClient``: an async client cannot outlive its event loop.
    """

    def __init__(
        self,
        base_url: str,
        default_project: str | None = None,
        timeout: float = DAEMON_CLIENT_TIMEOUT_S,
        api_key: str | None = None,
        admin_key: str | None = None,
        tool_timeout: float = DAEMON_TOOL_CALL_TIMEOUT_S,
    ):
        self.base_url = base_url.rstrip("/")
        self.default_project = default_project
        self.timeout = timeout
        # Read timeout of tool calls, which may run (or queue behind a sync) for minutes.
        self.tool_timeout = tool_timeout
        self.api_key = api_key
        # Management routes (/projects, /metrics, /daemon/*) need the admin key.
        self.admin_key = admin_key or api_key
        self._tools_cache: list[dict[str, Any]] | None = None
        self._tools_cache_project: str | None = None

    def _headers(self, admin: bool = False) -> dict[str, str]:
        key = self.admin_key if admin else self.api_key
        return {"Authorization": f"Bearer {key}"} if key else {}

    @contextlib.asynccontextmanager
    async def _http(self, admin: bool = False,
                    read_timeout: float | None = None) -> AsyncIterator[httpx.AsyncClient]:
        timeout = httpx.Timeout(self.timeout, read=read_timeout if read_timeout else self.timeout)
        async with httpx.AsyncClient(timeout=timeout, headers=self._headers(admin)) as client:
            yield client

    async def _request(self, method: str, path: str, admin: bool = False,
                       read_timeout: float | None = None, **kw: Any) -> Any:
        url = f"{self.base_url}{path}"
        try:
            try:
                async with self._http(admin, read_timeout) as client:
                    resp = await client.request(method, url, **kw)
            except httpx.ConnectError as e:
                # Nothing listening: the request never reached a daemon, so start the
                # local one (raises AiDbDaemonUnavailableError if it cannot) and retry.
                from ai_db.daemon.autostart import ensure_daemon
                await asyncio.to_thread(ensure_daemon, self.base_url, describe_http_error(e))
                async with self._http(admin, read_timeout) as client:
                    resp = await client.request(method, url, **kw)
        except httpx.TimeoutException as e:
            limit = (self.timeout if isinstance(e, httpx.ConnectTimeout) or not read_timeout
                     else read_timeout)
            raise AiDbDaemonError(
                f"daemon {method} {url} timed out ({describe_http_error(e)}, limit {limit:g}s); "
                "the daemon may still be running the call: see `ai-db daemon status` "
                "and the daemon log") from e
        except httpx.HTTPError as e:
            raise AiDbDaemonError(
                f"daemon {method} {url} failed: {describe_http_error(e)}") from e
        if resp.is_error:
            raise AiDbDaemonError(
                f"daemon {method} {path} -> HTTP {resp.status_code}: {_error_detail(resp)}")
        return resp.json()

    def _resolve_project(self, project: str | None) -> str:
        proj = project or self.default_project
        if not proj:
            from ai_db.utils import detect_project_name
            proj = detect_project_name(os.getcwd())
        return proj

    @staticmethod
    def _tool_result(data: dict[str, Any]) -> Any:
        text = data["content"][0]["text"]
        if data.get("isError"):
            raise AiDbDaemonError(text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    # --- tools -----------------------------------------------------------------

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        progress: Any = None,
    ) -> Any:
        """Sync wrapper for CLI compatibility."""
        return asyncio.run(self._execute_async(tool_name, arguments, progress))

    async def _execute_async(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> Any:
        args = dict(arguments or {})  # never mutate the caller's dict
        project = self._resolve_project(args.pop("project", None))
        # The daemon runs elsewhere (cwd "/"): relative paths resolve against ours.
        args.setdefault("cwd", os.getcwd())
        payload: dict[str, Any] = {"name": tool_name, "arguments": args}

        listener: asyncio.Task[None] | None = None
        if progress is not None:
            token = str(uuid.uuid4())
            payload["_meta"] = {"progressToken": token}
            listen = self._listen_progress_ws if WEBSOCKETS_AVAILABLE else self._listen_progress
            listener = asyncio.create_task(listen(project, token, progress))

        try:
            data = await self._request("POST", f"/projects/{project}/tools/call",
                                       read_timeout=self.tool_timeout, json=payload)
        finally:
            if listener is not None:
                try:
                    await asyncio.wait_for(listener, timeout=_PROGRESS_DRAIN_S)
                except asyncio.TimeoutError:
                    pass  # wait_for already cancelled the listener
        return self._tool_result(as_dict(data))

    @staticmethod
    def _forward(event: dict[str, Any], progress_cb: Callable[[int, int, str], None]) -> bool:
        """Forward one progress event; True when the stream is finished."""
        if event.get("type") == "progress":
            progress_cb(event.get("progress", 0), event.get("total", 0), event.get("message", ""))
        return event.get("type") == "done"

    async def _listen_progress(self, project: str, token: str,
                               progress_cb: Callable[[int, int, str], None]) -> None:
        """Server-Sent Events progress stream (used when websockets is not installed)."""
        url = f"{self.base_url}/projects/{project}/tools/call/progress"
        try:
            async with httpx.AsyncClient(timeout=None, headers=self._headers()) as client, \
                    client.stream("GET", url, params={"token": token}) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.startswith("data: ") and self._forward(json.loads(line[6:]), progress_cb):
                        return
        except (httpx.HTTPError, json.JSONDecodeError):
            pass  # progress is advisory: a missing/broken stream must not fail the tool call

    async def _listen_progress_ws(self, project: str, token: str,
                                  progress_cb: Callable[[int, int, str], None]) -> None:
        """WebSocket progress stream."""
        url = (f"{self.base_url.replace('http', 'ws', 1)}/projects/{project}"
               f"/tools/call/progress/ws?token={token}")
        try:
            async with ws_connect(url, additional_headers=self._headers(), ping_interval=20,
                                  ping_timeout=10, close_timeout=5) as ws:
                async for message in ws:
                    if self._forward(json.loads(message), progress_cb):
                        return
        except (WebSocketException, OSError, json.JSONDecodeError):
            pass  # progress is advisory: a missing/broken stream must not fail the tool call

    def list_tools(self) -> list[dict[str, Any]]:
        return asyncio.run(self._list_tools_async())

    async def _list_tools_async(self) -> list[dict[str, Any]]:
        project = self._resolve_project(None)
        if self._tools_cache is None or self._tools_cache_project != project:
            data = as_dict(await self._request("POST", f"/projects/{project}/tools/list"))
            self._tools_cache = as_list(data["tools"], "tools")
            self._tools_cache_project = project
        return self._tools_cache

    def get_tool(self, name: str) -> dict[str, Any] | None:
        return next((t for t in self.list_tools() if t["name"] == name), None)

    def register_tool(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Cannot register tools on daemon client")

    def execute_batch(self, calls: list[dict[str, Any]], project: str | None = None) -> list[Any]:
        """Execute multiple tool calls in a single request."""
        return asyncio.run(self._execute_batch_async(calls, project))

    async def _execute_batch_async(self, calls: list[dict[str, Any]],
                                   project: str | None = None) -> list[Any]:
        project = self._resolve_project(project)
        cwd = os.getcwd()
        payload = {"calls": [{**c, "arguments": {"cwd": cwd, **c.get("arguments", {})}}
                             for c in calls]}
        data = as_dict(await self._request("POST", f"/projects/{project}/tools/call/batch",
                                           read_timeout=self.tool_timeout, json=payload))
        return [self._tool_result(r) for r in data["results"]]

    def close(self) -> None:
        """Nothing to close: each call owns (and closes) its HTTP client."""

    # --- status and management -------------------------------------------------

    def health(self) -> dict[str, Any]:
        return as_dict(asyncio.run(self._request("GET", "/health")))

    def status(self) -> dict[str, Any]:
        project = self._resolve_project(None)
        return as_dict(asyncio.run(self._request("GET", f"/projects/{project}/status")))

    def list_projects(self) -> list[dict[str, Any]]:
        return as_list(asyncio.run(self._request("GET", "/projects", admin=True)))

    def register_project(self, name: str, db_path: str, config_path: str | None = None,
                         auto_start: bool = True) -> dict[str, Any]:
        payload = {"name": name, "db_path": db_path, "config_path": config_path,
                   "auto_start": auto_start}
        return as_dict(asyncio.run(self._request("POST", "/projects", admin=True, json=payload)))

    def unregister_project(self, name: str) -> dict[str, Any]:
        return as_dict(asyncio.run(self._request("DELETE", f"/projects/{name}", admin=True)))

    def cross_project_query(self, query: str, projects: list[str] | None = None,
                            top: int = 5) -> list[dict[str, Any]]:
        """Execute query across multiple projects."""
        arguments: dict[str, Any] = {"query": query, "top": top}
        if projects:
            arguments["projects"] = projects
        data = asyncio.run(self._request("POST", "/query/cross-project", admin=True,
                                         read_timeout=self.tool_timeout,
                                         json={"name": "query", "arguments": arguments}))
        return as_list(self._tool_result(as_dict(data)))

    def metrics(self) -> dict[str, Any]:
        return as_dict(asyncio.run(self._request("GET", "/metrics", admin=True)))

    def reload_config(self) -> dict[str, Any]:
        result = as_dict(asyncio.run(self._request("POST", "/daemon/reload", admin=True)))
        self._tools_cache = None
        self._tools_cache_project = None
        return result


def create_daemon_client(
    daemon_url: str | None = None,
    project: str | None = None,
    timeout: float = DAEMON_CLIENT_TIMEOUT_S,
    api_key: str | None = None,
) -> DaemonClient:
    """Create a DaemonClient from URL or environment."""
    url = daemon_url or os.environ.get("AI_DB_DAEMON_URL")
    if not url:
        raise ValueError("No daemon URL provided. Use --daemon-url or set AI_DB_DAEMON_URL")
    key = api_key or os.environ.get("AI_DB_API_KEY")
    return DaemonClient(url, default_project=project, timeout=timeout, api_key=key,
                        admin_key=os.environ.get("AI_DB_ADMIN_KEY"))
