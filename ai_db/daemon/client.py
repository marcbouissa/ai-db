"""HTTP client for ai-db daemon, matching ServiceDispatcher interface."""

import asyncio
import json
import os
import uuid
from typing import Any

import httpx
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False


class DaemonClient:
    """Drop-in replacement for ServiceDispatcher that routes to HTTP daemon."""

    def __init__(
        self,
        base_url: str,
        default_project: str | None = None,
        timeout: float = 30.0,
        api_key: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.default_project = default_project
        self.timeout = timeout
        self.api_key = api_key
        self._client: httpx.AsyncClient | None = None
        self._tools_cache: list[dict[str, Any]] | None = None
        self._tools_cache_project: str | None = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return httpx.AsyncClient(timeout=self.timeout, headers=headers)

    def _resolve_project(self, project: str | None) -> str:
        proj = project or self.default_project
        if not proj:
            # Auto-detect from CWD
            from ai_db.utils import detect_project_name
            proj = detect_project_name(os.getcwd())
        return proj

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
        progress: Any = None,
    ) -> Any:
        client = await self._ensure_client()

        # Resolve project
        project = None
        if arguments:
            project = arguments.pop("project", None)
        project = self._resolve_project(project)

        # Prepare request
        payload = {
            "name": tool_name,
            "arguments": arguments or {},
        }

        # Add progress token if progress callback provided
        if progress is not None:
            token = str(uuid.uuid4())
            payload["_meta"] = {"progressToken": token}

            # Start WebSocket connection for progress in background (preferred)
            if WEBSOCKETS_AVAILABLE:
                ws_task = asyncio.create_task(self._listen_progress_ws(project, token, progress))
            else:
                # Fallback to SSE
                ws_task = asyncio.create_task(self._listen_progress(project, token, progress))
        else:
            ws_task = None

        # Execute
        url = f"{self.base_url}/projects/{project}/tools/call"
        async with await self._ensure_client() as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

            # Wait for progress task to complete if we started one
            if progress is not None and ws_task is not None:
                try:
                    await asyncio.wait_for(ws_task, timeout=5.0)
                except asyncio.TimeoutError:
                    ws_task.cancel()
                    try:
                        await ws_task
                    except asyncio.CancelledError:
                        pass

        if data.get("isError"):
            raise RuntimeError(data["content"][0]["text"])

        # Parse text content back to structured data
        text = data["content"][0]["text"]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    async def _listen_progress(self, project: str, token: str, progress_cb):
        """Listen to SSE progress stream and forward to callback."""
        client = await self._ensure_client()
        url = f"{self.base_url}/projects/{project}/tools/call/progress"
        params = {"token": token}

        try:
            async with client.stream("GET", url, params=params, timeout=None) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: "
                        try:
                            event = json.loads(data_str)
                            if event.get("type") == "progress":
                                progress_cb(
                                    event.get("progress", 0),
                                    event.get("total", 0),
                                    event.get("message", "")
                                )
                            elif event.get("type") == "connected":
                                pass  # Connection established
                            elif event.get("type") == "keepalive":
                                pass  # Keepalive
                        except json.JSONDecodeError:
                            pass
        except httpx.HTTPStatusError:
            pass  # Progress endpoint not available or error
        except Exception:
            pass  # Ignore SSE errors

    async def _listen_progress_ws(self, project: str, token: str, progress_cb):
        """Listen to WebSocket progress stream and forward to callback."""
        if not WEBSOCKETS_AVAILABLE:
            return

        url = f"{self.base_url.replace('http', 'ws')}/projects/{project}/tools/call/progress/ws"
        params = f"?token={token}"

        try:
            async with websockets.connect(url + params, ping_interval=20, ping_timeout=10, close_timeout=5) as ws:
                # Wait for initial connected message
                try:
                    await asyncio.wait_for(ws.recv(), timeout=2.0)
                except asyncio.TimeoutError:
                    return  # No connected message, exit

                async for message in ws:
                    try:
                        event = json.loads(message)
                        if event.get("type") == "progress":
                            progress_cb(
                                event.get("progress", 0),
                                event.get("total", 0),
                                event.get("message", "")
                            )
                        elif event.get("type") == "keepalive":
                            pass
                    except json.JSONDecodeError:
                        pass
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception:
            pass

    def list_tools(self) -> list[dict[str, Any]]:
        """Sync wrapper."""
        return asyncio.run(self._list_tools_async())

    async def _list_tools_async(self) -> list[dict[str, Any]]:
        client = await self._ensure_client()
        project = self._resolve_project(None)

        # Use cache if available
        if self._tools_cache is not None and self._tools_cache_project == project:
            return self._tools_cache

        url = f"{self.base_url}/projects/{project}/tools/list"
        async with await self._ensure_client() as client:
            resp = await client.post(url)
            resp.raise_for_status()
            data = resp.json()

        self._tools_cache = data["tools"]
        self._tools_cache_project = project
        return self._tools_cache

    def get_tool(self, name: str) -> dict[str, Any] | None:
        tools = self.list_tools()
        for t in tools:
            if t["name"] == name:
                return t
        return None

    def register_tool(self, *args, **kwargs) -> None:
        """Not supported for daemon client."""
        raise NotImplementedError("Cannot register tools on daemon client")

    def execute_batch(
        self,
        calls: list[dict[str, Any]],
        project: str | None = None,
    ) -> list[Any]:
        """Execute multiple tool calls in a single request."""
        return asyncio.run(self._execute_batch_async(calls, project))

    async def _execute_batch_async(
        self,
        calls: list[dict[str, Any]],
        project: str | None = None,
    ) -> list[Any]:
        client = await self._ensure_client()
        project = self._resolve_project(project)

        # Convert to batch request format
        batch_payload = {"calls": calls}

        url = f"{self.base_url}/projects/{project}/tools/call/batch"
        resp = await client.post(url, json=batch_payload)
        resp.raise_for_status()
        data = resp.json()

        results = []
        for result in data["results"]:
            if result.get("isError"):
                raise RuntimeError(result["content"][0]["text"])
            text = result["content"][0]["text"]
            try:
                results.append(json.loads(text))
            except json.JSONDecodeError:
                results.append(text)
        return results

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            # Don't wait for aclose in sync context - just mark as closed
            # The async client will be garbage collected
            self._client = None

    async def aclose(self) -> None:
        """Async close."""
        if self._client:
            await self._client.aclose()
            self._client = None

    # Health/status methods
    def health(self) -> dict[str, Any]:
        return asyncio.run(self._health_async())

    async def _health_async(self) -> dict[str, Any]:
        client = await self._ensure_client()
        resp = await client.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def status(self) -> dict[str, Any]:
        return asyncio.run(self._status_async())

    async def _status_async(self) -> dict[str, Any]:
        client = await self._ensure_client()
        project = self._resolve_project(None)
        resp = await client.get(f"{self.base_url}/projects/{project}/status")
        resp.raise_for_status()
        return resp.json()

    def list_projects(self) -> list[dict[str, Any]]:
        return asyncio.run(self._list_projects_async())

    async def _list_projects_async(self) -> list[dict[str, Any]]:
        client = await self._ensure_client()
        resp = await client.get(f"{self.base_url}/projects")
        resp.raise_for_status()
        return resp.json()

    def register_project(
        self,
        name: str,
        db_path: str,
        config_path: str | None = None,
        auto_start: bool = True,
    ) -> dict[str, Any]:
        return asyncio.run(self._register_project_async(name, db_path, config_path, auto_start))

    async def _register_project_async(
        self,
        name: str,
        db_path: str,
        config_path: str | None,
        auto_start: bool,
    ) -> dict[str, Any]:
        client = await self._ensure_client()
        payload = {
            "name": name,
            "db_path": db_path,
            "config_path": config_path,
            "auto_start": auto_start,
        }
        resp = await client.post(f"{self.base_url}/projects", json=payload)
        resp.raise_for_status()
        return resp.json()

    def unregister_project(self, name: str) -> dict[str, Any]:
        return asyncio.run(self._unregister_project_async(name))

    def cross_project_query(
        self,
        query: str,
        projects: list[str] | None = None,
        top: int = 5,
    ) -> list[dict[str, Any]]:
        """Execute query across multiple projects."""
        return asyncio.run(self._cross_project_query_async(query, projects, top))

    async def _cross_project_query_async(
        self,
        query: str,
        projects: list[str] | None,
        top: int,
    ) -> list[dict[str, Any]]:
        client = await self._ensure_client()
        payload = {
            "name": "query",
            "arguments": {
                "query": query,
                "top": top,
            },
        }
        if projects:
            payload["arguments"]["projects"] = projects
        
        resp = await client.post(f"{self.base_url}/query/cross-project", json=payload)
        resp.raise_for_status()
        data = resp.json()

        if data.get("isError"):
            raise RuntimeError(data["content"][0]["text"])

        text = data["content"][0]["text"]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    def metrics(self) -> dict[str, Any]:
        """Get daemon metrics."""
        return asyncio.run(self._metrics_async())

    async def _metrics_async(self) -> dict[str, Any]:
        client = await self._ensure_client()
        resp = await client.get(f"{self.base_url}/metrics")
        resp.raise_for_status()
        return resp.json()

    def templates(self) -> list[dict[str, Any]]:
        """List available project templates."""
        return asyncio.run(self._templates_async())

    async def _templates_async(self) -> list[dict[str, Any]]:
        client = await self._ensure_client()
        resp = await client.get(f"{self.base_url}/templates")
        resp.raise_for_status()
        return resp.json()

    async def _unregister_project_async(self, name: str) -> dict[str, Any]:
        client = await self._ensure_client()
        resp = await client.delete(f"{self.base_url}/projects/{name}")
        resp.raise_for_status()
        return resp.json()

    def reload_config(self) -> dict[str, Any]:
        return asyncio.run(self._reload_config_async())

    async def _reload_config_async(self) -> dict[str, Any]:
        client = await self._ensure_client()
        resp = await client.post(f"{self.base_url}/daemon/reload")
        resp.raise_for_status()
        # Clear tools cache after reload
        self._tools_cache = None
        self._tools_cache_project = None
        return resp.json()


# Convenience function to create client from config/environment
def create_daemon_client(
    daemon_url: str | None = None,
    project: str | None = None,
    timeout: float = 30.0,
    api_key: str | None = None,
) -> DaemonClient:
    """Create a DaemonClient from URL or environment."""
    url = daemon_url or os.environ.get("AI_DB_DAEMON_URL")
    if not url:
        raise ValueError("No daemon URL provided. Use --daemon-url or set AI_DB_DAEMON_URL")
    key = api_key or os.environ.get("AI_DB_API_KEY")
    return DaemonClient(url, default_project=project, timeout=timeout, api_key=key)