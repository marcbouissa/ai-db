"""FastAPI daemon: one long-lived process holding warm VectorDB instances per project.

Clients (``DaemonClient``, the CLI with ``--daemon-url``, the MCP server) call tools over
HTTP. Each project gets its own VectorDB + ServiceDispatcher, loaded on first use.

Concurrency model: the event loop never runs blocking work. Tool calls, project loading,
model warming and WAL checkpoints run in worker threads (``asyncio.to_thread``); calls to
one project are serialised by that project's execution lock, so ``/health`` and other
projects stay responsive while a long ``sync`` runs.

Auth (fail closed once configured): ``AI_DB_API_KEYS="proj=key,..."`` sets per-project
keys and ``AI_DB_ADMIN_KEY`` the admin key. With neither set the daemon is open (it binds
to localhost by default). Otherwise every route except ``/health`` needs
``Authorization: Bearer <key>``: project routes accept that project's key or the admin
key; management routes (``/projects``, ``/metrics``, ``/query/cross-project``,
``/daemon/*``) accept only the admin key.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import secrets
import signal
import sqlite3
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response, StreamingResponse

from ai_db import VectorDB, load_config
from ai_db.config import AppConfig, DaemonProjectConfig
from ai_db.constants import (
    DAEMON_CONFIG_POLL_S,
    DAEMON_IDLE_CHECK_S,
    DAEMON_IDLE_TTL_S,
    DAEMON_PROGRESS_KEEPALIVE_S,
    DAEMON_RATE_LIMIT,
    DAEMON_RATE_WINDOW_S,
    DAEMON_WAL_CHECKPOINT_S,
    DEFAULT_DB_FILE,
)
from ai_db.daemon.models import (
    BatchToolCallRequest,
    BatchToolCallResponse,
    DaemonHealth,
    ListToolsResponse,
    MetricsResponse,
    ProjectMetricsResponse,
    ProjectResponse,
    ProjectStatus,
    RegisterProjectRequest,
    ToolCallRequest,
    ToolCallResponse,
    ToolDefinition,
)
from ai_db.dispatcher import ServiceDispatcher
from ai_db.errors import AiDbError, UnknownToolError
from ai_db.lazy import release_models
from ai_db.storage.sqlite_backend import SQLiteBackend

_logger = logging.getLogger("ai_db.daemon")

# /projects/{name}/status and /projects/{name}/tools/... are per-project routes; every
# other route except /health is a management route.
_PROJECT_ROUTE = re.compile(r"^/projects/(?P<name>[^/]+)/(?:status|tools(?:/.*)?)$")
_PROJECT_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


class UnknownProjectError(AiDbError):
    """The project is neither registered nor auto-discoverable."""


@dataclass
class ProjectMetrics:
    """Metrics for a single project."""
    requests_total: int = 0
    requests_errors: int = 0
    latency_sum_ms: float = 0.0
    last_access: float = 0.0
    model_load_time_ms: float = 0.0


# =============================================================================
# Project pool
# =============================================================================


class ProjectPool:
    """Lazily loaded VectorDB + dispatcher per project, with progress plumbing."""

    def __init__(self, config: AppConfig,
                 runtime_projects: dict[str, DaemonProjectConfig] | None = None):
        self.global_config = config
        # Registered through POST /projects; kept across config reloads.
        self.runtime_projects: dict[str, DaemonProjectConfig] = dict(runtime_projects or {})
        self.registered: dict[str, DaemonProjectConfig] = {
            **config.daemon.projects, **self.runtime_projects}
        self.projects: dict[str, VectorDB] = {}
        self.project_configs: dict[str, AppConfig] = {}
        self.dispatchers: dict[str, ServiceDispatcher] = {}
        # Locks are never deleted: a waiter and a newcomer must contend on the same one.
        self._load_locks: dict[str, asyncio.Lock] = {}
        self._exec_locks: dict[str, asyncio.Lock] = {}
        self._progress: dict[str, asyncio.Queue[dict[str, Any]]] = {}
        self._listeners: set[str] = set()
        self.metrics: dict[str, ProjectMetrics] = {}
        self.warmed: set[str] = set()
        self.start_time = time.time()

    def _load_lock(self, name: str) -> asyncio.Lock:
        return self._load_locks.setdefault(name, asyncio.Lock())

    def _exec_lock(self, name: str) -> asyncio.Lock:
        return self._exec_locks.setdefault(name, asyncio.Lock())

    def _metrics(self, name: str) -> ProjectMetrics:
        return self.metrics.setdefault(name, ProjectMetrics())

    def is_known(self, name: str) -> bool:
        return (name in self.registered or name in self.projects
                or self.global_config.daemon.auto_discover)

    # --- loading ---------------------------------------------------------------

    def _project_config(self, name: str) -> tuple[AppConfig, str]:
        proj = self.registered.get(name)
        if proj is not None:
            cfg = load_config(proj.config_path) if proj.config_path else self.global_config
            return cfg, os.path.expanduser(proj.db_path)
        raw_path = self.global_config.storage.options.get("path")
        return self.global_config, os.path.expanduser(raw_path) if raw_path else DEFAULT_DB_FILE

    async def get_or_create(self, name: str) -> tuple[VectorDB, ServiceDispatcher]:
        if not _PROJECT_NAME.match(name):
            raise UnknownProjectError(f"invalid project name {name!r}")
        if not self.is_known(name):
            raise UnknownProjectError(f"project '{name}' is not registered (auto_discover is off)")
        async with self._load_lock(name):
            if name not in self.projects:
                cfg, db_path = self._project_config(name)
                vectordb = await asyncio.to_thread(VectorDB, db_path, config=cfg)
                self.projects[name] = vectordb
                self.project_configs[name] = cfg
                self.dispatchers[name] = ServiceDispatcher(db=vectordb, config=cfg)
            self._metrics(name).last_access = time.time()
            return self.projects[name], self.dispatchers[name]

    async def warm_project(self, name: str) -> None:
        """Pre-load the embedding/rerank models so the first query is fast."""
        db, _ = await self.get_or_create(name)
        async with self._exec_lock(name):
            if name in self.warmed:
                return
            start = time.time()
            await asyncio.to_thread(db._ensure_embedder_loaded)
            await asyncio.to_thread(db._ensure_reranker_loaded)
            self._metrics(name).model_load_time_ms = (time.time() - start) * 1000
            self.warmed.add(name)
            _logger.info("[daemon] warmed project '%s' in %.0fms", name,
                         self._metrics(name).model_load_time_ms)

    async def remove(self, name: str) -> None:
        """Close and forget a loaded project (waits for its in-flight call)."""
        async with self._load_lock(name), self._exec_lock(name):
            db = self.projects.pop(name, None)
            self.project_configs.pop(name, None)
            self.dispatchers.pop(name, None)
            self.warmed.discard(name)
            if db is not None:
                await asyncio.to_thread(db.close)

    async def shutdown(self) -> None:
        for name in list(self.projects):
            await self.remove(name)
        await asyncio.to_thread(release_models)

    async def cleanup_idle(self, ttl_seconds: float = DAEMON_IDLE_TTL_S) -> None:
        now = time.time()
        for name in list(self.projects):
            if now - self._metrics(name).last_access > ttl_seconds:
                _logger.info("[daemon] unloading idle project '%s'", name)
                await self.remove(name)
        if not self.projects:
            # Projects share one model per config (the provider registries cache
            # it process-wide); free it only once no project can still use it.
            released = await asyncio.to_thread(release_models)
            if released:
                _logger.info("[daemon] no project loaded: released %d model(s)", released)

    async def checkpoint_all(self) -> None:
        for name, db in list(self.projects.items()):
            if not isinstance(db.backend, SQLiteBackend):
                continue  # only SQLite has a WAL to checkpoint
            async with self._exec_lock(name):
                await asyncio.to_thread(db.backend.wal_checkpoint, "TRUNCATE")

    def list_projects(self) -> dict[str, ProjectStatus]:
        result: dict[str, ProjectStatus] = {}
        for name, db in self.projects.items():
            try:
                st = db.status()
            except (AiDbError, sqlite3.Error):
                _logger.exception("[daemon] status failed for '%s'", name)
                result[name] = ProjectStatus(name=name, db_path=db.db_path, loaded=False)
                continue
            result[name] = ProjectStatus(
                name=name, db_path=db.db_path, chunks=st.get("chunks", 0), files=st.get("files", 0),
                symbols=st.get("symbols", 0), syntax_errors=st.get("syntax_errors", 0),
                size_kb=int(st.get("kb", 0)), loaded=True)
        for name, proj in self.registered.items():
            result.setdefault(name, ProjectStatus(
                name=name, db_path=os.path.expanduser(proj.db_path), loaded=False))
        return result

    # --- progress --------------------------------------------------------------

    def progress_queue(self, token: str) -> asyncio.Queue[dict[str, Any]]:
        """Queue for ``token``, created by whichever side (call or listener) comes first."""
        return self._progress.setdefault(token, asyncio.Queue())

    def attach_listener(self, token: str) -> asyncio.Queue[dict[str, Any]]:
        self._listeners.add(token)
        return self.progress_queue(token)

    def detach_listener(self, token: str) -> None:
        self._listeners.discard(token)
        self._progress.pop(token, None)

    def _progress_sink(self, token: str,
                       loop: asyncio.AbstractEventLoop) -> Callable[..., None]:
        """A *sync* callback for the dispatcher, which runs in a worker thread."""
        queue = self.progress_queue(token)

        def report(done: int, total: int, message: str = "") -> None:
            event = {"type": "progress", "progressToken": token, "progress": done,
                     "total": total, "message": message}
            loop.call_soon_threadsafe(queue.put_nowait, event)

        return report

    def _finish_progress(self, token: str) -> None:
        queue = self._progress.get(token)
        if queue is not None:
            queue.put_nowait({"type": "done", "progressToken": token})
        if token not in self._listeners:
            self._progress.pop(token, None)  # nobody listening: do not leak the queue

    # --- tool calls ------------------------------------------------------------

    async def run_tool(self, name: str, tool: str, arguments: dict[str, Any],
                       progress_token: str | None = None) -> Any:
        _, dispatcher = await self.get_or_create(name)
        args = dict(arguments)
        args.setdefault("project", name)
        loop = asyncio.get_running_loop()
        progress = self._progress_sink(progress_token, loop) if progress_token else None
        m = self._metrics(name)
        start = time.time()
        try:
            async with self._exec_lock(name):
                result = await asyncio.to_thread(dispatcher.execute, tool, args, progress)
        except Exception:
            m.requests_errors += 1
            raise
        finally:
            if progress_token:
                self._finish_progress(progress_token)
            m.requests_total += 1
            m.latency_sum_ms += (time.time() - start) * 1000
            m.last_access = time.time()
        return result


# =============================================================================
# Daemon state, lifecycle and reload
# =============================================================================


@dataclass
class _State:
    pool: ProjectPool | None = None
    tasks: list[asyncio.Task[None]] = field(default_factory=list)
    project_keys: dict[str, str] = field(default_factory=dict)
    admin_key: str | None = None
    reload_lock: asyncio.Lock | None = None


_state = _State()


def get_pool() -> ProjectPool:
    if _state.pool is None:
        raise RuntimeError("Daemon not initialized")
    return _state.pool


def _load_auth() -> None:
    keys: dict[str, str] = {}
    for pair in os.environ.get("AI_DB_API_KEYS", "").split(","):
        if "=" in pair:
            project, key = pair.split("=", 1)
            keys[project.strip()] = key.strip()
    _state.project_keys = keys
    _state.admin_key = os.environ.get("AI_DB_ADMIN_KEY") or None


async def _start_projects(pool: ProjectPool, previously_loaded: set[str] | None = None) -> None:
    """Load auto_start projects, auto-discovered ones and those loaded before a reload."""
    names = {n for n, p in pool.registered.items() if p.auto_start}
    names |= {n for n in previously_loaded or set() if pool.is_known(n)}
    if pool.global_config.daemon.auto_discover:
        for path in pool.global_config.auto_sync_paths:
            expanded = os.path.abspath(os.path.expanduser(path))
            if os.path.isdir(expanded):
                names.add(os.path.basename(expanded))
    for name in sorted(names):
        try:
            await pool.warm_project(name)
        except Exception:  # one project failing must not stop the daemon
            _logger.exception("[daemon] failed to start project '%s'", name)


async def reload_daemon_config() -> None:
    """The one reload path (config watcher and POST /daemon/reload)."""
    if _state.reload_lock is None:
        _state.reload_lock = asyncio.Lock()
    async with _state.reload_lock:
        old = get_pool()
        new_config = await asyncio.to_thread(load_config)
        new = ProjectPool(new_config, runtime_projects=old.runtime_projects)
        await _start_projects(new, previously_loaded=set(old.projects))
        _state.pool = new
        await old.shutdown()
        _logger.info("[daemon] configuration reloaded from %s", new_config.source_path)


async def _every(interval: float, job: Callable[[], Awaitable[None]], what: str) -> None:
    while True:
        await asyncio.sleep(interval)
        try:
            await job()
        except Exception:  # a periodic job failing must not stop the daemon
            _logger.exception("[daemon] %s failed", what)


def _config_watcher(config_path: str | None) -> Callable[[], Awaitable[None]]:
    path = os.path.expanduser(config_path) if config_path else None
    last = {"mtime": os.path.getmtime(path) if path and os.path.exists(path) else None}

    async def check() -> None:
        if path is None or not os.path.exists(path):
            return
        mtime = os.path.getmtime(path)
        if mtime != last["mtime"]:
            last["mtime"] = mtime
            await reload_daemon_config()

    return check


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # The CLI exports AI_DB_CONFIG (incl. `ai-db --config X daemon start`), so the
    # daemon reads the same file the user pointed at.
    config = await asyncio.to_thread(load_config)
    _load_auth()
    _state.pool = ProjectPool(config)
    _state.reload_lock = asyncio.Lock()
    await _start_projects(_state.pool)
    _state.tasks = [
        asyncio.create_task(_every(DAEMON_CONFIG_POLL_S, _config_watcher(config.source_path),
                                   "config reload")),
        asyncio.create_task(_every(DAEMON_IDLE_CHECK_S, lambda: get_pool().cleanup_idle(),
                                   "idle cleanup")),
        asyncio.create_task(_every(DAEMON_WAL_CHECKPOINT_S, lambda: get_pool().checkpoint_all(),
                                   "WAL checkpoint")),
    ]
    try:
        yield
    finally:
        for task in _state.tasks:
            task.cancel()
        for task in _state.tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        _state.tasks = []
        if _state.pool is not None:
            await _state.pool.shutdown()
            _state.pool = None


# =============================================================================
# Auth, rate limiting, request log
# =============================================================================


def _auth_failure(path: str, authorization: str | None) -> tuple[int, str] | None:
    """``(status, detail)`` when the request is not allowed, else None."""
    if path == "/health" or (not _state.project_keys and not _state.admin_key):
        return None
    match = _PROJECT_ROUTE.match(path)
    accepted = [k for k in (_state.admin_key,) if k]
    if match:
        project_key = _state.project_keys.get(match["name"])
        if project_key:
            accepted.append(project_key)
    if not authorization or not authorization.startswith("Bearer "):
        return 401, "Missing or invalid Authorization header"
    token = authorization[len("Bearer "):]
    if not any(secrets.compare_digest(token, k) for k in accepted):
        return 403, "Invalid API key for this route"
    return None


class RateLimiter:
    def __init__(self) -> None:
        self._buckets: dict[str, list[float]] = {}

    def is_allowed(self, key: str, limit: int, window: float) -> bool:
        now = time.time()
        bucket = [t for t in self._buckets.get(key, []) if now - t < window]
        allowed = len(bucket) < limit
        if allowed:
            bucket.append(now)
        self._buckets[key] = bucket
        return allowed


rate_limiter = RateLimiter()


def _route_project(path: str) -> str:
    match = _PROJECT_ROUTE.match(path)
    return match["name"] if match else "-"


app = FastAPI(title="ai-db Daemon", lifespan=lifespan)


@app.middleware("http")
async def guard(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """Auth, then per-project rate limit, then a JSON access-log line."""
    start = time.time()
    path = request.url.path
    project = _route_project(path)
    client_ip = request.client.host if request.client else "unknown"
    failure = _auth_failure(path, request.headers.get("Authorization"))
    if failure is not None:
        response: Response = JSONResponse(status_code=failure[0], content={"detail": failure[1]})
    elif path != "/health" and not rate_limiter.is_allowed(
            f"{project}:{client_ip}", DAEMON_RATE_LIMIT, DAEMON_RATE_WINDOW_S):
        response = JSONResponse(status_code=429,
                                content={"detail": f"Rate limit exceeded for project '{project}'"})
    else:
        response = await call_next(request)
    _logger.info(json.dumps({
        "method": request.method, "path": path, "project": project, "client_ip": client_ip,
        "status_code": response.status_code, "duration_ms": round((time.time() - start) * 1000, 2),
    }))
    return response


@app.exception_handler(UnknownProjectError)
async def _unknown_project(request: Request, exc: UnknownProjectError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def _internal_error(request: Request, exc: Exception) -> JSONResponse:
    _logger.error("[daemon] unhandled error on %s", request.url.path, exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": f"Internal server error: {exc!s}"})


# =============================================================================
# Routes
# =============================================================================


def _text(result: Any) -> str:
    return result if isinstance(result, str) else json.dumps(result, indent=2)


async def _call(name: str, req: ToolCallRequest) -> ToolCallResponse:
    token = (req.meta or {}).get("progressToken")
    try:
        result = await get_pool().run_tool(name, req.name, req.arguments,
                                           str(token) if token is not None else None)
    except UnknownProjectError:
        raise
    except UnknownToolError as e:
        return ToolCallResponse(content=[{"type": "text", "text": f"Unknown tool: {e}"}], isError=True)
    except Exception as e:
        _logger.exception("[daemon] tool '%s' failed", req.name)
        return ToolCallResponse(
            content=[{"type": "text", "text": f"Error executing tool '{req.name}': {e!s}"}],
            isError=True)
    return ToolCallResponse(content=[{"type": "text", "text": _text(result)}], isError=False)


@app.get("/health", response_model=DaemonHealth)
async def health() -> DaemonHealth:
    pool = get_pool()
    projects = await asyncio.to_thread(pool.list_projects)
    return DaemonHealth(status="ok", projects=projects, uptime_seconds=time.time() - pool.start_time)


@app.get("/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    pool = get_pool()
    per_project = {
        name: ProjectMetricsResponse(
            requests_total=m.requests_total, requests_errors=m.requests_errors,
            avg_latency_ms=m.latency_sum_ms / m.requests_total if m.requests_total else 0.0,
            model_load_time_ms=m.model_load_time_ms, last_access=m.last_access,
            warmed=name in pool.warmed)
        for name, m in pool.metrics.items()
    }
    total = sum(m.requests_total for m in pool.metrics.values())
    latency = sum(m.latency_sum_ms for m in pool.metrics.values())
    return MetricsResponse(
        global_requests=total, global_errors=sum(m.requests_errors for m in pool.metrics.values()),
        global_avg_latency_ms=latency / total if total else 0.0, projects=per_project,
        uptime_seconds=time.time() - pool.start_time)


@app.get("/projects", response_model=list[ProjectResponse])
async def list_projects() -> list[ProjectResponse]:
    pool = get_pool()
    result = []
    for name in sorted(set(pool.projects) | set(pool.registered)):
        proj = pool.registered.get(name)
        if proj is not None:
            result.append(ProjectResponse(name=name, db_path=proj.db_path,
                                          config_path=proj.config_path, auto_start=proj.auto_start))
        else:  # auto-discovered
            result.append(ProjectResponse(name=name, db_path=pool.projects[name].db_path,
                                          config_path=None, auto_start=True))
    return result


@app.post("/projects", response_model=ProjectResponse)
async def register_project(req: RegisterProjectRequest) -> ProjectResponse:
    pool = get_pool()
    if not _PROJECT_NAME.match(req.name):
        raise HTTPException(400, "Project name must be alphanumeric with dashes/underscores")
    if req.name in pool.projects or req.name in pool.registered:
        raise HTTPException(409, f"Project '{req.name}' already registered")
    proj = DaemonProjectConfig(db_path=req.db_path, config_path=req.config_path,
                               auto_start=req.auto_start)
    pool.runtime_projects[req.name] = proj
    pool.registered[req.name] = proj
    if req.auto_start:
        await pool.get_or_create(req.name)
    return ProjectResponse(name=req.name, db_path=req.db_path, config_path=req.config_path,
                           auto_start=req.auto_start)


@app.delete("/projects/{name}")
async def unregister_project(name: str) -> dict[str, str]:
    pool = get_pool()
    if name not in pool.projects and name not in pool.registered:
        raise HTTPException(404, f"Project '{name}' not found")
    await pool.remove(name)
    pool.registered.pop(name, None)
    pool.runtime_projects.pop(name, None)
    return {"status": "ok", "message": f"Project '{name}' unregistered"}


@app.post("/query/cross-project", response_model=ToolCallResponse)
async def cross_project_query(req: ToolCallRequest) -> ToolCallResponse:
    """Run ``query`` in several projects (default: all loaded) and merge by score."""
    pool = get_pool()
    requested = req.arguments.get("projects")
    if requested is None:
        names = sorted(pool.projects)
    else:
        names = requested if isinstance(requested, list) else [requested]
    merged: list[dict[str, Any]] = []
    errors: dict[str, str] = {}
    for name in names:
        try:
            result = await pool.run_tool(name, "query", {
                "query": req.arguments.get("query", ""), "top": req.arguments.get("top", 5)})
        except Exception as e:
            _logger.exception("[daemon] cross-project query failed for '%s'", name)
            errors[name] = str(e)
            continue
        for row in result if isinstance(result, list) else []:
            merged.append({**row, "source_project": name})
    merged.sort(key=lambda r: r.get("score", 0), reverse=True)
    text = json.dumps({"results": merged, "errors": errors} if errors else merged, indent=2)
    return ToolCallResponse(content=[{"type": "text", "text": text}], isError=bool(errors) and not merged)


@app.get("/projects/{name}/status", response_model=ProjectStatus)
async def project_status(name: str) -> ProjectStatus:
    projects = await asyncio.to_thread(get_pool().list_projects)
    if name not in projects:
        raise HTTPException(404, f"Project '{name}' not found")
    return projects[name]


@app.post("/projects/{name}/tools/list", response_model=ListToolsResponse)
async def list_tools(name: str) -> ListToolsResponse:
    _, dispatcher = await get_pool().get_or_create(name)
    return ListToolsResponse(tools=[ToolDefinition(**t) for t in dispatcher.list_tools()])


@app.post("/projects/{name}/tools/call", response_model=ToolCallResponse)
async def call_tool(name: str, req: ToolCallRequest) -> ToolCallResponse:
    return await _call(name, req)


@app.post("/projects/{name}/tools/call/batch", response_model=BatchToolCallResponse)
async def call_tool_batch(name: str, req: BatchToolCallRequest) -> BatchToolCallResponse:
    return BatchToolCallResponse(results=[await _call(name, call) for call in req.calls])


async def _progress_events(token: str) -> AsyncIterator[dict[str, Any]]:
    """``connected``, then progress events until the call reports ``done``."""
    pool = get_pool()
    queue = pool.attach_listener(token)
    try:
        yield {"type": "connected", "token": token}
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=DAEMON_PROGRESS_KEEPALIVE_S)
            except asyncio.TimeoutError:
                yield {"type": "keepalive"}
                continue
            yield event
            if event.get("type") == "done":
                return
    finally:
        pool.detach_listener(token)


@app.get("/projects/{name}/tools/call/progress")
async def tool_progress(name: str, token: str) -> StreamingResponse:
    """Server-Sent Events stream of a tool call's progress."""
    await get_pool().get_or_create(name)

    async def sse() -> AsyncIterator[str]:
        async for event in _progress_events(token):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})


@app.websocket("/projects/{name}/tools/call/progress/ws")
async def tool_progress_ws(websocket: WebSocket, name: str, token: str) -> None:
    """WebSocket stream of a tool call's progress (HTTP middleware does not run here)."""
    failure = _auth_failure(f"/projects/{name}/tools/call/progress",
                            websocket.headers.get("Authorization"))
    if failure is not None:
        await websocket.close(code=1008, reason=failure[1])
        return
    await get_pool().get_or_create(name)
    await websocket.accept()
    try:
        async for event in _progress_events(token):
            await websocket.send_json(event)
    except WebSocketDisconnect:
        pass
    finally:
        with contextlib.suppress(RuntimeError):  # client already closed the socket
            await websocket.close()


@app.post("/daemon/reload")
async def reload_config() -> dict[str, str]:
    await reload_daemon_config()
    return {"status": "ok", "message": "Configuration reloaded"}


@app.post("/daemon/shutdown")
async def shutdown() -> dict[str, str]:
    """Graceful shutdown (SIGTERM to self after the response is sent)."""
    async def stop_soon() -> None:
        await asyncio.sleep(0.1)
        os.kill(os.getpid(), signal.SIGTERM)

    _state.tasks.append(asyncio.create_task(stop_soon()))
    return {"status": "ok", "message": "Shutting down..."}
