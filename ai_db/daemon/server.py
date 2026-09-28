"""FastAPI daemon server with per-project VectorDB pool."""

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ai_db import VectorDB, load_config
from ai_db.config import AppConfig
from ai_db.constants import DEFAULT_DB_FILE
from ai_db.daemon.config import DaemonConfig, DaemonProjectConfig
from ai_db.daemon.models import (
    DaemonHealth,
    ListToolsResponse,
    ProjectResponse,
    ProjectStatus,
    RegisterProjectRequest,
    ToolCallRequest,
    ToolCallResponse,
    ToolDefinition,
)
from ai_db.dispatcher import ServiceDispatcher


class ProjectPool:
    """Manages lazy-loaded VectorDB instances per project."""

    def __init__(self, global_config: AppConfig, daemon_config: DaemonConfig):
        self.global_config = global_config
        self.daemon_config = daemon_config
        self.projects: dict[str, VectorDB] = {}
        self.project_configs: dict[str, AppConfig] = {}
        self.dispatchers: dict[str, ServiceDispatcher] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._start_time = time.time()
        # Progress callbacks: token -> callback
        self._progress_callbacks: dict[str, asyncio.Queue] = {}

    def _get_lock(self, name: str) -> asyncio.Lock:
        if name not in self._locks:
            self._locks[name] = asyncio.Lock()
        return self._locks[name]

    def register_progress_callback(self, token: str) -> asyncio.Queue:
        """Register a progress callback for a token, returns a queue to consume progress."""
        queue: asyncio.Queue = asyncio.Queue()
        self._progress_callbacks[token] = queue
        return queue

    def unregister_progress_callback(self, token: str) -> None:
        """Unregister a progress callback."""
        self._progress_callbacks.pop(token, None)

    def get_progress_callback(self, token: str):
        """Get a progress callback function for a token."""
        queue = self._progress_callbacks.get(token)
        if queue is None:
            return None

        async def report(done: int, total: int, message: str = "") -> None:
            await queue.put({"progressToken": token, "progress": done, "total": total, "message": message})

        return report

    async def get_or_create(self, name: str) -> tuple[VectorDB, ServiceDispatcher]:
        """Get or create VectorDB and dispatcher for a project."""
        lock = self._get_lock(name)
        async with lock:
            if name in self.projects:
                return self.projects[name], self.dispatchers[name]

            # Determine project config
            if name in self.project_configs:
                cfg = self.project_configs[name]
            elif name in self.daemon_config.projects:
                proj_cfg = self.daemon_config.projects[name]
                if proj_cfg.config_path:
                    cfg = load_config(proj_cfg.config_path)
                else:
                    cfg = self.global_config
            else:
                cfg = self.global_config

            # Determine db_path
            if name in self.daemon_config.projects:
                db_path = os.path.expanduser(self.daemon_config.projects[name].db_path)
            else:
                # Auto-discovered project - use default DB path
                db_path = cfg.storage.options.get("path", DEFAULT_DB_FILE)
                db_path = os.path.expanduser(db_path)

            # Create VectorDB and dispatcher
            vectordb = VectorDB(db_path, config=cfg)
            dispatcher = ServiceDispatcher(db=vectordb, config=cfg)

            self.projects[name] = vectordb
            self.project_configs[name] = cfg
            self.dispatchers[name] = dispatcher

            return vectordb, dispatcher

    async def remove(self, name: str) -> None:
        """Remove a project from the pool."""
        lock = self._get_lock(name)
        async with lock:
            if name in self.projects:
                self.projects[name].close()
                del self.projects[name]
            if name in self.project_configs:
                del self.project_configs[name]
            if name in self.dispatchers:
                del self.dispatchers[name]
            if name in self._locks:
                del self._locks[name]

    async def list_projects(self) -> dict[str, ProjectStatus]:
        """Get status of all projects (loaded and registered)."""
        result = {}
        # First, add loaded projects
        for name, db in self.projects.items():
            try:
                status = db.status()
                result[name] = ProjectStatus(
                    name=name,
                    db_path=db.db_path,
                    chunks=status.get("chunks", 0),
                    files=status.get("files", 0),
                    symbols=status.get("symbols", 0),
                    syntax_errors=status.get("syntax_errors", 0),
                    size_kb=status.get("kb", 0),
                    loaded=True,
                )
            except Exception:
                result[name] = ProjectStatus(
                    name=name,
                    db_path=db.db_path,
                    loaded=False,
                )
        # Then, add registered but unloaded projects
        for name, proj_cfg in self.daemon_config.projects.items():
            if name not in result:
                result[name] = ProjectStatus(
                    name=name,
                    db_path=os.path.expanduser(proj_cfg.db_path),
                    loaded=False,
                )
        return result

    async def shutdown(self) -> None:
        """Close all projects."""
        for name in list(self.projects.keys()):
            await self.remove(name)


# Global pool instance
_pool: ProjectPool | None = None


def get_pool() -> ProjectPool:
    if _pool is None:
        raise RuntimeError("Daemon not initialized")
    return _pool


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pool
    # Startup
    from ai_db.config import load_config as load_main_config
    from ai_db.daemon.config import DaemonConfig as InternalDaemonConfig

    config = load_main_config()
    daemon_cfg = InternalDaemonConfig.from_dict(config.daemon.to_dict())
    _pool = ProjectPool(config, daemon_cfg)

    # Auto-start configured projects
    for name, proj_cfg in daemon_cfg.projects.items():
        if proj_cfg.auto_start:
            try:
                await _pool.get_or_create(name)
            except Exception as e:
                print(f"[daemon] Failed to auto-start project '{name}': {e}")

    # Auto-discover projects from auto_sync_paths
    if daemon_cfg.auto_discover:
        for path in config.auto_sync_paths:
            expanded = os.path.abspath(os.path.expanduser(path))
            if os.path.exists(expanded):
                proj_name = os.path.basename(expanded)
                if proj_name not in _pool.projects and proj_name not in daemon_cfg.projects:
                    try:
                        await _pool.get_or_create(proj_name)
                    except Exception as e:
                        print(f"[daemon] Failed to auto-discover project '{proj_name}': {e}")

    yield

    # Shutdown
    if _pool:
        await _pool.shutdown()
        _pool = None


app = FastAPI(title="ai-db Daemon", lifespan=lifespan)


@app.get("/health", response_model=DaemonHealth)
async def health():
    pool = get_pool()
    projects = await pool.list_projects()
    return DaemonHealth(
        status="ok",
        projects=projects,
        uptime_seconds=time.time() - pool._start_time,
    )


@app.get("/projects", response_model=list[ProjectResponse])
async def list_projects():
    pool = get_pool()
    result = []
    for name in set(list(pool.projects.keys()) + list(pool.daemon_config.projects.keys())):
        proj_cfg = pool.daemon_config.projects.get(name)
        if proj_cfg:
            result.append(ProjectResponse(
                name=name,
                db_path=proj_cfg.db_path,
                config_path=proj_cfg.config_path,
                auto_start=proj_cfg.auto_start,
            ))
        else:
            # Auto-discovered
            result.append(ProjectResponse(
                name=name,
                db_path=pool.projects[name].db_path,
                config_path=None,
                auto_start=True,
            ))
    return result


@app.post("/projects", response_model=ProjectResponse)
async def register_project(req: RegisterProjectRequest):
    pool = get_pool()

    # Validate project name
    if not req.name or not req.name.replace("-", "").replace("_", "").isalnum():
        raise HTTPException(400, "Project name must be alphanumeric with dashes/underscores")

    # Check if already registered
    if req.name in pool.projects or req.name in pool.daemon_config.projects:
        raise HTTPException(409, f"Project '{req.name}' already registered")

    # Load config if provided
    cfg = pool.global_config
    if req.config_path:
        cfg = load_config(req.config_path)

    # Create project config
    proj_cfg = DaemonProjectConfig(
        db_path=req.db_path,
        config_path=req.config_path,
        auto_start=req.auto_start,
    )
    pool.daemon_config.projects[req.name] = proj_cfg
    pool.project_configs[req.name] = cfg

    # Create if auto_start
    if req.auto_start:
        await pool.get_or_create(req.name)

    return ProjectResponse(
        name=req.name,
        db_path=req.db_path,
        config_path=req.config_path,
        auto_start=req.auto_start,
    )


@app.delete("/projects/{name}")
async def unregister_project(name: str):
    pool = get_pool()

    if name not in pool.projects and name not in pool.daemon_config.projects:
        raise HTTPException(404, f"Project '{name}' not found")

    await pool.remove(name)

    if name in pool.daemon_config.projects:
        del pool.daemon_config.projects[name]
    if name in pool.project_configs:
        del pool.project_configs[name]

    return {"status": "ok", "message": f"Project '{name}' unregistered"}


@app.get("/projects/{name}/status", response_model=ProjectStatus)
async def project_status(name: str):
    pool = get_pool()
    projects = await pool.list_projects()
    if name not in projects:
        raise HTTPException(404, f"Project '{name}' not found")
    return projects[name]


@app.post("/projects/{name}/tools/list", response_model=ListToolsResponse)
async def list_tools(name: str):
    pool = get_pool()
    _, dispatcher = await pool.get_or_create(name)
    raw_tools = dispatcher.list_tools()
    tools = []
    for t in raw_tools:
        td = dict(t)
        if "inputSchema" not in td and "parameters_schema" in td:
            td["inputSchema"] = td["parameters_schema"]
        tools.append(ToolDefinition(**td))
    return ListToolsResponse(tools=tools)


@app.post("/projects/{name}/tools/call", response_model=ToolCallResponse)
async def call_tool(name: str, req: ToolCallRequest):
    pool = get_pool()
    _, dispatcher = await pool.get_or_create(name)

    try:
        # Add project to arguments if not present
        arguments = dict(req.arguments)
        if "project" not in arguments:
            arguments["project"] = name

        # Execute tool with progress support
        progress = None
        if req._meta and "progressToken" in req._meta:
            token = req._meta["progressToken"]
            progress = pool.get_progress_callback(token)

        result = dispatcher.execute(req.name, arguments, progress=progress)

        # Format result as text content
        if isinstance(result, str):
            text = result
        else:
            import json
            text = json.dumps(result, indent=2)

        return ToolCallResponse(
            content=[{"type": "text", "text": text}],
            isError=False,
        )
    except KeyError as e:
        return ToolCallResponse(
            content=[{"type": "text", "text": f"Unknown tool: {e}"}],
            isError=True,
        )
    except Exception as e:
        return ToolCallResponse(
            content=[{"type": "text", "text": f"Error executing tool '{req.name}': {e!s}"}],
            isError=True,
        )


@app.get("/projects/{name}/tools/call/progress")
async def tool_progress(name: str, token: str):
    """Server-Sent Events endpoint for tool progress updates."""
    pool = get_pool()

    # Ensure project exists
    await pool.get_or_create(name)

    # Register progress callback
    queue = pool.register_progress_callback(token)

    async def event_generator():
        try:
            # Send initial connected event
            yield f"data: {json.dumps({'type': 'connected', 'token': token})}\n\n"

            while True:
                # Wait for progress update
                try:
                    progress_data = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {json.dumps({'type': 'progress', **progress_data})}\n\n"
                    # If progress is complete (done == total), break
                    if progress_data.get("progress") == progress_data.get("total") and progress_data.get("total", 0) > 0:
                        break
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            pool.unregister_progress_callback(token)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/daemon/reload")
async def reload_config():
    """Reload configuration and re-register projects."""
    global _pool
    if _pool is None:
        raise HTTPException(500, "Daemon not initialized")

    # Reload main config
    new_config = load_config()
    old_pool = _pool

    # Create new pool
    _pool = ProjectPool(new_config, new_config.daemon)

    # Migrate running projects
    for name in old_pool.projects:
        if name in new_config.daemon.projects:
            try:
                await _pool.get_or_create(name)
            except Exception as e:
                print(f"[daemon] Failed to migrate project '{name}': {e}")

    # Shutdown old pool
    await old_pool.shutdown()

    return {"status": "ok", "message": "Configuration reloaded"}


@app.post("/daemon/shutdown")
async def shutdown():
    """Graceful shutdown."""
    import os
    import signal

    # Schedule shutdown after response
    async def do_shutdown():
        await asyncio.sleep(0.1)
        os.kill(os.getpid(), signal.SIGTERM)

    asyncio.create_task(do_shutdown())
    return {"status": "ok", "message": "Shutting down..."}


# Exception handlers
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {exc!s}"},
    )