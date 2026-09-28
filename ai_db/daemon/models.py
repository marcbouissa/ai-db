"""Pydantic models for daemon API requests/responses."""

from typing import Any

from pydantic import BaseModel, Field


class ToolCallRequest(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    _meta: dict[str, Any] | None = None


class ToolCallResponse(BaseModel):
    content: list[dict[str, str]]
    isError: bool = False


class BatchToolCallRequest(BaseModel):
    calls: list[ToolCallRequest]


class BatchToolCallResponse(BaseModel):
    results: list[ToolCallResponse]


class ToolDefinition(BaseModel):
    name: str
    description: str
    inputSchema: dict[str, Any]
    parameters_schema: dict[str, Any] | None = None
    category: str = "general"


class ListToolsResponse(BaseModel):
    tools: list[ToolDefinition]


class ProjectStatus(BaseModel):
    name: str
    db_path: str
    chunks: int = 0
    files: int = 0
    symbols: int = 0
    syntax_errors: int = 0
    size_kb: int = 0
    loaded: bool = True


class ProjectMetricsResponse(BaseModel):
    requests_total: int = 0
    requests_errors: int = 0
    avg_latency_ms: float = 0.0
    model_load_time_ms: float = 0.0
    last_access: float = 0.0
    warmed: bool = False


class DaemonHealth(BaseModel):
    status: str
    projects: dict[str, ProjectStatus]
    uptime_seconds: float


class RegisterProjectRequest(BaseModel):
    name: str
    db_path: str
    config_path: str | None = None
    auto_start: bool = True


class ProjectResponse(BaseModel):
    name: str
    db_path: str
    config_path: str | None = None
    auto_start: bool = True


class ProjectTemplate(BaseModel):
    name: str
    description: str
    config: dict[str, Any]


class ErrorResponse(BaseModel):
    detail: str


class MetricsResponse(BaseModel):
    global_requests: int = 0
    global_errors: int = 0
    global_avg_latency_ms: float = 0.0
    projects: dict[str, ProjectMetricsResponse]
    uptime_seconds: float