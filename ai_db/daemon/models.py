"""Pydantic models for daemon API requests/responses."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolCallRequest(BaseModel):
    # MCP sends the progress token as "_meta"; a leading underscore makes a pydantic
    # field private (silently dropped), so it is exposed as ``meta`` via an alias.
    model_config = ConfigDict(populate_by_name=True)

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    meta: dict[str, Any] | None = Field(default=None, alias="_meta")


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


class ErrorResponse(BaseModel):
    detail: str


class MetricsResponse(BaseModel):
    global_requests: int = 0
    global_errors: int = 0
    global_avg_latency_ms: float = 0.0
    projects: dict[str, ProjectMetricsResponse]
    uptime_seconds: float