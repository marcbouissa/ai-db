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