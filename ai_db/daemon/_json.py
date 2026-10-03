"""Typed validation helpers for JSON at the daemon HTTP boundary."""

from typing import Any

from ai_db.errors import AiDbProtocolError


def as_dict(value: Any, what: str = "response") -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AiDbProtocolError(f"Expected JSON object for {what}, got {type(value).__name__}")
    return value


def as_list(value: Any, what: str = "response") -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise AiDbProtocolError(f"Expected JSON array for {what}, got {type(value).__name__}")
    return value
