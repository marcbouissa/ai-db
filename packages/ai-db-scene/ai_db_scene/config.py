"""The scene plugin's settings: the ``plugins.scene`` section of the ai-db config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai_db.errors import AiDbConfigError
from ai_db_scene.constants import DEFAULT_JUDGE_MODEL

_KEYS = frozenset({"judge_model"})


@dataclass(frozen=True)
class SceneConfig:
    judge_model: str = DEFAULT_JUDGE_MODEL


def scene_config(app_config: Any) -> SceneConfig:
    """Validate and return ``plugins.scene`` of an ``AppConfig``."""
    raw: dict[str, Any] = app_config.plugins.get("scene", {})
    unknown = set(raw) - _KEYS
    if unknown:
        raise AiDbConfigError(f"unknown keys in 'plugins.scene': {sorted(unknown)}")
    judge_model = raw.get("judge_model", DEFAULT_JUDGE_MODEL)
    if not isinstance(judge_model, str) or not judge_model:
        raise AiDbConfigError("'plugins.scene.judge_model' must be a non-empty string")
    return SceneConfig(judge_model=judge_model)
