"""The scene plugin's settings: the ``plugins.scene`` section of the ai-db config."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from ai_db.errors import AiDbConfigError
from ai_db_scene.constants import DEFAULT_JUDGE_MODEL, POLYHAVEN_RESOLUTIONS

_KEYS = frozenset({"judge_model", "asset_dirs", "polyhaven", "export_dir", "aliases"})
_POLYHAVEN_KEYS = frozenset({"enabled", "license_acknowledged", "cache_dir", "resolution"})


@dataclass(frozen=True)
class PolyHavenConfig:
    """Poly Haven asset downloads. Off unless enabled *and* the licence is acknowledged: the
    assets are CC0, but using the API commercially needs an agreement with Poly Haven."""

    enabled: bool = False
    license_acknowledged: bool = False
    cache_dir: str | None = None  # None: $XDG_CACHE_HOME/ai-db/polyhaven
    resolution: str = "1k"


@dataclass(frozen=True)
class SceneConfig:
    judge_model: str = DEFAULT_JUDGE_MODEL
    #: folders of local .glb/.gltf/.blend assets used before part recipes (none by default)
    asset_dirs: tuple[str, ...] = ()
    polyhaven: PolyHavenConfig = field(default_factory=PolyHavenConfig)
    #: the only folder export_scene writes into (None: $AI_DB_SCENE_EXPORT_DIR, else SCENE_EXPORT_DIR)
    export_dir: str | None = None
    #: local-library file-name words -> category, on top of providers.CATEGORY_ALIASES
    aliases: tuple[tuple[str, str], ...] = ()


def _aliases(raw: Any) -> tuple[tuple[str, str], ...]:
    where = "plugins.scene.aliases"
    if not isinstance(raw, dict) or not all(isinstance(k, str) and k.strip() and isinstance(v, str)
                                            for k, v in raw.items()):
        raise AiDbConfigError(f"'{where}' must map file-name words to a category, got {raw!r}")
    from ai_db_scene.priors import categories

    unknown = sorted({v for v in raw.values()} - set(categories()))
    if unknown:
        raise AiDbConfigError(f"'{where}': unknown categories {unknown}")
    return tuple(sorted(raw.items()))


def _polyhaven(raw: Any) -> PolyHavenConfig:
    where = "plugins.scene.polyhaven"
    if not isinstance(raw, dict):
        raise AiDbConfigError(f"'{where}' must be an object like {{\"enabled\": true, \"license_acknowledged\": true}}")
    unknown = set(raw) - _POLYHAVEN_KEYS
    if unknown:
        raise AiDbConfigError(f"unknown keys in '{where}': {sorted(unknown)}; allowed: {sorted(_POLYHAVEN_KEYS)}")
    for key in ("enabled", "license_acknowledged"):
        if key in raw and not isinstance(raw[key], bool):
            raise AiDbConfigError(f"'{where}.{key}' must be true or false, got {raw[key]!r}")
    enabled, ack = raw.get("enabled", False), raw.get("license_acknowledged", False)
    if enabled and not ack:
        raise AiDbConfigError(
            f"'{where}.enabled' needs '{where}.license_acknowledged': true. Poly Haven's assets are CC0, "
            "but commercial use of its API needs a licence from Poly Haven (https://polyhaven.com/our-api); "
            "set license_acknowledged once you have one, or use asset_dirs for local files")
    cache_dir = raw.get("cache_dir")
    if cache_dir is not None and (not isinstance(cache_dir, str) or not cache_dir):
        raise AiDbConfigError(f"'{where}.cache_dir' must be a non-empty path string, got {cache_dir!r}")
    resolution = raw.get("resolution", "1k")
    if resolution not in POLYHAVEN_RESOLUTIONS:
        raise AiDbConfigError(f"'{where}.resolution' must be one of {list(POLYHAVEN_RESOLUTIONS)}, got {resolution!r}")
    return PolyHavenConfig(enabled, ack, cache_dir, resolution)


def _asset_dirs(raw: Any) -> tuple[str, ...]:
    where = "plugins.scene.asset_dirs"
    if not isinstance(raw, list) or not all(isinstance(d, str) and d for d in raw):
        raise AiDbConfigError(f"'{where}' must be a list of folder paths, got {raw!r}")
    out = []
    for d in raw:
        path = os.path.abspath(os.path.expanduser(d))
        if not os.path.isdir(path):  # a typo would silently disable the library
            raise AiDbConfigError(f"'{where}': {d!r} is not a folder")
        out.append(path)
    return tuple(out)


def scene_config(app_config: Any) -> SceneConfig:
    """Validate and return ``plugins.scene`` of an ``AppConfig``."""
    raw: dict[str, Any] = app_config.plugins.get("scene", {})
    unknown = set(raw) - _KEYS
    if unknown:
        raise AiDbConfigError(f"unknown keys in 'plugins.scene': {sorted(unknown)}; allowed: {sorted(_KEYS)}")
    judge_model = raw.get("judge_model", DEFAULT_JUDGE_MODEL)
    if not isinstance(judge_model, str) or not judge_model:
        raise AiDbConfigError("'plugins.scene.judge_model' must be a non-empty string")
    export_dir = raw.get("export_dir")
    if export_dir is not None and (not isinstance(export_dir, str) or not export_dir):
        raise AiDbConfigError(f"'plugins.scene.export_dir' must be a non-empty path string, got {export_dir!r}")
    return SceneConfig(judge_model=judge_model, asset_dirs=_asset_dirs(raw.get("asset_dirs", [])),
                       polyhaven=_polyhaven(raw.get("polyhaven", {})), export_dir=export_dir,
                       aliases=_aliases(raw.get("aliases", {})))
