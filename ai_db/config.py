"""Explicit, validated configuration for ai-db.

Rules:
- A config file is mandatory. If it is missing, ``load_config`` raises and tells the
  user to run ``ai-db init``. There are no runtime defaults; defaults live only in
  ``ai_db.config_template`` which ``ai-db init`` writes to disk.
- Unknown keys, unknown versions and invalid combinations raise ``AiDbConfigError``.
- Environment overrides are applied here and nowhere else.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from ai_db.errors import AiDbConfigError

CONFIG_VERSION = 1

RETRIEVAL_MODES = ("lexical", "hybrid")

# provider -> (required option keys, optional option keys)
EMBEDDING_PROVIDERS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "none": (frozenset(), frozenset()),
    "sentence_transformers": (
        frozenset({"model", "device", "batch_size"}),
        frozenset({"query_prompt", "_note"}),
    ),
    "openai_compatible": (
        frozenset({"model", "base_url", "api_key_env", "dimensions", "batch_size"}),
        frozenset({"_note"}),
    ),
    "voyage": (
        frozenset({"model", "api_key_env", "dimensions", "batch_size"}),
        frozenset({"base_url", "_note"}),
    ),
}

RERANK_PROVIDERS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "none": (frozenset(), frozenset()),
    "sentence_transformers": (frozenset({"model", "device"}), frozenset({"_note"})),
    "voyage": (frozenset({"model", "api_key_env"}), frozenset({"base_url", "_note"})),
    "cohere": (frozenset({"model", "api_key_env"}), frozenset({"base_url", "_note"})),
}

TOP_LEVEL_KEYS = frozenset(
    {"version", "storage", "retrieval", "embedding", "rerank", "access", "auto_sync_paths"}
)
REQUIRED_TOP_LEVEL = frozenset({"version", "storage", "retrieval", "embedding", "rerank"})


@dataclass(frozen=True)
class StorageConfig:
    provider: str
    options: dict[str, Any]


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    options: dict[str, Any]

    @property
    def enabled(self) -> bool:
        return self.provider != "none"


@dataclass(frozen=True)
class AppConfig:
    version: int
    storage: StorageConfig
    retrieval_mode: str
    embedding: ProviderConfig
    rerank: ProviderConfig
    cross_project: dict[str, list[str]] = field(default_factory=dict)
    auto_sync_paths: list[str] = field(default_factory=list)
    source_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "storage": {"provider": self.storage.provider, "options": dict(self.storage.options)},
            "retrieval": {"mode": self.retrieval_mode},
            "embedding": {"provider": self.embedding.provider, **self.embedding.options},
            "rerank": {"provider": self.rerank.provider, **self.rerank.options},
            "access": {"cross_project": {k: list(v) for k, v in self.cross_project.items()}},
            "auto_sync_paths": list(self.auto_sync_paths),
        }


def config_path(explicit: str | None = None) -> str:
    """Resolve where the config file lives (location only; never a second config)."""
    if explicit:
        return os.path.abspath(os.path.expanduser(explicit))
    env = os.environ.get("AI_DB_CONFIG")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return os.path.join(xdg, "ai-db", "config.json")
    return os.path.expanduser("~/.config/ai-db/config.json")


def _require_dict(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AiDbConfigError(f"'{where}' must be an object")
    return value


def _check_keys(section: dict[str, Any], allowed: frozenset[str], where: str) -> None:
    unknown = set(section) - allowed
    if unknown:
        raise AiDbConfigError(f"unknown key(s) in '{where}': {sorted(unknown)}")


def _parse_provider(
    raw: Any, where: str, registry: dict[str, tuple[frozenset[str], frozenset[str]]]
) -> ProviderConfig:
    section = _require_dict(raw, where)
    provider = section.get("provider")
    if not isinstance(provider, str):
        raise AiDbConfigError(f"'{where}.provider' must be a string")
    options = {k: v for k, v in section.items() if k != "provider"}
    if provider in registry:
        required, optional = registry[provider]
        missing = required - set(options)
        if missing:
            raise AiDbConfigError(f"'{where}' provider '{provider}' requires {sorted(missing)}")
        _check_keys(options, required | optional, where)
        if "batch_size" in options and (
            not isinstance(options["batch_size"], int) or options["batch_size"] < 1
        ):
            raise AiDbConfigError(f"'{where}.batch_size' must be a positive integer")
        if "dimensions" in options and (
            not isinstance(options["dimensions"], int) or options["dimensions"] < 1
        ):
            raise AiDbConfigError(f"'{where}.dimensions' must be a positive integer")
    # Providers not in the registry are third-party entry points; their options are
    # validated by the plugin itself when it is constructed.
    return ProviderConfig(provider=provider, options=options)


def parse_config(raw: Any, check_env: bool = True, source_path: str | None = None) -> AppConfig:
    """Validate a raw JSON object and return a frozen AppConfig."""
    data = _require_dict(raw, "<root>")
    _check_keys(data, TOP_LEVEL_KEYS, "<root>")
    missing = REQUIRED_TOP_LEVEL - set(data)
    if missing:
        raise AiDbConfigError(f"config is missing required section(s): {sorted(missing)}")
    if data["version"] != CONFIG_VERSION:
        raise AiDbConfigError(
            f"unsupported config version {data['version']!r}; this ai-db supports "
            f"version {CONFIG_VERSION}. Run: ai-db init --force"
        )

    storage_raw = _require_dict(data["storage"], "storage")
    _check_keys(storage_raw, frozenset({"provider", "options"}), "storage")
    if not isinstance(storage_raw.get("provider"), str):
        raise AiDbConfigError("'storage.provider' must be a string")
    storage = StorageConfig(
        provider=storage_raw["provider"],
        options=dict(_require_dict(storage_raw.get("options", {}), "storage.options")),
    )

    retrieval_raw = _require_dict(data["retrieval"], "retrieval")
    _check_keys(retrieval_raw, frozenset({"mode"}), "retrieval")
    mode = os.environ.get("AI_DB_RETRIEVAL_MODE") or retrieval_raw.get("mode")
    if mode not in RETRIEVAL_MODES:
        raise AiDbConfigError(f"'retrieval.mode' must be one of {RETRIEVAL_MODES}, got {mode!r}")

    embedding = _parse_provider(data["embedding"], "embedding", EMBEDDING_PROVIDERS)
    rerank = _parse_provider(data["rerank"], "rerank", RERANK_PROVIDERS)

    device_override = os.environ.get("AI_DB_DEVICE")
    if device_override:
        if embedding.provider == "sentence_transformers":
            embedding = ProviderConfig(embedding.provider, {**embedding.options, "device": device_override})
        if rerank.provider == "sentence_transformers":
            rerank = ProviderConfig(rerank.provider, {**rerank.options, "device": device_override})

    if mode == "hybrid" and not embedding.enabled:
        raise AiDbConfigError("retrieval.mode 'hybrid' requires an embedding provider (embedding.provider != 'none')")
    if mode == "lexical" and embedding.enabled:
        raise AiDbConfigError("embedding provider is configured but retrieval.mode is 'lexical'; set mode to 'hybrid' or embedding.provider to 'none'")

    if check_env:
        for section_name, prov in (("embedding", embedding), ("rerank", rerank)):
            env_name = prov.options.get("api_key_env")
            if env_name and not os.environ.get(env_name):
                raise AiDbConfigError(
                    f"{section_name} provider '{prov.provider}' needs environment variable {env_name}"
                )

    access_raw = _require_dict(data.get("access", {}), "access")
    _check_keys(access_raw, frozenset({"cross_project"}), "access")
    cross_raw = _require_dict(access_raw.get("cross_project", {}), "access.cross_project")
    cross: dict[str, list[str]] = {}
    for proj, allowed in cross_raw.items():
        if isinstance(allowed, str):
            allowed = [allowed]
        if not isinstance(allowed, list) or not all(isinstance(a, str) for a in allowed):
            raise AiDbConfigError(f"'access.cross_project.{proj}' must be a list of strings")
        cross[proj] = [a.strip() for a in allowed if a.strip()]

    paths = data.get("auto_sync_paths", [])
    if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
        raise AiDbConfigError("'auto_sync_paths' must be a list of strings")

    return AppConfig(
        version=CONFIG_VERSION,
        storage=storage,
        retrieval_mode=mode,
        embedding=embedding,
        rerank=rerank,
        cross_project=cross,
        auto_sync_paths=list(paths),
        source_path=source_path,
    )


def load_config(path: str | None = None, check_env: bool = True) -> AppConfig:
    """Load and validate the config file. Raises if it does not exist."""
    resolved = config_path(path)
    if not os.path.isfile(resolved):
        raise AiDbConfigError(f"no config found at {resolved}; run: ai-db init")
    with open(resolved, encoding="utf-8") as f:
        try:
            raw = json.load(f)
        except json.JSONDecodeError as exc:
            raise AiDbConfigError(f"config at {resolved} is not valid JSON: {exc}") from exc
    return parse_config(raw, check_env=check_env, source_path=resolved)


def masked_dict(cfg: AppConfig) -> dict[str, Any]:
    """Config as a dict with secrets hidden (api keys are never stored, only env names)."""
    out = cfg.to_dict()
    for section in ("embedding", "rerank"):
        env_name = out[section].get("api_key_env")
        if env_name:
            out[section]["api_key"] = "set" if os.environ.get(env_name) else "MISSING"
    return out
