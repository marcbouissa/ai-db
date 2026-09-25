"""The template ``ai-db init`` writes. The only place defaults exist."""

from __future__ import annotations

from typing import Any

from ai_db.config import CONFIG_VERSION, parse_config
from ai_db.embed.defaults import MODEL_NOTE, SUGGESTED_EMBEDDING, SUGGESTED_RERANK


def build_template(
    storage: str = "sqlite",
    embedding: str = "none",
    rerank: str = "none",
    mode: str | None = None,
) -> dict[str, Any]:
    """Return a validated config dict. ``mode`` is derived from ``embedding`` if omitted."""
    if mode is None:
        mode = "lexical" if embedding == "none" else "hybrid"

    emb: dict[str, Any] = {"provider": embedding}
    if embedding != "none":
        if embedding not in SUGGESTED_EMBEDDING:
            raise ValueError(f"no template for embedding provider '{embedding}'")
        emb.update(SUGGESTED_EMBEDDING[embedding])
        emb["_note"] = MODEL_NOTE

    rr: dict[str, Any] = {"provider": rerank}
    if rerank != "none":
        if rerank not in SUGGESTED_RERANK:
            raise ValueError(f"no template for rerank provider '{rerank}'")
        rr.update(SUGGESTED_RERANK[rerank])
        rr["_note"] = MODEL_NOTE

    options: dict[str, Any] = {"path": None} if storage == "sqlite" else {}
    data: dict[str, Any] = {
        "version": CONFIG_VERSION,
        "storage": {"provider": storage, "options": options},
        "retrieval": {"mode": mode},
        "embedding": emb,
        "rerank": rr,
        "access": {"cross_project": {}},
        "auto_sync_paths": [],
        "trace": {"wait_patterns": None, "wait_patterns_extend": None},
    }
    parse_config(data, check_env=False)
    return data


def migrate_legacy(old: dict[str, Any]) -> dict[str, Any]:
    """Convert a pre-v1 config (``cross_project_access`` / ``auto_sync_paths``) to v1."""
    if "version" in old:
        raise ValueError("config already has a 'version'; nothing to migrate")
    known = {"cross_project_access", "auto_sync_paths"}
    unknown = set(old) - known
    if unknown:
        raise ValueError(f"cannot migrate unknown legacy key(s): {sorted(unknown)}")
    data = build_template()
    cross = old.get("cross_project_access", {})
    data["access"]["cross_project"] = {
        k: ([v] if isinstance(v, str) else list(v)) for k, v in cross.items()
    }
    data["auto_sync_paths"] = list(old.get("auto_sync_paths", []))
    parse_config(data, check_env=False)
    return data
