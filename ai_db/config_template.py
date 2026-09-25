"""The template ``ai-db init`` writes. The only place defaults exist."""

from __future__ import annotations

import copy
from typing import Any

from ai_db.config import (
    CONFIG_VERSION,
    DEFAULT_DOC_WEIGHT,
    DEFAULT_INDEX_IGNORE,
    DEFAULT_RERANK_TOP_N,
    parse_config,
)
from ai_db.device import preferred_device
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
        # Pick the GPU when one is actually usable, else CPU. torch is imported
        # lazily inside preferred_device() and stays optional.
        if "device" in emb:
            emb["device"] = preferred_device()

    rr: dict[str, Any] = {"provider": rerank}
    if rerank != "none":
        if rerank not in SUGGESTED_RERANK:
            raise ValueError(f"no template for rerank provider '{rerank}'")
        rr.update(SUGGESTED_RERANK[rerank])
        rr["_note"] = MODEL_NOTE
        if "device" in rr:
            rr["device"] = preferred_device()

    options: dict[str, Any] = {"path": None} if storage == "sqlite" else {}
    if storage == "sqlite":
        # TODO 13.2: "exact" = brute-force cosine (default, always correct);
        # "vec0" = sqlite-vec ANN index, much faster at scale.
        options["vector_index"] = "exact"
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
    """Convert an unversioned pre-v1 config to the current version.

    Pre-v1 files used ``cross_project_access`` / ``auto_sync_paths`` at the top
    level and had no ``version`` key.
    """
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


def migrate_v1(data: dict[str, Any]) -> dict[str, Any]:
    """Convert a v1 config to v2 by adding the keys v2 introduced.

    v2 added ``retrieval.doc_weight``, ``rerank.top_n``, ``index.doc_weight``,
    ``index.ignore`` and the ``trace`` section. Existing values are preserved;
    new keys take their template defaults.
    """
    version = data.get("version")
    if version != 1:
        raise ValueError(f"migrate_v1 expects version 1, got {version!r}")

    out = copy.deepcopy(data)
    out["version"] = CONFIG_VERSION

    # retrieval.doc_weight
    retrieval = out.setdefault("retrieval", {})
    retrieval.setdefault("doc_weight", DEFAULT_DOC_WEIGHT)

    # rerank.top_n (v1 had no top_n; template default applies)
    rerank = out.setdefault("rerank", {})
    rerank.setdefault("top_n", DEFAULT_RERANK_TOP_N)

    # index.ignore / index.doc_weight
    index = out.setdefault("index", {})
    index.setdefault("doc_weight", DEFAULT_DOC_WEIGHT)
    index.setdefault("ignore", list(DEFAULT_INDEX_IGNORE))

    # trace section
    trace = out.setdefault("trace", {})
    trace.setdefault("wait_patterns", None)
    trace.setdefault("wait_patterns_extend", None)

    parse_config(out, check_env=False)
    return out


def migrate(data: dict[str, Any]) -> dict[str, Any]:
    """Upgrade any supported config to :data:`CONFIG_VERSION`.

    Dispatches on the file's own ``version`` so ``ai-db init --migrate`` works
    for unversioned, v1 and already-current files.
    """
    version = data.get("version")
    if version is None:
        return migrate_legacy(data)
    if version == 1:
        return migrate_v1(data)
    if version == CONFIG_VERSION:
        # already current: validate and hand it back
        parse_config(data, check_env=False)
        return copy.deepcopy(data)
    raise ValueError(
        f"cannot migrate from version {version!r}; this ai-db supports "
        f"versions {None!r} (unversioned), 1 and {CONFIG_VERSION}"
    )
