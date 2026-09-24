"""Query-result cache keys.

A cached result is valid only for the index generation it was computed at; every sync
that changes the index bumps the generation (see ``StorageBackend.bump_index_generation``).
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


def normalize_query(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def cache_key(tool: str, query: str, params: dict[str, Any], retrieval: dict[str, Any]) -> str:
    """sha256 over tool, normalized query, all parameters and the retrieval stack."""
    payload = {
        "tool": tool,
        "query": normalize_query(query),
        "params": params,
        "retrieval": retrieval,
    }
    raw = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
