"""Storage backend discovery.

Backends are discovered through the ``ai_db.storage`` entry-point group. Each entry
point loads a callable ``(options: dict) -> StorageBackend``. The built-in SQLite
backend is registered the same way (see ``pyproject.toml``), so there is exactly one
code path for built-in and third-party connectors.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Any

from ai_db.constants import DEFAULT_DB_FILE
from ai_db.errors import AiDbConfigError
from ai_db.storage.backend import StorageBackend
from ai_db.storage.sqlite_backend import SQLiteBackend

ENTRY_POINT_GROUP = "ai_db.storage"


def available_providers() -> dict[str, Any]:
    return {ep.name: ep for ep in entry_points(group=ENTRY_POINT_GROUP)}


def load_provider(name: str) -> Callable[[dict[str, Any]], StorageBackend]:
    providers = available_providers()
    if name not in providers:
        raise AiDbConfigError(
            f"unknown storage provider '{name}'; installed providers: {sorted(providers)}"
        )
    factory: Callable[[dict[str, Any]], StorageBackend] = providers[name].load()
    if not callable(factory):
        raise AiDbConfigError(f"storage entry point '{name}' does not load a callable")
    return factory


class StorageBackendFactory:
    """Builds storage backends from config or from an explicit SQLite path."""

    @classmethod
    def from_config(cls, cfg: Any, db_path: str | None = None) -> StorageBackend:
        """Build the backend named by ``cfg.storage.provider``.

        ``db_path`` (the CLI ``--db`` flag) overrides ``options.path``.
        """
        options = dict(cfg.storage.options)
        if db_path is not None:
            options["path"] = db_path
        backend = load_provider(cfg.storage.provider)(options)
        if not isinstance(backend, StorageBackend):
            raise AiDbConfigError(
                f"storage provider '{cfg.storage.provider}' returned {type(backend).__name__}, "
                "not a StorageBackend"
            )
        if cfg.retrieval_mode == "hybrid" and "vector" not in backend.capabilities():
            raise AiDbConfigError(
                f"retrieval.mode 'hybrid' needs a backend with the 'vector' capability; "
                f"'{cfg.storage.provider}' has {sorted(backend.capabilities())}"
            )
        return backend

    @classmethod
    def create(cls, connection_string: str | None = None) -> StorageBackend:
        """Create a SQLite backend from a filesystem path or ``sqlite:///`` URI.

        Raises ValueError for empty strings, malformed URIs and non-sqlite schemes
        (other databases are provided by entry-point plugins, see ``from_config``).
        """
        if connection_string is None:
            connection_string = DEFAULT_DB_FILE
        if not isinstance(connection_string, str) or not connection_string.strip():
            raise ValueError("Connection string cannot be empty or whitespace")
        return SQLiteBackend(parse_sqlite_location(connection_string.strip()))


def parse_sqlite_location(raw: str) -> str:
    """Turn a path or sqlite URI into a filesystem path or ``:memory:``."""
    if len(raw) >= 2 and raw[1] == ":" and raw[0].isalpha():
        return raw  # Windows drive path
    if "://" not in raw:
        return raw
    if raw.startswith("://"):
        raise ValueError(f"Malformed connection URI: missing scheme in '{raw}'")
    scheme, _, rest = raw.partition("://")
    scheme = scheme.lower()
    if scheme not in ("sqlite", "file"):
        raise ValueError(
            f"Unsupported storage backend URI scheme: '{scheme}'. Non-SQLite databases are "
            "installed as plugins and selected with storage.provider in the config."
        )
    if rest in ("", "/"):
        raise ValueError(f"Malformed SQLite URI: path is missing in '{raw}'")
    if rest.lower() in (":memory:", "/:memory:"):
        return ":memory:"
    if rest.startswith("/"):
        # sqlite:///relative/or/abs -> strip the authority slash
        sub = rest[1:]
        if not sub:
            raise ValueError(f"Malformed SQLite URI: path is missing in '{raw}'")
        return sub
    return rest
