"""Pluggable StorageBackend Factory for ai-db (Milestone 2, Feature 9).

Resolves StorageBackend instances based on URI connection strings, environment
fallbacks, or default filesystem configurations.
"""

import os
import re
from typing import Optional
from urllib.parse import urlparse

from ai_db.constants import DEFAULT_DB_FILE
from ai_db.storage.backend import StorageBackend
from ai_db.storage.sqlite_backend import SQLiteBackend


def _mask_uri(uri: str) -> str:
    """Mask password credentials in URI strings for safe error reporting."""
    return re.sub(r"(://[^:]*:)([^@]+)(@)", r"\1******\3", uri)


class StorageBackendFactory:
    """Factory resolving StorageBackend by connection string URI or filesystem path."""

    @classmethod
    def from_config(cls, cfg, db_path: Optional[str] = None) -> StorageBackend:
        """Build the backend named by ``cfg.storage.provider``."""
        from ai_db.errors import AiDbConfigError
        if cfg.storage.provider != "sqlite":
            raise AiDbConfigError(f"unknown storage provider '{cfg.storage.provider}'")
        path = db_path if db_path is not None else (cfg.storage.options.get("path") or DEFAULT_DB_FILE)
        return cls.create(path)

    @classmethod
    def create(cls, connection_string: Optional[str] = None) -> StorageBackend:
        """Instantiate and return a configured StorageBackend instance.

        Args:
            connection_string: Optional URI or path. If omitted, falls back to
                AI_DB_CONNECTION_STRING -> AI_DB_PATH -> DEFAULT_DB_FILE.

        Returns:
            Configured StorageBackend instance (SQLiteBackend or MySQLBackend).

        Raises:
            ValueError: If connection_string is empty, malformed, or uses unsupported scheme.
            ImportError: If required driver for external backend is not installed.
        """
        if connection_string is None:
            connection_string = os.environ.get("AI_DB_CONNECTION_STRING")
            if not connection_string:
                connection_string = os.environ.get("AI_DB_PATH", DEFAULT_DB_FILE)

        if not isinstance(connection_string, str) or not connection_string.strip():
            raise ValueError("Connection string cannot be empty or whitespace")

        raw = connection_string.strip()

        # Handle Windows drive paths (e.g. C:\data\db.sqlite)
        if len(raw) >= 2 and raw[1] == ":" and raw[0].isalpha():
            return SQLiteBackend(raw)

        # Handle bare filesystem paths without URI scheme
        if "://" not in raw:
            return SQLiteBackend(raw)

        # Malformed URI missing scheme (e.g. ":///")
        if raw.startswith("://"):
            raise ValueError(f"Malformed connection URI: missing scheme in '{_mask_uri(raw)}'")

        parsed = urlparse(raw)
        scheme = parsed.scheme.lower()
        if not scheme:
            raise ValueError(f"Malformed connection URI: missing scheme in '{_mask_uri(raw)}'")

        # SQLite schemes
        if scheme in ("sqlite", "file"):
            if raw.lower() in ("sqlite://", "sqlite:///", "file://", "file:///"):
                raise ValueError(f"Malformed SQLite URI: path is missing in '{raw}'")

            if raw.lower().startswith("sqlite:///"):
                sub = raw[len("sqlite:///"):]
                if sub.lower() in (":memory:", "/:memory:"):
                    return SQLiteBackend(":memory:")
                elif sub:
                    return SQLiteBackend(sub)
                else:
                    raise ValueError(f"Malformed SQLite URI: path is missing in '{raw}'")
            elif raw.lower() in ("sqlite://:memory:", "sqlite:///:memory:"):
                return SQLiteBackend(":memory:")
            else:
                full_path = f"{parsed.netloc}{parsed.path}"
                if full_path.lower() in (":memory:", "/:memory:"):
                    return SQLiteBackend(":memory:")
                if not full_path:
                    raise ValueError(f"Malformed SQLite URI: path is missing in '{raw}'")
                return SQLiteBackend(full_path)

        # MySQL / MariaDB schemes
        elif scheme in ("mysql", "mysql+pymysql", "mariadb"):
            if raw.lower() in ("mysql://", "mysql:///", "mariadb://", "mariadb:///"):
                raise ValueError(f"Malformed MySQL URI: missing host and database in '{_mask_uri(raw)}'")
            if not parsed.netloc and not parsed.path.lstrip("/"):
                raise ValueError(f"Malformed MySQL URI: missing host and database in '{_mask_uri(raw)}'")

            from ai_db.storage.mysql_backend import MySQLBackend
            return MySQLBackend(raw)

        # Unsupported schemes
        else:
            masked = _mask_uri(raw)
            raise ValueError(f"Unsupported storage backend URI scheme: '{scheme}' in '{masked}'")
