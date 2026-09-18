"""Backward-compatibility Database wrapper for ai-db (Milestone 2).

Inherits from SQLiteBackend to provide full compatibility for legacy callers
expecting `ai_db.storage.database.Database`.
"""

from typing import Optional, Dict, Any
from ai_db.constants import DEFAULT_DB_FILE
from ai_db.storage.sqlite_backend import SQLiteBackend
from ai_db.storage.state import get_session_state, set_session_state


class Database(SQLiteBackend):
    """Backward-compatible Database wrapper inheriting from SQLiteBackend."""

    def __init__(self, db_path: Optional[str] = DEFAULT_DB_FILE):
        super().__init__(db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        """Forward schema initialization to initialize()."""
        self.initialize()

    def _get_session_state(self, key: str) -> Optional[Any]:
        return self.get_state(key)

    def _set_session_state(self, key: str, value: Any) -> None:
        self.set_state(key, value)

    def prune_file(self, filepath: str) -> None:
        """Backward-compatible alias for delete_file."""
        self.delete_file(filepath)
