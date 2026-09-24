"""Minimal third-party storage connector used to test entry-point discovery."""

from ai_db.storage.sqlite_backend import SQLiteBackend


class DummyBackend(SQLiteBackend):
    """An in-memory SQLite backend that reports a different name."""

    @property
    def backend_name(self) -> str:
        return "dummy"


def create(options: dict) -> DummyBackend:
    if set(options) - {"path"}:
        raise ValueError(f"unexpected options {options}")
    return DummyBackend(":memory:")
