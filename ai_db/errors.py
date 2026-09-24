"""Typed exceptions raised by ai-db. Never caught-and-ignored inside the package."""


class AiDbError(Exception):
    """Base class for all ai-db errors."""


class AiDbConfigError(AiDbError):
    """Configuration is missing, invalid, or requires an unavailable dependency."""


class AiDbQueryError(AiDbError):
    """A search query could not be executed by the storage backend."""

    def __init__(self, query: str, cause: Exception):
        super().__init__(f"query failed: {cause} (query={query!r})")
        self.query = query
        self.cause = cause
