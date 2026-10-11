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


class AiDbStorageError(AiDbError):
    """Stored data is corrupt or the storage backend violated its contract."""


class AiDbProtocolError(AiDbError):
    """A remote (daemon) response had an unexpected JSON shape."""


class AiDbDaemonError(AiDbError):
    """The daemon could not be reached, or rejected a management request."""


class UnknownToolError(AiDbError, KeyError):
    """No tool is registered under the requested name.

    Transports catch this, not ``KeyError``: a ``KeyError`` raised *inside* a tool
    handler is a bug and must surface as one, not be reported as "Unknown tool".
    It still subclasses ``KeyError`` so existing ``except KeyError`` callers keep working.
    """
