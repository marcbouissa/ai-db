"""Lazy package exports (PEP 562): a package names what it re-exports and from where,
and a name is imported on first access instead of when the package is imported.

So importing one leaf module (``ai_db.constants``, ``ai_db.search.freshness``) no longer
runs every sibling module through the package ``__init__`` files. Packages keep an
``if TYPE_CHECKING:`` block with the real imports for type checkers.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any


def lazy_exports(package: str, namespace: dict[str, Any],
                 exports: dict[str, str]) -> tuple[Callable[[str], Any], Callable[[], list[str]]]:
    """``(__getattr__, __dir__)`` for ``package``; ``exports`` maps name -> module."""

    def __getattr__(name: str) -> Any:
        module = exports.get(name)
        if module is None:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        value = getattr(importlib.import_module(module), name)
        namespace[name] = value  # resolve once
        return value

    def __dir__() -> list[str]:
        return sorted(set(namespace) | set(exports))

    return __getattr__, __dir__
