"""
Scene Memory Factory.
======================
Follows StorageBackendFactory pattern for pluggable backends.
"""

from typing import Any
from ai_db.scene.storage.base import SceneMemoryBackend
from ai_db.scene.storage.json_file import JSONFileSceneBackend
from ai_db.scene.storage.sqlite_vec import SQLiteVecSceneBackend
from ai_db.logger import _logger


class SceneMemoryFactory:
    """Factory for creating scene memory backends."""
    
    _backends: dict[str, type[SceneMemoryBackend]] = {
        "json_file": JSONFileSceneBackend,
        "sqlite_vec": SQLiteVecSceneBackend,
    }
    
    @classmethod
    def create(cls, backend_name: str, config: dict[str, Any] | None = None) -> SceneMemoryBackend:
        """Create and initialize a scene memory backend."""
        backend_cls = cls._backends.get(backend_name)
        if not backend_cls:
            available = ", ".join(cls._backends.keys())
            raise ValueError(f"Unknown scene backend: {backend_name}. Available: {available}")
        
        backend = backend_cls()
        backend.initialize(config or {})
        _logger.info(f"Created scene memory backend: {backend_name}")
        return backend
    
    @classmethod
    def register_backend(cls, name: str, backend_cls: type[SceneMemoryBackend]) -> None:
        """Register a custom backend (for extensions)."""
        if not issubclass(backend_cls, SceneMemoryBackend):
            raise TypeError("Backend must inherit from SceneMemoryBackend")
        cls._backends[name] = backend_cls
        _logger.info(f"Registered custom scene backend: {name}")
    
    @classmethod
    def list_backends(cls) -> dict[str, dict[str, Any]]:
        """List available backends with capabilities."""
        return {
            name: {
                "class": backend_cls.__name__,
                "capabilities": getattr(backend_cls, "capabilities", set()),
            }
            for name, backend_cls in cls._backends.items()
        }
    
    @classmethod
    def get_backend_capabilities(cls, backend_name: str) -> set[str]:
        """Get capabilities for a backend without instantiating."""
        backend_cls = cls._backends.get(backend_name)
        if backend_cls:
            return getattr(backend_cls, "capabilities", set())
        return set()
