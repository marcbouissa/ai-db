"""
Base classes for file type extensions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Chunk:
    """A single chunk of parsed content."""
    content: str
    filepath: str
    start_line: int
    end_line: int
    chunk_type: str = "code"
    language: str | None = None
    metadata: dict[str, Any] | None = None


class ChunkerExtension(ABC):
    """Abstract base for file type chunkers."""
    
    # File extensions this chunker handles (e.g., [".md", ".markdown"])
    extensions: list[str] = []
    
    # Language identifier (e.g., "markdown", "usd", "segmentation")
    language: str = ""
    
    # Optional: tree-sitter language name for AST parsing
    ts_language: str | None = None
    
    @abstractmethod
    def chunk(self, filepath: Path, content: str) -> list[Chunk]:
        """Parse file content into chunks."""
        pass
    
    def can_handle(self, filepath: Path) -> bool:
        """Check if this chunker can handle the file."""
        # Check full suffix chain for compound extensions (e.g., .seg.png)
        for ext in self.extensions:
            if "".join(filepath.suffixes).lower().endswith(ext.lower()):
                return True
        return filepath.suffix.lower() in self.extensions
    
    def get_metadata(self, filepath: Path, chunk: Chunk) -> dict[str, Any]:
        """Extract additional metadata for indexing."""
        return {
            "language": self.language,
            "chunk_type": chunk.chunk_type,
            "extension": filepath.suffix.lower(),
        }


# Registry
_CHUNKERS: dict[str, ChunkerExtension] = {}


def register_chunker(chunker: ChunkerExtension) -> None:
    """Register a chunker for its extensions."""
    for ext in chunker.extensions:
        _CHUNKERS[ext.lower()] = chunker


def get_chunker(filepath: Path) -> ChunkerExtension | None:
    """Get registered chunker for a file."""
    # Try full suffix chain first (compound extensions like .seg.png)
    full_suffix = "".join(filepath.suffixes).lower()
    if full_suffix in _CHUNKERS:
        return _CHUNKERS[full_suffix]
    # Fall back to single suffix
    return _CHUNKERS.get(filepath.suffix.lower())


def list_chunkers() -> dict[str, ChunkerExtension]:
    """List all registered chunkers."""
    return dict(_CHUNKERS)