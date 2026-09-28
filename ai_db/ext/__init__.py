"""
ai-db Extensions: File Type Support
===================================
This package demonstrates how to add support for custom file types.
Extensions are loaded via entry points or direct registration.

To create a new file type extension:
1. Create a parser module with a Chunker implementation
2. Register it via entry point 'ai_db.chunkers' or call register_chunker()
3. Add tree-sitter queries if needed (for structured languages)
4. Register language detection patterns

Example extensions included:
- markdown: Standard markdown files with frontmatter support
- usd: Universal Scene Description (3D assets)
- seg: Image segmentation masks (PNG/RLE formats)
"""

from ai_db.ext.base import ChunkerExtension, register_chunker, get_chunker

__all__ = [
    "ChunkerExtension",
    "register_chunker",
    "get_chunker",
]