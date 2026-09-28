"""
Markdown file type extension.
Supports frontmatter, headings, code blocks, and structured content.
"""

import re
from pathlib import Path
from typing import Any

import frontmatter

from ai_db.ext.base import ChunkerExtension, Chunk, register_chunker


class MarkdownChunker(ChunkerExtension):
    """Chunker for Markdown files with frontmatter support."""
    
    extensions = [".md", ".markdown", ".mdx", ".mdown"]
    language = "markdown"
    ts_language = "markdown"
    
    def chunk(self, filepath: Path, content: str) -> list[Chunk]:
        chunks = []
        
        # Parse frontmatter
        try:
            post = frontmatter.loads(content)
            fm_data = post.metadata
            body = post.content
        except Exception:
            fm_data = {}
            body = content
        
        lines = body.splitlines(keepends=True)
        current_chunk = []
        current_start = 1
        heading_stack = []
        
        for i, line in enumerate(lines, 1):
            # Detect headings
            heading_match = re.match(r'^(#{1,6})\s+(.+)', line)
            if heading_match:
                # Flush previous chunk
                if current_chunk:
                    chunks.append(self._make_chunk(
                        filepath, current_chunk, current_start, i - 1, heading_stack
                    ))
                    current_chunk = []
                    current_start = i
                
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                heading_stack = heading_stack[:level - 1] + [title]
                current_chunk.append(line)
            elif re.match(r'^```', line):
                # Code block - keep with surrounding context
                current_chunk.append(line)
            else:
                current_chunk.append(line)
        
        # Flush remaining
        if current_chunk:
            chunks.append(self._make_chunk(
                filepath, current_chunk, current_start, len(lines), heading_stack
            ))
        
        return chunks
    
    def _make_chunk(
        self,
        filepath: Path,
        lines: list[str],
        start: int,
        end: int,
        heading_stack: list[str],
    ) -> Chunk:
        content = "".join(lines)
        
        # Determine chunk type based on content
        if re.match(r'^```', content.strip()):
            chunk_type = "code_block"
        elif heading_stack:
            chunk_type = "section"
        else:
            chunk_type = "text"
        
        return Chunk(
            content=content,
            filepath=str(filepath),
            start_line=start,
            end_line=end,
            chunk_type=chunk_type,
            language=self.language,
            metadata={
                "headings": heading_stack,
                "frontmatter": getattr(self, "_frontmatter", {}),
            },
        )
    
    def get_metadata(self, filepath: Path, chunk: Chunk) -> dict[str, Any]:
        meta = super().get_metadata(filepath, chunk)
        meta.update({
            "has_frontmatter": bool(chunk.metadata.get("frontmatter")) if chunk.metadata else False,
            "heading_path": " > ".join(chunk.metadata.get("headings", [])) if chunk.metadata else "",
        })
        return meta


# Auto-register on import
register_chunker(MarkdownChunker())