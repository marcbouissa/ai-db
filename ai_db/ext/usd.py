"""
USD (Universal Scene Description) file type extension.
Supports .usd, .usda, .usdc, .usdz files for 3D asset indexing.
"""

import re
from pathlib import Path
from typing import Any

from ai_db.ext.base import ChunkerExtension, Chunk, register_chunker


class USDChunker(ChunkerExtension):
    """Chunker for USD files (ASCII and binary)."""
    
    extensions = [".usd", ".usda", ".usdc", ".usdz"]
    language = "usd"
    
    # USD ASCII patterns
    PRIM_PATTERN = re.compile(r'^(def|over|class)\s+"([^"]+)"\s*{', re.MULTILINE)
    VARIANT_PATTERN = re.compile(r'variantSet\s+"([^"]+)"\s*{', re.MULTILINE)
    REF_PATTERN = re.compile(r'references\s*=\s*\[([^\]]+)\]', re.MULTILINE)
    PAYLOAD_PATTERN = re.compile(r'payload\s*=\s*\[([^\]]+)\]', re.MULTILINE)
    
    def chunk(self, filepath: Path, content: str) -> list[Chunk]:
        chunks = []
        
        # Try to parse as ASCII
        if self._is_ascii(content):
            chunks = self._chunk_ascii(filepath, content)
        else:
            # Binary USD - create a single metadata chunk
            chunks = [Chunk(
                content=f"[Binary USD file: {filepath.name}]",
                filepath=str(filepath),
                start_line=1,
                end_line=1,
                chunk_type="binary",
                language=self.language,
                metadata={"format": "usdc/usdz", "size_bytes": len(content)},
            )]
        
        return chunks
    
    def _is_ascii(self, content: bytes | str) -> bool:
        if isinstance(content, bytes):
            # Check for USD ASCII magic
            return content.startswith(b"#usda") or content.startswith(b"#usd ")
        return content.startswith("#usda") or content.startswith("#usd ")
    
    def _chunk_ascii(self, filepath: Path, content: str) -> list[Chunk]:
        chunks = []
        lines = content.splitlines(keepends=True)
        
        # Find all prim definitions
        for match in self.PRIM_PATTERN.finditer(content):
            prim_type = match.group(1)
            prim_path = match.group(2)
            start_pos = match.start()
            start_line = content[:start_pos].count('\n') + 1
            
            # Find matching brace
            end_line = self._find_matching_brace(content, match.end())
            
            # Extract prim content
            prim_content = "\n".join(lines[start_line - 1:end_line])
            
            # Extract references and payloads
            refs = self.REF_PATTERN.findall(prim_content)
            payloads = self.PAYLOAD_PATTERN.findall(prim_content)
            
            chunks.append(Chunk(
                content=prim_content,
                filepath=str(filepath),
                start_line=start_line,
                end_line=end_line,
                chunk_type="prim",
                language=self.language,
                metadata={
                    "prim_type": prim_type,
                    "prim_path": prim_path,
                    "references": refs,
                    "payloads": payloads,
                },
            ))
        
        # Also chunk variant sets
        for match in self.VARIANT_PATTERN.finditer(content):
            variant_name = match.group(1)
            start_pos = match.start()
            start_line = content[:start_pos].count('\n') + 1
            end_line = self._find_matching_brace(content, match.end())
            
            variant_content = "\n".join(lines[start_line - 1:end_line])
            
            chunks.append(Chunk(
                content=variant_content,
                filepath=str(filepath),
                start_line=start_line,
                end_line=end_line,
                chunk_type="variant_set",
                language=self.language,
                metadata={"variant_name": variant_name},
            ))
        
        # If no prims found, chunk by logical sections
        if not chunks:
            chunks = self._chunk_by_sections(filepath, lines)
        
        return chunks
    
    def _find_matching_brace(self, content: str, start: int) -> int:
        """Find the line number of the matching closing brace."""
        brace_count = 1
        pos = start
        line = content[:start].count('\n') + 1
        
        while pos < len(content) and brace_count > 0:
            if content[pos] == '{':
                brace_count += 1
            elif content[pos] == '}':
                brace_count -= 1
            if content[pos] == '\n':
                line += 1
            pos += 1
        
        return line
    
    def _chunk_by_sections(self, filepath: Path, lines: list[str]) -> list[Chunk]:
        """Fallback: chunk by logical sections (every ~100 lines)."""
        chunks = []
        chunk_size = 100
        
        for i in range(0, len(lines), chunk_size):
            chunk_lines = lines[i:i + chunk_size]
            chunks.append(Chunk(
                content="".join(chunk_lines),
                filepath=str(filepath),
                start_line=i + 1,
                end_line=min(i + chunk_size, len(lines)),
                chunk_type="section",
                language=self.language,
                metadata={},
            ))
        
        return chunks
    
    def get_metadata(self, filepath: Path, chunk: Chunk) -> dict[str, Any]:
        meta = super().get_metadata(filepath, chunk)
        meta.update({
            "prim_path": chunk.metadata.get("prim_path"),
            "prim_type": chunk.metadata.get("prim_type"),
            "has_references": bool(chunk.metadata.get("references")),
            "has_payloads": bool(chunk.metadata.get("payloads")),
        })
        return meta


# Auto-register on import
register_chunker(USDChunker())