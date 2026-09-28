# ai-db Extensions: Custom File Type Support

This directory contains examples of how to extend ai-db with support for custom file types.

## Included Extensions

| Extension | File Types | Description |
|-----------|------------|-------------|
| `markdown.py` | `.md`, `.markdown`, `.mdx` | Parses frontmatter, headings, code blocks |
| `usd.py` | `.usd`, `.usda`, `.usdc`, `.usdz` | 3D scene descriptions (Pixar USD) |
| `segmentation.py` | `.seg.png`, `.seg.npy`, `.rle`, `.coco.json`, `.yolo.txt` | Image segmentation masks |

## Quick Start

### 1. Auto-registration (Recommended)

Extensions auto-register on import. Just import them:

```python
# In your project's init or config
import ai_db.ext.markdown
import ai_db.ext.usd
import ai_db.ext.segmentation

# Now these file types are automatically chunked during sync
```

### 2. Manual Registration

```python
from ai_db.ext.base import register_chunker
from ai_db.ext.markdown import MarkdownChunker

register_chunker(MarkdownChunker())
```

### 3. Via Entry Points (for packages)

In your package's `pyproject.toml`:

```toml
[project.entry-points."ai_db.chunkers"]
markdown = "my_package.chunkers:MarkdownChunker"
usd = "my_package.chunkers:USDChunker"
```

## Creating Your Own Extension

```python
from ai_db.ext.base import ChunkerExtension, Chunk, register_chunker
from pathlib import Path
from typing import Any

class MyFormatChunker(ChunkerExtension):
    extensions = [".myext"]
    language = "myformat"
    
    def chunk(self, filepath: Path, content: str) -> list[Chunk]:
        # Parse your format
        return [Chunk(
            content=parsed_content,
            filepath=str(filepath),
            start_line=1,
            end_line=line_count,
            chunk_type="my_chunk_type",
            language=self.language,
            metadata={"custom": "metadata"},
        )]

register_chunker(MyFormatChunker())
```

## Requirements

- `python-frontmatter` for markdown frontmatter parsing
- `numpy` for `.npy` segmentation masks
- `opencv-python` or `PIL` for advanced PNG metadata (optional)

Install with:
```bash
pip install python-frontmatter numpy
```

## Integration with ai-db CLI

Once registered, the extensions work automatically:

```bash
# Sync will now parse .md, .usd, .seg.png files
ai-db sync /path/to/project

# Query across all file types
ai-db query "3D model with segmentation mask"

# Investigate includes structured content from all types
ai-db investigate "how are USD prims referenced"
```