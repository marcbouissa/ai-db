"""
Example: Using extensions together for a 3D + CV project.

This demonstrates a project with:
- Markdown documentation
- USD 3D assets
- Segmentation masks for CV training
"""

from pathlib import Path
from ai_db.ext.markdown import MarkdownChunker
from ai_db.ext.usd import USDChunker
from ai_db.ext.segmentation import SegmentationChunker
from ai_db.ext.base import get_chunker, list_chunkers


def demo_extension_loading():
    """Show how extensions are registered and used."""
    
    # Extensions auto-register on import
    print("Registered chunkers:")
    for ext, chunker in list_chunkers().items():
        print(f"  {ext} -> {chunker.__class__.__name__} ({chunker.language})")
    
    # Test chunker lookup
    test_files = [
        Path("README.md"),
        Path("assets/model.usda"),
        Path("masks/train/seg_001.seg.png"),
        Path("annotations/train.coco.json"),
        Path("labels/val.txt"),  # YOLO format
    ]
    
    print("\nChunker lookup:")
    for f in test_files:
        chunker = get_chunker(f)
        if chunker:
            print(f"  {f} -> {chunker.__class__.__name__}")
        else:
            print(f"  {f} -> No chunker (uses default)")


def demo_chunking():
    """Show chunking output for each type."""
    
    # Sample markdown
    md_content = """---
title: "Project Documentation"
tags: [3d, cv, usd]
---

# Overview

This project combines 3D assets with CV segmentation.

## 3D Models

See `assets/scene.usda` for the main scene.

```python
# Code example
def load_usd(path):
    return Usd.Stage.Open(path)
```

## Segmentation

Masks are in `masks/` directory.
"""
    
    # Sample USD ASCII
    usd_content = """#usda 1.0
(
    defaultPrim = "World"
)

def "World" {
    def "Robot" {
        def "Arm" {
            float length = 1.5
            references = [@./arm.usd@]
        }
    }
    
    variantSet "config" {
        "production" {
            payload = [@./robot_payload.usd@]
        }
        "preview" {
            payload = [@./robot_preview.usd@]
        }
    }
}
"""
    
    # Sample COCO annotation
    coco_content = """{
    "categories": [
        {"id": 1, "name": "person"},
        {"id": 2, "name": "car"}
    ],
    "annotations": [
        {"id": 1, "category_id": 1, "bbox": [100, 200, 50, 100], "area": 5000, "segmentation": [[100,200,150,200,150,300,100,300]], "iscrowd": 0}
    ]
}
"""
    
    # Test chunking
    
    md_chunker = MarkdownChunker()
    usd_chunker = USDChunker()
    seg_chunker = SegmentationChunker()
    
    print("\n=== Markdown Chunks ===")
    for chunk in md_chunker.chunk(Path("test.md"), md_content):
        print(f"  [{chunk.chunk_type}] L{chunk.start_line}-{chunk.end_line}: {chunk.content[:80]}...")
        if chunk.metadata:
            print(f"    Metadata: {chunk.metadata}")
    
    print("\n=== USD Chunks ===")
    for chunk in usd_chunker.chunk(Path("scene.usda"), usd_content):
        print(f"  [{chunk.chunk_type}] L{chunk.start_line}-{chunk.end_line}: {chunk.metadata.get('prim_path', 'N/A')}")
    
    print("\n=== COCO Segmentation Chunks ===")
    for chunk in seg_chunker.chunk(Path("train.coco.json"), coco_content):
        print(f"  [{chunk.chunk_type}] Class: {chunk.metadata.get('category')}, Area: {chunk.metadata.get('area')}")


if __name__ == "__main__":
    demo_extension_loading()
    demo_chunking()