"""
Image Segmentation file type extension.
Supports segmentation masks in various formats:
- PNG with palette/alpha masks
- RLE (Run-Length Encoding) formats
- COCO JSON annotations
- YOLO format masks
- NumPy .npy masks
"""

import json
from pathlib import Path
from typing import Any

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from ai_db.ext.base import ChunkerExtension, Chunk, register_chunker


class SegmentationChunker(ChunkerExtension):
    """Chunker for segmentation mask files."""
    
    extensions = [
        ".seg.png", ".mask.png", ".seg.npy", ".rle", 
        ".coco.json", ".yolo.txt", ".voc.xml"
    ]
    language = "segmentation"
    
    def chunk(self, filepath: Path, content: str | bytes) -> list[Chunk]:
        chunks = []
        
        suffix = "".join(filepath.suffixes).lower()
        
        if suffix in (".seg.png", ".mask.png", ".png"):
            chunks = self._chunk_png_mask(filepath, content)
        elif suffix == ".seg.npy" or suffix == ".npy":
            chunks = self._chunk_numpy_mask(filepath, content)
        elif suffix == ".rle":
            chunks = self._chunk_rle(filepath, content)
        elif suffix == ".coco.json":
            chunks = self._chunk_coco(filepath, content)
        elif suffix == ".yolo.txt":
            chunks = self._chunk_yolo(filepath, content)
        else:
            # Generic fallback
            chunks = [Chunk(
                content=f"[Segmentation file: {filepath.name}]",
                filepath=str(filepath),
                start_line=1,
                end_line=1,
                chunk_type="metadata",
                language=self.language,
                metadata={"format": suffix, "size_bytes": len(content)},
            )]
        
        return chunks
    
    def _chunk_png_mask(self, filepath: Path, content: bytes | str) -> list[Chunk]:
        """Extract metadata from PNG mask."""
        if isinstance(content, str):
            content = content.encode()
        
        chunks = []
        
        # Parse PNG chunks for metadata
        if content.startswith(b'\x89PNG\r\n\x1a\n'):
            metadata = self._parse_png_metadata(content)
            chunks.append(Chunk(
                content=f"PNG Segmentation Mask: {filepath.name}",
                filepath=str(filepath),
                start_line=1,
                end_line=1,
                chunk_type="mask_metadata",
                language=self.language,
                metadata=metadata,
            ))
        
        return chunks
    
    def _parse_png_metadata(self, data: bytes) -> dict[str, Any]:
        """Extract basic PNG metadata."""
        metadata = {}
        i = 8  # Skip signature
        
        while i < len(data):
            if i + 8 > len(data):
                break
            length = int.from_bytes(data[i:i+4], 'big')
            chunk_type = data[i+4:i+8].decode('ascii', errors='ignore')
            
            if chunk_type == 'IHDR':
                if i + 8 + length <= len(data):
                    ihdr = data[i+8:i+8+length]
                    metadata['width'] = int.from_bytes(ihdr[0:4], 'big')
                    metadata['height'] = int.from_bytes(ihdr[4:8], 'big')
                    metadata['bit_depth'] = ihdr[8]
                    metadata['color_type'] = ihdr[9]
            elif chunk_type == 'tRNS':
                metadata['has_transparency'] = True
            elif chunk_type == 'pHYs':
                if i + 8 + length <= len(data):
                    phys = data[i+8:i+8+length]
                    metadata['dpi_x'] = int.from_bytes(phys[0:4], 'big')
                    metadata['dpi_y'] = int.from_bytes(phys[4:8], 'big')
            elif chunk_type == 'IEND':
                break
            
            i += 12 + length + 4  # length + type + data + crc
        
        return metadata
    
    def _chunk_numpy_mask(self, filepath: Path, content: bytes | str) -> list[Chunk]:
        """Parse NumPy .npy mask file."""
        if not NUMPY_AVAILABLE:
            return [Chunk(
                content=f"[NumPy mask: {filepath.name} - numpy not available]",
                filepath=str(filepath),
                start_line=1,
                end_line=1,
                chunk_type="mask_metadata",
                language=self.language,
                metadata={"format": "npy", "error": "numpy not installed"},
            )]
        
        try:
            arr = np.load(filepath)
            metadata = {
                "shape": arr.shape,
                "dtype": str(arr.dtype),
                "unique_values": np.unique(arr).tolist()[:20],
                "num_classes": len(np.unique(arr)),
                "size_bytes": arr.nbytes,
            }
            
            return [Chunk(
                content=f"Segmentation Mask: {filepath.name}\nShape: {arr.shape}\nClasses: {len(np.unique(arr))}",
                filepath=str(filepath),
                start_line=1,
                end_line=1,
                chunk_type="mask_metadata",
                language=self.language,
                metadata=metadata,
            )]
        except Exception as e:
            return [Chunk(
                content=f"[Failed to load NumPy mask: {e}]",
                filepath=str(filepath),
                start_line=1,
                end_line=1,
                chunk_type="error",
                language=self.language,
                metadata={"error": str(e)},
            )]
    
    def _chunk_rle(self, filepath: Path, content: str) -> list[Chunk]:
        """Parse RLE (Run-Length Encoding) mask."""
        try:
            # RLE format: "x y width height count1 count2 ..."
            parts = content.strip().split()
            if len(parts) >= 4:
                x, y, w, h = map(int, parts[:4])
                counts = list(map(int, parts[4:]))
                
                # Decode to get class distribution
                total_pixels = sum(counts)
                num_runs = len(counts)
                
                return [Chunk(
                    content=f"RLE Mask: {filepath.name}\nBBox: ({x},{y},{w},{h})\nRuns: {num_runs}\nPixels: {total_pixels}",
                    filepath=str(filepath),
                    start_line=1,
                    end_line=1,
                    chunk_type="rle_mask",
                    language=self.language,
                    metadata={
                        "bbox": [x, y, w, h],
                        "num_runs": num_runs,
                        "total_pixels": total_pixels,
                        "format": "rle",
                    },
                )]
        except Exception:
            pass
        
        return [Chunk(
            content=f"[RLE Mask: {filepath.name}]",
            filepath=str(filepath),
            start_line=1,
            end_line=1,
            chunk_type="rle_mask",
            language=self.language,
            metadata={"format": "rle"},
        )]
    
    def _chunk_coco(self, filepath: Path, content: str) -> list[Chunk]:
        """Parse COCO format annotations."""
        try:
            data = json.loads(content)
            annotations = data.get('annotations', [])
            categories = {c['id']: c['name'] for c in data.get('categories', [])}
            
            chunks = []
            for ann in annotations:
                cat_name = categories.get(ann.get('category_id'), 'unknown')
                seg = ann.get('segmentation', [])
                bbox = ann.get('bbox', [])
                area = ann.get('area', 0)
                
                chunks.append(Chunk(
                    content=f"COCO Annotation: {cat_name}\nBBox: {bbox}\nArea: {area}",
                    filepath=str(filepath),
                    start_line=1,
                    end_line=1,
                    chunk_type="coco_annotation",
                    language=self.language,
                    metadata={
                        "category": cat_name,
                        "category_id": ann.get('category_id'),
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": ann.get('iscrowd', 0),
                        "format": "coco",
                    },
                ))
            
            return chunks
        except Exception as e:
            return [Chunk(
                content=f"[Failed to parse COCO: {e}]",
                filepath=str(filepath),
                start_line=1,
                end_line=1,
                chunk_type="error",
                language=self.language,
                metadata={"error": str(e)},
            )]
    
    def _chunk_yolo(self, filepath: Path, content: str) -> list[Chunk]:
        """Parse YOLO format segmentation (polygon format)."""
        chunks = []
        lines = content.strip().split('\n')
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 7:  # class_id + at least 3 points (6 coords)
                class_id = int(parts[0])
                coords = list(map(float, parts[1:]))
                points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
                
                chunks.append(Chunk(
                    content=f"YOLO Segmentation: class {class_id}\nPoints: {len(points)}",
                    filepath=str(filepath),
                    start_line=1,
                    end_line=1,
                    chunk_type="yolo_segmentation",
                    language=self.language,
                    metadata={
                        "class_id": class_id,
                        "num_points": len(points),
                        "points": points[:10],  # First 10 points
                        "format": "yolo",
                    },
                ))
        
        if not chunks:
            chunks = [Chunk(
                content=f"[YOLO Segmentation: {filepath.name}]",
                filepath=str(filepath),
                start_line=1,
                end_line=1,
                chunk_type="yolo_segmentation",
                language=self.language,
                metadata={"format": "yolo"},
            )]
        
        return chunks
    
    def get_metadata(self, filepath: Path, chunk: Chunk) -> dict[str, Any]:
        meta = super().get_metadata(filepath, chunk)
        meta.update({
            "mask_format": chunk.chunk_type,
            "num_classes": chunk.metadata.get("num_classes") if chunk.metadata else None,
        })
        return meta


# Auto-register on import
register_chunker(SegmentationChunker())