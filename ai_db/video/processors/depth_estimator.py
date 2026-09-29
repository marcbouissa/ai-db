"""
Depth Estimator Video Processor.
=================================
Uses Depth Anything V2 or Metric3D for per-frame depth estimation.
Requires video-depth extra: torch, transformers, accelerate
"""

from pathlib import Path
from typing import Any

from ai_db.video.processors.base import (
    VideoProcessor, VideoProcessorResult, DepthMap, register_video_processor
)


class DepthEstimator(VideoProcessor):
    """Estimate per-pixel depth from keyframes."""
    
    name = "depth_estimator"
    version = "1.0.0"
    requires = ["torch", "transformers", "accelerate"]
    provides = ["depth"]
    heavy = True

    def __init__(self):
        super().__init__()
        self._pipe = None
        self._model_id = "depth-anything/Depth-Anything-V2-Small-hf"
    
    def _load_model(self):
        if self._pipe is None:
            from transformers import pipeline
            import torch
            self._pipe = pipeline(
                "depth-estimation",
                model=self._model_id,
                device=0 if torch.cuda.is_available() else -1,
            )

    def process(self, video_path: Path, config: dict[str, Any]) -> VideoProcessorResult:
        self._load_model()
        
        keyframes = config.get("keyframes", [])
        if not keyframes:
            return VideoProcessorResult(
                metadata={"error": "No keyframes provided to depth estimator"}
            )
        
        output_dir = Path(config.get("output_dir", ".ai_db/video_depth"))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        depth_maps = []
        for kf in keyframes:
            # Load image
            from PIL import Image
            image = Image.open(kf.filepath)
            
            # Estimate depth
            result = self._pipe(image)
            depth = result["depth"]
            
            # Save depth map
            import numpy as np
            depth_array = np.array(depth)
            depth_path = output_dir / f"{Path(kf.filepath).stem}_depth.npy"
            np.save(depth_path, depth_array)
            
            # Also save as visualization PNG
            depth_vis = (depth_array / depth_array.max() * 255).astype(np.uint8)
            Image.fromarray(depth_vis).save(
                output_dir / f"{Path(kf.filepath).stem}_depth.png"
            )
            
            depth_maps.append(DepthMap(
                keyframe_index=kf.index,
                filepath=str(depth_path),
                scale=1.0,  # Relative depth, not metric
                metadata={
                    "source_frame": kf.filepath,
                    "min_depth": float(depth_array.min()),
                    "max_depth": float(depth_array.max()),
                    "model": self._model_id,
                }
            ))
        
        return VideoProcessorResult(
            depth_maps=depth_maps,
            metadata={
                "video_path": str(video_path),
                "model": self._model_id,
                "frames_processed": len(depth_maps),
            }
        )


# Auto-register on import
register_video_processor(DepthEstimator())
