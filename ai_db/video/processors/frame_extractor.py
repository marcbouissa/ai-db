"""
Frame Extractor Video Processor.
=================================
Extracts keyframes from video at regular intervals or scene changes.
Zero external dependencies beyond opencv-python-headless (in video-base extra).
"""

from pathlib import Path
from typing import Any

from ai_db.video.processors.base import (
    VideoProcessor, VideoProcessorResult, Keyframe, register_video_processor
)


class FrameExtractor(VideoProcessor):
    """Extract keyframes from video."""
    
    name = "frame_extractor"
    version = "1.0.0"
    requires = ["opencv-python-headless"]
    provides = ["keyframes"]
    heavy = False

    def process(self, video_path: Path, config: dict[str, Any]) -> VideoProcessorResult:
        import cv2
        
        interval = config.get("interval", 2.0)          # seconds between keyframes
        max_frames = config.get("max_frames", 100)      # max keyframes to extract
        output_dir = Path(config.get("output_dir", ".ai_db/video_frames"))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        frame_interval = int(fps * interval) if fps > 0 else 30
        frame_interval = max(1, frame_interval)
        
        keyframes = []
        frame_idx = 0
        saved = 0
        
        while saved < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_idx / fps if fps > 0 else frame_idx * interval
            h, w = frame.shape[:2]
            
            frame_name = f"{video_path.stem}_frame_{saved:04d}.jpg"
            frame_path = output_dir / frame_name
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            keyframes.append(Keyframe(
                index=saved,
                timestamp=timestamp,
                filepath=str(frame_path),
                width=w,
                height=h,
                metadata={
                    "source_frame": frame_idx,
                    "fps": fps,
                }
            ))
            
            saved += 1
            frame_idx += frame_interval
            if frame_idx >= total_frames:
                break
        
        cap.release()
        
        return VideoProcessorResult(
            keyframes=keyframes,
            metadata={
                "video_path": str(video_path),
                "duration": duration,
                "fps": fps,
                "total_frames": total_frames,
                "extracted": len(keyframes),
                "interval": interval,
            }
        )


# Auto-register on import
register_video_processor(FrameExtractor())
