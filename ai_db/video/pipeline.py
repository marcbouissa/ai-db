"""
Video Processing Pipeline.
===========================
Orchestrates multiple video processors in sequence, passing results between them.
Mirrors the Indexer pattern for multi-stage processing.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_db.video.processors.base import (
    VideoProcessor, Keyframe, list_video_processors
)
from ai_db.logger import _logger


@dataclass
class PipelineStage:
    """A single stage in the video processing pipeline."""
    processor_name: str
    config: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)  # What this stage needs from previous


@dataclass
class PipelineResult:
    """Complete result from pipeline execution."""
    video_path: str
    stages_run: list[str] = field(default_factory=list)
    keyframes: list[Keyframe] = field(default_factory=list)
    depth_maps: list = field(default_factory=list)
    camera_poses: list = field(default_factory=list)
    reconstructions: list = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)


class VideoPipeline:
    """
    Video processing pipeline that chains multiple processors.
    
    Usage:
        pipeline = VideoPipeline()
        result = pipeline.run(
            "video.mp4",
            processors=["frame_extractor", "vlm_analyzer", "depth_estimator"],
            config={"frame_extractor": {"interval": 1.0}}
        )
    """
    
    # Default pipeline configurations
    DEFAULT_PIPELINES = {
        "basic": ["frame_extractor"],
        "describe": ["frame_extractor", "vlm_analyzer"],
        "depth": ["frame_extractor", "depth_estimator"],
        "full": ["frame_extractor", "vlm_analyzer", "depth_estimator"],
        "reconstruct": ["frame_extractor", "depth_estimator", "colmap_sfm"],
    }
    
    def __init__(self):
        self.processors = list_video_processors()
    
    def run(
        self,
        video_path: str | Path,
        processors: list[str] | None = None,
        pipeline: str | None = None,
        config: dict[str, Any] | None = None,
        output_dir: str | Path | None = None,
    ) -> PipelineResult:
        """
        Run video processing pipeline.
        
        Args:
            video_path: Path to input video
            processors: List of processor names to run in order
            pipeline: Named pipeline preset ("basic", "describe", "depth", "full", "reconstruct")
            config: Per-processor configuration dict
            output_dir: Base output directory
            
        Returns:
            PipelineResult with all extracted data
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Resolve processor list
        if pipeline:
            if pipeline not in self.DEFAULT_PIPELINES:
                raise ValueError(f"Unknown pipeline: {pipeline}. Available: {list(self.DEFAULT_PIPELINES.keys())}")
            processors = self.DEFAULT_PIPELINES[pipeline]
        elif not processors:
            processors = self.DEFAULT_PIPELINES["basic"]
        
        config = config or {}
        output_dir = Path(output_dir or ".ai_db/video_output")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize result
        result = PipelineResult(video_path=str(video_path))
        shared_config = {"output_dir": str(output_dir), **config.get("global", {})}
        
        # Run each processor in sequence
        for proc_name in processors:
            processor = self.processors.get(proc_name)
            if not processor:
                error = f"Processor not found: {proc_name}"
                _logger.error(error)
                result.errors[proc_name] = error
                continue
            
            # Check dependencies
            missing = processor.check_dependencies()
            if missing:
                error = f"Missing dependencies for {proc_name}: {missing}"
                _logger.error(error)
                result.errors[proc_name] = error
                continue
            
            # Build processor config
            proc_config = {**shared_config, **config.get(proc_name, {})}
            
            # Pass results from previous stages
            if result.keyframes:
                proc_config["keyframes"] = result.keyframes
            if result.depth_maps:
                proc_config["depth_maps"] = result.depth_maps
            if result.camera_poses:
                proc_config["camera_poses"] = result.camera_poses
            if result.reconstructions:
                proc_config["reconstructions"] = result.reconstructions
            
            _logger.info(f"Running video processor: {proc_name}")
            try:
                proc_result = processor.process(video_path, proc_config)
                
                # Merge results
                if proc_result.keyframes:
                    result.keyframes.extend(proc_result.keyframes)
                if proc_result.depth_maps:
                    result.depth_maps.extend(proc_result.depth_maps)
                if proc_result.camera_poses:
                    result.camera_poses.extend(proc_result.camera_poses)
                if proc_result.reconstructions:
                    result.reconstructions.extend(proc_result.reconstructions)
                
                result.metadata[proc_name] = proc_result.metadata
                result.stages_run.append(proc_name)
                
            except Exception as e:
                error = f"{proc_name} failed: {e}"
                _logger.error(error)
                result.errors[proc_name] = str(e)
        
        return result
    
    def list_available(self) -> dict[str, VideoProcessor]:
        """List all available processors with their capabilities."""
        return {
            name: {
                "version": p.version,
                "provides": p.provides,
                "requires": p.requires,
                "heavy": p.heavy,
            }
            for name, p in self.processors.items()
        }


def create_pipeline_from_preset(preset: str) -> list[str]:
    """Get processor list for a named preset."""
    pipeline = VideoPipeline()
    if preset not in pipeline.DEFAULT_PIPELINES:
        raise ValueError(f"Unknown preset: {preset}")
    return pipeline.DEFAULT_PIPELINES[preset]
