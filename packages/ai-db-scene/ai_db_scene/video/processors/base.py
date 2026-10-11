"""
Video Processor Extensions for ai-db.
======================================
Follows the exact same pattern as ChunkerExtension in ai_db.ext.base.
Extensions are loaded via entry points 'ai_db_scene.video_processors' or direct registration.
"""

import importlib.metadata
import importlib.util
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

# Never print() here. This module is imported by the stdio MCP server, where
# stdout is the JSON-RPC channel: a single stray line makes the client fail its
# parse and drop the connection. Diagnostics go to stderr via the shared logger.
from ai_db.logger import _logger


@dataclass
class Keyframe:
    """A single extracted keyframe from video."""
    index: int
    timestamp: float          # seconds
    filepath: str             # path to saved frame image
    width: int
    height: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DepthMap:
    """Per-pixel depth estimation for a frame."""
    keyframe_index: int
    filepath: str             # path to saved depth map (npy/png)
    scale: float              # metric scale factor if known
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CameraPose:
    """Camera pose for a frame (from SfM/matchmove)."""
    keyframe_index: int
    position: tuple[float, float, float]    # world position (x, y, z)
    rotation: tuple[float, float, float, float]  # quaternion (x, y, z, w)
    focal_length: float | None = None       # in pixels
    sensor_width: float | None = None       # in mm
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Reconstruction:
    """3D reconstruction output (point cloud, mesh, Gaussian splat, etc.)."""
    type: str                 # "point_cloud" | "mesh" | "gaussian_splat" | "nerf"
    filepath: str             # path to output file (.ply, .obj, .splat, etc.)
    camera_poses: list[CameraPose] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VideoProcessorResult:
    """Aggregated result from a video processor pipeline."""
    keyframes: list[Keyframe] = field(default_factory=list)
    depth_maps: list[DepthMap] = field(default_factory=list)
    camera_poses: list[CameraPose] = field(default_factory=list)
    reconstructions: list[Reconstruction] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class VideoProcessor(ABC):
    """
    Base class for video processors.
    
    Registered via the entry point group 'ai_db_scene.video_processors' and loaded on
    first use of the registry (never at import: processors may need torch).
    """
    
    # Unique processor name (used for registration & CLI)
    name: str = ""
    
    # Version for compatibility
    version: str = "1.0.0"
    
    # Importable module names this processor needs (e.g. "cv2", not the PyPI name
    # "opencv-python-headless"). Checked with find_spec, which does not import them.
    requires: ClassVar[list[str]] = []
    
    # What this processor provides: "keyframes", "depth", "poses", "reconstruction"
    provides: ClassVar[list[str]] = []
    
    # Whether this is a "heavy" processor (GPU, slow)
    heavy: bool = False

    @abstractmethod
    def process(self, video_path: Path, config: dict[str, Any]) -> VideoProcessorResult:
        """
        Process a video file and return results.
        
        Args:
            video_path: Path to input video file
            config: Processor-specific configuration
            
        Returns:
            VideoProcessorResult with extracted data
        """

    def check_dependencies(self) -> list[str]:
        """Check if required dependencies are installed. Returns missing deps."""
        return [mod for mod in self.requires if importlib.util.find_spec(mod) is None]

    def can_handle(self, video_path: Path) -> bool:
        """Check if this processor can handle the video (by extension)."""
        video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".m4v"}
        return video_path.suffix.lower() in video_exts


# ============================================================================
# Registry (mirrors _CHUNKERS in ai_db.ext.base)
# ============================================================================

_VIDEO_PROCESSORS: dict[str, VideoProcessor] = {}


def register_video_processor(processor: VideoProcessor) -> None:
    """Register a video processor for its name."""
    if not processor.name:
        raise ValueError("VideoProcessor must have a name")
    _VIDEO_PROCESSORS[processor.name] = processor


_LOADED = False


def _ensure_loaded() -> None:
    global _LOADED
    if not _LOADED:
        _LOADED = True
        load_video_processors_from_entry_points()


def get_video_processor(name: str) -> VideoProcessor | None:
    """Get registered video processor by name (loads the entry points on first use)."""
    _ensure_loaded()
    return _VIDEO_PROCESSORS.get(name)


def list_video_processors() -> dict[str, VideoProcessor]:
    """List all registered video processors (loads the entry points on first use)."""
    _ensure_loaded()
    return dict(_VIDEO_PROCESSORS)


def load_video_processors_from_entry_points() -> int:
    """
    Discover and load video processors from entry points 'ai_db_scene.video_processors'.
    Mirrors the chunker loading pattern.
    Returns number of processors loaded.
    """
    loaded = 0
    for ep in importlib.metadata.entry_points(group="ai_db_scene.video_processors"):
        try:
            processor_cls = ep.load()
            if not isinstance(processor_cls, type) or not issubclass(processor_cls, VideoProcessor):
                _logger.warning("Video processor entry point '%s' is not a VideoProcessor subclass", ep.name)
                continue

            processor = processor_cls()
            missing = processor.check_dependencies()
            if missing:
                _logger.info("Video processor '%s' missing deps: %s -- skipping", processor.name, missing)
                continue

            register_video_processor(processor)
            loaded += 1
            _logger.info("Loaded video processor: %s v%s", processor.name, processor.version)
        except (ImportError, AttributeError, TypeError) as e:
            _logger.warning("Failed to load video processor %s: %s", ep.name, e)
    
    return loaded

