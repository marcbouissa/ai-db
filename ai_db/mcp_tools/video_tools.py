"""
MCP Tools for Video Processing.
================================
Registers with ServiceDispatcher following existing tool patterns.
"""

from typing import Any
from pathlib import Path
from ai_db.dispatcher import ServiceDispatcher
from ai_db.video.pipeline import VideoPipeline
from ai_db.logger import _logger


def register_video_tools(dispatcher: ServiceDispatcher) -> None:
    """Register video processing tools with the dispatcher."""
    
    # 1. video_to_scene
    dispatcher.register_tool(
        name="video_to_scene",
        description="Process video and extract 3D scene information (keyframes, depth, descriptions, camera poses)",
        parameters_schema={
            "type": "object",
            "properties": {
                "video_path": {"type": "string", "description": "Path to input video file"},
                "pipeline": {"type": "string", "enum": ["basic", "describe", "depth", "full", "reconstruct"], "description": "Preset pipeline to run"},
                "processors": {"type": "array", "items": {"type": "string"}, "description": "Custom list of processors (overrides pipeline)"},
                "config": {"type": "object", "description": "Per-processor configuration"},
                "output_dir": {"type": "string", "description": "Output directory for extracted data"},
                "blender_collection": {"type": "string", "default": "VideoImport", "description": "Blender collection name for imported assets"},
            },
            "required": ["video_path"],
        },
        handler=_handle_video_to_scene,
        category="video",
    )

    # 2. list_video_processors
    dispatcher.register_tool(
        name="list_video_processors",
        description="List available video processors and their capabilities",
        parameters_schema={"type": "object", "properties": {}},
        handler=_handle_list_video_processors,
        category="video",
    )

    # 3. video_extract_keyframes
    dispatcher.register_tool(
        name="video_extract_keyframes",
        description="Extract keyframes from video (standalone frame extraction)",
        parameters_schema={
            "type": "object",
            "properties": {
                "video_path": {"type": "string", "description": "Path to input video file"},
                "interval": {"type": "number", "default": 2.0, "description": "Seconds between keyframes"},
                "max_frames": {"type": "integer", "default": 100, "description": "Maximum keyframes to extract"},
                "output_dir": {"type": "string", "description": "Output directory for frames"},
            },
            "required": ["video_path"],
        },
        handler=_handle_video_extract_keyframes,
        category="video",
    )

    _logger.info("Registered video processing MCP tools")


# =========================================================================
# Handlers
# =========================================================================

def _handle_video_to_scene(args: dict[str, Any]) -> Any:
    pipeline = VideoPipeline()
    
    result = pipeline.run(
        video_path=args["video_path"],
        pipeline=args.get("pipeline"),
        processors=args.get("processors"),
        config=args.get("config"),
        output_dir=args.get("output_dir"),
    )
    
    # Format response
    response = {
        "video_path": result.video_path,
        "stages_run": result.stages_run,
        "keyframes_extracted": len(result.keyframes),
        "depth_maps": len(result.depth_maps),
        "camera_poses": len(result.camera_poses),
        "reconstructions": len(result.reconstructions),
        "errors": result.errors,
        "metadata": result.metadata,
    }
    
    # Add keyframe info
    if result.keyframes:
        response["keyframes"] = [
            {
                "index": kf.index,
                "timestamp": kf.timestamp,
                "filepath": kf.filepath,
                "width": kf.width,
                "height": kf.height,
            }
            for kf in result.keyframes
        ]
    
    if result.errors:
        response["status"] = "partial"
    else:
        response["status"] = "success"
    
    return response


def _handle_list_video_processors(args: dict[str, Any]) -> Any:
    pipeline = VideoPipeline()
    processors = pipeline.list_available()
    return {
        "processors": processors,
        "presets": pipeline.DEFAULT_PIPELINES,
    }


def _handle_video_extract_keyframes(args: dict[str, Any]) -> Any:
    from ai_db.video.processors.base import get_video_processor
    
    processor = get_video_processor("frame_extractor")
    if not processor:
        return {"error": "frame_extractor not available. Install opencv-python-headless."}
    
    missing = processor.check_dependencies()
    if missing:
        return {"error": f"Missing dependencies: {missing}"}
    
    result = processor.process(
        Path(args["video_path"]),
        {
            "interval": args.get("interval", 2.0),
            "max_frames": args.get("max_frames", 100),
            "output_dir": args.get("output_dir"),
        }
    )
    
    return {
        "keyframes": [
            {
                "index": kf.index,
                "timestamp": kf.timestamp,
                "filepath": kf.filepath,
                "width": kf.width,
                "height": kf.height,
            }
            for kf in result.keyframes
        ],
        "count": len(result.keyframes),
        "metadata": result.metadata,
    }
