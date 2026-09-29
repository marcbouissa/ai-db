"""
Video Processor Factory.
=========================
Loads processors from entry points (mirrors chunker loading in ai_db.ext).
"""

from ai_db.video.processors.base import load_video_processors_from_entry_points


def load_video_processors() -> int:
    """Load all video processors from entry points. Returns count loaded."""
    return load_video_processors_from_entry_points()
