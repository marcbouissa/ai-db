"""
VLM Analyzer Video Processor.
==============================
Uses Vision Language Models (GPT-4V, Claude, LLaVA) to analyze keyframes
and generate 3D scene descriptions.
Requires video-vlm extra: openai, anthropic
"""

from pathlib import Path
from typing import Any

from ai_db.video.processors.base import (
    VideoProcessor, VideoProcessorResult, register_video_processor
)


class VLMAnalyzer(VideoProcessor):
    """Analyze keyframes with Vision LLM to extract 3D scene understanding."""
    
    name = "vlm_analyzer"
    version = "1.0.0"
    requires = ["openai"]  # or anthropic
    provides = ["scene_description", "object_list", "spatial_layout"]
    heavy = True

    def __init__(self):
        super().__init__()
        self._client = None
        self._model = "gpt-4-vision-preview"
    
    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI()
            except ImportError:
                try:
                    import anthropic
                    self._client = anthropic.Anthropic()
                    self._model = "claude-3-opus-20240229"
                except ImportError:
                    raise RuntimeError("VLM analyzer requires openai or anthropic package")
        return self._client

    def process(self, video_path: Path, config: dict[str, Any]) -> VideoProcessorResult:
        # This processor expects keyframes from a previous processor
        # In pipeline mode, it receives keyframes via config
        keyframes = config.get("keyframes", [])
        if not keyframes:
            return VideoProcessorResult(
                metadata={"error": "No keyframes provided to VLM analyzer"}
            )
        
        client = self._get_client()
        is_anthropic = hasattr(client, 'messages')
        
        # Build analysis prompt
        prompt = config.get("prompt", """
        Analyze these video keyframes as a 3D scene. Describe:
        1. Overall scene type (indoor/outdoor, room type, environment)
        2. Major objects with approximate 3D positions (left/center/right, near/far)
        3. Camera movement (static, pan, dolly, orbit)
        4. Lighting conditions
        5. Materials and textures visible
        Output as structured JSON.
        """)
        
        # Process keyframes in batches (API limits)
        batch_size = config.get("batch_size", 4)
        descriptions = []
        
        for i in range(0, len(keyframes), batch_size):
            batch = keyframes[i:i+batch_size]
            
            if is_anthropic:
                # Anthropic format
                content = [{"type": "text", "text": prompt}]
                for kf in batch:
                    import base64
                    with open(kf.filepath, "rb") as f:
                        img_data = base64.b64encode(f.read()).decode()
                    content.append({
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": img_data}
                    })
                
                response = client.messages.create(
                    model=self._model,
                    max_tokens=2000,
                    messages=[{"role": "user", "content": content}]
                )
                descriptions.append(response.content[0].text)
            else:
                # OpenAI format
                content = [{"type": "text", "text": prompt}]
                for kf in batch:
                    import base64
                    with open(kf.filepath, "rb") as f:
                        img_data = base64.b64encode(f.read()).decode()
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}
                    })
                
                response = client.chat.completions.create(
                    model=self._model,
                    max_tokens=2000,
                    messages=[{"role": "user", "content": content}]
                )
                descriptions.append(response.choices[0].message.content)
        
        return VideoProcessorResult(
            metadata={
                "video_path": str(video_path),
                "descriptions": descriptions,
                "model": self._model,
                "keyframes_analyzed": len(keyframes),
            }
        )


# Auto-register on import
register_video_processor(VLMAnalyzer())
