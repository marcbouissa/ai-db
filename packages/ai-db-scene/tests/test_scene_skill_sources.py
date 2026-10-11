"""draft_skill video/asset sources of the scene plugin (synthetic files, fake Blender)."""

from __future__ import annotations

import json
import struct
from importlib.metadata import entry_points

import pytest
from ai_db_scene import skill_sources as ss

VTT = """WEBVTT

00:00:01.000 --> 00:00:04.000
Add a <b>cube</b> and scale it to 2 m.

00:00:04.000 --> 00:00:06.000
Add a <b>cube</b> and scale it to 2 m.

00:01:05.500 --> 00:01:09.000
Add a Bevel modifier with 3 segments.
"""

GLTF = {"asset": {"version": "2.0"}, "extensionsUsed": ["KHR_materials_clearcoat"],
        "meshes": [{"name": "Mug", "primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "material": 0}]}],
        "accessors": [{"min": [-0.05, 0.0, -0.05], "max": [0.05, 0.1, 0.05]}, {}],
        "nodes": [{"name": "Mug", "mesh": 0, "scale": [1, 1, 1]}],
        "images": [{"uri": "mug_albedo.png"}], "textures": [{"source": 0}],
        "materials": [{"name": "Ceramic", "pbrMetallicRoughness": {
            "baseColorFactor": [0.9, 0.9, 0.85, 1], "metallicFactor": 0.0, "roughnessFactor": 0.3,
            "baseColorTexture": {"index": 0}}}]}


def test_hook_registered():
    assert "scene" in {ep.name for ep in entry_points(group="ai_db.skill_sources")}
    assert ss.read_source({"url": "https://x"}) is None


def test_video_subtitles_become_timestamped_steps(tmp_path):
    video = tmp_path / "tut.mp4"
    video.write_bytes(b"\x00")
    (tmp_path / "tut.vtt").write_text(VTT)
    out = ss.read_source({"video": str(video), "license": "CC-BY 4.0"})
    assert out["sections"]["workflow"] == ["[00:01] Add a cube and scale it to 2 m.",
                                           "[01:05] Add a Bevel modifier with 3 segments."]
    assert out["source"] == {"video": str(video), "license": "CC-BY 4.0"}


def test_srt_and_refusals(tmp_path):
    srt = tmp_path / "a.srt"
    srt.write_text("1\n00:00:02,000 --> 00:00:03,000\nHello\n\n2\n00:00:10,000 --> 00:00:11,000\nWorld\n")
    assert ss.parse_subtitles(srt.read_text()) == [(2.0, "Hello"), (10.0, "World")]
    with pytest.raises(ValueError, match="URL"):
        ss.read_source({"video": "https://youtube.com/watch?v=x"})
    with pytest.raises(ValueError, match="not a file"):
        ss.read_source({"video": str(tmp_path / "missing.mp4")})


def test_video_without_subtitles_falls_back_to_keyframes(tmp_path, monkeypatch):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"\x00")
    monkeypatch.setattr(ss, "_keyframes", lambda p: [{"timestamp": 12.0, "filepath": "/tmp/f1.png"}])
    out = ss.read_source({"video": str(video)})
    assert out["sections"]["examples"] == ["keyframe [00:12]: /tmp/f1.png"]
    assert "no .vtt/.srt" in out["warnings"][0]


def test_gltf_and_glb_summaries(tmp_path):
    g = tmp_path / "mug.gltf"
    g.write_text(json.dumps(GLTF))
    lines = ss.read_source({"asset": str(g)})["sections"]["asset_build"]
    assert lines[0] == "dimensions (mesh space, m): x 0.100, y 0.100, z 0.100"
    assert any(ln.startswith("mesh Mug: 1 primitive(s), attributes NORMAL, POSITION") for ln in lines)
    assert any("material Ceramic" in ln and "roughness 0.3" in ln and "mug_albedo.png" in ln for ln in lines)
    assert lines[-1] == "extensions: KHR_materials_clearcoat"
    body = json.dumps(GLTF).encode()
    body += b" " * (-len(body) % 4)
    glb = tmp_path / "mug.glb"
    glb.write_bytes(b"glTF" + struct.pack("<II", 2, 20 + len(body)) + struct.pack("<II", len(body), 0x4E4F534A) + body)
    assert ss.read_source({"asset": str(glb)})["sections"]["asset_build"] == lines


class _FakeBlender:
    def __init__(self):
        self.code = ""

    def run_code(self, code, timeout=30.0):
        self.code = code
        compile(code, "summary", "exec")  # the generated script is valid Python
        return {"objects": [{"name": "Mug", "type": "MESH", "dimensions": [0.1, 0.1, 0.12], "children": [],
                             "modifiers": [{"name": "Bevel", "type": "BEVEL", "props": {"width": 0.002, "segments": 3}},
                                           {"name": "GN", "type": "NODES", "props": {}, "node_group": {
                                               "name": "Handle", "nodes": [{"name": "Curve Circle", "type": "GeometryNodeCurvePrimitiveCircle",
                                                                            "inputs": {"Radius": 0.04}}],
                                               "links": ["Curve Circle.Curve -> Group Output.Geometry"]}}],
                             "materials": [{"name": "Ceramic", "tree": {"name": "Shader Nodetree", "nodes": [
                                 {"name": "Principled BSDF", "type": "ShaderNodeBsdfPrincipled", "inputs": {"Roughness": 0.3}}],
                                 "links": []}}]}]}


def test_blender_object_summary_with_fake_client():
    fake = _FakeBlender()
    out = ss.read_asset({"asset": "Mug"}, client=fake)
    lines = out["sections"]["asset_build"]
    assert "TARGET = 'Mug'" in fake.code
    assert lines[0] == "object Mug (MESH), dimensions [0.1, 0.1, 0.12] m"
    assert 'modifier 1: BEVEL \'Bevel\' {"width": 0.002, "segments": 3}' in lines[1]
    assert any("geometry nodes node GeometryNodeCurvePrimitiveCircle 'Curve Circle' {\"Radius\": 0.04}" in ln for ln in lines)
    assert any("shader node ShaderNodeBsdfPrincipled" in ln for ln in lines)


def test_blend_file_uses_library_load(tmp_path):
    blend = tmp_path / "mug.blend"
    blend.write_bytes(b"BLENDER")
    fake = _FakeBlender()
    ss.read_asset({"asset": str(blend)}, client=fake)
    assert "libraries.load" in fake.code and str(blend) in fake.code and "libraries.remove" in fake.code
