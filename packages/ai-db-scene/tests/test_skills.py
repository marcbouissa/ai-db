"""Skills shipped with ai-db-scene: well-formed, indexed, and their code runs in Blender."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from ai_db_scene import skills_dir

SKILLS = sorted(Path(skills_dir()).glob("*/SKILL.md"))


def _code(skill: str) -> str:
    """The first ```python block of a skill: the library the usage examples build on."""
    text = (Path(skills_dir()) / skill / "SKILL.md").read_text()
    return re.search(r"```python\n(.*?)```", text, re.DOTALL).group(1)


def test_skills_have_frontmatter_and_triggers():
    assert {p.parent.name for p in SKILLS} >= {"pbr-shader-builder", "blender-material-library",
                                                "ai-db-scene-pipeline"}
    for p in SKILLS:
        text = p.read_text()
        assert text.startswith("---\nname: " + p.parent.name + "\n"), p
        assert "\ndescription: " in text and "triggers:" in text, p


def test_skill_code_compiles():
    for p in SKILLS:
        for block in re.findall(r"```python\n(.*?)```", p.read_text(), re.DOTALL):
            compile(block, str(p), "exec")


def test_plugin_contributes_the_skills_dir(monkeypatch):
    from ai_db.plugins import plugin_skill_dirs
    assert skills_dir() in plugin_skill_dirs()


def test_skills_are_indexed_and_routed(tmp_path):
    from ai_db.config import parse_config
    from ai_db.config_template import build_template
    from ai_db.facade import VectorDB
    db = VectorDB(str(tmp_path / "k.db"), config=parse_config(build_template()))
    try:
        res = db.sync_skills(skill_dirs=[skills_dir()], verbose=False)
        assert res["added"] >= 3
        top = db.route_skills("build a PBR material from albedo roughness normal maps", top_k=1)
        assert top and top[0]["name"] == "pbr-shader-builder"
    finally:
        db.close()


_MAKE_TEXTURES = """
import bpy, os
folder = FOLDER
os.makedirs(folder, exist_ok=True)
for stem, px in (("oak_diff_2k", (0.6, 0.4, 0.2, 1)), ("oak_rough_2k", (0.7, 0.7, 0.7, 1)),
                 ("oak_nor_dx_2k", (0.5, 0.5, 1.0, 1)), ("oak_ao_2k", (0.9, 0.9, 0.9, 1)),
                 ("oak_disp_2k", (0.5, 0.5, 0.5, 1)), ("readme", (0, 0, 0, 1))):
    img = bpy.data.images.new(stem, 8, 8)
    img.pixels = list(px) * 64
    img.filepath_raw = os.path.join(folder, stem + (".png" if stem != "readme" else ".txt"))
    img.file_format = "PNG"
    img.save()
bpy.ops.mesh.primitive_cube_add(size=1)
bpy.context.active_object.name = "PBRCube"
result = {"ok": True}
"""


@pytest.mark.blender
def test_pbr_builder_runs_in_blender(tmp_path):
    from ai_db_scene.blender_client import BlenderClient, try_get_blender_client
    if try_get_blender_client() is None:
        pytest.skip("no Blender with the addon on the configured port")
    c = BlenderClient(timeout=120)
    folder = str(tmp_path / "tex")
    c.run_code(_MAKE_TEXTURES.replace("FOLDER", repr(folder)))
    res = c.run_code(_code("pbr-shader-builder") + f"""
maps = find_pbr_maps({folder!r})
mat = build_pbr_material("oak_bench", maps, projection="box", scale=2.0)
add_grunge(mat, amount=0.4)
assign_material("PBRCube", mat)
nt = mat.node_tree
bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
result = {{"maps": sorted(maps), "linked": sorted(s.name for s in bsdf.inputs if s.is_linked),
          "spaces": sorted({{n.image.colorspace_settings.name for n in nt.nodes if n.type == "TEX_IMAGE"}}),
          "cube": [m.name for m in bpy.data.objects["PBRCube"].data.materials]}}
""")
    assert res["maps"] == ["ao", "base_color", "height", "normal_dx", "roughness"]
    assert {"Base Color", "Roughness", "Normal"} <= set(res["linked"])
    assert res["spaces"] == ["Non-Color", "sRGB"]
    assert res["cube"] == ["oak_bench"]
    c.run_code("import bpy\nbpy.data.objects.remove(bpy.data.objects['PBRCube'])\nresult = {}")
