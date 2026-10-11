---
name: pbr-shader-builder
description: Build physically based (PBR) Blender materials from texture map sets (albedo, roughness, metallic, normal GL/DX, AO, height, opacity, gloss) with automatic map detection, correct colour spaces, box or UV projection, bump or true displacement, and a procedural grunge layer for wear. Runs through execute_blender_code.
license: MIT
metadata:
  author: ai-db
  version: "1.0.0"
  domain: blender
  role: specialist
  triggers: pbr, pbr material, pbr shader, texture set, albedo roughness normal, normal map, displacement, height map, ambient occlusion, metallic workflow, polyhaven textures, ambientcg, grunge, wear, weathering, shader nodes
  related-skills: blender-material-library, materials, texture-workflow, uv-workflow, lookdev
---

# PBR shader builder

Turn a folder of texture maps into a correct Principled BSDF node tree in one call, or add
procedural wear to any material. The code below is tested against Blender 5.x by ai-db's
test suite; run it unchanged with `execute_blender_code` (or `run_code`).

## When to Use

- You have a texture set (Poly Haven, ambientCG, Quixel, photogrammetry) and need a material.
- An object needs believable surface variation without textures (grunge, wear, dust).
- Placeholders from `apply_layout` should get a real texture set instead of a procedural material.

For quick looks without texture files, use `apply_material` (see blender-material-library).

## Rules that make PBR look right

1. **Colour space:** only base colour (albedo) and emission are `sRGB`. Roughness, metallic,
   normal, AO, height, opacity are data: `Non-Color`. Wrong colour space = washed-out or
   too-shiny surfaces.
2. **Normal maps** go through a *Normal Map* node, never straight into `Normal`.
   Blender is OpenGL (+Y): a DirectX map (`_dx`, `_normaldx`) needs its green channel inverted.
3. **Gloss** maps are inverted roughness.
4. **AO** multiplies into base colour only for EEVEE/stylised looks; Cycles computes its own
   occlusion, so keep `ao_strength` low (default 0.5) or 0.
5. **Height**: `bump` (default) works everywhere and needs no dense mesh; `displacement`
   needs subdivided geometry and Cycles (`displacement_method = BOTH`).
6. **Projection**: `box` (default) needs no UVs and suits architecture and placeholders;
   use `uv` for unwrapped assets.
7. **Scale** is texture repeats per metre (box) or per UV unit (uv): a 2 m wood plank texture
   on a 4 m floor wants `scale=0.5`.

## Code

```python
import bpy
import os
import re

_MAPS = {  # role -> name tokens (matched between _ - . or string ends, case-insensitive)
    "base_color": ("albedo", "basecolor", "base_color", "basecolour", "diffuse", "diff", "col", "color", "colour"),
    "roughness": ("roughness", "rough", "rgh"),
    "gloss": ("gloss", "glossiness"),
    "metallic": ("metallic", "metalness", "metal", "met"),
    "normal_dx": ("normaldx", "normal_dx", "nor_dx", "nrm_dx"),
    "normal": ("normal", "normalgl", "normal_gl", "nor_gl", "nor", "nrm"),
    "ao": ("ao", "ambientocclusion", "ambient_occlusion", "occlusion"),
    "height": ("height", "displacement", "disp", "bump"),
    "opacity": ("opacity", "alpha", "mask"),
    "emission": ("emission", "emissive", "emit"),
}
_IMAGE_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr", ".tga", ".webp")


def find_pbr_maps(folder):
    """{role: path} for the texture files in `folder` (first match per role; *_dx beats normal)."""
    found = {}
    for fname in sorted(os.listdir(folder)):
        stem, ext = os.path.splitext(fname.lower())
        if ext not in _IMAGE_EXT:
            continue
        for role, tokens in _MAPS.items():
            if role in found:
                continue
            if any(re.search(r"(^|[_\-. ])" + re.escape(t) + r"($|[_\-. ])", stem) for t in tokens):
                found[role] = os.path.join(folder, fname)
                break
    if "normal_dx" in found and "normal" in found and found["normal"] == found["normal_dx"]:
        del found["normal"]
    return found


def build_pbr_material(name, maps, projection="box", scale=1.0, displacement="bump",
                       bump_strength=0.3, ao_strength=0.5, box_blend=0.2):
    """Principled BSDF material from {role: image path}; returns the material."""
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (900, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (550, 0)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    coord = nt.nodes.new("ShaderNodeTexCoord")
    coord.location = (-900, 0)
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.location = (-700, 0)
    mapping.inputs["Scale"].default_value = (scale, scale, scale)
    nt.links.new(coord.outputs["UV" if projection == "uv" else "Object"], mapping.inputs["Vector"])

    def tex(role, y):
        node = nt.nodes.new("ShaderNodeTexImage")
        node.location = (-400, y)
        node.label = role
        node.image = bpy.data.images.load(maps[role], check_existing=True)
        node.image.colorspace_settings.name = "sRGB" if role in ("base_color", "emission") else "Non-Color"
        if projection == "box":
            node.projection = "BOX"
            node.projection_blend = box_blend
        nt.links.new(mapping.outputs["Vector"], node.inputs["Vector"])
        return node

    color = None
    if "base_color" in maps:
        color = tex("base_color", 400).outputs["Color"]
        if "ao" in maps and ao_strength > 0:
            ao = tex("ao", 250)
            mix = nt.nodes.new("ShaderNodeMix")
            mix.data_type = "RGBA"
            mix.blend_type = "MULTIPLY"
            mix.location = (150, 350)
            mix.inputs["Factor"].default_value = ao_strength
            nt.links.new(color, mix.inputs["A"])
            nt.links.new(ao.outputs["Color"], mix.inputs["B"])
            color = mix.outputs["Result"]
        nt.links.new(color, bsdf.inputs["Base Color"])
    if "roughness" in maps:
        nt.links.new(tex("roughness", 100).outputs["Color"], bsdf.inputs["Roughness"])
    elif "gloss" in maps:
        inv = nt.nodes.new("ShaderNodeInvert")
        inv.location = (150, 100)
        nt.links.new(tex("gloss", 100).outputs["Color"], inv.inputs["Color"])
        nt.links.new(inv.outputs["Color"], bsdf.inputs["Roughness"])
    if "metallic" in maps:
        nt.links.new(tex("metallic", -50).outputs["Color"], bsdf.inputs["Metallic"])
    if "opacity" in maps:
        nt.links.new(tex("opacity", -200).outputs["Color"], bsdf.inputs["Alpha"])
    if "emission" in maps:
        nt.links.new(tex("emission", -350).outputs["Color"], bsdf.inputs["Emission Color"])
        bsdf.inputs["Emission Strength"].default_value = 1.0

    normal = None
    nkey = "normal" if "normal" in maps else ("normal_dx" if "normal_dx" in maps else None)
    if nkey:
        ncol = tex(nkey, -500).outputs["Color"]
        if nkey == "normal_dx":  # DirectX -> OpenGL: invert green
            sep = nt.nodes.new("ShaderNodeSeparateColor")
            sep.location = (-150, -600)
            inv = nt.nodes.new("ShaderNodeMath")
            inv.operation = "SUBTRACT"
            inv.inputs[0].default_value = 1.0
            inv.location = (0, -650)
            comb = nt.nodes.new("ShaderNodeCombineColor")
            comb.location = (150, -600)
            nt.links.new(ncol, sep.inputs["Color"])
            nt.links.new(sep.outputs["Red"], comb.inputs["Red"])
            nt.links.new(sep.outputs["Green"], inv.inputs[1])
            nt.links.new(inv.outputs["Value"], comb.inputs["Green"])
            nt.links.new(sep.outputs["Blue"], comb.inputs["Blue"])
            ncol = comb.outputs["Color"]
        nmap = nt.nodes.new("ShaderNodeNormalMap")
        nmap.location = (300, -500)
        if projection != "uv":
            nmap.space = "OBJECT"  # box projection has no UV tangents
        nt.links.new(ncol, nmap.inputs["Color"])
        normal = nmap.outputs["Normal"]

    if "height" in maps and displacement == "displacement":
        disp = nt.nodes.new("ShaderNodeDisplacement")
        disp.location = (550, -450)
        disp.inputs["Scale"].default_value = 0.05
        nt.links.new(tex("height", -800).outputs["Color"], disp.inputs["Height"])
        nt.links.new(disp.outputs["Displacement"], out.inputs["Displacement"])
        if hasattr(mat, "displacement_method"):
            mat.displacement_method = "BOTH"
    elif "height" in maps:
        bump = nt.nodes.new("ShaderNodeBump")
        bump.location = (350, -750)
        bump.inputs["Strength"].default_value = bump_strength
        nt.links.new(tex("height", -800).outputs["Color"], bump.inputs["Height"])
        if normal is not None:
            nt.links.new(normal, bump.inputs["Normal"])
        normal = bump.outputs["Normal"]
    if normal is not None:
        nt.links.new(normal, bsdf.inputs["Normal"])
    if "opacity" in maps and hasattr(mat, "surface_render_method"):
        mat.surface_render_method = "DITHERED"
    return mat


def add_grunge(mat, amount=0.5, scale=8.0, bump_strength=0.15, seed=0.0):
    """Procedural wear on any Principled material: roughness variation, darkened base colour
    in the crevices of a noise pattern, and a fine bump. amount 0..1."""
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.location = (-200, -1000)
    noise.noise_dimensions = "4D"  # before W: the W input only exists in 4D
    noise.inputs["Scale"].default_value = scale
    noise.inputs["Detail"].default_value = 8.0
    noise.inputs["W"].default_value = seed
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.location = (0, -1000)
    ramp.color_ramp.elements[0].position = 0.45
    ramp.color_ramp.elements[1].position = 0.7
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    mask = ramp.outputs["Color"]

    def mixed(socket_name, value_b, y):
        sock = bsdf.inputs[socket_name]
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA" if socket_name == "Base Color" else "FLOAT"
        mix.location = (300, y)
        fac = nt.nodes.new("ShaderNodeMath")
        fac.operation = "MULTIPLY"
        fac.inputs[1].default_value = amount
        fac.location = (150, y - 100)
        nt.links.new(mask, fac.inputs[0])
        nt.links.new(fac.outputs["Value"], mix.inputs["Factor"])
        a_in, b_in = (mix.inputs["A"], mix.inputs["B"]) if mix.data_type == "RGBA" else (mix.inputs[2], mix.inputs[3])
        if sock.is_linked:
            nt.links.new(sock.links[0].from_socket, a_in)
        else:
            a_in.default_value = sock.default_value
        b_in.default_value = value_b
        out = mix.outputs["Result"] if mix.data_type == "RGBA" else mix.outputs[0]
        nt.links.new(out, sock)

    mixed("Roughness", 0.9, -1100)
    mixed("Base Color", (0.05, 0.045, 0.04, 1.0), -1300)
    bump = nt.nodes.new("ShaderNodeBump")
    bump.location = (350, -1500)
    bump.inputs["Strength"].default_value = bump_strength
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    if bsdf.inputs["Normal"].is_linked:
        nt.links.new(bsdf.inputs["Normal"].links[0].from_socket, bump.inputs["Normal"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def assign_material(obj_name, mat):
    """Make `mat` the only material of an object (and of every mesh below an asset root)."""
    root = bpy.data.objects[obj_name]
    for o in [root, *root.children_recursive]:
        if o.type == "MESH":
            o.data.materials.clear()
            o.data.materials.append(mat)
```

## Usage

Append one of these to the code above in the same `execute_blender_code` call:

```python
# texture set on disk -> material on an object
maps = find_pbr_maps("/path/to/textures/wood_floor_2k")
mat = build_pbr_material("wood_floor", maps, projection="box", scale=0.5)
assign_material("Room_Floor", mat)
result = {"material": mat.name, "maps": sorted(maps)}
```

```python
# worn metal without textures: start from any Principled material, then add grunge
mat = bpy.data.materials.new("worn_steel")
mat.use_nodes = True
p = mat.node_tree.nodes["Principled BSDF"]
p.inputs["Metallic"].default_value = 1.0
p.inputs["Roughness"].default_value = 0.35
add_grunge(mat, amount=0.6, scale=6.0)
assign_material("tool_cabinet", mat)
```

## Checks after building

- `validate_scene` (materials check) passes: every mesh has a material.
- Render a probe with `render` at low samples and look: plastic-looking wood means the
  roughness map is sRGB or missing; inverted dents mean a DirectX normal map was read as GL.
- Box projection seams: raise `box_blend` (0.2 → 0.4).
