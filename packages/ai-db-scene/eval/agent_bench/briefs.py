"""The benchmark briefs and their must-have checklists.

A checklist item is (label, how many, name/category tokens): an object counts for the item
when one of the tokens appears in its name or its ``category`` property. Counting by name
works for every arm, including agents that never tag categories.
"""

from __future__ import annotations

BRIEFS: dict[str, dict] = {
    # not in run_arms.sh's default set: run on demand (bench.py run <arm> oval_office)
    "oval_office": {
        "brief": ("The Oval Office: an oval room in classic style with a large desk in front of three tall "
                  "windows, two sofas facing each other across a coffee table on an oval rug, two armchairs "
                  "by the fireplace on the opposite wall, and paintings on the walls."),
        "checklist": [("desk", 1, ("desk",)), ("sofas", 2, ("sofa", "couch")), ("coffee table", 1, ("coffee",)),
                      ("rug", 1, ("rug", "carpet")), ("armchairs", 2, ("armchair", "arm_chair")),
                      ("fireplace", 1, ("fireplace", "mantel")), ("windows", 3, ("window",)),
                      ("paintings", 2, ("picture", "painting", "frame"))],
    },
    "living_room": {
        "brief": ("A cozy living room: a sofa facing a TV on a TV stand, a coffee table in front of "
                  "the sofa on a rug, two armchairs and a floor lamp."),
        "checklist": [("sofa", 1, ("sofa", "couch")), ("tv stand", 1, ("tv_stand", "tvstand", "media", "console")),
                      ("tv", 1, ("tv", "television", "screen")), ("coffee table", 1, ("coffee",)),
                      ("rug", 1, ("rug", "carpet")), ("armchair", 2, ("armchair", "arm_chair")),
                      ("floor lamp", 1, ("lamp",))],
    },
    "classroom": {
        "brief": ("A classroom with a teacher desk and chair, a whiteboard on the front wall, and twelve "
                  "student desks with chairs in rows facing the whiteboard."),
        "checklist": [("desks", 13, ("desk",)), ("chairs", 13, ("chair",)),
                      ("whiteboard", 1, ("whiteboard", "board"))],
    },
    "cafe": {
        "brief": ("A small cafe with four round tables with two chairs each, a counter with two bar "
                  "stools, and three potted plants."),
        "checklist": [("tables", 4, ("table",)), ("chairs", 8, ("chair",)), ("counter", 1, ("counter", "bar")),
                      ("stools", 2, ("stool",)), ("plants", 3, ("plant",))],
    },
    "round_lobby": {
        "brief": ("A round hotel lobby with a reception desk, four armchairs around a low table, two "
                  "potted plants and a pendant lamp."),
        "checklist": [("reception desk", 1, ("reception", "desk")), ("armchairs", 4, ("armchair", "chair")),
                      ("low table", 1, ("table",)), ("plants", 2, ("plant",)),
                      ("pendant lamp", 1, ("pendant", "lamp", "light"))],
    },
    "park_corner": {
        "brief": ("A park corner outdoors: three benches facing a fountain, two trees, a trash bin and "
                  "two street lamps."),
        "checklist": [("benches", 3, ("bench",)), ("fountain", 1, ("fountain",)), ("trees", 2, ("tree",)),
                      ("trash bin", 1, ("trash", "bin", "litter")), ("street lamps", 2, ("lamp", "light", "post"))],
    },
    "warehouse": {
        "brief": ("An industrial warehouse: four pallet racks in rows, pallets with crates, a forklift, a "
                  "workbench, safety barriers and ceiling lights."),
        "checklist": [("pallet racks", 4, ("rack",)), ("pallets", 2, ("pallet",)), ("crates", 2, ("crate", "box")),
                      ("forklift", 1, ("forklift",)), ("workbench", 1, ("workbench", "bench")),
                      ("safety barriers", 2, ("barrier", "railing", "guard")),
                      ("ceiling lights", 2, ("ceiling", "light", "lamp"))],
    },
    "beach_cove": {
        "brief": ("A tropical beach cove outdoors: a sandy beach sloping into calm turquoise water, "
                  "five palm trees, dense green bushes and ferns along the back, a wooden boardwalk "
                  "from the bushes to the water, two sun loungers under a parasol, and a few rocks "
                  "at the water's edge."),
        "checklist": [("sand", 1, ("sand", "beach")), ("water", 1, ("water", "sea", "ocean", "lagoon")),
                      ("palm trees", 5, ("palm", "tree")), ("bushes", 3, ("bush", "shrub", "fern", "hedge")),
                      ("boardwalk", 1, ("boardwalk", "walkway", "deck", "path")),
                      ("sun loungers", 2, ("lounger", "deckchair", "sunbed")),
                      ("parasol", 1, ("parasol", "umbrella")), ("rocks", 2, ("rock", "stone", "boulder"))],
    },
}

TASK = """Build this 3D scene in Blender: {brief}

Requirements: real-world scale in metres with the floor at z=0; every requested object present
and recognisable; a material on every object; lighting; an active scene camera that frames the
scene. Work only through the provided Blender and ai-db tools: never use a shell, files or
scripts outside them. When the scene is complete, reply DONE followed by a one-paragraph summary."""

HINT_BLENDER = ("Blender tools are reached through discover_servers / discover_tools / "
                "get_tool_schema / invoke_tool.")
HINT_AIDB = ("ai-db tools are also available: a scene pipeline (get_scene_spec_schema -> "
             "submit_scene_spec -> solve_layout -> apply_layout -> validate_scene), scene memory "
             "(snapshot_scene, restore_scene) and route_skill for Blender how-to skills.")

#: the ai-db tools an agent sees in the "with ai-db" arm (code-search tools are irrelevant here)
AIDB_TOOLS = ("get_scene_spec_schema,submit_scene_spec,solve_layout,apply_layout,validate_scene,"
              "resolve_asset,search_exemplars,record_run,snapshot_scene,restore_scene,list_scenes,"
              "get_scene_versions,search_scenes_by_prompt,route_skill")


HINT_AIDB2 = ("ai-db tools are also available: get_scene_spec_schema, then build_scene(spec) builds the "
              "whole scene in one call (it repairs layout conflicts itself, adds real furniture, materials, "
              "daylight, lamp lights and a camera, validates and fixes) and saves a version; edit_scene(scene_id, "
              "steps) refactors it step by step. Also scene memory (snapshot_scene, restore_scene) and "
              "route_skill for Blender how-to skills.")
#: the "aidb2" arm: ai-db after the benchmark-driven fixes (build_scene, edit_scene)
AIDB2_TOOLS = AIDB_TOOLS + ",build_scene,edit_scene"


HINT_AIDB5 = ("ai-db tools are also available. START HERE: build_scene builds the whole scene in one call. "
              "Pass the scene spec's fields (setting, objects, relations) as build_scene's own top-level "
              "arguments, or the spec_id that submit_scene_spec returned; get_scene_spec_schema shows the "
              "fields, the object categories and example specs. build_scene repairs layout conflicts, adds real "
              "furniture, materials, daylight, lamp lights and a camera, validates, fixes and saves a version "
              "(scene_id). Read its result's verdict and next: when the verdict says done, stop and reply DONE; "
              "otherwise do what next says (usually edit_scene(scene_id, steps) for the named fixes). Also scene "
              "memory (snapshot_scene, restore_scene) and route_skill for Blender how-to skills.")


def tool_name(server: str, tool: str, runner: str) -> str:
    """A tool as the agent's runner offers it: OpenCode prefixes MCP tools with the server
    (``aidb_build_scene``, ``blender_discover_servers``); the Strata agent and Claude are told
    bare names. nemotron-3.5-lightning-aidb5 called bare discover_servers and build_scene in
    OpenCode (6 'tool not found' errors) before it found the prefixed ones."""
    return f"{server}_{tool}" if runner == "opencode" else tool


def hint_aidb6(runner: str) -> str:
    """aidb6: aidb5's hint with the runner's tool names and the brief passed to build_scene."""
    def a(t: str) -> str:
        return tool_name("aidb", t, runner)

    return (f"ai-db tools are also available. START HERE: {a('build_scene')} builds the whole scene in one call. "
            "Pass the scene spec's fields (setting, objects, relations) as its own top-level arguments, plus "
            "brief: the request above, word for word (it then checks every requested object and count); "
            f"{a('get_scene_spec_schema')} shows the fields, the object categories and example specs. "
            f"{a('build_scene')} repairs layout conflicts, adds real furniture, materials, daylight, lamp lights "
            "and a camera, validates, fixes and saves a version (scene_id). Read its result's verdict and next "
            "(they come first): when the verdict says done, stop and reply DONE; otherwise do what next says "
            f"(usually {a('edit_scene')}(scene_id, steps) for the named fixes). Also scene memory "
            f"({a('snapshot_scene')}, {a('restore_scene')}) and {a('route_skill')} for Blender how-to skills.")


def prompt(brief_id: str, with_aidb: bool, version: int = 1, runner: str = "strata") -> str:
    """version 1: the aidb arm; 2: aidb2..aidb4 (``build_scene(spec)``); 5: aidb5 (the current
    build_scene: top-level spec fields or spec_id, START HERE, verdict/next); 6: aidb6 (tool names
    as ``runner`` offers them - opencode, strata, claude - and brief passed to build_scene).
    Versions up to 5 keep their exact text (bare names) so earlier arms stay comparable."""
    text = TASK.format(brief=BRIEFS[brief_id]["brief"]) + "\n\n"
    if version >= 6:
        b = [tool_name("blender", t, runner) for t in ("discover_servers", "discover_tools", "get_tool_schema", "invoke_tool")]
        text += f"Blender tools are reached through {' / '.join(b)}."
    else:
        text += HINT_BLENDER
    if not with_aidb:
        return text
    return text + " " + (hint_aidb6(runner) if version >= 6 else {1: HINT_AIDB, 2: HINT_AIDB2, 5: HINT_AIDB5}[version])
