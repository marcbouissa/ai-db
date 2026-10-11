"""create_skill / draft_skill / list_generated_skills / delete_skill (ai_db.skill_gen)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import ai_db.skill_gen as sg
from ai_db.config import parse_config
from ai_db.config_template import build_template
from ai_db.dispatcher import ServiceDispatcher

BODY = "# Mug modelling\n\n## Workflow\n\n1. Cylinder, 2. solidify, 3. handle from a torus.\n"


class _EP:
    def __init__(self, name, fn):
        self.name, self.value, self._fn = name, f"x:{name}", fn

    def load(self):
        return self._fn


@pytest.fixture
def disp(tmp_path, monkeypatch):
    # no installed plugin hook may run (the scene one would open the user's scene store)
    monkeypatch.setattr(sg, "entry_points", lambda group: [])
    raw = build_template()
    raw["skills"] = {"generated_dir": str(tmp_path / "gen")}
    d = ServiceDispatcher(db_path=str(tmp_path / "k.db"), config=parse_config(raw, check_env=False))
    return d


def test_create_then_route_finds_it(disp, tmp_path):
    out = disp.execute("create_skill", {"name": "mug-modelling", "description": "Model a coffee mug with a handle in Blender.",
                                        "body": BODY, "triggers": ["model a mug", "coffee mug"], "domain": "blender"})
    path = Path(out["path"])
    assert out["status"] == "created" and path == tmp_path / "gen" / "mug-modelling" / "SKILL.md"
    text = path.read_text()
    assert text.startswith("---\nname: mug-modelling\ndescription: Model a coffee mug")
    assert "  triggers: model a mug, coffee mug\n" in text
    hits = disp.execute("route_skill", {"prompt": "how do I model a coffee mug", "min_confidence": 0.0})
    assert hits and hits[0]["name"] == "mug-modelling"
    assert [s["name"] for s in disp.execute("list_generated_skills", {})] == ["mug-modelling"]

    with pytest.raises(ValueError, match="overwrite"):
        disp.execute("create_skill", {"name": "mug-modelling", "description": "x" * 30, "body": BODY})
    assert disp.execute("create_skill", {"name": "mug-modelling", "description": "y" * 30, "body": BODY,
                                         "overwrite": True})["status"] == "updated"

    disp.execute("delete_skill", {"name": "mug-modelling"})
    assert not path.exists() and disp.execute("list_generated_skills", {}) == []
    assert not disp.execute("route_skill", {"prompt": "coffee mug", "min_confidence": 0.0})


@pytest.mark.parametrize("args, msg", [
    ({"name": "Bad_Name"}, "kebab-case"),
    ({"name": "../evil"}, "kebab-case"),
    ({"description": "short"}, "description"),
    ({"description": 'has "quote" in it, long enough'}, "one line"),
    ({"body": ""}, "empty"),
    ({"body": "no heading here"}, "heading"),
    ({"body": "---\nname: x\n---\n# T"}, "frontmatter"),
    ({"body": "# T\n" + "x" * sg.SKILL_BODY_MAX}, "max"),
    ({"triggers": ["a: b"]}, "trigger"),
    ({"related_skills": ["Not Kebab"]}, "related"),
    ({"domain": "Two Words"}, "domain"),
])
def test_create_rejects(disp, args, msg):
    base = {"name": "ok-skill", "description": "A perfectly fine description.", "body": BODY}
    with pytest.raises(ValueError, match=msg):
        disp.execute("create_skill", {**base, **args})


def test_refuses_to_shadow_packaged_skill(disp, tmp_path, monkeypatch):
    other = tmp_path / "pkg" / "pbr-shader-builder"
    other.mkdir(parents=True)
    (other / "SKILL.md").write_text("---\nname: pbr-shader-builder\ndescription: packaged\n---\n# P\n")
    db = disp._get_db()
    db.sync_skills(skill_dirs=[str(tmp_path / "pkg")], verbose=False)
    with pytest.raises(ValueError, match="outside the generated folder"):
        disp.execute("create_skill", {"name": "pbr-shader-builder", "description": "A perfectly fine description.",
                                      "body": BODY})
    with pytest.raises(ValueError, match="no generated skill"):
        disp.execute("delete_skill", {"name": "pbr-shader-builder"})
    assert (other / "SKILL.md").exists()


def test_default_generated_dir_is_under_first_skill_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(sg.constants, "DEFAULT_SKILL_DIRS", [str(tmp_path / "s")])
    assert sg.generated_skills_dir(None) == str(tmp_path / "s" / "generated")
    assert sg.generated_skills_dir(SimpleNamespace(skills_generated_dir=str(tmp_path / "g"))) == str(tmp_path / "g")


def test_config_rejects_unknown_skills_key():
    raw = build_template()
    raw["skills"] = {"generated_dir": "/x", "nope": 1}
    with pytest.raises(Exception, match="nope"):
        parse_config(raw, check_env=False)


def test_draft_skill_gathers_skills_contexts_and_plugins(disp, monkeypatch):
    disp.execute("create_skill", {"name": "mug-modelling", "description": "Model a coffee mug with a handle in Blender.",
                                  "body": BODY, "triggers": ["coffee mug"]})
    disp.execute("context_save", {"session_id": "s1", "title": "mug session",
                                  "summary": "Mug: bevel the rim before solidify, handle is a half torus."})

    def hook(topic, config):
        return {"pitfalls": [f"plugin pitfall for {topic}"], "sources": ["run:abc"]}

    def broken(topic, config):
        raise RuntimeError("boom")

    monkeypatch.setattr(sg, "entry_points", lambda group: [_EP("demo", hook), _EP("bad", broken)])
    d = disp.execute("draft_skill", {"topic": "coffee mug"})
    assert d["name"] == "coffee-mug"
    assert "mug-modelling" in [s["name"] for s in d["existing_skills"]]
    assert "mug-modelling" in d["possible_duplicates"]
    assert "bevel the rim" in d["body"] and "plugin pitfall for coffee mug" in d["body"]
    for title in ("## When to use", "## Workflow", "## Pitfalls", "## Examples"):
        assert title in d["body"]
    assert d["plugins"]["bad"] == {"error": "boom"}
    assert "demo:run:abc" in d["sources"]
    assert d["skill_md"].startswith("---\nname: coffee-mug\n")
    # the draft's body passes create_skill's validation once the description is written
    sg.validate_skill(d["name"], "Model a coffee mug, from the drafted notes.", d["body"], d["triggers"])
    json.dumps(d)


def test_mcp_allowlist_hides_skill_tools(disp, monkeypatch):
    from ai_db.transports.mcp import StdioMCPServer

    monkeypatch.setenv("AI_DB_MCP_TOOLS", "draft_skill")
    srv = StdioMCPServer(dispatcher=disp)
    names = [t["name"] for t in srv.handle_request({"id": 1, "method": "tools/list"})["result"]["tools"]]
    assert names == ["draft_skill"]
    res = srv.handle_request({"id": 2, "method": "tools/call", "params": {"name": "create_skill", "arguments": {}}})
    assert res["result"]["isError"]


def test_html_definition_lists_become_options():
    """Reference manuals (Blender's) document options as <dl>; heading anchors (¶) are dropped."""
    from ai_db.skill_sources import extract_text_sections, html_to_text

    html = ("<h1>Bevel Modifier<a class='headerlink'>¶</a></h1><dl><dt>Affect</dt><dd><dl>"
            "<dt>Edges</dt><dd><p>Bevel the edges.</p></dd></dl></dd>"
            "<dt>Segments</dt><dd><p>The number of edge loops.</p></dd></dl>")
    sections, facts = extract_text_sections(html_to_text(html))
    assert facts["title"] == "Bevel Modifier"
    opts = [e for e in sections["examples"] if e.startswith("Option")]
    assert opts == ["Option - Edges: Bevel the edges.", "Option - Segments: The number of edge loops."]


def test_unrelated_top_skill_is_not_a_duplicate():
    assert sg._same_topic("bevel modifier for hard surface edges",
                          {"name": "bevel-hard-surface", "description": "Bevel modifier settings"})
    assert not sg._same_topic("bevel modifier for hard surface edges",
                              {"name": "high-end-visual-design", "description": "Design websites like an agency."})
