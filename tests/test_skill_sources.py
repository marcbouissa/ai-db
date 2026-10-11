"""draft_skill sources: text, URL (mocked fetch), HTML, URL safety, metadata.sources."""

from __future__ import annotations

import socket

import pytest

import ai_db.skill_gen as sg
import ai_db.skill_sources as ss
from ai_db.config import parse_config
from ai_db.config_template import build_template
from ai_db.dispatcher import ServiceDispatcher

HTML = """<html><head><title>x</title><script>var a=1;</script></head><body>
<nav>Home | Blog</nav>
<h1>Low-poly mug in Blender</h1><p>Released under CC-BY 4.0.</p>
<h2>Steps</h2>
<ol><li>Add a cylinder with 16 vertices.</li><li>Inset the top face and extrude down.</li>
<li>Add a Solidify modifier, thickness: 0.02 m.</li></ol>
<pre>bpy.ops.mesh.primitive_cylinder_add(vertices=16)</pre>
<footer>(c) someone</footer></body></html>"""


@pytest.fixture
def disp(tmp_path, monkeypatch):
    monkeypatch.setattr(sg, "entry_points", lambda group: [])
    raw = build_template()
    raw["skills"] = {"generated_dir": str(tmp_path / "gen")}
    return ServiceDispatcher(db_path=str(tmp_path / "k.db"), config=parse_config(raw, check_env=False))


def test_html_to_text_keeps_structure():
    text = ss.html_to_text(HTML)
    assert "# Low-poly mug in Blender" in text and "1. Add a cylinder" in text and "3. Add a Solidify" in text
    assert "```\nbpy.ops.mesh" in text and "var a" not in text and "Home | Blog" not in text


def test_extract_text_sections():
    found, facts = ss.extract_text_sections(ss.html_to_text(HTML))
    assert found["workflow"][0] == "1. Add a cylinder with 16 vertices."
    assert found["examples"][0].startswith("```\nbpy.ops.mesh.primitive_cylinder_add")
    assert any("thickness = 0.02 m" in e for e in found["examples"])
    assert facts == {"title": "Low-poly mug in Blender", "license": "CC-BY 4.0"}


def test_draft_skill_from_url_and_text(disp, monkeypatch):
    calls = []

    def fake_fetch(url):
        calls.append(url)
        return "text/html; charset=utf-8", HTML.encode()

    monkeypatch.setattr(ss, "http_fetch", lambda url, allow_private=False: fake_fetch(url))
    d = disp.execute("draft_skill", {"topic": "low poly mug", "sources": [
        {"url": "https://example.org/mug"},
        {"text": "## Notes\n\n1. Bevel the rim.\n2. Shade smooth.", "title": "my notes", "license": "CC0"},
        {"url": "ftp://example.org/x"},
        {"weird": "x"}]})
    assert calls == ["https://example.org/mug"]
    assert "1. Add a cylinder with 16 vertices." in d["body"] and "2. Shade smooth." in d["body"]
    assert "```\nbpy.ops.mesh" in d["body"]
    assert d["metadata_sources"] == [
        {"url": "https://example.org/mug", "license": "CC-BY 4.0", "title": "Low-poly mug in Blender"},
        {"text": "my notes", "license": "CC0", "title": "my notes"}]
    assert len(d["warnings"]) == 2 and "no reader" in d["warnings"][1]
    assert '  sources:\n    - {"url": "https://example.org/mug"' in d["skill_md"]
    out = disp.execute("create_skill", {"name": "low-poly-mug", "description": "Model a low-poly mug in Blender.",
                                        "body": d["body"], "sources": d["metadata_sources"]})
    with open(out["path"], encoding="utf-8") as f:
        text = f.read()
    assert '    - {"text": "my notes", "license": "CC0", "title": "my notes"}' in text
    with pytest.raises(ValueError, match="source"):
        disp.execute("create_skill", {"name": "x-y", "description": "Model a low-poly mug in Blender.",
                                      "body": d["body"], "sources": [{"evil": "x"}], "overwrite": True})


def test_fetch_refuses_non_text(disp, monkeypatch):
    monkeypatch.setattr(ss, "http_fetch", lambda url, allow_private=False: ("model/gltf-binary", b"glTF"))
    d = disp.execute("draft_skill", {"topic": "mug", "sources": [{"url": "https://example.org/m.glb"}]})
    assert "only text" in d["warnings"][0] and d["metadata_sources"] == []


def test_check_url_refuses_private_and_non_http(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port: [(0, 0, 0, "", ("10.0.0.5", port))])
    with pytest.raises(ValueError, match="non-public"):
        ss.check_url("http://intranet.example/")
    ss.check_url("http://intranet.example/", allow_private=True)
    for bad in ("file:///etc/passwd", "ftp://x/y", "javascript:alert(1)"):
        with pytest.raises(ValueError, match="http"):
            ss.check_url(bad)
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port: [(0, 0, 0, "", ("127.0.0.1", port))])
    with pytest.raises(ValueError, match="non-public"):
        ss.http_fetch("http://localhost:8080/")
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port: [(0, 0, 0, "", ("93.184.216.34", port))])
    ss.check_url("https://example.org/")


def test_config_allow_private_urls():
    raw = build_template()
    raw["skills"] = {"allow_private_urls": True}
    cfg = parse_config(raw, check_env=False)
    assert cfg.skills_allow_private_urls and cfg.to_dict()["skills"] == {"allow_private_urls": True}
    raw["skills"] = {"allow_private_urls": "yes"}
    with pytest.raises(Exception, match="allow_private_urls"):
        parse_config(raw, check_env=False)
