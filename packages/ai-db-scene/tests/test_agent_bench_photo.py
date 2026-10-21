"""The photo -> scene experiment (eval/agent_bench/photo_exp.py, 2026-10-21): image input for the
local vision model, the photo arms and their prompts, the B cell's deepen call, the room-shape
checklist item and the Qwen2.5-VL tool chat template. No GPU, no Blender: the server is mocked."""

from __future__ import annotations

import base64
import importlib
import io
import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest

BENCH_DIR = Path(__file__).resolve().parents[1] / "eval" / "agent_bench"
PROMPTS = {"basic": "Recreate the room in this photo.", "deepen_instruction": "Describe it in prose.",
           "lead_deep": "An oval room with a desk.",
           "checklist": [["oval room", 1, ["room_shape:ellipse"]], ["desk", 1, ["desk"]]]}


@pytest.fixture(scope="module")
def mods() -> dict[str, Any]:
    sys.path.insert(0, str(BENCH_DIR))
    try:
        return {n: importlib.import_module(n) for n in ("strata_agent", "photo_exp", "bench")}
    finally:
        sys.path.remove(str(BENCH_DIR))


def _png(tmp_path: Path, w: int, h: int) -> str:
    from PIL import Image

    p = tmp_path / "ref.png"
    Image.new("RGB", (w, h), (200, 180, 150)).save(p)
    return str(p)


def _decode(part: dict[str, Any]) -> Any:
    from PIL import Image

    url = part["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))


def test_image_is_scaled_to_1280_long_side(mods, tmp_path):
    im = _decode(mods["strata_agent"].image_part(_png(tmp_path, 2000, 1338)))
    assert im.size == (1280, 856)


def test_small_image_is_not_upscaled(mods, tmp_path):
    im = _decode(mods["strata_agent"].image_part(_png(tmp_path, 640, 400)))
    assert im.size == (640, 400)


def test_first_user_message_carries_the_image(mods, tmp_path, monkeypatch):
    sa = mods["strata_agent"]
    assert sa.user_content("hi", None) == "hi"
    parts = sa.user_content("hi", _png(tmp_path, 100, 80))
    assert [p["type"] for p in parts] == ["image_url", "text"] and parts[1]["text"] == "hi"

    sent: list[list[dict[str, Any]]] = []

    def fake_chat(messages, tools, base_url, model):
        sent.append(json.loads(json.dumps(messages)))
        return {"content": "DONE: built"}, {"prompt_tokens": 10, "completion_tokens": 2, "finish_reason": "stop"}

    monkeypatch.setattr(sa, "_chat", fake_chat)
    out = sa.run("build it", {}, str(tmp_path / "t.jsonl"), 60, base_url="http://x", model="m",
                 image=_png(tmp_path, 100, 80))
    assert out["ok"]
    first_user = sent[0][1]
    assert first_user["role"] == "user" and first_user["content"][0]["type"] == "image_url"
    assert first_user["content"][1] == {"type": "text", "text": "build it"}


def test_photo_arms_parse_and_round_trip(mods):
    pe = mods["photo_exp"]
    assert pe.parse_arm("oval-photo-B-aidb-qwen2.5vl-7b-r2") == {
        "cell": "B", "mode": "aidb", "with_aidb": True, "model": "qwen2.5vl-7b", "repeat": 2}
    assert pe.parse_arm("qwen3-14b-local-aidb7") is None
    arms = pe.builds(2)
    assert len(arms) == 12 and len(set(arms)) == 12
    assert all(pe.parse_arm(a) for a in arms)
    assert pe.builds(1)[:2] == ["oval-photo-A-raw-qwen2.5vl-7b-r1", "oval-photo-A-aidb-qwen2.5vl-7b-r1"]


def test_bench_knows_the_vision_model_and_the_brief(mods):
    bench = mods["bench"]
    assert bench.LOCAL_MODELS["qwen2.5vl-7b"][1] == "qwen2.5-vl-7b"
    if (Path(mods["photo_exp"].PROMPTS)).exists():
        assert bench.briefs.BRIEFS["oval_photo"]["checklist"][0] == ("oval room", 1, ("room_shape:ellipse",))


def test_cell_prompts(mods, tmp_path, monkeypatch):
    pe = mods["photo_exp"]
    monkeypatch.setattr(pe, "PHOTO_DIR", str(tmp_path))
    a_raw = pe.prompt("A", False, 1, PROMPTS)
    assert a_raw.startswith(pe.PHOTO_NOTE) and PROMPTS["basic"] in a_raw
    assert "discover_servers / discover_tools" in a_raw and "build_scene" not in a_raw
    c_aidb = pe.prompt("C", True, 1, PROMPTS)
    assert PROMPTS["lead_deep"] in c_aidb and "START HERE: build_scene" in c_aidb
    with pytest.raises(SystemExit):
        pe.prompt("B", False, 1, PROMPTS)  # no deepened brief yet
    (tmp_path / "deepen_r1.json").write_text(json.dumps({"text": "A cream oval room ..."}))
    b_raw, b_aidb = pe.prompt("B", False, 1, PROMPTS), pe.prompt("B", True, 1, PROMPTS)
    assert "A cream oval room ..." in b_raw and "A cream oval room ..." in b_aidb  # same brief for both


def test_deepen_sends_photo_and_saves_brief_time_tokens(mods, tmp_path, monkeypatch):
    pe = mods["photo_exp"]
    monkeypatch.setattr(pe, "PHOTO_DIR", str(tmp_path))
    bodies: list[dict[str, Any]] = []

    def post(base_url, body):
        bodies.append(body)
        return {"choices": [{"message": {"content": " An oval room, cream walls. "}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1500, "completion_tokens": 7}}

    rec = pe.deepen(1, base_url="http://x", model="m", image=_png(tmp_path, 300, 200), prompts=PROMPTS, post=post)
    assert rec["text"] == "An oval room, cream walls." and rec["tokens"] == {"input": 1500, "output": 7}
    assert rec["words"] == 5 and rec["seconds"] >= 0
    content = bodies[0]["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert PROMPTS["basic"] in content[1]["text"] and PROMPTS["deepen_instruction"] in content[1]["text"]
    assert "tools" not in bodies[0]
    # saved once: a second call (the B-aidb build of the same repeat) reuses it
    assert pe.deepen(1, post=lambda *_: pytest.fail("called twice"), prompts=PROMPTS) == rec


def _ellipse(rx: float, ry: float, n: int = 64) -> list[tuple[float, float]]:
    return [(rx * math.cos(2 * math.pi * i / n), ry * math.sin(2 * math.pi * i / n)) for i in range(n)]


def test_room_shapes(mods):
    pe = mods["photo_exp"]
    assert pe.classify_outline(_ellipse(5.5, 4.5))["shape"] == "ellipse"
    assert pe.classify_outline(_ellipse(5.5, 4.5, 24))["shape"] == "ellipse"
    rect = [(x, y) for x in (0, 0.5, 6, 11) for y in (0, 9)]
    assert pe.classify_outline(rect)["shape"] == "rect"
    rotated = [(x * 0.8 - y * 0.6, x * 0.6 + y * 0.8) for x, y in [(0, 0), (11, 0), (11, 9), (0, 9)]]
    assert pe.classify_outline(rotated)["shape"] == "rect"
    octagon = _ellipse(5, 5, 8)
    assert pe.classify_outline(octagon)["shape"] == "polygon"
    assert pe.classify_outline([(0, 0), (1, 1)])["shape"] == "none"


def test_room_shape_prefers_walls_over_floor(mods):
    pe = mods["photo_exp"]
    groups = {"wall": [{"name": "OvalWall", "points": _ellipse(5.5, 4.5)}],
              "floor": [{"name": "Floor", "points": [(-6, -5), (6, -5), (6, 5), (-6, 5)]}]}
    out = pe.room_shape(groups)
    assert out["shape"] == "ellipse" and out["from"] == "wall"
    assert pe.room_shape({"wall": [], "floor": groups["floor"]})["shape"] == "rect"
    assert pe.room_shape({"wall": [], "floor": []})["shape"] == "none"


def test_room_points_code_is_valid_python(mods):
    compile(mods["photo_exp"].room_points_code(), "room_points", "exec")


def test_oval_room_item_scores_the_room_shape(mods, monkeypatch):
    bench = mods["bench"]
    monkeypatch.setitem(bench.briefs.BRIEFS, "shape_test", {
        "brief": "x", "checklist": [("oval room", 1, ("room_shape:ellipse",)), ("desk", 1, ("desk",))]})
    inv = {"objects": [{"name": "desk", "type": "MESH", "category": "desk", "parent": None, "mats": 1,
                        "tris": 40, "faces": 20, "size_m": [1.8, 0.9, 0.76]}], "camera": "c", "lights": 1}
    items = bench.checklist("shape_test", inv)
    assert items[0]["found"] == 0 and items[1]["found"] == 1
    inv["room_shape"] = {"shape": "ellipse"}
    assert bench.checklist("shape_test", inv)[0]["found"] == 1
    inv["room_shape"] = {"shape": "rect"}
    assert bench.checklist("shape_test", inv)[0]["real"] == 0.0


def _render_template(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> str:
    import jinja2

    env = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda v: json.dumps(v, ensure_ascii=False)  # as transformers/vLLM do
    tpl = env.from_string((BENCH_DIR / "qwen2_5_vl_tools.jinja").read_text())
    return tpl.render(messages=messages, tools=tools, add_generation_prompt=True)


def test_vl_tool_template_renders_tools_images_calls_and_results():
    tools = [{"type": "function", "function": {"name": "blender__invoke_tool", "parameters": {"type": "object"}}}]
    msgs = [{"role": "system", "content": "You build scenes."},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Recreate it."}]},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "1", "type": "function",
                             "function": {"name": "blender__invoke_tool", "arguments": {"tool": "x"}}}]},
            {"role": "tool", "tool_call_id": "1", "content": "ok"},
            {"role": "tool", "tool_call_id": "2", "content": [{"type": "text", "text": "ok2"}]}]
    out = _render_template(msgs, tools)
    assert out.startswith("<|im_start|>system\nYou build scenes.\n\n# Tools")
    assert '"name": "blender__invoke_tool"' in out and "<tools>" in out
    assert "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Recreate it.<|im_end|>" in out
    assert '<tool_call>\n{"name": "blender__invoke_tool", "arguments": {"tool": "x"}}\n</tool_call><|im_end|>' in out
    assert "<|im_start|>user\n<tool_response>\nok\n</tool_response>\n<tool_response>\nok2\n</tool_response><|im_end|>" in out
    assert out.count("You build scenes.") == 1
    assert out.endswith("<|im_start|>assistant\n")
    # arguments as a JSON string are passed through as they are
    msgs[2]["tool_calls"][0]["function"]["arguments"] = '{"tool": "y"}'
    assert '"arguments": {"tool": "y"}}' in _render_template(msgs, tools)


def test_vl_template_without_tools_keeps_the_vl_default():
    out = _render_template([{"role": "user", "content": [{"type": "image_url"}, {"type": "text", "text": "hi"}]}],
                           None)
    assert out.startswith("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
                          "<|vision_start|><|image_pad|><|vision_end|>hi<|im_end|>")


def test_opencode_photo_arms_and_prompts(mods, tmp_path, monkeypatch):
    """OpenCode photo arms: the model part is the OpenCode model id, the tool names are OpenCode's
    (<server>_<tool>) and the B brief lives in the model's own deepen file."""
    pe = mods["photo_exp"]
    m = "longcat-2.5-preview-free"
    assert pe.parse_arm(f"oval-photo-C-raw-{m}-r1") == {
        "cell": "C", "mode": "raw", "with_aidb": False, "model": m, "repeat": 1}
    assert not pe.is_local(m) and pe.is_local(pe.MODEL_KEY)
    assert pe.builds(1, m)[0] == f"oval-photo-A-raw-{m}-r1"
    c_aidb = pe.prompt("C", True, 1, PROMPTS, model=m)
    assert "blender_discover_servers / blender_discover_tools" in c_aidb and "START HERE: aidb_build_scene" in c_aidb
    monkeypatch.setattr(pe, "PHOTO_DIR", str(tmp_path))
    assert pe.deepen_path(1, m).endswith(f"deepen_{m}_r1.json") and pe.deepen_path(1).endswith("deepen_r1.json")
    (tmp_path / f"deepen_{m}_r1.json").write_text(json.dumps({"text": "A cream oval room ..."}))
    assert "A cream oval room ..." in pe.prompt("B", False, 1, PROMPTS, model=m)
