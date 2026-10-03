"""Judge: parsing, repair retry, blind pairwise, config errors. No real API calls."""

from __future__ import annotations

import json
import random
from types import SimpleNamespace

import pytest

from ai_db.config import DEFAULT_JUDGE_MODEL, parse_config
from ai_db.config_template import build_template
from ai_db.errors import AiDbConfigError
from ai_db.scene.judge import (
    RUBRIC,
    AnthropicJudge,
    FakeJudge,
    Judge,
    JudgeParseError,
    parse_judge_output,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 8
GOOD = json.dumps({"scores": {c: 4 for c in RUBRIC}, "issues": [
    {"criterion": "lighting", "message": "dark", "suggested_fix": "add key light"}]})


class FakeClient:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(content=[SimpleNamespace(text=self.replies.pop(0))])


def test_parse_ok_and_fenced():
    r = parse_judge_output("```json\n" + GOOD + "\n```")
    assert r.overall == 4.0 and r.issues[0]["suggested_fix"] == "add key light"


@pytest.mark.parametrize("bad", [
    "not json", '{"scores": {}}', json.dumps({"scores": {**{c: 4 for c in RUBRIC}, "composition": 6}}),
    json.dumps({"scores": {**{c: 4 for c in RUBRIC}, "extra": 1}}),
    json.dumps({"scores": {c: 4 for c in RUBRIC}, "issues": [{"criterion": "zzz", "message": "m"}]}),
])
def test_parse_strict(bad):
    with pytest.raises(ValueError):
        parse_judge_output(bad)


def test_score_images_base64_and_model(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(PNG)
    c = FakeClient([GOOD])
    j = AnthropicJudge(model="m1", client=c)
    r = j.score([PNG, str(img)], "brief", {"s": 1})
    assert r.overall == 4.0 and c.calls[0]["model"] == "m1"
    blocks = c.calls[0]["messages"][0]["content"]
    assert [b["type"] for b in blocks] == ["image", "image", "text"]
    assert blocks[0]["source"]["media_type"] == "image/png"


def test_repair_retry_once_then_fail():
    c = FakeClient(["garbage", GOOD])
    assert AnthropicJudge(client=c).score([PNG], "b", None).overall == 4.0
    assert len(c.calls) == 2 and "invalid" in c.calls[1]["messages"][-1]["content"]
    c2 = FakeClient(["garbage", "still garbage"])
    with pytest.raises(JudgeParseError):
        AnthropicJudge(client=c2).score([PNG], "b", None)
    assert len(c2.calls) == 2


@pytest.mark.parametrize("seed", range(6))
def test_compare_blind_and_derandomized(seed):
    # Model always prefers whichever candidate shows the A images; check mapping via image data.
    a, b = PNG + b"A", PNG + b"B"

    class Prefers(FakeClient):
        def create(self, **kw):
            blocks = kw["messages"][0]["content"]
            imgs = [x for x in blocks if x["type"] == "image"]
            import base64
            first = base64.b64decode(imgs[0]["source"]["data"])
            self.calls.append(kw)
            # prefers the real "A" image regardless of position
            w = "x" if first.endswith(b"A") else "y"
            return SimpleNamespace(content=[SimpleNamespace(text=json.dumps({"winner": w, "reasoning": "r"}))])

    j = AnthropicJudge(client=Prefers([]), rng=random.Random(seed))
    assert j.compare([a], [b], "brief").winner == "a"
    assert j.compare([b], [a], "brief").winner == "b"


def test_compare_order_is_randomized():
    seen = set()
    for seed in range(20):
        c = FakeClient([json.dumps({"winner": "tie"})])
        AnthropicJudge(client=c, rng=random.Random(seed)).compare([PNG + b"A"], [PNG + b"B"], "b")
        seen.add(c.calls[0]["messages"][0]["content"][1]["source"]["data"])
    assert len(seen) == 2


def test_missing_sdk_or_key_raises(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "anthropic", None)
    with pytest.raises(AiDbConfigError, match="anthropic SDK"):
        AnthropicJudge()
    monkeypatch.delitem(__import__("sys").modules, "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    try:
        import anthropic  # noqa: F401
    except ImportError:
        pytest.skip("anthropic not installed")
    with pytest.raises(AiDbConfigError, match="ANTHROPIC_API_KEY"):
        AnthropicJudge()


def test_fake_judge_and_default_model():
    f = FakeJudge()
    assert isinstance(f, Judge)
    assert f.score([PNG], "b", None).overall == 4.0 and f.compare([PNG], [PNG], "b").winner == "a"
    cfg = parse_config(json.loads(build_template()) if isinstance(build_template(), str) else build_template())
    assert cfg.scene.judge_model == DEFAULT_JUDGE_MODEL
