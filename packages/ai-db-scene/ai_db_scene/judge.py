"""Visual judge (pipeline Stage 6): scores renders against a fixed rubric.

``AnthropicJudge`` calls the Anthropic API (optional ``[judge]`` extra) and fails loudly when the
SDK or ``ANTHROPIC_API_KEY`` is missing; there is no silent fallback. ``FakeJudge`` is for tests.
"""

from __future__ import annotations

import base64
import json
import os
import random
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ai_db.errors import AiDbConfigError, AiDbError
from ai_db_scene.config import scene_config
from ai_db_scene.constants import DEFAULT_JUDGE_MODEL

RUBRIC: dict[str, str] = {
    "composition": "Framing, balance and placement of objects relative to the camera focus; no clutter or clipping.",
    "proportions_realism": "Object sizes and relative scale plausible for real-world counterparts; nothing floating or sunk.",
    "material_coherence": "Materials and colors fit each object and the requested style; consistent across the scene.",
    "lighting": "Readable exposure, believable key/fill lighting, mood matching the brief.",
    "brief_fidelity": "Every requested object, relation and mood from the brief is present and recognizable.",
}
SCORE_MIN, SCORE_MAX = 1, 5

Image = bytes | str | Path


class JudgeParseError(AiDbError):
    """The judge's output could not be parsed into a valid result, even after one repair."""


@dataclass
class JudgeResult:
    scores: dict[str, int]
    overall: float
    issues: list[dict[str, Any]] = field(default_factory=list)  # {criterion, message, suggested_fix?}

    def to_dict(self) -> dict[str, Any]:
        return {"scores": dict(self.scores), "overall": self.overall, "issues": [dict(i) for i in self.issues]}


@dataclass
class PairwiseResult:
    winner: str  # "a" | "b" | "tie"
    reasoning: str = ""


@runtime_checkable
class Judge(Protocol):
    def score(
        self, images: list[Image], brief: str, spec: dict[str, Any] | None, rubric: dict[str, str] = RUBRIC
    ) -> JudgeResult: ...

    def compare(self, images_a: list[Image], images_b: list[Image], brief: str) -> PairwiseResult: ...


# --- parsing (pure) --------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json(text: str) -> Any:
    m = _FENCE.search(text)
    body = m.group(1) if m else text
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object found")
    return json.loads(body[start : end + 1])


def parse_judge_output(text: str, rubric: dict[str, str] = RUBRIC) -> JudgeResult:
    """Strict parse: every rubric criterion needs an integer score in 1..5. Raises ValueError."""
    data = _extract_json(text)
    if not isinstance(data, dict) or not isinstance(data.get("scores"), dict):
        raise TypeError("expected an object with a 'scores' object")
    scores: dict[str, int] = {}
    for c in rubric:
        v = data["scores"].get(c)
        if isinstance(v, bool) or not isinstance(v, int) or not SCORE_MIN <= v <= SCORE_MAX:
            raise ValueError(f"scores.{c} must be an integer {SCORE_MIN}-{SCORE_MAX}, got {v!r}")
        scores[c] = v
    extra = set(data["scores"]) - set(rubric)
    if extra:
        raise ValueError(f"unknown criteria in scores: {sorted(extra)}")
    issues = data.get("issues", [])
    if not isinstance(issues, list):
        raise TypeError("'issues' must be a list")
    clean: list[dict[str, Any]] = []
    for i in issues:
        if not isinstance(i, dict) or i.get("criterion") not in rubric or not isinstance(i.get("message"), str):
            raise ValueError(f"bad issue entry: {i!r}")
        item = {"criterion": i["criterion"], "message": i["message"]}
        if isinstance(i.get("suggested_fix"), str):
            item["suggested_fix"] = i["suggested_fix"]
        clean.append(item)
    overall = round(sum(scores.values()) / len(scores), 2)
    return JudgeResult(scores, overall, clean)




def _image_block(img: Image) -> dict[str, Any]:
    raw = img if isinstance(img, bytes) else Path(img).read_bytes()
    if raw.startswith(b"\x89PNG"):
        media = "image/png"
    elif raw.startswith(b"\xff\xd8"):
        media = "image/jpeg"
    elif raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        media = "image/webp"
    else:
        raise ValueError("unsupported image format (need PNG, JPEG or WebP)")
    return {"type": "image", "source": {"type": "base64", "media_type": media,
                                          "data": base64.b64encode(raw).decode("ascii")}}


# --- Anthropic implementation ------------------------------------------------------


class AnthropicJudge:
    def __init__(self, model: str | None = None, client: Any = None, max_tokens: int = 1500,
                 rng: random.Random | None = None) -> None:
        self.model = model or os.environ.get("AI_DB_JUDGE_MODEL") or DEFAULT_JUDGE_MODEL
        self.max_tokens = max_tokens
        self._rng = rng or random.Random()
        if client is None:
            try:
                import anthropic  # type: ignore[import-not-found,unused-ignore]
            except ImportError as e:
                raise AiDbConfigError("AnthropicJudge needs the anthropic SDK: pip install 'ai-db[judge]'") from e
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise AiDbConfigError("AnthropicJudge needs ANTHROPIC_API_KEY to be set")
            client = anthropic.Anthropic()
        self._client = client

    @classmethod
    def from_config(cls, cfg: Any, **kw: Any) -> AnthropicJudge:
        return cls(model=scene_config(cfg).judge_model, **kw)

    def _ask(self, content: list[dict[str, Any]], parse: Callable[[str], Any]) -> Any:
        messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
        text = ""
        for attempt in range(2):  # original + one repair retry
            resp = self._client.messages.create(model=self.model, max_tokens=self.max_tokens, messages=messages)
            text = "".join(getattr(b, "text", "") for b in resp.content)
            try:
                return parse(text)
            except (ValueError, TypeError) as e:
                if attempt == 1:
                    raise JudgeParseError(f"judge output invalid after repair retry: {e}") from e
                messages += [
                    {"role": "assistant", "content": text or "(empty)"},
                    {"role": "user", "content": f"Your reply was invalid: {e}. Reply again with ONLY the JSON object."},
                ]
        raise JudgeParseError("unreachable")  # pragma: no cover

    def score(
        self, images: list[Image], brief: str, spec: dict[str, Any] | None, rubric: dict[str, str] = RUBRIC
    ) -> JudgeResult:
        crit = "\n".join(f"- {k}: {v}" for k, v in rubric.items())
        prompt = (
            f"You are a strict 3D scene reviewer. Score the rendered views against the brief.\n\nBrief: {brief}\n"
            f"Scene spec: {json.dumps(spec) if spec else 'n/a'}\n\nRubric (integer 1-5 each, 5 best):\n{crit}\n\n"
            'Reply with ONLY JSON: {"scores": {<criterion>: int, ...}, "issues": [{"criterion": str, '
            '"message": str, "suggested_fix": str}]}. Use the exact criterion names.'
        )
        content = [_image_block(i) for i in images] + [{"type": "text", "text": prompt}]
        res: JudgeResult = self._ask(content, lambda t: parse_judge_output(t, rubric))
        return res

    def compare(self, images_a: list[Image], images_b: list[Image], brief: str) -> PairwiseResult:
        """Blind pairwise: presentation order is randomized, the result is mapped back to a/b."""
        swapped = self._rng.random() < 0.5
        first, second = (images_b, images_a) if swapped else (images_a, images_b)
        content: list[dict[str, Any]] = [{"type": "text", "text": "Candidate X:"}]
        content += [_image_block(i) for i in first]
        content += [{"type": "text", "text": "Candidate Y:"}]
        content += [_image_block(i) for i in second]
        crit = ", ".join(RUBRIC)
        content.append({"type": "text", "text": (
            f"Which candidate better satisfies this brief? Judge on: {crit}.\nBrief: {brief}\n"
            'Reply with ONLY JSON: {"winner": "x"|"y"|"tie", "reasoning": str}.')})

        def parse(t: str) -> PairwiseResult:
            data = _extract_json(t)
            w = str(data.get("winner", "")).strip().lower() if isinstance(data, dict) else ""
            if w not in ("x", "y", "tie"):
                raise ValueError(f"'winner' must be x, y or tie, got {w!r}")
            if w == "tie":
                return PairwiseResult("tie", str(data.get("reasoning", "")))
            pos_first = w == "x"
            a_wins = pos_first != swapped
            return PairwiseResult("a" if a_wins else "b", str(data.get("reasoning", "")))

        pr: PairwiseResult = self._ask(content, parse)
        return pr


# --- test double -----------------------------------------------------------------------


class FakeJudge:
    """Deterministic judge for tests; records calls."""

    def __init__(self, scores: dict[str, int] | None = None, issues: list[dict[str, Any]] | None = None,
                 winner: str = "a") -> None:
        self.scores = scores or {c: 4 for c in RUBRIC}
        self.issues = issues or []
        self.winner = winner
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def score(
        self, images: Sequence[Image], brief: str, spec: dict[str, Any] | None, rubric: dict[str, str] = RUBRIC
    ) -> JudgeResult:
        self.calls.append(("score", (len(images), brief)))
        s = {c: self.scores.get(c, 4) for c in rubric}
        return JudgeResult(s, round(sum(s.values()) / len(s), 2), list(self.issues))

    def compare(self, images_a: Sequence[Image], images_b: Sequence[Image], brief: str) -> PairwiseResult:
        self.calls.append(("compare", (len(images_a), len(images_b), brief)))
        return PairwiseResult(self.winner, "fake")
