"""investigate output formats: json, compact, stub, sexp.

`investigate` was JSON-only -- no `--format` on the CLI, no `format` in the
dispatcher schema, and `indent=2` hardcoded in both the CLI and the MCP server.
A pack already decides per evidence item whether it can afford a full body, and
then shipped that body inside JSON alongside every span, ref and `why` string.

Measured on this repo at `--budget 8000`, 5 queries: compact is 19% smaller,
stub is 89% smaller (8.9x). Bodies stay reachable via the `ref:` handles, which
is what the handles are for.
"""

from __future__ import annotations

import json

import pytest

from ai_db.analyzer.formatters import (
    PACK_FORMATS,
    Formatters,
    format_pack,
    format_pack_as_sexp,
    format_pack_as_stub,
)
from ai_db.errors import AiDbConfigError


@pytest.fixture()
def pack() -> dict:
    """A minimal pack with every optional section populated."""
    return {
        "query": "how are search results ranked",
        "mode": "explain",
        "entry_points": [
            {"qualified_name": "QueryEngine.search", "filepath": "/repo/ai_db/search/query.py",
             "lines": "L67-71", "why": "rank 1; name matches 'search'; bm25 #7; graph 0.76"},
            {"qualified_name": "Ranker.rank", "filepath": "/repo/ai_db/search/ranking.py",
             "lines": "L106-123", "why": "rank 2; bm25 #2"},
        ],
        "evidence": [
            {"ref": "ref:fbecc029", "qualified_name": "QueryEngine.search",
             "filepath": "/repo/ai_db/search/query.py", "lines": "L67-71",
             "role": "seed", "stub": "def search(self): ...", "body": None},
            {"ref": "ref:aaaabbbb", "qualified_name": "Ranker.rank",
             "filepath": "/repo/ai_db/search/ranking.py", "lines": "L106-123",
             "role": "caller", "stub": "def rank(self): ...", "body": None},
        ],
        "call_graph": {"nodes": ["QueryEngine.search", "Ranker.rank"],
                       "edges": [["QueryEngine.search", "Ranker.rank"]]},
        "tests": [{"qualified_name": "test_rank", "filepath": "/repo/tests/test_r.py"}],
        "recent_changes": [],
        "files_touched": ["/repo/ai_db/search/query.py"],
        "changes": [],
        "unresolved": ["perf_counter", "timed"],
        "omitted": [],
        "omitted_count": 45,
        "retrieval": {"mode": "lexical"},
        "token_count": 1197,
        "budget_tokens": 1200,
    }


# ==============================================================================
# compact
# ==============================================================================

def test_compact_is_the_same_data_on_one_line(pack):
    out = format_pack(pack, "compact")
    assert "\n" not in out.strip()
    assert json.loads(out) == pack, "compact must not change the data"


def test_compact_is_smaller_than_pretty_json(pack):
    pretty = format_pack(pack, "json")
    compact = format_pack(pack, "compact")
    assert len(compact) < len(pretty)
    # The gap is pure whitespace, so it should be roughly the indent overhead.
    assert 0.6 < len(compact) / len(pretty) < 0.95


# ==============================================================================
# stub
# ==============================================================================

def test_stub_keeps_the_answer_and_drops_the_scaffolding(pack):
    out = format_pack_as_stub(pack)
    assert "QueryEngine.search" in out
    assert "rank 1; name matches 'search'" in out, "provenance is the point"
    assert "QueryEngine.search -> Ranker.rank" in out, "graph edges must survive"
    assert "perf_counter" in out, "unresolved names are actionable"
    assert "45 items omitted" in out


def test_stub_omits_bodies_but_keeps_their_handles(pack):
    out = format_pack_as_stub(pack)
    assert "def search(self): ..." not in out, "a stub must not inline bodies"
    assert "ref:fbecc029" in out, "the handle is how you get the body back"
    assert len(out) < len(format_pack(pack, "json")) / 2


def test_stub_distinguishes_body_from_stub_from_meta(pack):
    out = format_pack_as_stub(pack)
    assert "[stub]" in out
    pack["evidence"][0]["body"] = "def search(self): return 1"
    assert "[body]" in format_pack_as_stub(pack)
    pack["evidence"][0].pop("stub")
    pack["evidence"][0].pop("body")
    assert "[meta]" in format_pack_as_stub(pack)


def test_stub_caps_items_per_section(pack, monkeypatch):
    """A pack holds up to a budget's worth of evidence; printing all of it
    defeats the point of rendering it compactly."""
    monkeypatch.setattr("ai_db.constants.PACK_STUB_MAX_ITEMS", 2)
    pack["evidence"] = [
        {"qualified_name": f"f{i}", "lines": "L1-2", "ref": f"ref:{i:08d}"}
        for i in range(10)
    ]
    out = format_pack_as_stub(pack)
    assert "more" in out
    assert "f9" not in out


def test_stub_handles_a_pack_with_nothing_optional(pack):
    minimal = {"query": "q", "mode": "locate", "entry_points": [], "evidence": [],
               "call_graph": {"nodes": [], "edges": []}, "unresolved": [],
               "omitted_count": 0, "token_count": 5, "budget_tokens": 100}
    out = format_pack_as_stub(minimal)
    assert "investigate locate" in out


# ==============================================================================
# sexp
# ==============================================================================

def test_sexp_is_a_balanced_navigable_tree(pack):
    out = format_pack_as_sexp(pack)
    assert out.startswith("(:investigate")
    assert out.count("(") == out.count(")"), "s-expressions must balance"
    assert "(:entry QueryEngine.search" in out
    assert "(:calls QueryEngine.search Ranker.rank)" in out
    assert "(:omitted 45)" in out


def test_sexp_quotes_only_what_needs_it(pack):
    out = format_pack_as_sexp(pack)
    assert '"" ' not in out
    assert '"how are search results ranked"' in out, "spaces force quoting"
    assert '"rank 1; name matches' in out


# ==============================================================================
# dispatch
# ==============================================================================

@pytest.mark.parametrize("style", sorted(PACK_FORMATS))
def test_every_declared_format_renders(pack, style):
    assert format_pack(pack, style).strip()


def test_unknown_format_is_rejected_with_the_valid_set(pack):
    with pytest.raises(AiDbConfigError) as exc:
        format_pack(pack, "yaml")
    msg = str(exc.value)
    assert "yaml" in msg
    for style in PACK_FORMATS:
        assert style in msg, "the error must list what is valid"


def test_format_is_reachable_through_the_formatters_class(pack):
    """The class is exposed as VectorDB.formatters; keep the two in step."""
    assert Formatters.format_pack(pack, "compact") == format_pack(pack, "compact")


# ==============================================================================
# path shortening
# ==============================================================================

def test_paths_are_shortened_to_the_working_directory(pack, tmp_path, monkeypatch):
    """A checkout directory can be named anything. Matching on a name produced
    'ai-db/ai_db/...' when the repo itself was called ai-db."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ai_db").mkdir()
    (tmp_path / "ai_db" / "x.py").write_text("", encoding="utf-8")
    pack["entry_points"][0]["filepath"] = str(tmp_path / "ai_db" / "x.py")
    out = format_pack_as_stub(pack)
    assert "ai_db/x.py" in out
    assert str(tmp_path) not in out


def test_path_outside_cwd_falls_back_to_basename(pack):
    out = format_pack_as_stub(pack)
    assert "/repo/ai_db/search/query.py" not in out
    assert "query.py" in out
