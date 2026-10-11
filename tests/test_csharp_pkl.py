"""C# and Apple Pkl grammars: symbols, refs, chunks, outline, indexing."""

import os

import pytest

from ai_db import VectorDB
from ai_db.constants import INDEXABLE_EXTENSIONS
from ai_db.parser.ast_visitor import extract_file_outline
from ai_db.parser.chunker import chunk_file
from ai_db.parser.languages import EXT_TO_LANG, SUPPORTED_LANGUAGES
from ai_db.parser.ts_graph import extract_graph

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "csharp_pkl")
CS = os.path.join(FIX, "Sample.cs")
PKL = os.path.join(FIX, "config.pkl")


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _graph(path):
    symbols, refs, errors = extract_graph(path, _read(path))
    return {(s["symbol_type"], s["name"]) for s in symbols}, refs, errors


def test_languages_are_registered_and_indexable():
    assert {"csharp", "pkl"} <= SUPPORTED_LANGUAGES
    assert EXT_TO_LANG[".cs"] == "csharp" and EXT_TO_LANG[".pkl"] == "pkl"
    # every extension with a grammar must also be picked up by the indexer
    assert set(EXT_TO_LANG) <= INDEXABLE_EXTENSIONS


# --- C# --------------------------------------------------------------------------


def test_csharp_symbols():
    symbols, _, errors = _graph(CS)
    assert errors == []
    assert {("interface", "IRepository"), ("class", "BaseService"), ("class", "OrderService"),
            ("struct", "Point"), ("type", "Order"), ("type", "Status"), ("type", "Handler"),
            ("property", "Count"), ("method", "Run"), ("method", "SaveAsync"),
            ("method", "OrderService"), ("class", "Helper"), ("method", "Static")} <= symbols


def test_csharp_inheritance_includes_generic_bases():
    _, refs, _ = _graph(CS)
    inherit = {(r["caller_name"], r["callee_name"]) for r in refs if r["ref_type"] == "inherit"}
    assert inherit == {("module.Shop.Orders.IRepository", "IDisposable"),
                       ("module.Shop.Orders.OrderService", "BaseService"),
                       ("module.Shop.Orders.OrderService", "IRepository")}


def test_csharp_calls_imports_and_await():
    _, refs, _ = _graph(CS)
    calls = {(r["caller_name"], r["callee_name"]) for r in refs if r["ref_type"] == "call"}
    run = "module.Shop.Orders.OrderService.Run"
    assert {(run, "Get"), (run, "Info"), (run, "Static"), (run, "Process"),
            ("module.Shop.Orders.OrderService.Get", "Order")} <= calls
    delay = next(r for r in refs if r["callee_name"] == "Delay")
    assert delay["await_kind"] == "await"
    imports = {r["callee_name"] for r in refs if r["ref_type"] == "import"}
    assert imports == {"System", "System.Collections.Generic", "System.Text.StringBuilder"}


def test_csharp_syntax_error_is_reported():
    _, _, errors = extract_graph("Bad.cs", "class A { void M( { } }")
    assert errors


# --- Pkl -------------------------------------------------------------------------


def test_pkl_symbols():
    symbols, _, errors = _graph(PKL)
    assert errors == []
    assert {("type", "Port"), ("class", "Database"), ("class", "Service"), ("class", "Base"),
            ("method", "url"), ("function", "double"), ("property", "host"),
            ("property", "replicas"), ("property", "db"), ("property", "label")} <= symbols


def test_pkl_refs():
    _, refs, _ = _graph(PKL)
    by_type = {}
    for r in refs:
        by_type.setdefault(r["ref_type"], set()).add((r["caller_name"], r["callee_name"]))
    assert {c for _, c in by_type["import"]} == {"base.pkl", "pkl:math", "lib/Server.pkl"}
    assert by_type["inherit"] == {("module.Service", "Base")}
    assert {("module.Service.replicas", "max"), ("module.db", "Database"),
            ("module.servers", "Server"), ("module.servers", "double"),
            ("module.label", "url")} <= by_type["call"]


def test_pkl_outline_labels_properties():
    outline = dict(extract_file_outline(PKL, _read(PKL)))
    assert outline[10] == "class Database"
    assert outline[12] == "  property host"
    assert outline[14] == "  def url"


# --- shared invariants -----------------------------------------------------------


@pytest.mark.parametrize("path", [CS, PKL])
def test_method_callers_match_chunk_names(path):
    """investigate/trace look refs up by the chunk's qualified name, so every call made
    inside a method must name that method exactly as the chunker does."""
    content = _read(path)
    chunk_names = {c["qualified_name"] for c in chunk_file(path, content)}
    _, refs, _ = extract_graph(path, content)
    method_callers = {r["caller_name"].removeprefix("module.") for r in refs
                      if r["ref_type"] == "call" and r["caller_name"].count(".") >= 2
                      and not r["caller_name"].endswith((".replicas", ".db", ".servers", ".label"))}
    assert method_callers and method_callers <= chunk_names, method_callers - chunk_names


def test_index_and_query_both_languages(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "Sample.cs").write_text(_read(CS))
    (repo / "config.pkl").write_text(_read(PKL))
    db = VectorDB(str(tmp_path / "x.db"))
    try:
        assert db.sync(str(repo), project="p", verbose=False)["added"] == 2
        names = {r[0] for r in db.backend.conn.execute("SELECT name FROM symbols")}
        assert {"OrderService", "SaveAsync", "Database", "replicas"} <= names
        hits = db.query("database connection", project="p")
        assert any(h["abs_path"].endswith("config.pkl") for h in hits)
    finally:
        db.close()
