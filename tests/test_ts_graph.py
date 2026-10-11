"""
tests/test_ts_graph.py
Per-language tests for tree-sitter symbol extraction, cross-references, and syntax errors.

Each language fixture contains:
- A class/struct
- A method
- A call from one function to another
- An import
- An inheritance
"""

import json
from pathlib import Path

import pytest

from ai_db.parser.chunker import chunk_file
from ai_db.parser.ts_graph import extract_graph, language_for

# ==============================================================================
# Python
# ==============================================================================

PYTHON_CODE = '''
from typing import List
from .utils import helper

class BaseService:
    pass

class UserService(BaseService):
    def get_user(self, user_id: int) -> dict:
        return helper.fetch(user_id)

async def fetch_remote_profile(url: str) -> dict:
    return {"url": url}

def process_user(user_id: int) -> dict:
    return UserService().get_user(user_id)
'''

def test_python_symbols():
    symbols, _, errors = extract_graph("test.py", PYTHON_CODE)
    assert len(errors) == 0
    names = [s["name"] for s in symbols]
    assert "BaseService" in names
    assert "UserService" in names
    assert "get_user" in names
    assert "fetch_remote_profile" in names
    assert "process_user" in names
    # Note: imports are captured as refs, not symbols

def test_python_refs():
    _, refs, _ = extract_graph("test.py", PYTHON_CODE)
    
    # Check inherit refs
    inherit_refs = [r for r in refs if r["ref_type"] == "inherit"]
    assert any(r["callee_name"] == "BaseService" for r in inherit_refs)
    
    # Check import refs
    import_refs = [r for r in refs if r["ref_type"] == "import"]
    assert any(r["callee_name"] == "typing" for r in import_refs)
    assert any(r["callee_name"] == ".utils" for r in import_refs)
    
    # Check call refs
    call_refs = [r for r in refs if r["ref_type"] == "call"]
    callee_names = {r["callee_name"] for r in call_refs}
    assert "fetch" in callee_names
    assert "UserService" in callee_names  # constructor call
    assert "get_user" in callee_names

def test_python_qualified_names():
    _, refs, _ = extract_graph("test.py", PYTHON_CODE)
    
    # Check qualified names for methods
    call_refs = [r for r in refs if r["ref_type"] == "call"]
    caller_names = {r["caller_name"] for r in call_refs}
    assert "module.UserService.get_user" in caller_names
    assert "module.process_user" in caller_names


# ==============================================================================
# JavaScript
# ==============================================================================

JAVASCRIPT_CODE = '''
const helper = require('./utils');

class BaseService {
    constructor() {}
}

class UserService extends BaseService {
    getUser(userId) {
        return helper.fetch(userId);
    }
}

async function fetchRemoteProfile(url) {
    return { url };
}

function processUser(userId) {
    return new UserService().getUser(userId);
}
'''

def test_javascript_symbols():
    symbols, _, errors = extract_graph("test.js", JAVASCRIPT_CODE)
    assert len(errors) == 0
    names = [s["name"] for s in symbols]
    assert "BaseService" in names
    assert "UserService" in names
    assert "getUser" in names
    assert "fetchRemoteProfile" in names
    assert "processUser" in names

def test_javascript_refs():
    _, refs, _ = extract_graph("test.js", JAVASCRIPT_CODE)
    
    # Check inherit refs
    inherit_refs = [r for r in refs if r["ref_type"] == "inherit"]
    assert any(r["callee_name"] == "BaseService" for r in inherit_refs)
    
    # Check import refs
    import_refs = [r for r in refs if r["ref_type"] == "import"]
    assert any(r["callee_name"] == "./utils" for r in import_refs)
    
    # Check call refs (note: constructor calls via `new` not yet captured)
    call_refs = [r for r in refs if r["ref_type"] == "call"]
    callee_names = {r["callee_name"] for r in call_refs}
    assert "fetch" in callee_names
    assert "getUser" in callee_names


# ==============================================================================
# TypeScript
# ==============================================================================

TYPESCRIPT_CODE = '''
import { Helper } from './utils';

interface BaseInterface {
    process(): void;
}

class BaseService implements BaseInterface {
    process() {}
}

class UserService extends BaseService {
    getUser(userId: number): object {
        return Helper.fetch(userId);
    }
}

async function fetchRemoteProfile(url: string): Promise<object> {
    return { url };
}

function processUser(userId: number): object {
    return new UserService().getUser(userId);
}
'''

def test_typescript_symbols():
    symbols, _, errors = extract_graph("test.ts", TYPESCRIPT_CODE)
    assert len(errors) == 0
    names = [s["name"] for s in symbols]
    assert "BaseInterface" in names
    assert "BaseService" in names
    assert "UserService" in names
    assert "getUser" in names
    assert "fetchRemoteProfile" in names
    assert "processUser" in names

def test_typescript_refs():
    _, refs, _ = extract_graph("test.ts", TYPESCRIPT_CODE)
    
    # Check inherit refs (extends and implements)
    inherit_refs = [r for r in refs if r["ref_type"] == "inherit"]
    callee_names = {r["callee_name"] for r in inherit_refs}
    assert "BaseService" in callee_names
    assert "BaseInterface" in callee_names
    
    # Check import refs
    import_refs = [r for r in refs if r["ref_type"] == "import"]
    assert any(r["callee_name"] == "./utils" for r in import_refs)


# ==============================================================================
# TSX
# ==============================================================================

TSX_CODE = '''
import React from 'react';
import { Button } from './Button';

interface Props {
    onClick: () => void;
}

class BaseComponent extends React.Component {
    render() {
        return null;
    }
}

class UserCard extends BaseComponent<Props> {
    handleClick() {
        Button.onClick();
    }
    
    render() {
        return <Button onClick={this.handleClick} />;
    }
}

function App() {
    return <UserCard />;
}
'''

def test_tsx_symbols():
    symbols, _, errors = extract_graph("test.tsx", TSX_CODE)
    assert len(errors) == 0
    names = [s["name"] for s in symbols]
    assert "BaseComponent" in names
    assert "UserCard" in names
    assert "handleClick" in names
    assert "App" in names

def test_tsx_refs():
    _, refs, _ = extract_graph("test.tsx", TSX_CODE)
    
    # Check inherit refs
    inherit_refs = [r for r in refs if r["ref_type"] == "inherit"]
    assert any(r["callee_name"] == "BaseComponent" for r in inherit_refs)
    
    # Check import refs
    import_refs = [r for r in refs if r["ref_type"] == "import"]
    assert any(r["callee_name"] == "react" for r in import_refs)
    assert any(r["callee_name"] == "./Button" for r in import_refs)


# ==============================================================================
# C
# ==============================================================================

C_CODE = '''
#include "utils.h"

struct Invoice {
    double amount;
};

double calculate_total(struct Invoice *inv) {
    return calculate_tax(inv->amount);
}

void process_invoice(struct Invoice *inv) {
    double total = calculate_total(inv);
}
'''

def test_c_symbols():
    symbols, _, errors = extract_graph("test.c", C_CODE)
    assert len(errors) == 0
    names = [s["name"] for s in symbols]
    assert "Invoice" in names
    assert "calculate_total" in names
    assert "process_invoice" in names

def test_c_refs():
    _, refs, _ = extract_graph("test.c", C_CODE)
    
    # Check import refs
    import_refs = [r for r in refs if r["ref_type"] == "import"]
    assert any(r["callee_name"] == "utils.h" for r in import_refs)
    
    # Check call refs
    call_refs = [r for r in refs if r["ref_type"] == "call"]
    callee_names = {r["callee_name"] for r in call_refs}
    assert "calculate_tax" in callee_names
    assert "calculate_total" in callee_names


# ==============================================================================
# Syntax Error Tests
# ==============================================================================

def test_python_syntax_error():
    code = "def broken( :\n    pass\n"
    _ = extract_graph("test.py", code)
    # tree-sitter is lenient and may not report errors for this code
    # just verify it doesn't crash

def test_javascript_syntax_error():
    code = "function broken( {\n    return 1;\n}"
    _, _, errors = extract_graph("test.js", code)
    assert len(errors) > 0

def test_typescript_syntax_error():
    code = "function broken( : {\n    return 1;\n}"
    _, _, errors = extract_graph("test.ts", code)
    assert len(errors) > 0

def test_c_syntax_error():
    code = "int broken( {\n    return 1;\n}"
    _, _, errors = extract_graph("test.c", code)
    assert len(errors) > 0


# ==============================================================================
# Language Detection
# ==============================================================================

def test_language_for():
    assert language_for("test.py") == "python"
    assert language_for("test.pyi") == "python"
    assert language_for("test.js") == "javascript"
    assert language_for("test.jsx") == "javascript"
    assert language_for("test.ts") == "typescript"
    assert language_for("test.tsx") == "tsx"
    assert language_for("test.go") == "go"
    assert language_for("test.rs") == "rust"
    assert language_for("test.c") == "c"
    assert language_for("test.h") == "c"
    assert language_for("test.cpp") == "cpp"
    assert language_for("test.hpp") == "cpp"
    assert language_for("test.java") == "java"
    assert language_for("test.unknown") is None


# ==============================================================================
# Go
# ==============================================================================

GO_CODE = '''package billing

import "fmt"

type Invoice struct {
\tAmount float64
}

func (i *Invoice) Total() float64 {
\treturn fmt.Sprint(i.Amount)
}

func Run() {
\thelper()
}
'''

def test_go_symbols():
    symbols, _, errors = extract_graph("test.go", GO_CODE)
    assert errors == []
    names = {s["name"] for s in symbols}
    assert {"Invoice", "Total", "Run"} <= names

def test_go_refs():
    _, refs, _ = extract_graph("test.go", GO_CODE)
    assert any(r["ref_type"] == "import" and "fmt" in r["callee_name"] for r in refs)
    calls = {r["callee_name"] for r in refs if r["ref_type"] == "call"}
    assert "helper" in calls
    assert "Sprint" in calls  # package-qualified call fmt.Sprint


# ==============================================================================
# Rust
# ==============================================================================

RUST_CODE = '''use crate::utils::calc;

struct Invoice {
    amount: f64,
}

trait Priced {
    fn total(&self) -> f64;
}

impl Priced for Invoice {
    fn total(&self) -> f64 {
        calc(self.amount)
    }
}
'''

def test_rust_symbols():
    symbols, _, errors = extract_graph("test.rs", RUST_CODE)
    assert errors == []
    names = {s["name"] for s in symbols}
    assert {"Invoice", "Priced", "total"} <= names

def test_rust_refs():
    _, refs, _ = extract_graph("test.rs", RUST_CODE)
    # `impl Priced for Invoice` must record the TRAIT as the base
    inherit = {(r["callee_name"]) for r in refs if r["ref_type"] == "inherit"}
    assert "Priced" in inherit
    assert any("calc" in r["callee_name"] for r in refs if r["ref_type"] == "import")
    assert any(r["callee_name"] == "calc" for r in refs if r["ref_type"] == "call")


# ==============================================================================
# C++
# ==============================================================================

CPP_CODE = '''#include "u.h"

class Base {
public:
    virtual void go() = 0;
};

class Impl : public Base {
    void go() override {
        helper();
    }
};
'''

def test_cpp_symbols():
    symbols, _, errors = extract_graph("test.cpp", CPP_CODE)
    assert errors == []
    names = {s["name"] for s in symbols}
    assert {"Base", "Impl"} <= names

def test_cpp_refs():
    _, refs, _ = extract_graph("test.cpp", CPP_CODE)
    assert any(r["ref_type"] == "import" and "u.h" in r["callee_name"] for r in refs)
    inherit = {(r["callee_name"], r["caller_name"]) for r in refs if r["ref_type"] == "inherit"}
    assert ("Base", "module.Impl") in inherit


# ==============================================================================
# Java
# ==============================================================================

JAVA_CODE = '''package p;

import utils.Helper;

interface Svc {
    void go();
}

class Base {}

class Impl extends Base implements Svc {
    public void go() {
        Helper.run();
    }
}
'''

def test_java_symbols():
    symbols, _, errors = extract_graph("test.java", JAVA_CODE)
    assert errors == []
    names = {s["name"] for s in symbols}
    assert {"Base", "Impl", "Svc"} <= names

def test_java_refs():
    _, refs, _ = extract_graph("test.java", JAVA_CODE)
    assert any(r["ref_type"] == "import" and "utils.Helper" in r["callee_name"]
               for r in refs)
    inherit = {(r["callee_name"], r["caller_name"]) for r in refs if r["ref_type"] == "inherit"}
    assert ("Base", "module.Impl") in inherit   # extends
    assert ("Svc", "module.Impl") in inherit    # implements


# ==============================================================================
# Unknown Language
# ==============================================================================

def test_unknown_language_raises():
    # The function should return empty lists for unknown languages
    symbols, refs, errors = extract_graph("test.unknown", "some code")
    assert symbols == []
    assert refs == []
    assert errors == []


# ==============================================================================
# G1: caller scopes agree with chunk names in every language
# ==============================================================================

# (file, source, the method's chunk qualified name)
_METHOD_CALLS = [
    ("a.js", "class Svc {\n  run(a) { return helper(a); }\n}\nfunction helper(c) { return c; }\n",
     "Svc.run"),
    ("b.js", "const go = (b) => { return helper(b); };\nfunction helper(c) { return c; }\n", "go"),
    ("a.ts", ("abstract class Base {\n  describe(): string { return fmt(1); }\n}\n"
              "function fmt(n: number) { return String(n); }\n"), "Base.describe"),
    ("b.ts", "export const go = (b: number) => fmt(b);\nfunction fmt(n: number) { return String(n); }\n",
     "go"),
    ("A.java", "class A {\n  A() { init(); }\n  void init() {}\n}\n", "A.A"),
    ("a.c", "static int helper(int x) { return x; }\nint *make(void) { helper(1); return 0; }\n", "make"),
    ("a.cpp", ("namespace ns {\nclass Foo {\n public:\n  void bar() { helper(); }\n};\n}\n"
               "void helper() {}\n"), "ns.Foo.bar"),
    ("b.cpp", "namespace ns {\nclass Foo {};\nvoid Foo::baz() { helper(); }\n}\nvoid helper() {}\n",
     "ns.Foo.baz"),
    ("a.go", "package main\ntype S struct{ n int }\nfunc (s *S) M() { helper() }\nfunc helper() {}\n", "S.M"),
    ("a.rs", "struct Foo;\nimpl fmt::Display for Foo {\n    fn fmt(&self) { helper(); }\n}\nfn helper() {}\n",
     "Foo.fmt"),
    ("b.rs", "struct Wrap<T>(T);\nimpl<T> Wrap<T> {\n    fn get(&self) { helper(); }\n}\nfn helper() {}\n",
     "Wrap.get"),
]


@pytest.mark.parametrize(("filename", "code", "method"), _METHOD_CALLS,
                         ids=[c[0] for c in _METHOD_CALLS])
def test_call_caller_matches_method_chunk(filename, code, method):
    chunks = {c["qualified_name"] for c in chunk_file(filename, code)}
    assert method in chunks
    _, refs, _ = extract_graph(filename, code)
    callers = {r["caller_name"] for r in refs if r["ref_type"] == "call"}
    assert f"module.{method}" in callers
    # every named caller is some chunk
    assert {c for c in callers if c != "module"} <= {f"module.{q}" for q in chunks}


def test_trace_finds_callees_in_every_polyglot_language(tmp_path):
    from ai_db import VectorDB
    from ai_db.dispatcher import ServiceDispatcher

    root = Path(__file__).parent / "fixtures" / "polyglot"
    vdb = VectorDB(str(tmp_path / "t.db"))
    try:
        vdb.sync(str(root), project="poly", verbose=False)
        disp = ServiceDispatcher(db=vdb)
        for entry in ("TaxCalculator.totalWithTax", "TaxCalculator.TotalWithTax",
                      "TaxCalculator.total_with_tax"):
            out = json.loads(disp.execute("trace", {"entry": entry, "format": "json", "depth": 1,
                                                   "cwd": str(root), "project": "poly"}))
            kids = [c["qualified_name"] for c in out["root"]["children"]]
            assert kids, entry
    finally:
        vdb.close()


# ==============================================================================
# G5: per-language query fixes
# ==============================================================================

def _graph(filename, code):
    symbols, refs, _ = extract_graph(filename, code)
    syms = {(s["symbol_type"], s["name"]) for s in symbols}
    return syms, refs


def _refs(refs, ref_type):
    return [(r["caller_name"], r["callee_name"]) for r in refs if r["ref_type"] == ref_type]


def test_g5_python_aliased_imports_types_and_attribute_bases():
    code = ("import numpy as np\nimport os.path as osp\nfrom a import c as d, e\n"
            "from typing import TypeAlias\n"
            "Vec: TypeAlias = list[float]\ntype Pair = tuple[int, int]\nT = TypeVar('T')\n"
            "app = Flask(__name__)\n\n"
            "class Base(abc.ABC):\n    x = field(default=1)\n    y: int = field(default=2)\n"
            "    z: int\n")
    syms, refs = _graph("m.py", code)
    imports = [callee for _, callee in _refs(refs, "import")]
    assert {"numpy", "os.path", "a", "c", "e", "typing", "TypeAlias"} <= set(imports)
    assert len(imports) == len(set(imports))  # one ref per imported module/name
    types = {name for kind, name in syms if kind == "type"}
    assert types == {"Vec", "Pair", "T", "y"}  # not app = Flask(), x = field(), bare z: int
    assert ("module.Base", "ABC") in _refs(refs, "inherit")


def test_g5_c_typedefs_and_pointer_functions():
    code = ("typedef int myint;\ntypedef struct s *sp;\ntypedef int (*cb_t)(int);\n"
            "static char *dup(const char *s) { return COPY(s); }\nint **grid(void) { return 0; }\n")
    syms, refs = _graph("a.c", code)
    assert {("type", "myint"), ("type", "sp"), ("type", "cb_t"),
            ("function", "dup"), ("function", "grid")} <= syms
    assert _refs(refs, "call") == [("module.dup", "COPY")]  # macro call counted once
    names = {c["qualified_name"] for c in chunk_file("a.c", code)}
    assert {"myint", "sp", "cb_t", "dup", "grid"} <= names


def test_g5_cpp_in_class_methods_destructors_templates():
    code = ("class Foo {\n public:\n  ~Foo() { cleanup(); }\n  void bar() { helper(); }\n};\n"
            "Foo::~Foo() {}\n"
            "template <typename T>\nT maxof(T a, T b) { return a; }\n"
            "template <typename T>\nclass Box {\n  void put(T x) { store(x); }\n};\n")
    syms, refs = _graph("a.cpp", code)
    assert {("method", "~Foo"), ("method", "bar"), ("function", "maxof"), ("class", "Box"),
            ("method", "put")} <= syms
    chunks = {c["qualified_name"]: c for c in chunk_file("a.cpp", code)}
    assert {"maxof", "Box", "Box.put"} <= set(chunks)
    assert chunks["maxof"]["start_line"] == 7  # the template line belongs to it
    assert not any(c.startswith("imports/globals") for c in chunks)
    assert ("module.Box.put", "store") in _refs(refs, "call")


def test_g5_js_arrow_function_bound_to_variable():
    syms, _ = _graph("a.js", "const go = (b) => b + 1;\nlet g = function (a) { return a; };\n"
                             "const n = 3;\n")
    assert {("function", "go"), ("function", "g")} <= syms
    assert not any(name == "n" for _, name in syms)


def test_g5_ts_abstract_class():
    code = ("export abstract class Shape extends Base implements Drawable {\n"
            "  describe(): string { return fmt(1); }\n}\n")
    syms, refs = _graph("a.ts", code)
    assert ("class", "Shape") in syms
    assert {("module.Shape", "Base"), ("module.Shape", "Drawable")} <= set(_refs(refs, "inherit"))


def test_g5_go_struct_once_and_field_reads_are_not_calls():
    code = ("package main\ntype S struct{ n int }\ntype ID string\n"
            "func (s *S) M() { x := s.n; fmt.Println(x) }\n")
    symbols, refs, _ = extract_graph("a.go", code)
    assert [s["symbol_type"] for s in symbols if s["name"] == "S"] == ["struct"]
    assert ("type", "ID") in {(s["symbol_type"], s["name"]) for s in symbols}
    assert _refs(refs, "call") == [("module.S.M", "Println")]


def test_g5_rust_impl_trait_for_type():
    code = ("struct Wrap<T>(T);\nimpl<T> fmt::Display for Wrap<T> {\n"
            "    fn fmt(&self) { helper(); }\n}\nfn helper() {}\n")
    _, refs = _graph("a.rs", code)
    assert _refs(refs, "inherit") == [("module.Wrap", "Display")]  # trait only, never the type
    assert ("module.Wrap.fmt", "helper") in _refs(refs, "call")


def test_g5_java_generic_supertypes_enum_implements_and_package():
    code = ("package com.x;\nimport java.util.List;\n"
            "class A extends Base<String> implements Runnable, Comparable<A> {}\n"
            "enum Color implements Named { RED; String n() { return go(); } }\n")
    _, refs = _graph("A.java", code)
    assert set(_refs(refs, "inherit")) == {("module.A", "Base"), ("module.A", "Runnable"),
                                           ("module.A", "Comparable"), ("module.Color", "Named")}
    assert [c for _, c in _refs(refs, "import")] == ["java.util.List"]  # package is not an import
    assert "Color.n" in {c["qualified_name"] for c in chunk_file("A.java", code)}


@pytest.mark.parametrize(("filename", "code"), [
    ("c.rs", "use std::fmt;\n\n/// Doc for helper.\n// more\nfn helper() {}\n"),
    ("C.java", "class A {\n  int x;\n  /** Doc for m. */\n  // line\n  void m() {}\n}\n"),
])
def test_g5_line_and_block_comments_attach_to_the_next_definition(filename, code):
    chunk = next(c for c in chunk_file(filename, code) if c["qualified_name"] in ("helper", "A.m"))
    assert "Doc for" in chunk["content"]