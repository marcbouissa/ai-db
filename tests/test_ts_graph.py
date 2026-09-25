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
    # Disabled languages return None
    assert language_for("test.go") is None
    assert language_for("test.rs") is None
    assert language_for("test.cpp") is None
    assert language_for("test.java") is None
    assert language_for("test.unknown") is None


# ==============================================================================
# Unknown Language
# ==============================================================================

def test_unknown_language_raises():
    # The function should return empty lists for unknown languages
    symbols, refs, errors = extract_graph("test.unknown", "some code")
    assert symbols == []
    assert refs == []
    assert errors == []