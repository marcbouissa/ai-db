"""
tests/test_parser.py
Comprehensive 4-tier test suite for the Parser and Code Analysis subsystem.

Tiers:
  - Tier 1: Feature Coverage (AST extraction, signatures, outlines, syntax, chunking)
  - Tier 2: Boundary & Corner Cases (empty, whitespace, syntax errors, polyglot, missing linters)
  - Tier 3: Pairwise Combinations (chunk-symbol alignment, syntax fallback outlines, cross-refs)
  - Tier 4: Real-World Workflows (incremental code editing lifecycle, polyglot repository ingest)
"""
from ai_db.parser.annotations import extract_annotations
from ai_db.parser.ast_visitor import extract_file_outline, extract_symbols
from ai_db.parser.chunker import chunk_file
from ai_db.parser.cross_refs import extract_cross_refs
from ai_db.parser.linters import ExternalLinter
from ai_db.parser.syntax import validate_python_syntax

# ==============================================================================
# Tier 1: Feature Coverage (Happy Path Isolation)
# ==============================================================================

class TestParserTier1FeatureCoverage:
    """Happy-path tests for core parsing, extraction, and validation features."""

    def test_validate_python_syntax_valid(self):
        """Valid Python module parses without errors."""
        content = (
            "def calculate_total(items: list[float], tax_rate: float = 0.05) -> float:\n"
            "    return sum(items) * (1.0 + tax_rate)\n"
        )
        result = validate_python_syntax(content, "accounting.py")
        assert result is None

    def test_validate_python_syntax_invalid(self):
        """Malformed Python code returns precise line, col, and message."""
        content = (
            "def broken_function( :\n"
            "    pass\n"
        )
        result = validate_python_syntax(content, "broken.py")
        assert result is not None
        line, col, msg = result
        assert line == 1
        assert isinstance(col, int)
        assert len(msg) > 0

    def test_extract_symbols_classes_and_functions(self):
        """Extracts classes with inheritance, sync functions, and async functions."""
        code = (
            "class BaseService:\n"
            "    pass\n\n"
            "class UserService(BaseService):\n"
            "    def get_user(self, user_id: int):\n"
            "        return {'id': user_id}\n\n"
            "async def fetch_remote_profile(url: str) -> dict:\n"
            "    return {}\n"
        )
        symbols = extract_symbols("services.py", code)
        names = [s["name"] for s in symbols]
        assert "BaseService" in names
        assert "UserService" in names
        assert "get_user" in names
        assert "fetch_remote_profile" in names

        # Verify symbol types and signatures
        user_svc = next(s for s in symbols if s["name"] == "UserService")
        assert user_svc["symbol_type"] == "class"
        assert "class UserService(BaseService)" in user_svc["signature"]
        assert user_svc["line"] == 4

        fetch_sym = next(s for s in symbols if s["name"] == "fetch_remote_profile")
        assert fetch_sym["symbol_type"] == "def"
        assert fetch_sym["signature"].startswith("async def fetch_remote_profile")

    def test_extract_symbols_nested_scopes(self):
        """Nested methods and functions within class bodies are captured with proper lines."""
        code = (
            "class Pipeline:\n"
            "    def run(self):\n"
            "        def step_one():\n"
            "            return 1\n"
            "        return step_one()\n"
        )
        symbols = extract_symbols("pipeline.py", code)
        names = [s["name"] for s in symbols]
        assert "Pipeline" in names
        assert "run" in names
        assert "step_one" in names

    def test_extract_file_outline_structure(self):
        """File outline returns sorted line numbers, signatures, and constants."""
        code = (
            "VERSION = '1.0.0'\n"
            "MAX_LIMIT = 500\n\n"
            "class DatabaseManager:\n"
            "    def connect(self, dsn):\n"
            "        pass\n\n"
            "    async def ping(self):\n"
            "        pass\n\n"
            "def initialize():\n"
            "    pass\n"
        )
        outline = extract_file_outline("db_mgr.py", code)
        assert len(outline) >= 4
        lines = [item[0] for item in outline]
        assert lines == sorted(lines)

        labels = [item[1] for item in outline]
        assert any("MAX_LIMIT" in l for l in labels)
        assert any("class DatabaseManager" in l for l in labels)
        assert any("connect" in l for l in labels)
        assert any("ping" in l for l in labels)
        assert any("def initialize" in l for l in labels)

    def test_chunk_file_python_ast(self):
        """Python file is broken into imports/globals, class blocks, and function blocks."""
        code = (
            "import os\n"
            "import sys\n\n"
            "GLOBAL_DEBUG = False\n\n"
            "class Processor:\n"
            "    def process(self, data):\n"
            "        return data.strip()\n\n"
            "def standalone_func():\n"
            "    return 42\n"
        )
        chunks = chunk_file("processor.py", code)
        assert len(chunks) >= 3

        chunk_names = [c["name"] for c in chunks]
        assert "imports/globals" in chunk_names
        assert any("class Processor" in name for name in chunk_names)
        assert any("def standalone_func" in name for name in chunk_names)

        for c in chunks:
            assert c["chunk_type"] in ("code", "module", "class_header")
            assert c["start_line"] <= c["end_line"]
            assert len(c["content"]) > 0

    def test_chunk_file_markdown_headers(self):
        """Markdown documents are chunked according to markdown '#' header sections."""
        md_content = (
            "# Project Overview\n\n"
            "This document describes the platform architecture.\n\n"
            "## Architecture Components\n\n"
            "The system consists of core parser and storage modules.\n\n"
            "### Database Subsystem\n\n"
            "SQLite with WAL mode is used for high concurrency.\n"
        )
        chunks = chunk_file("README.md", md_content)
        assert len(chunks) == 3
        headers = [c["name"] for c in chunks]
        assert "Project Overview" in headers
        assert "Architecture Components" in headers
        assert "Database Subsystem" in headers

        for c in chunks:
            assert c["chunk_type"] == "md"
            assert c["start_line"] <= c["end_line"]

    def test_chunk_file_package_metadata(self):
        """Metadata files like pyproject.toml are classified as lib_meta and capped."""
        toml_content = (
            "[project]\n"
            "name = 'ai-db'\n"
            "version = '2.0.0'\n"
            "description = 'Code intelligence engine'\n"
        )
        chunks = chunk_file("pyproject.toml", toml_content)
        assert len(chunks) == 1
        assert chunks[0]["chunk_type"] == "lib_meta"
        assert chunks[0]["name"] == "pkg:pyproject.toml"
        assert len(chunks[0]["content"]) <= 1500


# ==============================================================================
# Tier 2: Boundary & Corner Cases
# ==============================================================================

class TestParserTier2BoundaryAndCorner:
    """Boundary conditions, unparseable source, empty inputs, and missing tooling."""

    def test_validate_python_syntax_empty_file(self):
        """Empty string is valid Python AST, returns None."""
        assert validate_python_syntax("", "empty.py") is None

    def test_validate_python_syntax_whitespace_only(self):
        """Whitespace-only string is valid Python AST, returns None."""
        assert validate_python_syntax("   \n\t  \n  ", "spaces.py") is None

    def test_extract_symbols_unparseable_python(self):
        """When AST parsing fails due to syntax errors, regex fallback preserves symbol extraction."""
        broken_code = (
            "class PartiallyBrokenService:\n"
            "    def valid_method(self):\n"
            "        return True\n\n"
            "    def broken_syntax( :   # Intentional syntax failure\n"
            "        pass\n"
        )
        symbols = extract_symbols("broken_service.py", broken_code)
        names = [s["name"] for s in symbols]
        assert "PartiallyBrokenService" in names
        assert "valid_method" in names or "broken_syntax" in names

    def test_extract_symbols_polyglot_code(self):
        """Extracts symbols from TypeScript and JavaScript files via regex."""
        ts_code = (
            "export class AuthService {\n"
            "    public async authenticate(token: string) {\n"
            "        return true;\n"
            "    }\n"
            "}\n\n"
            "export const formatToken = (raw: string) => {\n"
            "    return raw.trim();\n"
            "};\n"
        )
        symbols = extract_symbols("auth.ts", ts_code)
        names = [s["name"] for s in symbols]
        assert "AuthService" in names
        assert "formatToken" in names

    def test_extract_symbols_empty_string(self):
        """Extracting symbols from empty file returns empty list."""
        assert extract_symbols("empty.py", "") == []
        assert extract_symbols("empty.ts", "") == []

    def test_chunk_file_zero_bytes(self):
        """Chunking empty content returns empty list."""
        assert chunk_file("empty.py", "") == []
        assert chunk_file("empty.md", "") == []

    def test_chunk_file_large_single_block(self):
        """Files with long unheaded blocks are partitioned into windowed chunks."""
        lines = [f"x_{i} = {i} * 2" for i in range(120)]
        content = "\n".join(lines)
        chunks = chunk_file("constants.py", content)
        assert len(chunks) >= 2
        for c in chunks:
            assert c["start_line"] <= c["end_line"]

    def test_external_linter_missing_binary(self):
        """Linter gracefully returns None when configured tool is not in PATH."""
        linter = ExternalLinter(config_validators={
            ".custom": ["__non_existent_binary_tool_12345__", "{file}"]
        })
        res = linter.validate("test.custom", "some content")
        assert res is None


# ==============================================================================
# Tier 3: Pairwise Combinations
# ==============================================================================

class TestParserTier3Combinations:
    """Pairwise interaction between chunking, AST symbols, outlines, and cross-refs."""

    def test_parser_chunking_and_symbol_alignment(self):
        """Verifies that chunk line intervals envelop the corresponding extracted symbol definitions."""
        code = (
            "def alpha():\n"
            "    pass\n\n"
            "class Beta:\n"
            "    def gamma(self):\n"
            "        pass\n"
        )
        symbols = extract_symbols("module.py", code)
        chunks = chunk_file("module.py", code)

        for sym in symbols:
            # Each symbol must fall within at least one chunk range
            enclosing = [
                c for c in chunks
                if c["start_line"] <= sym["line"] <= c["end_line"]
            ]
            assert len(enclosing) > 0, f"Symbol {sym['name']} at line {sym['line']} has no enclosing chunk"

    def test_parser_syntax_error_then_outline_fallback(self):
        """Outlines continue to extract structural headers even when syntax errors break AST."""
        broken_code = (
            "class BrokenASTClass:\n"
            "    def broken( :\n"
            "        pass\n"
        )
        outline = extract_file_outline("broken.py", broken_code)
        assert len(outline) > 0
        labels = [item[1] for item in outline]
        assert any("BrokenASTClass" in l for l in labels)

    def test_parser_cross_refs_with_nested_classes(self):
        """Extracts inheritance and call site references accurately from AST."""
        code = (
            "class SuperHandler:\n"
            "    def handle(self):\n"
            "        pass\n\n"
            "class CustomHandler(SuperHandler):\n"
            "    def handle(self):\n"
            "        super().handle()\n"
        )
        refs = extract_cross_refs("handler.py", code)
        callees = [r["callee_name"] for r in refs]
        assert "SuperHandler" in callees

    def test_parser_annotations_extraction(self):
        """Extracts comment annotations (TODO/FIXME/HACK) and module/function docstrings."""
        code = (
            "'''Module level docstring for auditing.'''\n\n"
            "# TODO: optimize memory footprint\n"
            "def worker():\n"
            "    '''Worker function docstring.'''\n"
            "    # FIXME: handle network timeout\n"
            "    pass\n"
        )
        annotations = extract_annotations("audit.py", code)
        kinds = [a["kind"] for a in annotations]
        assert "todo" in kinds
        assert "fixme" in kinds
        assert "docstring" in kinds

        todo_item = next(a for a in annotations if a["kind"] == "todo")
        assert "optimize memory footprint" in todo_item["content"]


# ==============================================================================
# Tier 4: Real-World Application Workflows
# ==============================================================================

class TestParserTier4Workflows:
    """End-to-end multi-step scenarios representing realistic developer operations."""

    def test_workflow_incremental_code_editing(self):
        """
        Simulates an active code edit session:
        1. Clean valid state -> clean syntax, AST symbols parsed.
        2. Broken state mid-typing -> syntax error flagged, fallback regex extracts symbols.
        3. Restored valid state -> syntax error cleared, AST parsing restored.
        """
        v1_clean = (
            "class PaymentGateway:\n"
            "    def charge(self, amount: float) -> bool:\n"
            "        return True\n"
        )
        assert validate_python_syntax(v1_clean, "pay.py") is None
        syms_v1 = extract_symbols("pay.py", v1_clean)
        assert any(s["name"] == "charge" for s in syms_v1)

        v2_broken = (
            "class PaymentGateway:\n"
            "    def charge(self, amount: float) -> bool:\n"
            "        return True\n\n"
            "    def refund(self,   # incomplete edit\n"
        )
        err = validate_python_syntax(v2_broken, "pay.py")
        assert err is not None
        # Fallback regex still discovers PaymentGateway and charge
        syms_v2 = extract_symbols("pay.py", v2_broken)
        assert any(s["name"] == "PaymentGateway" for s in syms_v2)

        v3_fixed = (
            "class PaymentGateway:\n"
            "    def charge(self, amount: float) -> bool:\n"
            "        return True\n\n"
            "    def refund(self, tx_id: str) -> bool:\n"
            "        return True\n"
        )
        assert validate_python_syntax(v3_fixed, "pay.py") is None
        syms_v3 = extract_symbols("pay.py", v3_fixed)
        v3_names = [s["name"] for s in syms_v3]
        assert "PaymentGateway" in v3_names
        assert "charge" in v3_names
        assert "refund" in v3_names

    def test_workflow_polyglot_repository_ingest(self):
        """
        Simulates parsing a heterogeneous multi-language repository:
        Python module, TypeScript API, and Documentation markdown.
        """
        files = {
            "src/core.py": "class CoreEngine:\n    def execute(self):\n        pass\n",
            "frontend/api.ts": "export class ApiClient {\n    public async post(path: string) {}\n}\n",
            "docs/guide.md": "# Quickstart Guide\n\nRun the engine with --start.\n\n## Troubleshooting\n\nCheck logs.\n",
        }

        results = {}
        for path, content in files.items():
            symbols = extract_symbols(path, content)
            chunks = chunk_file(path, content)
            results[path] = {"symbols": symbols, "chunks": chunks}

        # Python verification
        py_data = results["src/core.py"]
        assert any(s["name"] == "CoreEngine" for s in py_data["symbols"])
        assert any("class CoreEngine" in c["name"] for c in py_data["chunks"])

        # TypeScript verification
        ts_data = results["frontend/api.ts"]
        assert any(s["name"] == "ApiClient" for s in ts_data["symbols"])
        assert len(ts_data["chunks"]) > 0

        # Markdown verification
        md_data = results["docs/guide.md"]
        assert len(md_data["chunks"]) == 2
        assert md_data["chunks"][0]["chunk_type"] == "md"
        assert md_data["chunks"][0]["name"] == "Quickstart Guide"
