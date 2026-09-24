"""Tree-sitter structural chunking."""

from ai_db.constants import MAX_CHUNK_TOKENS
from ai_db.parser.chunker import chunk_file, count_tokens


def _by_name(chunks):
    return {c["qualified_name"]: c for c in chunks}


def test_python_nested_class_methods_are_own_chunks():
    code = (
        "import os\n\n"
        "class Outer:\n"
        "    \"\"\"Doc.\"\"\"\n"
        "    x = 1\n\n"
        "    def a(self):\n"
        "        return 1\n\n"
        "    class Inner:\n"
        "        def b(self):\n"
        "            return 2\n"
    )
    chunks = chunk_file("m.py", code)
    by = _by_name(chunks)
    assert {"imports/globals", "Outer", "Outer.a", "Outer.Inner", "Outer.Inner.b"} <= set(by)
    header = by["Outer"]
    assert header["chunk_type"] == "class_header"
    assert "x = 1" in header["content"] and "return 1" not in header["content"]
    assert chunks[by["Outer.a"]["parent_index"]]["qualified_name"] == "Outer"
    assert chunks[by["Outer.Inner.b"]["parent_index"]]["qualified_name"] == "Outer.Inner"
    assert chunks[header["parent_index"]]["name"] == "imports/globals"


def test_decorators_and_leading_comments_attach_to_definition():
    code = "import x\n\n# explains f\n@decorator\ndef f():\n    pass\n"
    by = _by_name(chunk_file("d.py", code))
    assert by["f"]["start_line"] == 3
    assert "# explains f" in by["f"]["content"] and "@decorator" in by["f"]["content"]
    assert "explains" not in by["imports/globals"]["content"]


def test_oversized_function_split_with_headers():
    body = "\n".join(f"    value_{i} = compute_something({i}, 'padding text here')" for i in range(300))
    code = f"def huge(a, b):\n{body}\n"
    chunks = chunk_file("big.py", code)
    parts = [c for c in chunks if c["qualified_name"] == "huge"]
    assert len(parts) > 1
    for i, p in enumerate(parts, start=1):
        assert p["content"].startswith(f"# big.py::huge (part {i}/{len(parts)})")
        assert p["token_count"] <= MAX_CHUNK_TOKENS
        if i > 1:
            assert p["content"].splitlines()[1] == "def huge(a, b):"
            assert p["parent_index"] == chunks.index(parts[0])
    covered = sorted(ln for p in parts for ln in range(p["start_line"], p["end_line"] + 1))
    assert covered == list(range(1, 302))


def test_typescript_class_methods_and_arrow_functions():
    code = (
        "import x from 'y';\n"
        "export class ApiClient {\n"
        "  public async post(path: string) { return 1; }\n"
        "}\n"
        "export const handler = async (e) => {\n"
        "  return e;\n"
        "};\n"
    )
    by = _by_name(chunk_file("api.ts", code))
    assert by["ApiClient"]["chunk_type"] == "class_header"
    assert "ApiClient.post" in by
    assert by["handler"]["chunk_type"] == "code"
    assert by["handler"]["language"] == "typescript"


def test_go_method_receiver_qualifies_name():
    code = "package main\n\ntype S struct{ a int }\n\nfunc (s *S) M() {\n}\n\nfunc F() {}\n"
    by = _by_name(chunk_file("a.go", code))
    assert {"S", "S.M", "F"} <= set(by)


def test_rust_impl_methods():
    code = "struct S;\nimpl S {\n    fn m(&self) {}\n}\n"
    by = _by_name(chunk_file("a.rs", code))
    assert "S.m" in by


def test_markdown_header_path_and_code_fence():
    md = "# Install\n\ntext\n\n## MySQL\n\n```sh\n# not a header\n```\n\n# Usage\n\nuse\n"
    chunks = chunk_file("README.md", md)
    assert [c["qualified_name"] for c in chunks] == ["Install", "Install > MySQL", "Usage"]
    assert [c["name"] for c in chunks] == ["Install", "MySQL", "Usage"]
    assert "# not a header" in chunks[1]["content"]


def test_plain_text_windows_overlap():
    text = "\n".join(f"line {i}" for i in range(1, 131))
    chunks = chunk_file("notes.txt", text)
    assert [(c["start_line"], c["end_line"]) for c in chunks] == [(1, 60), (51, 110), (101, 130)]


def test_syntax_error_still_chunked():
    chunks = chunk_file("bad.py", "def ok():\n    return 1\n\ndef broken(:\n    pass\n")
    assert any(c["qualified_name"] == "ok" for c in chunks)


def test_chunk_metadata_fields():
    c = chunk_file("m.py", "def f():\n    return 1\n")[0]
    assert c["token_count"] == count_tokens(c["content"])
    assert len(c["content_hash"]) == 64
