"""Structural chunking with tree-sitter.

One chunking path per file kind:

- package metadata (``package.json``, ``METADATA``, ``pyproject.toml``): one ``lib_meta`` chunk
- markdown: one chunk per header section, named by header, qualified by header path
- source code in a selected language (``ai_db.parser.languages``): tree-sitter structural chunks
  (module runs, functions, class headers + one chunk per method, recursively)
- any other text file: fixed line windows with overlap

Code chunks never exceed ``MAX_CHUNK_TOKENS`` (tiktoken ``o200k_base``). Oversized nodes
are split at child-statement boundaries; every continuation part starts with a
``<comment> <file>::<qualified name> (part i/n)`` header plus the signature line.

Each chunk is a dict with: chunk_type, name, qualified_name, language, start_line,
end_line, content, token_count, content_hash, parent_index (index of the parent chunk in
the returned list, or None).
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Collection
from dataclasses import dataclass, field
from functools import cache
from typing import Any

from ai_db.constants import (
    MAX_CHUNK_TOKENS,
    TEXT_WINDOW_LINES,
    TEXT_WINDOW_OVERLAP,
)
from ai_db.parser.languages import SUPPORTED_LANGUAGES, enabled_language_for
from ai_db.parser.languages import language_for as _language_for
from ai_db.parser.naming import SCOPE_CONTAINERS, definition_name, qualified_name
from ai_db.tokens import count_tokens
from ai_db.utils import strip_code_bloat

META_FILES = ("package.json", "METADATA", "pyproject.toml")
META_MAX_CHARS = 1500


@cache
def _parser(lang: str) -> Any:
    from tree_sitter_language_pack import get_parser

    return get_parser(lang)


@dataclass(frozen=True)
class LangSpec:
    comment: str
    funcs: frozenset[str]
    containers: frozenset[str]  # classes, impls, interfaces, namespaces
    leaves: frozenset[str] = frozenset()  # standalone type/struct/enum declarations
    wrappers: dict[str, str] = field(default_factory=dict)  # node type -> inner field
    var_decls: frozenset[str] = frozenset()  # const f = () => {}


_JS_FUNCS = frozenset({"function_declaration", "generator_function_declaration", "method_definition"})
_JS_WRAPPERS = {"export_statement": "declaration"}
_JS_VARS = frozenset({"lexical_declaration", "variable_declaration"})
_TS_LEAVES = frozenset({"type_alias_declaration", "enum_declaration"})

LANG_SPECS: dict[str, LangSpec] = {
    "python": LangSpec(
        comment="#",
        funcs=frozenset({"function_definition"}),
        containers=SCOPE_CONTAINERS["python"],
        wrappers={"decorated_definition": "definition"},
    ),
    "javascript": LangSpec(
        comment="//", funcs=_JS_FUNCS, containers=SCOPE_CONTAINERS["javascript"],
        wrappers=_JS_WRAPPERS, var_decls=_JS_VARS,
    ),
    "typescript": LangSpec(
        comment="//", funcs=_JS_FUNCS, containers=SCOPE_CONTAINERS["typescript"], leaves=_TS_LEAVES,
        wrappers=_JS_WRAPPERS, var_decls=_JS_VARS,
    ),
    "tsx": LangSpec(
        comment="//", funcs=_JS_FUNCS, containers=SCOPE_CONTAINERS["tsx"], leaves=_TS_LEAVES,
        wrappers=_JS_WRAPPERS, var_decls=_JS_VARS,
    ),
    "go": LangSpec(
        comment="//",
        funcs=frozenset({"function_declaration", "method_declaration"}),
        containers=SCOPE_CONTAINERS["go"],
        leaves=frozenset({"type_declaration"}),
    ),
    "rust": LangSpec(
        comment="//",
        funcs=frozenset({"function_item"}),
        containers=SCOPE_CONTAINERS["rust"],
        leaves=frozenset({"struct_item", "enum_item", "type_item"}),
    ),
    "c": LangSpec(
        comment="//",
        funcs=frozenset({"function_definition"}),
        containers=SCOPE_CONTAINERS["c"],
        leaves=frozenset({"struct_specifier", "enum_specifier", "type_definition"}),
    ),
    "cpp": LangSpec(
        comment="//",
        funcs=frozenset({"function_definition"}),
        containers=SCOPE_CONTAINERS["cpp"],
        leaves=frozenset({"enum_specifier", "type_definition"}),
        # `template <...>` wraps the function/class it declares (no field: last child)
        wrappers={"template_declaration": "declaration"},
    ),
    "java": LangSpec(
        comment="//",
        funcs=frozenset({"method_declaration", "constructor_declaration"}),
        containers=SCOPE_CONTAINERS["java"],
    ),
    "csharp": LangSpec(
        comment="//",
        funcs=frozenset({"method_declaration", "constructor_declaration", "destructor_declaration",
                         "operator_declaration"}),
        # Namespaces are containers so their classes are chunked (and qualified) like
        # the call graph names them (ai_db.parser.naming).
        containers=SCOPE_CONTAINERS["csharp"],
        leaves=frozenset({"enum_declaration", "delegate_declaration"}),
    ),
    # Apple Pkl. Module-level properties (the config values) stay in the module
    # chunks; classes, methods and type aliases get their own.
    "pkl": LangSpec(
        comment="//",
        funcs=frozenset({"classMethod"}),
        containers=SCOPE_CONTAINERS["pkl"],
        leaves=frozenset({"typeAlias"}),
    ),
}


def language_for(filepath: str) -> str | None:
    return _language_for(filepath)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _Builder:
    """Accumulates chunk dicts for one file."""

    def __init__(self, filepath: str, lines: list[str], language: str, comment: str):
        self.filepath = filepath
        self.basename = os.path.basename(filepath)
        self.lines = lines
        self.language = language
        self.comment = comment
        self.chunks: list[dict[str, Any]] = []
        self._line_tokens = [count_tokens(line) + 1 for line in lines]

    def tokens(self, start: int, end: int) -> int:
        """Token estimate for 1-based inclusive line range (exact per line)."""
        return sum(self._line_tokens[start - 1:end])

    def text(self, start: int, end: int, skip: list[tuple[int, int]] | None = None) -> str:
        out = []
        for ln in range(start, end + 1):
            if skip and any(a <= ln <= b for a, b in skip):
                continue
            out.append(self.lines[ln - 1])
        return strip_code_bloat("\n".join(out).strip("\n"))

    def add(self, chunk_type: str, name: str, qualified: str, start: int, end: int,
            content: str, parent: int | None) -> int | None:
        if not content.strip():
            return None
        self.chunks.append({
            "chunk_type": chunk_type,
            "name": name,
            "qualified_name": qualified,
            "language": self.language,
            "start_line": start,
            "end_line": end,
            "content": content,
            "token_count": count_tokens(content),
            "content_hash": _hash(content),
            "parent_index": parent,
        })
        return len(self.chunks) - 1

    def add_split(self, chunk_type: str, name: str, qualified: str, start: int, end: int,
                  boundaries: list[int], parent: int | None,
                  skip: list[tuple[int, int]] | None = None) -> int | None:
        """Add one chunk, or several parts if the span exceeds MAX_CHUNK_TOKENS.

        ``boundaries`` are 1-based line numbers where a split is allowed.
        Returns the index of the first part.
        """
        skip = skip or []
        live = [ln for ln in range(start, end + 1) if not any(a <= ln <= b for a, b in skip)]
        total = sum(self._line_tokens[ln - 1] for ln in live)
        if total <= MAX_CHUNK_TOKENS:
            return self.add(chunk_type, name, qualified, start, end, self.text(start, end, skip), parent)

        signature = self.lines[start - 1].strip()
        header_budget = count_tokens(signature) + 24
        budget = max(64, MAX_CHUNK_TOKENS - header_budget)
        allowed = set(boundaries)

        hard = MAX_CHUNK_TOKENS - header_budget
        parts: list[list[int]] = []
        current: list[int] = []
        current_tokens = 0
        for ln in live:
            t = self._line_tokens[ln - 1]
            grown = current_tokens + t
            # Past the soft budget: cut at the next allowed boundary; never pass the hard limit.
            if current and grown > budget and (ln in allowed or grown > hard):
                parts.append(current)
                current, current_tokens = [], 0
            current.append(ln)
            current_tokens += t
        if current:
            parts.append(current)

        first: int | None = None
        n = len(parts)
        for i, part in enumerate(parts, start=1):
            body = strip_code_bloat("\n".join(self.lines[ln - 1] for ln in part).strip("\n"))
            head = f"{self.comment} {self.basename}::{qualified} (part {i}/{n})"
            content = f"{head}\n{body}" if i == 1 else f"{head}\n{signature}\n{body}"
            idx = self.add(chunk_type, f"{name} (part {i}/{n})", qualified,
                           part[0], part[-1], content, parent if first is None else first)
            if first is None:
                first = idx
        return first


class _CodeChunker:
    def __init__(self, builder: _Builder, spec: LangSpec, source: bytes):
        self.b = builder
        self.spec = spec
        self.source = source

    # --- naming ---------------------------------------------------------

    def _text(self, node: Any) -> str:
        return self.source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    def _name(self, node: Any) -> str:
        name = definition_name(node, self.source, self.b.language)
        return name if name is not None else f"L{node.start_point[0] + 1}"

    def _arrow_name(self, node: Any) -> str | None:
        """Name for ``const f = () => ...`` declarations, else None."""
        for decl in node.named_children:
            if decl.type != "variable_declarator":
                continue
            value = decl.child_by_field_name("value")
            if value is not None and value.type in ("arrow_function", "function_expression", "function"):
                name = decl.child_by_field_name("name")
                return self._text(name) if name is not None else None
        return None

    # --- classification ---------------------------------------------------

    def _classify(self, node: Any) -> tuple[str, Any, str | None]:
        """Return (kind, inner_node, name) with kind in func|container|leaf|loose|comment."""
        inner = node
        if node.type in self.spec.wrappers:
            found = node.child_by_field_name(self.spec.wrappers[node.type])
            if found is None and node.named_children:
                found = node.named_children[-1]
            if found is not None:
                inner = found
        if inner.type in self.spec.funcs:
            return "func", inner, self._name(inner)
        if inner.type in self.spec.containers:
            return "container", inner, self._name(inner)
        if inner.type in self.spec.leaves:
            return "leaf", inner, self._name(inner)
        if inner.type in self.spec.var_decls:
            arrow = self._arrow_name(inner)
            if arrow:
                return "func", inner, arrow
        if node.type in _COMMENT_NODES:
            return "comment", node, None
        return "loose", node, None

    # --- walking ------------------------------------------------------------

    def chunk_module(self, root: Any) -> None:
        total = len(self.b.lines)
        loose: list[tuple[int, int, list[int]]] = []  # contiguous runs: (start, end, boundaries)
        items: list[tuple[str, Any, Any, str, int]] = []  # kind, outer, inner, name, start

        pending_comment_start: int | None = None
        prev_loose = False
        for node in root.named_children:
            start = node.start_point[0] + 1
            end = _end_line(node)
            kind, inner, name = self._classify(node)
            if kind == "comment":
                if pending_comment_start is None:
                    pending_comment_start = start
                _extend_run(loose, start, end, prev_loose)
                prev_loose = True
                continue
            if kind == "loose":
                pending_comment_start = None
                _extend_run(loose, start, end, prev_loose)
                prev_loose = True
                continue
            prev_loose = False
            if pending_comment_start is not None:
                # leading comments belong to the following definition
                _trim_run(loose, pending_comment_start)
                start = pending_comment_start
                pending_comment_start = None
            items.append((kind, node, inner, name or "?", start))

        module_idx: int | None = None
        first_item_line = items[0][4] if items else total + 1
        for run_start, run_end, bounds in loose:
            is_prefix = run_start < first_item_line and module_idx is None
            label = "imports/globals" if is_prefix else f"globals L{run_start}-{run_end}"
            idx = self.b.add_split("module", label, label, run_start, run_end, bounds, None)
            if is_prefix:
                module_idx = idx

        for kind, outer, inner, name, start in items:
            self._emit(kind, outer, inner, name, start, module_idx)

    def _emit(self, kind: str, outer: Any, inner: Any, name: str, start: int,
              parent: int | None) -> None:
        end = _end_line(outer)
        # The call graph names a caller scope with the same function (ai_db.parser.naming).
        qualified = qualified_name(inner, self.source, self.b.language, own=name)
        if kind == "func":
            self.b.add_split("code", f"def {qualified}", qualified, start, end,
                             _child_starts(inner), parent)
            return
        if kind == "leaf":
            self.b.add_split("code", f"type {qualified}", qualified, start, end,
                             _child_starts(inner), parent)
            return

        # container
        body = inner.child_by_field_name("body")
        if body is None:
            for child in inner.named_children:
                if child.type in ("declaration_list", "class_body", "field_declaration_list",
                                  "interface_body", "enum_body", "block", "classBody"):
                    body = child
        members: list[tuple[str, Any, Any, str, int]] = []
        if body is not None:
            pending: int | None = None
            for child in _members(body):
                c_kind, c_inner, c_name = self._classify(child)
                c_start = child.start_point[0] + 1
                if c_kind == "comment":
                    pending = c_start if pending is None else pending
                    continue
                if c_kind in ("func", "container", "leaf"):
                    members.append((c_kind, child, c_inner, c_name or "?", pending or c_start))
                pending = None

        label = f"class {qualified}"
        if not members:
            self.b.add_split("code", label, qualified, start, end, _child_starts(body or inner), parent)
            return
        skip = [(m_start, _end_line(m_outer)) for _, m_outer, _, _, m_start in members]
        header_idx = self.b.add_split("class_header", label, qualified, start, end,
                                      _child_starts(body or inner), parent, skip=skip)
        for m_kind, m_outer, m_inner, m_name, m_start in members:
            self._emit(m_kind, m_outer, m_inner, m_name, m_start, header_idx)


# Comment node types across grammars (Rust and Java split line and block comments).
_COMMENT_NODES = frozenset({"comment", "line_comment", "block_comment"})


def _members(body: Any) -> list[Any]:
    """A container body's member nodes (a Java enum's methods sit one level down, in
    ``enum_body_declarations``)."""
    out: list[Any] = []
    for child in body.named_children:
        out.extend(child.named_children if child.type == "enum_body_declarations" else [child])
    return out


def _end_line(node: Any) -> int:
    row, col = node.end_point
    # a node ending at column 0 ends on the previous line
    return int(row if col == 0 and row > node.start_point[0] else row + 1)


def _child_starts(node: Any) -> list[int]:
    starts: list[int] = []
    stack = [node]
    while stack:
        n = stack.pop()
        for c in n.named_children:
            starts.append(c.start_point[0] + 1)
            if c.type in ("block", "statement_block", "compound_statement", "class_body",
                          "declaration_list", "field_declaration_list", "classBody"):
                stack.append(c)
    return sorted(set(starts))


def _extend_run(runs: list[tuple[int, int, list[int]]], start: int, end: int,
                continues: bool) -> None:
    """Add a loose node; it joins the last run when no definition came in between."""
    if runs and continues:
        s, _, b = runs[-1]
        runs[-1] = (s, max(end, runs[-1][1]), [*b, start])
    else:
        runs.append((start, end, [start]))


def _trim_run(runs: list[tuple[int, int, list[int]]], cut: int) -> None:
    """Remove lines >= cut from the last loose run (they became a leading comment)."""
    if not runs:
        return
    s, e, b = runs[-1]
    if s >= cut:
        runs.pop()
    elif e >= cut:
        runs[-1] = (s, cut - 1, [x for x in b if x < cut])


def _chunk_code(filepath: str, content: str, lines: list[str], lang: str) -> list[dict[str, Any]]:
    spec = LANG_SPECS[lang]
    source = content.encode("utf-8")
    tree = _parser(lang).parse(source)
    builder = _Builder(filepath, lines, lang, spec.comment)
    _CodeChunker(builder, spec, source).chunk_module(tree.root_node)
    return builder.chunks


def _chunk_markdown(filepath: str, content: str, lines: list[str]) -> list[dict[str, Any]]:
    source = content.encode("utf-8")
    tree = _parser("markdown").parse(source)
    builder = _Builder(filepath, lines, "markdown", "<!--")

    def heading_text(section: Any) -> str:
        for child in section.named_children:
            if child.type in ("atx_heading", "setext_heading"):
                raw = source[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
                first = raw.strip().splitlines()[0] if raw.strip() else ""
                return first.lstrip("#").strip() or "Untitled"
        return "Overview"

    def walk(section: Any, path: list[str]) -> None:
        title = heading_text(section)
        subsections = [c for c in section.named_children if c.type == "section"]
        start = section.start_point[0] + 1
        end = _end_line(section)
        skip = [(c.start_point[0] + 1, _end_line(c)) for c in subsections]
        qualified = " > ".join([*path, title])
        builder.add_split("md", title, qualified, start, end,
                          [c.start_point[0] + 1 for c in section.named_children], None, skip=skip)
        for sub in subsections:
            walk(sub, [*path, title])

    root = tree.root_node
    preamble = [c for c in root.named_children if c.type != "section"]
    if preamble:
        start = preamble[0].start_point[0] + 1
        end = _end_line(preamble[-1])
        builder.add_split("md", "Overview", "Overview", start, end,
                          [c.start_point[0] + 1 for c in preamble], None)
    for section in (c for c in root.named_children if c.type == "section"):
        walk(section, [])
    return builder.chunks


def _chunk_text(filepath: str, lines: list[str]) -> list[dict[str, Any]]:
    builder = _Builder(filepath, lines, "text", "#")
    total = len(lines)
    step = TEXT_WINDOW_LINES - TEXT_WINDOW_OVERLAP
    start = 1
    while start <= total:
        end = min(start + TEXT_WINDOW_LINES - 1, total)
        label = f"L{start}-{end}"
        builder.add("txt", label, label, start, end, builder.text(start, end), None)
        if end == total:
            break
        start += step
    return builder.chunks


def chunk_file(filepath: str, content: str,
               languages: Collection[str] = SUPPORTED_LANGUAGES) -> list[dict[str, Any]]:
    """Split one file into retrieval chunks (see module docstring). Files whose language
    is not in ``languages`` are chunked as plain text."""
    lines = content.splitlines()
    if not lines:
        return []
    filename = os.path.basename(filepath)

    if filename in META_FILES:
        text_block = strip_code_bloat(content)[:META_MAX_CHARS]
        if not text_block.strip():
            return []
        return [{
            "chunk_type": "lib_meta",
            "name": f"pkg:{filename}",
            "qualified_name": f"pkg:{filename}",
            "language": "meta",
            "start_line": 1,
            "end_line": len(lines),
            "content": text_block,
            "token_count": count_tokens(text_block),
            "content_hash": _hash(text_block),
            "parent_index": None,
        }]

    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".md":
        return _chunk_markdown(filepath, content, lines)
    lang = enabled_language_for(filepath, languages)
    if lang is not None:
        return _chunk_code(filepath, content, lines, lang)
    return _chunk_text(filepath, lines)
