"""Names of definitions, shared by the chunker and the call graph.

A chunk's ``qualified_name`` and a call ref's caller scope (``module.<qualified>``) must
agree, or ``investigate``/``trace`` find no callees for that chunk. Both are computed
here, from the same per-language container set, so they agree by construction:

- ``definition_name``: a definition's own name (Go methods are ``Recv.m``, Rust impls
  are named by their type, C/C++ functions by their declarator chain, ``Foo::bar`` ->
  ``Foo.bar``, JS/TS functions bound to a variable by the variable).
- ``qualified_name``: the names of the enclosing containers plus the own name.
"""

from __future__ import annotations

from typing import Any

from ai_db.parser.languages import declared_name

# Per language: the node types that qualify the names of the definitions inside them
# (classes, impls, traits, modules, namespaces). The chunker emits one header chunk per
# container and recurses into its body.
SCOPE_CONTAINERS: dict[str, frozenset[str]] = {
    "python": frozenset({"class_definition"}),
    "javascript": frozenset({"class_declaration"}),
    "typescript": frozenset({"class_declaration", "abstract_class_declaration",
                             "interface_declaration"}),
    "tsx": frozenset({"class_declaration", "abstract_class_declaration",
                      "interface_declaration"}),
    "go": frozenset(),
    "rust": frozenset({"impl_item", "trait_item", "mod_item"}),
    "c": frozenset(),
    "cpp": frozenset({"class_specifier", "struct_specifier", "namespace_definition"}),
    "java": frozenset({"class_declaration", "interface_declaration", "enum_declaration",
                       "record_declaration"}),
    "csharp": frozenset({"namespace_declaration", "class_declaration", "struct_declaration",
                         "interface_declaration", "record_declaration"}),
    "pkl": frozenset({"clazz"}),
}

# Function values that take the name of the variable they are bound to.
_BOUND_FUNCTIONS = frozenset({"arrow_function", "function_expression", "function"})


def _text(source: bytes, node: Any) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _rust_type_name(source: bytes, typ: Any) -> str:
    """``Wrap<T>`` -> ``Wrap``, ``fmt::Foo`` -> ``Foo``, ``&Foo`` -> ``Foo``."""
    while typ.type in ("generic_type", "reference_type", "scoped_type_identifier"):
        inner = typ.child_by_field_name("name" if typ.type == "scoped_type_identifier" else "type")
        if inner is None:
            break
        typ = inner
    return _text(source, typ)


def bound_variable(node: Any) -> Any | None:
    """The ``variable_declarator`` a JS/TS function value is assigned to, else None."""
    parent = node.parent
    if node.type in _BOUND_FUNCTIONS and parent is not None and parent.type == "variable_declarator":
        return parent
    return None


def definition_name(node: Any, source: bytes, lang: str) -> str | None:
    """A definition's own name (unqualified), or None when it has none."""
    t = node.type
    if t == "method_declaration" and lang == "go":
        recv = node.child_by_field_name("receiver")
        name = node.child_by_field_name("name")
        recv_type = ""
        if recv is not None:
            stack = [recv]
            while stack:
                n = stack.pop()
                if n.type == "type_identifier":
                    recv_type = _text(source, n)
                    break
                stack.extend(reversed(n.named_children))
        base = _text(source, name) if name is not None else "?"
        return f"{recv_type}.{base}" if recv_type else base
    if t == "type_declaration":
        for spec in node.named_children:
            spec_name = spec.child_by_field_name("name")
            if spec_name is not None:
                return _text(source, spec_name)
        return None
    if t == "impl_item":
        # `impl Trait for Type` methods belong to Type (the trait is an interface).
        typ = node.child_by_field_name("type")
        return _rust_type_name(source, typ) if typ is not None else None
    if t == "function_definition" and lang in ("c", "cpp"):
        decl = node.child_by_field_name("declarator")
        while decl is not None and decl.type != "function_declarator":
            decl = decl.child_by_field_name("declarator")
        if decl is not None:
            inner = decl.child_by_field_name("declarator")
            if inner is not None:
                return _text(source, inner).replace("::", ".")
        return None
    if t == "type_definition" and lang in ("c", "cpp"):
        # `typedef int t;`, `typedef struct s *sp;`, `typedef int (*cb)(int);`
        decl = node.child_by_field_name("declarator")
        while decl is not None and decl.type != "type_identifier":
            nxt = decl.child_by_field_name("declarator")
            decl = nxt if nxt is not None else (decl.named_children[0] if decl.named_children else None)
        return _text(source, decl) if decl is not None else None
    var = bound_variable(node)
    if var is not None:
        name = var.child_by_field_name("name")
        return _text(source, name) if name is not None else None
    name = declared_name(node)
    return _text(source, name) if name is not None else None


def qualified_name(node: Any, source: bytes, lang: str, own: str | None = None) -> str:
    """``Container.Inner.name`` for a definition node (``own`` overrides its own name)."""
    containers = SCOPE_CONTAINERS.get(lang, frozenset())
    parts: list[str] = []
    parent = node.parent
    while parent is not None:
        if parent.type in containers:
            parts.append(definition_name(parent, source, lang) or "?")
        parent = parent.parent
    parts.reverse()
    parts.append(own if own is not None else (definition_name(node, source, lang) or "?"))
    return ".".join(parts)
