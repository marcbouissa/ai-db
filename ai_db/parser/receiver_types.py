"""
ai_db.parser.receiver_types
Best-effort static resolution of a Python call receiver to a class name, so that
``self.execute()`` inside ``class A`` is recorded as ``A.execute`` and distinguishable
from ``cursor.execute()``.  Returns None whenever the type cannot be established
cheaply (the ref is then stored with ``callee_resolved = NULL``).
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tree_sitter import Node

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _text(node: Node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _class_like(name: str) -> bool:
    return bool(name) and name[0].isupper()


def _enclosing(node: Node, type_name: str) -> Node | None:
    parent = node.parent
    while parent is not None:
        if parent.type == type_name:
            return parent
        parent = parent.parent
    return None


def _type_name(annotation: str) -> str | None:
    """Reduce an annotation (``Foo``, ``pkg.Foo``, ``Foo | None``, ``Optional[Foo]``,
    ``"Foo"``) to a bare class name; containers and unions of several classes -> None."""
    ann = annotation.strip().strip("'\"")
    m = re.fullmatch(r"(?:typing\.)?Optional\[(.+)\]", ann)
    if m:
        ann = m.group(1).strip().strip("'\"")
    parts = [p.strip() for p in ann.split("|") if p.strip() not in ("None", "")]
    if len(parts) != 1:
        return None
    last = parts[0].strip("'\"").rsplit(".", 1)[-1]
    if not _IDENT.match(last) or not _class_like(last):
        return None
    return last


def _ctor_class(value: Node, source: bytes) -> str | None:
    """``Foo(...)`` / ``pkg.Foo(...)`` -> ``Foo`` when the callee looks like a class."""
    if value.type != "call":
        return None
    fn = value.child_by_field_name("function")
    if fn is None:
        return None
    name = _text(fn, source).rsplit(".", 1)[-1]
    return name if _IDENT.match(name) and _class_like(name) else None


def _assigned_type(stmt: Node, source: bytes) -> tuple[str, str] | None:
    """(target_text, class_name) for ``x = Foo()`` / ``x: Foo = ...`` statements."""
    left = stmt.child_by_field_name("left")
    if left is None:
        return None
    ann = stmt.child_by_field_name("type")
    if ann is not None:
        tn = _type_name(_text(ann, source))
        return (_text(left, source), tn) if tn else None
    right = stmt.child_by_field_name("right")
    if right is None:
        return None
    cls = _ctor_class(right, source)
    return (_text(left, source), cls) if cls else None


def _walk_assignments(root: Node, source: bytes, skip_nested_defs: bool) -> dict[str, str]:
    found: dict[str, str] = {}
    stack = [root]
    while stack:
        n = stack.pop()
        for child in n.children:
            if skip_nested_defs and child.type in ("function_definition", "class_definition"):
                continue
            if child.type == "assignment":
                got = _assigned_type(child, source)
                if got:
                    found.setdefault(got[0], got[1])
            stack.append(child)
    return found


def _param_types(func: Node, source: bytes) -> dict[str, str]:
    types: dict[str, str] = {}
    params = func.child_by_field_name("parameters")
    if params is None:
        return types
    for p in params.children:
        if p.type in ("typed_parameter", "typed_default_parameter"):
            ann = p.child_by_field_name("type")
            name_node = p.child_by_field_name("name") or next(
                (c for c in p.children if c.type == "identifier"), None)
            if ann is not None and name_node is not None:
                tn = _type_name(_text(ann, source))
                if tn:
                    types[_text(name_node, source)] = tn
    return types


def _self_attrs_from_params(cls: Node, source: bytes) -> dict[str, str]:
    """``self.db = db`` where ``db: Database`` is a parameter of that method -> ``{"self.db": "Database"}``."""
    found: dict[str, str] = {}
    stack = [cls]
    while stack:
        n = stack.pop()
        for child in n.children:
            if child.type == "function_definition":
                params = _param_types(child, source)
                if params:
                    for target, value in _plain_assignments(child, source):
                        if target.startswith("self.") and value in params:
                            found.setdefault(target, params[value])
                continue  # nested defs are handled when reached, not via stack
            stack.append(child)
    return found


def _plain_assignments(func: Node, source: bytes) -> list[tuple[str, str]]:
    """(target_text, value_text) for ``a = b`` statements in ``func`` (not in nested defs)."""
    out: list[tuple[str, str]] = []
    stack = [func]
    while stack:
        n = stack.pop()
        for child in n.children:
            if child.type in ("function_definition", "class_definition") and child is not func:
                continue
            if child.type == "assignment":
                left, right = child.child_by_field_name("left"), child.child_by_field_name("right")
                if left is not None and right is not None and right.type == "identifier":
                    out.append((_text(left, source), _text(right, source)))
            stack.append(child)
    return out


class ReceiverResolver:
    """``resolve_receiver_class`` for every call of one parsed file.

    The per-scope tables (a class's ``self.x`` assignments, a function's parameters and
    locals, the module's assignments) are built once per scope node and reused for every
    call in it. Walking them again for each call was 94 % of ``extract_graph`` on a large
    module (cli.py: 4.1 s of 14.8 s for the whole repository).
    """

    def __init__(self, source: bytes):
        self.source = source
        self._assignments: dict[tuple[int, bool], dict[str, str]] = {}
        self._self_attrs: dict[int, dict[str, str]] = {}
        self._params: dict[int, dict[str, str]] = {}

    def _assigned(self, root: Node, skip_nested_defs: bool) -> dict[str, str]:
        key = (root.id, skip_nested_defs)
        if key not in self._assignments:
            self._assignments[key] = _walk_assignments(root, self.source, skip_nested_defs)
        return self._assignments[key]

    def _attrs(self, cls: Node) -> dict[str, str]:
        if cls.id not in self._self_attrs:
            self._self_attrs[cls.id] = _self_attrs_from_params(cls, self.source)
        return self._self_attrs[cls.id]

    def _param_map(self, func: Node) -> dict[str, str]:
        if func.id not in self._params:
            self._params[func.id] = _param_types(func, self.source)
        return self._params[func.id]

    def resolve(self, call_node: Node, receiver: str | None) -> str | None:
        """Class name the receiver of ``call_node`` denotes, or None when unknown."""
        if not receiver:
            return None
        source = self.source
        func = _enclosing(call_node, "function_definition")
        cls = _enclosing(call_node, "class_definition")
        if receiver in ("self", "cls"):
            if cls is None:
                return None
            name = cls.child_by_field_name("name")
            return _text(name, source) if name is not None else None
        if receiver.startswith("self."):
            if cls is None:
                return None
            return (self._assigned(cls, skip_nested_defs=False).get(receiver)
                    or self._attrs(cls).get(receiver))
        if "." in receiver or not _IDENT.match(receiver):
            return None
        if func is not None:
            param = self._param_map(func).get(receiver)
            if param:
                return param
            local = self._assigned(func, skip_nested_defs=True).get(receiver)
            if local:
                return local
        root = call_node
        while root.parent is not None:
            root = root.parent
        module_level = self._assigned(root, skip_nested_defs=True).get(receiver)
        if module_level:
            return module_level
        return receiver if _class_like(receiver) else None  # ``ClassName.method(...)``


def resolve_receiver_class(call_node: Node, receiver: str | None, source: bytes) -> str | None:
    """One-off resolution; use ``ReceiverResolver`` for the calls of a whole file."""
    return ReceiverResolver(source).resolve(call_node, receiver)
