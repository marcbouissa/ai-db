import json
import os
from typing import Any

from ai_db.errors import AiDbConfigError

#: Output styles for an ``investigate`` pack. ``json`` stays the default because
#: every existing caller parses it; the rest are opt-in.
PACK_FORMATS = frozenset({"json", "compact", "stub", "sexp"})


class Formatters:
    @staticmethod
    def format_as_stub(data: dict[str, Any]) -> str:
        """Formats analysis output into native code skeletons (.pyi/stub style).
        Best for LLM code analysis, reasoning, and type inspection.
        """
        lines = []

        def render_file_stub(fpath: str, fres: dict[str, Any]):
            meta = fres.get("meta", {})
            conf_str = f" conf:{meta.get('conf', 1.0)}" if "conf" in meta else ""
            cached_str = " (cached)" if meta.get("cached") else ""
            lines.append(f"# file: {fpath}{cached_str}{conf_str}")

            diff = fres.get("diff")
            if diff:
                added = diff.get("added", [])
                changed = diff.get("changed", [])
                removed = diff.get("removed", [])
                if added or changed or removed:
                    lines.append(f"# diff: +{len(added)} ~{len(changed)} -{len(removed)}")

            for s in fres.get("symbols", []):
                name = s.get("name", "")
                kind = s.get("kind", "def")
                sig = s.get("sig", "")
                span = s.get("span", [0, 0])
                ref = s.get("ref", "")
                ref_tag = f" @{ref}" if ref else ""
                span_tag = f"L{span[0]}-{span[1]}" if span and len(span) == 2 else ""

                if kind == "range":
                    lines.append(f"# --- range {span_tag}{ref_tag} ---")
                    if s.get("body"):
                        lines.append(s["body"].rstrip())
                elif kind == "class":
                    clean_sig = sig.strip() if sig else f"class {name}"
                    if not clean_sig.endswith(":"):
                        clean_sig += ":"
                    lines.append(f"{clean_sig} ...  # {span_tag}{ref_tag}")
                    for m in s.get("methods", []):
                        m_sig = m.get("sig", f"def {m.get('name')}(...)").strip()
                        m_span = m.get("span", [0, 0])
                        m_ref = m.get("ref", "")
                        m_ref_tag = f" @{m_ref}" if m_ref else ""
                        lines.append(f"    {m_sig}: ...  # L{m_span[0]}-{m_span[1]}{m_ref_tag}")
                else:
                    clean_sig = sig.strip() if sig else f"def {name}(...)"
                    if clean_sig.startswith("class "):
                        if not clean_sig.endswith(":"):
                            clean_sig += ":"
                        lines.append(f"{clean_sig} ...  # {span_tag}{ref_tag}")
                    else:
                        lines.append(f"{clean_sig}: ...  # {span_tag}{ref_tag}")

                if s.get("body") and kind != "range":
                    # Indent body block
                    body_lines = s["body"].strip().splitlines()
                    if len(body_lines) > 25:
                        for b_line in body_lines[:10]:
                            lines.append(f"    {b_line}")
                        lines.append(f"    # ... [{len(body_lines)-20} lines elided] ...")
                        for b_line in body_lines[-10:]:
                            lines.append(f"    {b_line}")
                    else:
                        for b_line in body_lines:
                            lines.append(f"    {b_line}")

            lines.append("")

        if "results" in data:
            for fpath, fres in data["results"].items():
                render_file_stub(fpath, fres)
        elif "symbols" in data:
            render_file_stub(data.get("file", "unknown"), data)

        return "\n".join(lines).strip()

    @staticmethod
    def format_as_sexp(data: dict[str, Any]) -> str:
        """Formats analysis output into compact S-Expressions (Lisp/EDN).
        Best for absolute minimum token consumption in tree navigation.
        """
        def sexp_escape(val: Any) -> str:
            if val is None:
                return "nil"
            s = str(val).replace('"', '\\"').replace("\n", " ")
            return f'"{s}"'

        def render_file_sexp(fpath: str, fres: dict[str, Any]) -> str:
            sym_parts = []
            for s in fres.get("symbols", []):
                name = s.get("name", "")
                kind = s.get("kind", "def")
                span = s.get("span", [0, 0])
                ref = s.get("ref", "")
                sig = s.get("sig", "")
                span_str = f"({span[0]} {span[1]})" if span and len(span) == 2 else "nil"

                sub_methods = []
                for m in s.get("methods", []):
                    m_span = m.get("span", [0, 0])
                    m_span_str = f"({m_span[0]} {m_span[1]})" if m_span and len(m_span) == 2 else "nil"
                    sub_methods.append(f"(:method {m.get('name')} {m_span_str} {m.get('ref', '')} {sexp_escape(m.get('sig', ''))})")

                methods_block = f" :methods ({' '.join(sub_methods)})" if sub_methods else ""
                body_block = f" :body {sexp_escape(s.get('body'))}" if s.get("body") else ""
                sym_parts.append(f"(:{kind} {name} {span_str} {ref} {sexp_escape(sig)}{methods_block}{body_block})")

            meta = fres.get("meta", {})
            meta_str = f"(:meta :in {meta.get('tokens_in', 0)} :out {meta.get('tokens_out', 0)} :conf {meta.get('conf', 1.0)})"
            return f"(:file {sexp_escape(fpath)} {meta_str} :symbols ({' '.join(sym_parts)}))"

        if "results" in data:
            file_sexps = [render_file_sexp(fp, fr) for fp, fr in data["results"].items()]
            return f"(:batch :targets-count {len(file_sexps)} :files ({' '.join(file_sexps)}))"
        elif "symbols" in data:
            return render_file_sexp(data.get("file", "unknown"), data)
        elif "entry_points" in data:
            # An investigation pack is a different schema from analyze output.
            return format_pack_as_sexp(data)
        return "()"

    @staticmethod
    def format_pack(pack: dict[str, Any], style: str) -> str:
        """Render an ``investigate`` pack in ``style``.

        The pack is not AST-shaped like analyze output, so it needs its own
        renderers rather than a reshape into symbols. It carries an
        entry-point list, evidence with optional bodies or stubs, a call graph,
        tests, and diff spans.

        ``stub`` is the point of this: a pack already decides per evidence item
        whether it can afford a full body, and then ships that body inside JSON
        alongside every span, ref and ``why`` string. Rendering as a stub keeps
        the decisions and drops the scaffolding. Bodies stay one ``expand`` away
        via the ``ref:`` handle, which is the whole point of having handles.
        """
        if style == "compact":
            return json.dumps(pack, separators=(",", ":"), ensure_ascii=False)
        if style == "stub":
            return format_pack_as_stub(pack)
        if style == "sexp":
            return format_pack_as_sexp(pack)
        if style == "json":
            return json.dumps(pack, indent=2, ensure_ascii=False)
        raise AiDbConfigError(
            f"unknown investigate format {style!r}; "
            f"expected one of {sorted(PACK_FORMATS)}"
        )


def _pack_rel(filepath: str) -> str:
    """Shorten an absolute path to something readable.

    Packs routinely span absolute paths that all share one long prefix, and
    repeating it costs more than the answer. Relative to the working directory
    when possible; otherwise the basename, which is still more useful than a
    200-character prefix repeated once per item. Deliberately does not try to
    guess a repo root by name: a checkout directory can be called anything, and
    matching on a name produced ``ai-db/ai_db/...`` on this repo.
    """
    if not filepath:
        return ""
    try:
        cwd = os.getcwd()
    except OSError:
        return os.path.basename(filepath)
    if filepath.startswith(cwd + os.sep):
        return filepath[len(cwd) + 1:]
    if os.sep in filepath:
        return os.path.basename(filepath)
    return filepath


def format_pack_as_stub(pack: dict[str, Any]) -> str:
    """Compact, agent-readable rendering of an investigation pack.

    No bodies: they are the bulk of the payload and are available on demand via
    ``ai-db expand <ref>``. What stays is the ranked answer, why each item was
    chosen, and the graph relationships -- the parts a reader acts on.
    """
    from ai_db.constants import PACK_STUB_MAX_ITEMS, PACK_STUB_WHY_CHARS

    out: list[str] = []
    mode = pack.get("mode", "?")
    tokens = pack.get("token_count", 0)
    budget = pack.get("budget_tokens", 0)
    query = pack.get("query", "")

    entry_points = pack.get("entry_points", [])
    out.append(f"# investigate {mode}: {query!r}  [{tokens}/{budget} tokens]")
    out.append(f"# {len(entry_points)} entry points")

    by_name = {}
    for ep in entry_points:
        by_name[ep.get("qualified_name")] = ep
        why = (ep.get("why") or "")[:PACK_STUB_WHY_CHARS]
        out.append(f"  {ep.get('qualified_name')}  {_pack_rel(ep.get('filepath', ''))}"
                   f" {ep.get('lines', '')}")
        if why:
            out.append(f"      why: {why}")

    evidence = pack.get("evidence", [])
    if evidence:
        out.append(f"# evidence {len(evidence)}"
                   + (f", {pack.get('omitted_count', 0)} omitted"
                      if pack.get("omitted_count") else ""))
        seen = set()
        shown = 0
        for ev in evidence:
            name = ev.get("qualified_name", "")
            if name in seen:
                continue
            seen.add(name)
            if shown >= PACK_STUB_MAX_ITEMS:
                out.append(f"  ... {len(seen) - shown} more")
                break
            shown += 1
            ref = ev.get("ref", "")
            out.append(f"  {name}  {ev.get('lines', '')}"
                       + (f"  {ref}" if ref else "")
                       + (f"  [{ev.get('body') and 'body' or ev.get('stub') and 'stub' or 'meta'}]"
                          if True else ""))
            if ev.get("role") and ev["role"] != "seed":
                out.append(f"      role: {ev['role']}")

    graph = pack.get("call_graph") or {}
    edges = graph.get("edges") or []
    if graph.get("nodes") or edges:
        out.append(f"# call graph  {len(graph.get('nodes') or [])} nodes, {len(edges)} edges")
        for edge in edges[:PACK_STUB_MAX_ITEMS]:
            out.append(f"  {edge[0]} -> {edge[1]}")

    tests = pack.get("tests") or []
    if tests:
        out.append(f"# tests {len(tests)}")
        for t in tests[:PACK_STUB_MAX_ITEMS]:
            name = t.get("qualified_name") or t.get("name") or "?"
            out.append(f"  {name}  {_pack_rel(t.get('filepath', ''))}")

    changes = pack.get("changes") or []
    if changes:
        out.append(f"# changes {len(changes)}")
        for c in changes[:PACK_STUB_MAX_ITEMS]:
            out.append(f"  {_pack_rel(c.get('filepath', ''))} {c.get('lines', '')}")

    unresolved = pack.get("unresolved") or []
    if unresolved:
        out.append("# unresolved")
        out.append("  " + ", ".join(str(u) for u in unresolved[:PACK_STUB_MAX_ITEMS]))

    if pack.get("omitted_count"):
        out.append(f"# {pack['omitted_count']} items omitted for budget"
                   f" (ai-db expand <ref> for any of them)")
    return "\n".join(out)


def format_pack_as_sexp(pack: dict[str, Any]) -> str:
    """S-expression rendering of a pack, for callers that want the structure
    to be machine-navigable without JSON's key noise."""
    from ai_db.constants import PACK_STUB_MAX_ITEMS

    def node(s: str) -> str:
        return f'"{s}"' if (" " in str(s) or '"' in str(s)) else str(s)

    parts = [
        "(:investigate",
        node(pack.get("mode", "?")),
        node(pack.get("query", "")),
        f'(:tokens {pack.get("token_count", 0)} {pack.get("budget_tokens", 0)})',
    ]
    for ep in (pack.get("entry_points") or [])[:PACK_STUB_MAX_ITEMS]:
        parts.append(
            f"(:entry {node(ep.get('qualified_name'))} "
            f"{node(_pack_rel(ep.get('filepath', '')))} {node(ep.get('lines', ''))} "
            f"{node((ep.get('why') or '')[:120])})"
        )
    for ev in (pack.get("evidence") or [])[:PACK_STUB_MAX_ITEMS]:
        parts.append(
            f"(:evidence {node(ev.get('qualified_name'))} {node(ev.get('lines', ''))} "
            f"{node(ev.get('ref', ''))} "
            f":kind {'body' if ev.get('body') else 'stub' if ev.get('stub') else 'meta'})"
        )
    for edge in ((pack.get("call_graph") or {}).get("edges") or [])[:PACK_STUB_MAX_ITEMS]:
        parts.append(f"(:calls {node(edge[0])} {node(edge[1])})")
    for t in (pack.get("tests") or [])[:PACK_STUB_MAX_ITEMS]:
        parts.append(f"(:test {node(t.get('qualified_name') or t.get('name') or '?')})")
    unresolved = pack.get("unresolved") or []
    if unresolved:
        parts.append("(:unresolved " + " ".join(node(u) for u in unresolved) + ")")
    parts.append(f"(:omitted {pack.get('omitted_count', 0)})")
    parts.append(")")
    return " ".join(parts)



format_as_stub = Formatters.format_as_stub
format_as_sexp = Formatters.format_as_sexp
format_pack = Formatters.format_pack


#: Output styles for `analyze`. Mirrors the CLI's --format choices.
ANALYZE_FORMATS = frozenset({"json", "stub", "sexp", "outline", "prose"})


def format_analyze(data: dict[str, Any], style: str) -> str:
    """Render ``analyze`` output in one of its styles.

    Lives here rather than in the CLI so the token accounting can count the
    exact string a caller receives. While rendering was inline in the CLI, the
    formatted size could not be measured without duplicating the rendering --
    and duplicated rendering drifts, at which point the number is worse than no
    number.
    """
    if style == "json":
        return json.dumps(data, indent=2, ensure_ascii=False)
    if style == "stub":
        return Formatters.format_as_stub(data)
    if style == "sexp":
        return Formatters.format_as_sexp(data)
    if style == "outline":
        return _analyze_outline(data)
    if style == "prose":
        return _analyze_prose(data)
    raise AiDbConfigError(
        f"unknown analyze format {style!r}; expected one of {sorted(ANALYZE_FORMATS)}"
    )


def _analyze_outline(data: dict[str, Any]) -> str:
    lines: list[str] = []
    items = data.get("symbols", []) if "symbols" in data else []
    if not items and "results" in data:
        for fpath, fres in data["results"].items():
            lines.append(f"=== {fpath} ({fres['meta']['tokens_out']} tokens) ===")
            for s in fres.get("symbols", []):
                sig = s.get("sig", s["name"])
                span = f"L{s['span'][0]}-L{s['span'][1]}" if "span" in s else ""
                lines.append(f"  [{s.get('ref', '')}] {span:10} {sig}")
        return "\n".join(lines)
    for s in items:
        sig = s.get("sig", s["name"])
        span = f"L{s['span'][0]}-L{s['span'][1]}" if "span" in s else ""
        lines.append(f"[{s.get('ref', '')}] {span:10} {sig}")
    return "\n".join(lines)


def _analyze_prose(data: dict[str, Any]) -> str:
    out: list[str] = []
    meta = data.get("meta", {})
    out.append(f"### Analysis Result ({meta.get('tokens_out', 0)} tokens, "
               f"cached={meta.get('cached', False)})")
    if "symbols" in data:
        for s in data["symbols"]:
            out.append(f"- **{s['name']}** ({s['kind']}) `{s.get('ref', '')}`: {s.get('sig', '')}")
            if s.get("body"):
                out.append(f"```\n{s['body']}\n```")
    elif "results" in data:
        for fpath, fres in data["results"].items():
            out.append(f"\n#### {fpath}")
            for s in fres.get("symbols", []):
                out.append(f"- **{s['name']}** `{s.get('ref', '')}`: {s.get('sig', '')}")
                if s.get("body"):
                    out.append(f"```\n{s['body']}\n```")
    return "\n".join(out)


def annotate_formatted_tokens(data: dict[str, Any], style: str) -> str:
    """Render and record the real formatted size on ``meta``.

    ``meta.tokens_out`` is a depth-derived estimate of how much *content* was
    selected, and it is identical for every format -- which is why it could
    never support a comparison between them. ``tokens_out_formatted`` is the
    token count of the string the caller actually receives, so the two together
    show the overhead the format itself adds.

    Returns the rendered text so the caller prints exactly what was counted.

    One inherent asymmetry, stated rather than hidden: the field is set *after*
    rendering, so a ``json`` payload cannot contain its own token count. For MCP
    and HTTP, which deliver the dict, the field is in the payload and the count
    describes that payload. For the CLI printing JSON, the count describes the
    string as printed, which is one field's worth smaller than the dict.
    Re-rendering after setting the field would fix neither -- the second string
    is longer than the one counted.
    """
    rendered = format_analyze(data, style)
    meta = data.setdefault("meta", {})
    meta["format"] = style
    try:
        from ai_db.parser.chunker import count_tokens
        meta["tokens_out_formatted"] = count_tokens(rendered)
    except Exception:  # noqa: BLE001 - a tokenizer failure must not lose the output
        meta["tokens_out_formatted"] = None
    return rendered
