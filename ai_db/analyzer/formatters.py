from typing import Any


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
        return "()"



format_as_stub = Formatters.format_as_stub
format_as_sexp = Formatters.format_as_sexp
