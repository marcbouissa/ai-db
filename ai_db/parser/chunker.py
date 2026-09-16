import os
import re
import ast
from typing import List, Dict, Any
from ai_db.utils import strip_code_bloat

def chunk_file(filepath: str, content: str) -> List[Dict[str, Any]]:
    """Chunks files into logical sections: classes, functions, or markdown sections."""
    ext = os.path.splitext(filepath)[1].lower()
    filename = os.path.basename(filepath)
    lines = content.splitlines()
    total_lines = len(lines)
    chunks = []

    if total_lines == 0:
        return []

    # Package metadata files (package.json, METADATA)
    if filename in ("package.json", "METADATA", "pyproject.toml"):
        text_block = strip_code_bloat(content)
        if text_block:
            chunks.append({
                "chunk_type": "lib_meta",
                "name": f"pkg:{filename}",
                "start_line": 1,
                "end_line": total_lines,
                "content": text_block[:1500]  # Cap metadata to avoid excessive JSON junk
            })
        return chunks

    # Markdown chunking by headers
    if ext in (".md", ".rst"):
        current_header = "Overview"
        current_lines = []
        start_line = 1

        for i, line in enumerate(lines, 1):
            if line.startswith("#"):
                if current_lines:
                    text_block = strip_code_bloat("\n".join(current_lines).strip())
                    if text_block:
                        chunks.append({
                            "chunk_type": "md",
                            "name": current_header,
                            "start_line": start_line,
                            "end_line": i - 1,
                            "content": text_block
                        })
                current_header = line.strip("# ").strip()
                current_lines = [line]
                start_line = i
            else:
                current_lines.append(line)

        if current_lines:
            text_block = strip_code_bloat("\n".join(current_lines).strip())
            if text_block:
                chunks.append({
                    "chunk_type": "md",
                    "name": current_header,
                    "start_line": start_line,
                    "end_line": total_lines,
                    "content": text_block
                })
        return chunks

    # Python AST-guided chunking when available
    if ext in (".py", ".pyi"):
        try:
            tree = ast.parse(content, filename=filepath)
            body_nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
            if body_nodes:
                first_lineno = body_nodes[0].lineno
                if first_lineno > 1:
                    pre_content = strip_code_bloat("\n".join(lines[:first_lineno - 1]).strip())
                    if pre_content:
                        chunks.append({
                            "chunk_type": "code",
                            "name": "imports/globals",
                            "start_line": 1,
                            "end_line": first_lineno - 1,
                            "content": pre_content
                        })

                for node in body_nodes:
                    start_l = node.lineno
                    end_l = getattr(node, "end_lineno", None)
                    if end_l is None:
                        end_l = total_lines
                    block_text = strip_code_bloat("\n".join(lines[start_l - 1 : end_l]).strip())
                    if block_text:
                        sig_prefix = "class" if isinstance(node, ast.ClassDef) else "def"
                        chunks.append({
                            "chunk_type": "code",
                            "name": f"{sig_prefix} {node.name}",
                            "start_line": start_l,
                            "end_line": end_l,
                            "content": block_text
                        })
                return chunks
        except Exception:
            pass

    # Code and Typings chunking (Python fallback / JS / TS / C++ / D.TS / PYI)
    if ext in (".py", ".pyi", ".js", ".ts", ".jsx", ".tsx", ".c", ".cpp", ".rs", ".go") or filepath.endswith(".d.ts"):
        func_regex = re.compile(
            r"^(?:async\s+)?(?:def\s+|class\s+|function\s+|interface\s+|type\s+|declare\s+|export\s+|const\s+\w+\s*=\s*(?:async\s*)?\([^)]*\)\s*=>|public\s+|fn\s+)(\w+)"
        )
        current_symbol = "hdr"
        current_lines = []
        start_line = 1

        for i, line in enumerate(lines, 1):
            m = func_regex.match(line.strip())
            if m and len(current_lines) > 20:
                text_block = strip_code_bloat("\n".join(current_lines).strip())
                if text_block:
                    chunks.append({
                        "chunk_type": "code",
                        "name": current_symbol,
                        "start_line": start_line,
                        "end_line": i - 1,
                        "content": text_block
                    })
                current_symbol = m.group(0).strip()
                current_lines = [line]
                start_line = i
            else:
                current_lines.append(line)
                if len(current_lines) >= 70:
                    text_block = strip_code_bloat("\n".join(current_lines).strip())
                    chunks.append({
                        "chunk_type": "code",
                        "name": f"{current_symbol} L{start_line}-{i}",
                        "start_line": start_line,
                        "end_line": i,
                        "content": text_block
                    })
                    current_lines = []
                    start_line = i + 1

        if current_lines:
            text_block = strip_code_bloat("\n".join(current_lines).strip())
            if text_block:
                chunks.append({
                    "chunk_type": "code",
                    "name": current_symbol,
                    "start_line": start_line,
                    "end_line": total_lines,
                    "content": text_block
                })
        return chunks

    # Default fallback: windowed line chunks
    window_size = 50
    for i in range(0, total_lines, window_size):
        sub_lines = lines[i : i + window_size]
        text_block = strip_code_bloat("\n".join(sub_lines).strip())
        if text_block:
            chunks.append({
                "chunk_type": "txt",
                "name": f"L{i + 1}-{min(i + window_size, total_lines)}",
                "start_line": i + 1,
                "end_line": min(i + window_size, total_lines),
                "content": text_block
            })

    return chunks

