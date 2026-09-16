import os
import zlib
import sqlite3
from typing import List, Dict, Any, Optional
from ai_db.utils import get_allowed_projects, tokenize

class QueryEngine:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def query(self, search_text: str, top_k: int = 5, relative_to: Optional[str] = None,
              project: Optional[str] = None, allowed_projects: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        tokens = tokenize(search_text)
        if not tokens:
            return []

        # Resolve allowed project scopes
        allowed = get_allowed_projects(project or "global", allowed_projects)
        placeholders = ",".join("?" for _ in allowed)

        fts_query = " OR ".join(tokens)
        cur = self.conn.cursor()
        try:
            cur.execute(
                f"""
                SELECT fts_index.chunk_id, fts_index.filepath, fts_index.name,
                       bm25(fts_index) as bm25_rank,
                       chunks.start_line, chunks.end_line, chunks.zcontent, chunks.chunk_type, chunks.project
                FROM fts_index
                JOIN chunks ON fts_index.chunk_id = chunks.id
                WHERE fts_index MATCH ? AND chunks.project IN ({placeholders})
                ORDER BY bm25_rank
                LIMIT ?
                """,
                [fts_query] + allowed + [top_k * 2]
            )
            rows = cur.fetchall()
        except Exception:
            cur.execute(
                f"""
                SELECT id as chunk_id, filepath, name, 0.0 as bm25_rank,
                       start_line, end_line, zcontent, chunk_type, project
                FROM chunks
                WHERE project IN ({placeholders})
                LIMIT ?
                """,
                allowed + [top_k]
            )
            rows = cur.fetchall()

        results = []
        for r in rows:
            path_display = r["filepath"]
            if relative_to:
                try:
                    path_display = os.path.relpath(path_display, relative_to)
                except Exception:
                    pass

            try:
                decompressed = zlib.decompress(r["zcontent"]).decode("utf-8", errors="replace")
            except Exception:
                decompressed = ""

            results.append({
                "chunk_id": r["chunk_id"],
                "file": path_display,
                "abs_path": r["filepath"],
                "name": r["name"],
                "type": r["chunk_type"],
                "project": r["project"],
                "lines": f"L{r['start_line']}-{r['end_line']}",
                "score": round(-float(r["bm25_rank"]), 3) if r["bm25_rank"] is not None else 1.0,
                "snippet": decompressed[:280].strip() + ("..." if len(decompressed) > 280 else "")
            })

        return results[:top_k]

    def query_symbol(self, name: str, relative_to: Optional[str] = None,
                     project: Optional[str] = None, allowed_projects: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        allowed = get_allowed_projects(project or "global", allowed_projects)
        placeholders = ",".join("?" for _ in allowed)

        cur = self.conn.cursor()
        cur.execute(
            f"""
            SELECT name, symbol_type, filepath, line, signature, project
            FROM symbols
            WHERE (name = ? OR name LIKE ?) AND project IN ({placeholders})
            ORDER BY CASE WHEN name = ? THEN 0 ELSE 1 END, filepath, line
            LIMIT 50
            """,
            [name, f"%{name}%"] + allowed + [name]
        )
        rows = cur.fetchall()
        results = []
        for r in rows:
            path_display = r["filepath"]
            if relative_to:
                try:
                    path_display = os.path.relpath(path_display, relative_to)
                except Exception:
                    pass
            results.append({
                "name": r["name"],
                "symbol_type": r["symbol_type"],
                "file": path_display,
                "abs_path": r["filepath"],
                "line": r["line"],
                "signature": r["signature"],
                "project": r["project"]
            })
        return results

    def check_syntax(self, target_path: Optional[str] = None, relative_to: Optional[str] = None,
                     project: Optional[str] = None, allowed_projects: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        allowed = get_allowed_projects(project or "global", allowed_projects)
        placeholders = ",".join("?" for _ in allowed)

        cur = self.conn.cursor()
        if target_path:
            abs_target = os.path.abspath(target_path)
            if os.path.isfile(abs_target):
                cur.execute(
                    f"SELECT filepath, line, col, message, project FROM syntax_errors WHERE filepath = ? AND project IN ({placeholders})",
                    [abs_target] + allowed
                )
            else:
                cur.execute(
                    f"SELECT filepath, line, col, message, project FROM syntax_errors WHERE filepath LIKE ? AND project IN ({placeholders}) ORDER BY filepath, line",
                    [f"{abs_target}%"] + allowed
                )
        else:
            cur.execute(
                f"SELECT filepath, line, col, message, project FROM syntax_errors WHERE project IN ({placeholders}) ORDER BY filepath, line",
                allowed
            )

        rows = cur.fetchall()
        results = []
        for r in rows:
            path_display = r["filepath"]
            if relative_to:
                try:
                    path_display = os.path.relpath(path_display, relative_to)
                except Exception:
                    pass
            results.append({
                "file": path_display,
                "abs_path": r["filepath"],
                "line": r["line"],
                "col": r["col"],
                "message": r["message"],
                "project": r["project"]
            })
        return results

