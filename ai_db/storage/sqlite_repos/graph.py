"""Symbols, cross-references and symbol centrality (the 'graph' capability).

A mixin of :class:`ai_db.storage.sqlite_backend.SQLiteBackend`: every repository shares
the backend's connection and ``transaction()`` (see ``_base.SQLiteRepository``)."""

from __future__ import annotations

import math
from typing import Any

from ai_db.storage.models import (
    ChunkRecord,
    SymbolRecord,
    SymbolRefRecord,
)
from ai_db.storage.sqlite_repos._base import (
    SQLiteRepository,
    _writes,
)


class GraphRepository(SQLiteRepository):
    @_writes
    def rebuild_symbol_centrality(self, projects: list[str] | None = None) -> None:
        """Recompute normalized call/inherit in-degree per (project, symbol name).

        ``projects=None`` recomputes everything; a list recomputes only those
        projects, leaving every other project's rows untouched (TODO 13.4).
        """
        self._check_closed()
        cur = self.conn.cursor()

        scoped = projects is not None
        if scoped:
            wanted = list(dict.fromkeys(projects or []))
            if not wanted:
                return
            placeholders = ",".join("?" for _ in wanted)
            cur.execute(
                f"DELETE FROM symbol_centrality WHERE project IN ({placeholders})", wanted)
            cur.execute(f"""
                SELECT project, callee_name, COUNT(*) AS n FROM symbol_refs
                WHERE ref_type IN ('call', 'inherit')
                  AND project IN ({placeholders})
                GROUP BY project, callee_name
            """, wanted)
        else:
            cur.execute("DELETE FROM symbol_centrality")
            cur.execute("""
                SELECT project, callee_name, COUNT(*) AS n FROM symbol_refs
                WHERE ref_type IN ('call', 'inherit')
                GROUP BY project, callee_name
            """)

        rows = cur.fetchall()
        max_by_project: dict[str, int] = {}
        for r in rows:
            max_by_project[r["project"]] = max(max_by_project.get(r["project"], 0), r["n"])
        cur.executemany(
            "INSERT INTO symbol_centrality (project, name, in_degree, score) VALUES (?, ?, ?, ?)",
            [(r["project"], r["callee_name"], r["n"],
              math.log1p(r["n"]) / math.log1p(max_by_project[r["project"]])) for r in rows],
        )

    def get_symbol_centrality(self, names: list[str],
                              allowed_projects: list[str] | None = None) -> dict[str, float]:
        """Max normalized in-degree score per bare symbol name (0..1)."""
        self._check_closed()
        names = sorted(set(names))
        if not names or (allowed_projects is not None and not allowed_projects):
            return {}
        params: list[Any] = list(names)
        where = f"name IN ({','.join('?' * len(names))})"
        if allowed_projects is not None:
            where += f" AND project IN ({','.join('?' * len(allowed_projects))})"
            params.extend(allowed_projects)
        cur = self.conn.execute(
            f"SELECT name, MAX(score) AS s FROM symbol_centrality WHERE {where} GROUP BY name", params)
        return {r["name"]: float(r["s"]) for r in cur.fetchall()}

    def get_refs_from(self, filepath: str, caller_scope: str | None,
                      ref_types: tuple[str, ...] = ("call",)) -> list[SymbolRefRecord]:
        """References made inside ``caller_scope`` (e.g. ``module.Class.method``) of a file;
        ``caller_scope=None`` returns references from every scope of the file."""
        self._check_closed()
        scope_sql = "" if caller_scope is None else " AND caller_name = ?"
        scope_params = [] if caller_scope is None else [caller_scope]
        cur = self.conn.execute(
            f"""SELECT id, caller_filepath, caller_name, caller_line, callee_name, ref_type, project,
                       call_line, call_col, seq, await_kind, guard, receiver,
                       callee_qualifier, callee_resolved
                FROM symbol_refs WHERE caller_filepath = ?{scope_sql}
                AND ref_type IN ({",".join("?" * len(ref_types))})
                ORDER BY COALESCE(call_line, caller_line), COALESCE(call_col, 0), COALESCE(seq, 0)""",
            [filepath, *scope_params, *ref_types],
        )
        # The call-site columns matter here: trace orders calls by them and marks
        # awaits/guards from them (they were never selected, so every child showed its
        # caller's def line, in arbitrary order, with no async markers).
        return [SymbolRefRecord(id=r["id"], caller_filepath=r["caller_filepath"],
                                caller_name=r["caller_name"], caller_line=r["caller_line"],
                                callee_name=r["callee_name"], ref_type=r["ref_type"],
                                project=r["project"], call_line=r["call_line"],
                                call_col=r["call_col"], seq=r["seq"], await_kind=r["await_kind"],
                                guard=r["guard"], receiver=r["receiver"],
                                callee_qualifier=r["callee_qualifier"],
                                callee_resolved=r["callee_resolved"]) for r in cur.fetchall()]

    def find_chunks_by_symbol(self, names: list[str], allowed_projects: list[str] | None = None,
                              limit_per_name: int = 8) -> dict[str, list[ChunkRecord]]:
        """Definition chunks whose last qualified-name component is in ``names``."""
        self._check_closed()
        names = sorted(set(names))
        if not names or (allowed_projects is not None and not allowed_projects):
            return {}
        params: list[Any] = list(names)
        where = f"symbol_name IN ({','.join('?' * len(names))})"
        if allowed_projects is not None:
            where += f" AND project IN ({','.join('?' * len(allowed_projects))})"
            params.extend(allowed_projects)
        cur = self.conn.execute(
            f"SELECT id, symbol_name FROM chunks WHERE {where} ORDER BY symbol_name, id", params)
        ids_by_name: dict[str, list[int]] = {}
        for r in cur.fetchall():
            bucket = ids_by_name.setdefault(r["symbol_name"], [])
            if len(bucket) < limit_per_name:
                bucket.append(r["id"])
        all_ids = [i for ids in ids_by_name.values() for i in ids]
        by_id = {c.id: c for c in self.get_chunks_by_ids(all_ids)}
        return {n: [by_id[i] for i in ids if i in by_id] for n, ids in ids_by_name.items()}

    def get_chunk_by_qualified_name(self, filepath: str, qualified_name: str) -> ChunkRecord | None:
        """First chunk (lowest start line) of ``qualified_name`` in ``filepath``."""
        self._check_closed()
        cur = self.conn.execute(
            "SELECT id FROM chunks WHERE filepath = ? AND qualified_name = ? ORDER BY start_line LIMIT 1",
            (filepath, qualified_name))
        row = cur.fetchone()
        if row is None:
            return None
        return self.get_chunks_by_ids([row["id"]])[0]

    @_writes
    def insert_symbols(self, symbols: list[SymbolRecord]) -> None:
        self._check_closed()
        cur = self.conn.cursor()
        cur.executemany(
            """
            INSERT OR IGNORE INTO symbols (name, symbol_type, filepath, line, signature, project)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [(s.name, s.symbol_type, s.filepath, s.line, s.signature, s.project) for s in symbols]
        )

    def query_symbols(
        self,
        name: str,
        allowed_projects: list[str] | None = None,
        limit: int = 50,
        include_prefix: bool = False,
    ) -> list[SymbolRecord]:
        self._check_closed()
        if allowed_projects is not None and len(allowed_projects) == 0:
            return []
        cur = self.conn.cursor()
        if include_prefix:
            escaped = name.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            clauses = ["(name = ? OR name LIKE ? ESCAPE '\\')"]
            params: list[Any] = [name, f"{escaped}%"]
        else:
            clauses = ["name = ?"]
            params = [name]

        if allowed_projects is not None:
            placeholders = ",".join("?" for _ in allowed_projects)
            clauses.append(f"project IN ({placeholders})")
            params.extend(allowed_projects)

        where = " AND ".join(clauses)
        params.extend([name, limit])

        cur.execute(
            f"""
            SELECT id, name, symbol_type, filepath, line, signature, project
            FROM symbols
            WHERE {where}
            ORDER BY CASE WHEN name = ? THEN 0 ELSE 1 END, line ASC
            LIMIT ?
            """,
            params
        )
        return [
            SymbolRecord(
                id=r["id"],
                name=r["name"],
                symbol_type=r["symbol_type"],
                filepath=r["filepath"],
                line=r["line"],
                signature=r["signature"],
                project=r["project"]
            )
            for r in cur.fetchall()
        ]

    @_writes
    def insert_symbol_refs(self, refs: list[SymbolRefRecord]) -> None:
        self._check_closed()
        cur = self.conn.cursor()
        cur.executemany(
            """
            INSERT OR IGNORE INTO symbol_refs
                (caller_filepath, caller_name, caller_line, callee_name, ref_type, project,
                 call_line, call_col, callee_qualifier, callee_resolved,
                 seq, await_kind, guard, receiver)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [(r.caller_filepath, r.caller_name, r.caller_line, r.callee_name, r.ref_type, r.project,
              r.call_line if r.call_line is not None else r.caller_line, r.call_col or 0,
              r.callee_qualifier, r.callee_resolved,
              r.seq, r.await_kind, r.guard, r.receiver) for r in refs]
        )

    @staticmethod
    def _callers_where(callee_name: str, allowed_projects: list[str] | None) -> tuple[str, list[Any]]:
        # A dotted name ('Class.method') matches the resolved callee; a bare name matches
        # every call of that attribute name, whatever the receiver.
        if "." in callee_name:
            clauses, params = ["callee_resolved = ?"], [callee_name]
        else:
            clauses, params = ["callee_name = ?"], [callee_name]
        if allowed_projects is not None:
            clauses.append(f"project IN ({','.join('?' for _ in allowed_projects)})")
            params.extend(allowed_projects)
        return " AND ".join(clauses), params

    def query_symbol_callers(
        self,
        callee_name: str,
        allowed_projects: list[str] | None = None,
        limit: int = 100,
    ) -> tuple[list[SymbolRefRecord], int]:
        """Call sites of ``callee_name`` plus the total match count (before ``limit``)."""
        self._check_closed()
        if allowed_projects is not None and len(allowed_projects) == 0:
            return [], 0
        where, params = self._callers_where(callee_name, allowed_projects)
        total = self.conn.execute(f"SELECT COUNT(*) FROM symbol_refs WHERE {where}", params).fetchone()[0]
        cur = self.conn.execute(
            f"""
            SELECT id, caller_filepath, caller_name, caller_line, callee_name, ref_type, project,
                   call_line, call_col, callee_qualifier, callee_resolved
            FROM symbol_refs
            WHERE {where}
            ORDER BY callee_resolved IS NULL, callee_resolved, caller_filepath, call_line, call_col
            LIMIT ?
            """,
            [*params, limit]
        )
        return [
            SymbolRefRecord(
                id=r["id"],
                caller_filepath=r["caller_filepath"],
                caller_name=r["caller_name"],
                caller_line=r["caller_line"],
                callee_name=r["callee_name"],
                ref_type=r["ref_type"],
                project=r["project"],
                call_line=r["call_line"],
                call_col=r["call_col"],
                callee_qualifier=r["callee_qualifier"],
                callee_resolved=r["callee_resolved"],
            )
            for r in cur.fetchall()
        ], int(total)

    def count_symbol_callers_by_resolved(
        self,
        callee_name: str,
        allowed_projects: list[str] | None = None,
    ) -> dict[str | None, int]:
        """Call-site counts per ``callee_resolved`` (None = receiver class unknown)."""
        self._check_closed()
        if allowed_projects is not None and len(allowed_projects) == 0:
            return {}
        where, params = self._callers_where(callee_name, allowed_projects)
        cur = self.conn.execute(
            f"SELECT callee_resolved AS r, COUNT(*) AS n FROM symbol_refs WHERE {where} GROUP BY callee_resolved",
            params)
        return {r["r"]: int(r["n"]) for r in cur.fetchall()}
