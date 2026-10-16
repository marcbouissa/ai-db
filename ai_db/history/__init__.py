"""Git-history and convention mining (read-only), for review-rule generation.

Entry point: ``python -m ai_db.history {mine,conventions} REPO [--out FILE]``.

- ``mine``: commits split at a cutoff into train/test, classified (fix/revert/feat/...),
  zero-context hunks, SZZ-lite (which commits introduced the lines a fix removed),
  co-changed file pairs, fix hotspots, reverts.
- ``conventions``: identifier-style adherence per (language, symbol kind) and a few
  language checks, measured with the tree-sitter parser on files read from a revision.

Not yet wired into ``ai-db``'s CLI/MCP (those files carry other uncommitted work).
Wiring, when they are free:
- ``ai_db/cli.py``: a ``history`` subcommand with ``mine``/``conventions`` sub-parsers
  that delegates to ``ai_db.history.__main__.build_parser``/``main``.
- ``ai_db/dispatcher.py`` + ``mcp_server.py``: a ``history`` tool taking
  ``{action: "mine"|"conventions", repo, rev, cutoff, cutoff_ratio}`` and returning a
  summary (kinds, hotspots, cochange, reverts, naming) -- the full hunks are too large for
  an MCP response, so it should write them to a file and return its path.
"""

from ai_db.history.conventions import ConventionOptions, conventions
from ai_db.history.gitlog import classify
from ai_db.history.mine import MineOptions, mine

__all__ = ["ConventionOptions", "MineOptions", "classify", "conventions", "mine"]
