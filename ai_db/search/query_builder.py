"""Query-side text processing for lexical search: tokenization, identifier
expansion, stopwords and FTS5 query construction."""

from __future__ import annotations

import re

from ai_db.constants import CODE_STOPWORDS

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def split_identifier(word: str) -> list[str]:
    """``getUserName`` / ``get_user_name`` / ``HTTPServer`` -> lowercase parts."""
    parts: list[str] = []
    for piece in word.split("_"):
        if piece:
            parts.extend(p.lower() for p in _CAMEL.split(piece) if p)
    return parts


def expand_terms(text: str) -> list[str]:
    """Query terms: words, identifier parts and joined identifier forms, minus stopwords.

    ``getUserName`` -> ``get``, ``user``, ``name``, ``getusername``, ``get_user_name``.
    Order is preserved (first occurrence) so NEAR/AND groups follow the user's wording.
    """
    seen: dict[str, None] = {}

    def add(term: str) -> None:
        term = term.lower()
        if len(term) >= 2 and term not in CODE_STOPWORDS:
            seen.setdefault(term, None)

    for word in re.findall(r"\w+", text):
        parts = split_identifier(word)
        for p in parts:
            add(p)
        if len(parts) > 1:
            add("".join(parts))
            add("_".join(parts))
    return list(seen)


def base_terms(text: str) -> list[str]:
    """Only the split words (no joined forms) — used for AND/NEAR groups."""
    seen: dict[str, None] = {}
    for word in re.findall(r"\w+", text):
        for p in split_identifier(word):
            if len(p) >= 2 and p not in CODE_STOPWORDS:
                seen.setdefault(p, None)
    return list(seen)


def _quote(term: str) -> str:
    return '"' + term.replace("\x00", "").replace('"', '""') + '"'


def build_fts(terms: list[str], core: list[str] | None = None) -> str:
    """FTS5 MATCH expression.

    ``(core AND ...) OR NEAR(core ..., 10) OR (t1* OR t2* ...)`` where ``core`` are the
    base words and ``terms`` every expanded term. Documents matching all words close
    together accumulate the most BM25 weight; prefix matches keep recall.
    """
    terms = [t for t in (t.replace("\x00", "").strip() for t in terms) if t]
    if not terms:
        raise ValueError("build_fts needs at least one term")
    core = [t for t in (core or terms) if t in terms] or terms
    prefix = " OR ".join(f"{_quote(t)}*" for t in terms)
    if len(core) == 1:
        return prefix
    conj = " AND ".join(_quote(t) for t in core)
    near = "NEAR(" + " ".join(_quote(t) for t in core) + ", 10)"
    return f"({conj}) OR {near} OR ({prefix})"


def identifier_words(text: str) -> str:
    """Space-joined split parts of every identifier in ``text`` (document-side column)."""
    words: dict[str, None] = {}
    for ident in _IDENT.findall(text):
        parts = split_identifier(ident)
        if len(parts) > 1:
            for p in parts:
                words.setdefault(p, None)
    return " ".join(words)
