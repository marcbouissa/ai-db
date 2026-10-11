"""Sources for ``draft_skill``: blog posts and tutorials as text, plus plugin source kinds.

A source is one of:

- ``{"text": ..., "title"?, "url"?, "license"?}``: text the agent already has (markdown,
  plain text or HTML);
- ``{"url": ...}``: fetched as text only: http(s), a size cap and a timeout, HTML turned into
  readable text. Loopback, private, link-local and reserved addresses are refused (also after
  a redirect) unless ``skills.allow_private_urls`` is true;
- anything else (``{"video": path}``, ``{"asset": path}``, ...) goes to the plugins of the
  ``ai_db.skill_sources`` entry-point group: ``(source: dict, config) -> dict | None``,
  None meaning "not mine". The dict has ``sections`` ({section: [lines]}), ``source``
  (what to record in ``metadata.sources``) and ``warnings``.

Deterministic: headings, ordered steps, code blocks, parameter values and a stated licence
are extracted with regular expressions; nothing calls a model.
"""

from __future__ import annotations

import ipaddress
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from html.parser import HTMLParser
from importlib.metadata import entry_points
from typing import Any

from ai_db.errors import AiDbConfigError

SKILL_SOURCE_GROUP = "ai_db.skill_sources"
FETCH_MAX_BYTES = 2_000_000
FETCH_TIMEOUT_S = 15.0
FETCH_USER_AGENT = "ai-db draft_skill (text only)"
SOURCE_TEXT_MAX_CHARS = 400_000
MAX_SOURCES = 10
MAX_STEPS = 25
MAX_HEADINGS = 15
MAX_CODE_BLOCKS = 3
CODE_BLOCK_MAX_CHARS = 800
MAX_PARAMS = 15
_TEXT_TYPES = ("text/html", "text/plain", "text/markdown", "application/xhtml+xml", "application/json")

#: ``fetch(url) -> (content_type, bytes)``; tests replace it.
Fetch = Callable[[str], tuple[str, bytes]]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Any:
        raise urllib.error.HTTPError(newurl, code, f"redirect to {newurl}", headers, fp)


def check_url(url: str, allow_private: bool = False) -> None:
    """Refuse non-http(s) URLs and, unless allowed, hosts on private/loopback networks."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError(f"only http(s) URLs are fetched, got {url!r}")
    if allow_private:
        return
    try:
        infos = socket.getaddrinfo(parts.hostname, parts.port or (443 if parts.scheme == "https" else 80))
    except OSError as e:
        raise ValueError(f"cannot resolve {parts.hostname!r}: {e}") from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast \
                or ip.is_unspecified:
            raise ValueError(f"{parts.hostname!r} resolves to a non-public address ({ip}); "
                             "set skills.allow_private_urls to fetch it")


def http_fetch(url: str, allow_private: bool = False, max_redirects: int = 3) -> tuple[str, bytes]:
    """GET ``url`` as text: redirects followed by hand so each hop is checked again."""
    opener = urllib.request.build_opener(_NoRedirect)
    for _ in range(max_redirects + 1):
        check_url(url, allow_private)
        req = urllib.request.Request(url, headers={"User-Agent": FETCH_USER_AGENT, "Accept": "text/html,text/plain"})
        try:
            with opener.open(req, timeout=FETCH_TIMEOUT_S) as resp:  # scheme and host checked above
                ctype = resp.headers.get("Content-Type", "")
                data = resp.read(FETCH_MAX_BYTES + 1)
                return ctype, data[:FETCH_MAX_BYTES]
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307, 308) and e.filename:
                url = urllib.parse.urljoin(url, e.filename)
                continue
            raise
    raise ValueError(f"too many redirects for {url!r}")


class _Html2Text(HTMLParser):
    _SKIP = frozenset({"script", "style", "nav", "footer", "header", "aside", "noscript", "svg", "form"})
    _BLOCK = frozenset({"p", "div", "section", "article", "br", "tr", "table", "ul", "ol", "blockquote"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.skip = 0
        self.pre = 0
        self.lists: list[list[int]] = []  # per open list: [ordered, counter]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self.skip += 1
        elif self.skip:
            return
        elif re.fullmatch(r"h[1-6]", tag):
            self.out.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag == "pre":
            self.pre += 1
            self.out.append("\n\n```\n")
        elif tag in ("ol", "ul"):
            self.lists.append([1 if tag == "ol" else 0, 0])
        elif tag == "li":
            if self.lists and self.lists[-1][0]:
                self.lists[-1][1] += 1
                self.out.append(f"\n{self.lists[-1][1]}. ")
            else:
                self.out.append("\n- ")
        elif tag == "dt":  # definition lists: how reference manuals document options
            self.out.append("\n" + _OPTION_MARK)
        elif tag == "dd":
            self.out.append(" :: ")
        elif tag in self._BLOCK:
            self.out.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP:
            self.skip = max(0, self.skip - 1)
        elif self.skip:
            return
        elif tag == "pre":
            self.pre = max(0, self.pre - 1)
            self.out.append("\n```\n\n")
        elif tag in ("ol", "ul") and self.lists:
            self.lists.pop()
            self.out.append("\n")
        elif re.fullmatch(r"h[1-6]", tag) or tag in self._BLOCK:
            self.out.append("\n")

    def handle_data(self, data: str) -> None:
        if self.skip:
            return
        data = data.replace("\u00b6", "")  # Sphinx/MkDocs heading anchors
        self.out.append(data if self.pre else re.sub(r"\s+", " ", data))


def html_to_text(html: str) -> str:
    """Readable markdown-ish text: headings as #, ordered lists numbered, <pre> fenced."""
    p = _Html2Text()
    p.feed(html)
    p.close()
    text = "".join(p.out)
    lines = [ln.rstrip() for ln in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


_OPTION_MARK = "\u2063opt "  # invisible separator: marks a <dt> line for extract_text_sections
# the definition is on the same or the next line; a <dt> right after (a nested list) is not one
_OPTION = re.compile("^" + _OPTION_MARK + r"(.{1,60}?)\s*::[ \t]*(?:\n[ \t]*)?(?!" + _OPTION_MARK[0] + r")(\S.*)$",
                     re.MULTILINE)
_STEP = re.compile(r"^\s*(?:step\s*)?(\d{1,2})\s*[.):]\s+(.{3,})$", re.IGNORECASE | re.MULTILINE)
_HEAD = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$", re.MULTILINE)
_FENCE = re.compile(r"```[\w+-]*\n(.*?)```", re.DOTALL)
_PARAM = re.compile(r"\b([A-Za-z][A-Za-z ]{1,30}?)\s*(?:=|:|\bto\b)\s*(-?\d+(?:\.\d+)?\s*(?:mm|cm|m|%|°|deg|px|k)?)(?!\w)")
_LICENSE = re.compile(r"\b(CC0(?:[- ]1\.0)?|CC[- ]BY(?:[- ](?:SA|NC|ND))*(?:[- ]\d\.\d)?|Creative Commons [A-Za-z0-9 .-]{2,40}?|"
                      r"MIT License|Apache(?: License)?,? (?:Version )?2\.0|BSD[- ]\d[- ]Clause|Public Domain|GPL-?\d(?:\.\d)?)\b",
                      re.IGNORECASE)


def extract_text_sections(text: str) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Workflow / Examples lines and found facts (title, license) from a tutorial's text."""
    text = text[:SOURCE_TEXT_MAX_CHARS]
    code = [c.strip() for c in _FENCE.findall(text) if c.strip()]
    prose = _FENCE.sub("\n", text)
    options = [f"Option - {' '.join(t.split()).rstrip(':')}: {' '.join(d.split()).lstrip(': ')[:200]}"
               for t, d in _OPTION.findall(prose) if d.strip(": ")]
    prose = prose.replace(_OPTION_MARK, "")
    heads = [h[1].strip() for h in _HEAD.findall(prose)]
    steps = [f"{num}. {' '.join(step.split())[:240]}" for num, step in _STEP.findall(prose)]
    params: list[str] = []
    for name, value in _PARAM.findall(prose):
        item = f"{' '.join(name.split())} = {' '.join(value.split())}"
        if item not in params:
            params.append(item)
    workflow = steps[:MAX_STEPS] or [f"(section) {h}" for h in heads[:MAX_HEADINGS]]
    examples = [f"```\n{c[:CODE_BLOCK_MAX_CHARS]}\n```" for c in code[:MAX_CODE_BLOCKS]]
    examples += options[:MAX_PARAMS]
    if params:
        examples.append("Parameter values from the source: " + "; ".join(params[:MAX_PARAMS]))
    facts: dict[str, str] = {}
    if heads:
        facts["title"] = heads[0][:120]
    lic = _LICENSE.search(text)
    if lic:
        facts["license"] = lic.group(1).strip()
    return {k: v for k, v in {"workflow": workflow, "examples": examples}.items() if v}, facts


def _plugin_source(source: dict[str, Any], config: Any) -> dict[str, Any] | None:
    for ep in sorted(entry_points(group=SKILL_SOURCE_GROUP), key=lambda e: e.name):
        try:
            reader = ep.load()
        except ImportError as e:
            raise AiDbConfigError(f"skill-source plugin '{ep.name}' ({ep.value}) failed to import: {e}") from e
        out = reader(source, config)
        if out is not None:
            return dict(out)
    return None


def read_sources(sources: Any, config: Any = None, fetch: Fetch | None = None) -> dict[str, Any]:
    """Sections, recorded sources and warnings of ``sources`` (see module docstring)."""
    if sources is None:
        return {"sections": {}, "sources": [], "warnings": []}
    if not isinstance(sources, list) or not all(isinstance(s, dict) for s in sources):
        raise ValueError("sources must be a list of objects ({text}, {url}, {video}, {asset})")
    if len(sources) > MAX_SOURCES:
        raise ValueError(f"at most {MAX_SOURCES} sources, got {len(sources)}")
    allow_private = bool(getattr(config, "skills_allow_private_urls", False))
    fetch = fetch or (lambda u: http_fetch(u, allow_private))
    sections: dict[str, list[str]] = {}
    recorded: list[dict[str, str]] = []
    warnings: list[str] = []
    for k, src in enumerate(sources):
        try:
            if "text" in src or "url" in src:
                text, url = src.get("text"), src.get("url")
                if text is None:
                    if urllib.parse.urlsplit(str(url)).scheme not in ("http", "https"):
                        raise ValueError(f"only http(s) URLs are fetched, got {url!r}")
                    ctype, data = fetch(str(url))
                    if ctype and not ctype.split(";")[0].strip().lower().startswith(_TEXT_TYPES):
                        raise ValueError(f"not a text page ({ctype}); only text is read")
                    text = data.decode("utf-8", errors="replace")
                    if "html" in ctype.lower() or text.lstrip()[:15].lower().startswith(("<!doctype", "<html")):
                        text = html_to_text(text)
                elif not isinstance(text, str):
                    raise ValueError("text must be a string")
                elif re.search(r"<(?:p|h[1-6]|div|li|pre)\b", text[:5000], re.IGNORECASE):
                    text = html_to_text(text)
                found, facts = extract_text_sections(text)
                rec = {"url": str(url)} if url else {"text": str(src.get("title") or facts.get("title") or f"text #{k + 1}")}
                lic = src.get("license") or facts.get("license")
                if lic:
                    rec["license"] = str(lic)
                if src.get("title") or facts.get("title"):
                    rec["title"] = str(src.get("title") or facts.get("title"))
            else:
                out = _plugin_source(src, config)
                if out is None:
                    raise ValueError(f"no reader for source keys {sorted(src)}")
                found = out.get("sections") or {}
                rec = {k2: str(v) for k2, v in (out.get("source") or {}).items() if v is not None}
                warnings += [str(w) for w in out.get("warnings") or []]
        except (ValueError, OSError, urllib.error.URLError) as e:
            warnings.append(f"source {k + 1}: {e}")
            continue
        for key, lines in found.items():
            sections.setdefault(key, []).extend(str(x) for x in lines)
        recorded.append(rec)
    return {"sections": sections, "sources": recorded, "warnings": warnings}
