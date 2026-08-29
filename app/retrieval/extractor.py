"""Pull the readable article out of a page and throw the furniture away.

trafilatura does the real work; the regex fallback covers the cases where it
returns nothing (unusual markup, plain text served as HTML).
"""

from __future__ import annotations

import html
import re

import trafilatura

from app.utils.logging import get_logger

log = get_logger("assistant.extract")

_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style|noscript|svg|nav|footer|header|aside|form)\b.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
_TAG_RE = re.compile(r"<[^>]+>")
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.DOTALL | re.IGNORECASE)
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_NL_RE = re.compile(r"\n{3,}")
# Nav/consent noise that survives naive stripping.
_BOILERPLATE_RE = re.compile(
    r"^\s*(cookie|accept all|subscribe|sign in|log in|advertisement|share this"
    r"|skip to (main )?content)\b.*$",
    re.IGNORECASE | re.MULTILINE,
)


class Extracted:
    __slots__ = ("text", "title", "published")

    def __init__(self, text: str, title: str = "", published: str | None = None) -> None:
        self.text = text
        self.title = title
        self.published = published


def _clean(text: str) -> str:
    text = _BOILERPLATE_RE.sub("", text)
    text = _WS_RE.sub(" ", text)
    text = _NL_RE.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def _fallback(page_html: str) -> str:
    stripped = _SCRIPT_STYLE_RE.sub(" ", page_html)
    stripped = re.sub(r"</(p|div|li|h[1-6]|tr|br)\s*>", "\n", stripped, flags=re.IGNORECASE)
    return _clean(html.unescape(_TAG_RE.sub(" ", stripped)))


def extract(page_html: str, url: str = "") -> Extracted:
    """Return the main content. `text` is empty when there is nothing usable."""
    if not page_html or not page_html.strip():
        return Extracted("")

    title = ""
    published = None
    text = ""

    try:
        result = trafilatura.bare_extraction(
            page_html,
            url=url or None,
            include_comments=False,
            include_tables=True,
            with_metadata=True,
            favor_precision=True,
        )
        if result:
            as_dict = result if isinstance(result, dict) else result.as_dict()
            text = (as_dict.get("text") or "").strip()
            title = (as_dict.get("title") or "").strip()
            published = as_dict.get("date") or None
    except Exception as exc:  # trafilatura raises a variety of parser errors
        log.debug("trafilatura failed on %s: %s", url, exc)

    if not text:
        text = _fallback(page_html)

    if not title:
        match = _TITLE_RE.search(page_html)
        title = _clean(html.unescape(_TAG_RE.sub("", match.group(1)))) if match else ""

    return Extracted(_clean(text), title, published)
