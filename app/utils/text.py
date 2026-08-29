"""Text shaping for Apple's text-to-speech.

The model is *told* to write speech (see rag/prompt.py), but a local 8B model
will sometimes emit a bullet list or a URL anyway. This is the deterministic
safety net applied to every `source="siri"` answer.
"""

from __future__ import annotations

import re

_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_BULLET_RE = re.compile(r"^\s{0,4}([-*+]|\d{1,2}[.)])\s+", re.MULTILINE)
_BLOCKQUOTE_RE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((?:[^)]*)\)")
_BOLD_ITALIC_RE = re.compile(r"(\*{1,3}|_{1,3})(?=\S)(.+?)(?<=\S)\1", re.DOTALL)
_BARE_URL_RE = re.compile(r"<?\bhttps?://\S+|\bwww\.\S+", re.IGNORECASE)
_HR_RE = re.compile(r"^\s*([-*_])\1{2,}\s*$", re.MULTILINE)
_CITATION_RE = re.compile(r"[\[(]\s*(?:source|來源|参考|參考)?\s*\d+\s*[\])]", re.IGNORECASE)
_MULTI_NL_RE = re.compile(r"\n{2,}")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


def to_speech_text(text: str, *, max_chars: int = 1200) -> str:
    """Flatten Markdown-ish model output into something a TTS voice can read."""
    if not text:
        return ""
    out = _CODE_FENCE_RE.sub(" ", text)
    out = _TABLE_ROW_RE.sub("", out)
    out = _HR_RE.sub("", out)
    out = _MD_LINK_RE.sub(r"\1", out)          # keep link text, drop the target
    out = _BARE_URL_RE.sub("", out)            # never speak a URL
    out = _CITATION_RE.sub("", out)
    out = _HEADING_RE.sub("", out)
    out = _BLOCKQUOTE_RE.sub("", out)
    out = _BULLET_RE.sub("", out)
    out = _BOLD_ITALIC_RE.sub(r"\2", out)
    out = _INLINE_CODE_RE.sub(r"\1", out)
    # Bullets became separate lines; join them into sentences so the voice does
    # not pause oddly, using the punctuation already at the end of each line.
    lines = [ln.strip() for ln in out.splitlines()]
    joined: list[str] = []
    for line in lines:
        if not line:
            continue
        if joined and joined[-1][-1] not in ".!?。！？；;:：,，":
            joined[-1] += "。" if _is_cjk(joined[-1]) else "."
        joined.append(line)
    out = " ".join(joined)
    out = _MULTI_SPACE_RE.sub(" ", _MULTI_NL_RE.sub(" ", out)).strip()
    return _truncate_on_sentence(out, max_chars)


def _is_cjk(text: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in text)


def _truncate_on_sentence(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    window = text[:max_chars]
    cut = max(window.rfind(ch) for ch in ".!?。！？")
    return (window[: cut + 1] if cut > max_chars * 0.5 else window).strip()
