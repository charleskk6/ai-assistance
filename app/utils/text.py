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
# Prose citations the model reaches for despite being told not to. Removing the
# connective too ("因為根據 Source 2 顯示，" -> "") keeps the sentence readable.
_PROSE_CITATION_RES = [
    re.compile(r"(?:因為)?根據(?:上述|以上)?\s*(?:Source|來源|资料|資料)\s*\d*\s*(?:顯示|显示|所述|指出)?\s*[，,]?\s*", re.IGNORECASE),
    re.compile(r"(?:根據|按照)\s*(?:提供(?:嘅|的))?\s*(?:資料|资料|證據|证据|evidence)\s*[，,]?\s*"),
    re.compile(r"[,]?\s*\b(?:as|according)\s+(?:stated\s+)?(?:in|to)\s+(?:the\s+)?(?:first|second|third|fourth|\d+(?:st|nd|rd|th)?)?\s*sources?\s*\d*", re.IGNORECASE),
    # Only appositive use ("Source 2, the release is..."), never the subject of a
    # sentence - deleting that leaves a verb with nothing in front of it.
    re.compile(r"\bsources?\s*\d+\s*[,:，：]\s*", re.IGNORECASE),
]

# Deletions leave stranded punctuation: ", ." or a clause starting with a comma.
_STRANDED_PUNCT_RES = [
    (re.compile(r"\s+([,.，。!?！？])"), r"\1"),
    (re.compile(r"([,，])\s*([.。!?！？])"), r"\2"),
    (re.compile(r"(^|[.。!?！？]\s*)[,，、:：]+\s*"), r"\1"),
]
_MULTI_NL_RE = re.compile(r"\n{2,}")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


def to_speech_text(
    text: str, *, max_chars: int = 1200, drop_trailing_fragment: bool = False
) -> str:
    """Flatten Markdown-ish model output into something a TTS voice can read.

    Set `drop_trailing_fragment` when the runtime reported that the model was cut
    off at its token ceiling; the unfinished last sentence is then removed.
    """
    if not text:
        return ""
    out = _CODE_FENCE_RE.sub(" ", text)
    out = _TABLE_ROW_RE.sub("", out)
    out = _HR_RE.sub("", out)
    out = _MD_LINK_RE.sub(r"\1", out)          # keep link text, drop the target
    out = _BARE_URL_RE.sub("", out)            # never speak a URL
    out = _CITATION_RE.sub("", out)
    for pattern in _PROSE_CITATION_RES:
        out = pattern.sub("", out)
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
    for pattern, repl in _STRANDED_PUNCT_RES:
        out = pattern.sub(repl, out)
    out = _MULTI_SPACE_RE.sub(" ", _MULTI_NL_RE.sub(" ", out)).strip()
    out = _truncate_on_sentence(out, max_chars)
    return _end_on_sentence(out) if drop_trailing_fragment else out


def _is_cjk(text: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in text)


_TERMINATORS = ".!?。！？…"


def _end_on_sentence(text: str) -> str:
    """Drop the unfinished last sentence of a cut-off answer.

    A voice reading "…it needs a databa" sounds broken in a way a slightly
    shorter answer does not. Only applied when the runtime said the model hit its
    token ceiling, and only when a real answer survives the trim.
    """
    if not text or text[-1] in _TERMINATORS or text[-1] in "\"'）)】」』":
        return text
    cut = max(text.rfind(ch) for ch in _TERMINATORS)
    kept = cut + 1
    # Keep the trim only when a real answer survives it: enough text in absolute
    # terms, and most of what the model wrote. Otherwise the fragment is the bulk
    # of the answer and dropping it would leave nothing worth speaking.
    if kept >= 40 and kept >= len(text) * 0.5:
        return text[:kept]
    return text


def _truncate_on_sentence(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    window = text[:max_chars]
    cut = max(window.rfind(ch) for ch in ".!?。！？")
    return (window[: cut + 1] if cut > max_chars * 0.5 else window).strip()
