"""Turn a token stream into speakable chunks.

``SentenceSplitter`` emits a sentence as soon as its terminator arrives so TTS
can start while the LLM is still generating. The *first* chunk may be flushed
early at a clause boundary (comma, dash) once it is long enough, which cuts
time-to-first-audio on long opening sentences.
"""

from __future__ import annotations

import re

_ABBREV = {"mr", "mrs", "ms", "dr", "st", "vs", "etc", "e.g", "i.e", "approx", "no", "prof", "mt"}
_END_RE = re.compile(r'([.!?…]+["\')\]]*)(\s+|$)')
_CLAUSE_RE = re.compile(r"([,;:]|\s[–—-])\s")


class SentenceSplitter:
    def __init__(self, first_chunk_min_chars: int = 40, max_chars: int = 220) -> None:
        self.first_min = first_chunk_min_chars
        self.max_chars = max_chars
        self._buf = ""
        self._emitted = 0

    def push(self, text: str) -> list[str]:
        self._buf += text
        out: list[str] = []
        while True:
            piece = self._next()
            if piece is None:
                break
            out.append(piece)
        return out

    def flush(self) -> list[str]:
        rest = self._buf.strip()
        self._buf = ""
        if rest:
            self._emitted += 1
            return [rest]
        return []

    def _take(self, end: int) -> str:
        piece, self._buf = self._buf[:end].strip(), self._buf[end:]
        self._emitted += 1
        return piece

    def _next(self) -> str | None:
        buf = self._buf
        for m in _END_RE.finditer(buf):
            if not m.group(2) and m.end() == len(buf):
                # Terminator at the very end of the buffer: wait for the next token
                # (it could be "..." continuing, or a decimal like "3.5").
                return None
            word = buf[: m.start()].rsplit(None, 1)[-1].lower() if buf[: m.start()].strip() else ""
            if m.group(1) == "." and word.rstrip(".") in _ABBREV:
                continue
            if m.group(1) == "." and re.search(r"\d$", buf[: m.start()]) and re.match(r"\d", buf[m.end():] or "x"):
                continue
            piece = self._take(m.end())
            if piece:
                return piece
            buf = self._buf
        if self._emitted == 0 and len(buf) >= self.first_min:
            cm = None
            for cm in _CLAUSE_RE.finditer(buf):
                pass
            if cm and cm.start() >= self.first_min // 2:
                return self._take(cm.end())
        if len(buf) > self.max_chars:
            cut = buf.rfind(" ", 0, self.max_chars)
            return self._take(cut if cut > 0 else self.max_chars)
        return None


_EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF☀-➿️]")


def clean_for_speech(text: str) -> str:
    """Strip markup and stage directions the model may emit; keep normal punctuation."""
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.S)
    text = re.sub(r"\[(?:coach note|note)[^\]]*\]", " ", text, flags=re.I)
    text = re.sub(r"\*\*?|__|`|#+\s*", "", text)
    text = re.sub(r"\((?:laughs|smiles|chuckles)\)|\*[^*]+\*", " ", text, flags=re.I)
    text = _EMOJI_RE.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class ThinkFilter:
    """Removes <think>…</think> spans from a token stream (hybrid reasoning models)."""

    def __init__(self) -> None:
        self._buf = ""
        self._inside = False

    def push(self, text: str) -> str:
        self._buf += text
        out = []
        while self._buf:
            if self._inside:
                end = self._buf.find("</think>")
                if end < 0:
                    self._buf = self._buf[-8:]
                    return "".join(out)
                self._buf = self._buf[end + 8:]
                self._inside = False
            else:
                start = self._buf.find("<think>")
                if start < 0:
                    # keep a possible partial "<think" tag at the end
                    keep = max((i for i in range(1, 7) if "<think>".startswith(self._buf[-i:])), default=0)
                    out.append(self._buf[: len(self._buf) - keep])
                    self._buf = self._buf[len(self._buf) - keep:]
                    return "".join(out)
                out.append(self._buf[:start])
                self._buf = self._buf[start + 7:]
                self._inside = True
        return "".join(out)

    def flush(self) -> str:
        rest = "" if self._inside else self._buf
        self._buf = ""
        return rest
