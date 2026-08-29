"""Decide whether a question needs a web search.

Deterministic and free: no LLM call to classify an LLM call. The rule is simple -
route to the web only when the answer plausibly depends on information that
changed after the model was trained.

Three layers, in priority order:
  1. An explicit /local or /web prefix always wins.
  2. Timeless-intent signals ("explain", "解釋") veto a web search, so
     "explain the latest React features" stays local rather than searching.
  3. Freshness signals (English and Cantonese/Chinese) trigger a web search.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Route = Literal["local", "web"]

_OVERRIDE_RE = re.compile(r"^\s*/(local|web)\b[:,]?\s*", re.IGNORECASE)

# --- freshness signals ------------------------------------------------------
# Word-boundary matched for Latin scripts; substring matched for CJK, which has
# no spaces. Kept as two lists for exactly that reason.

_FRESH_EN = [
    "latest", "newest", "current", "currently", "today", "tonight", "yesterday",
    "tomorrow", "this week", "this month", "this year", "right now", "as of now",
    "recent", "recently", "just released", "newly released", "up to date",
    "up-to-date", "news", "headline", "headlines", "breaking",
    "release", "released", "releases", "release notes", "changelog",
    "new version", "current version", "stable version", "latest version",
    "price", "prices", "pricing", "cost of", "how much does", "stock price",
    "exchange rate", "weather", "forecast", "score", "who won", "election",
    "what happened", "what's happening", "whats happening", "what is happening",
    "this morning", "still available", "deprecated", "end of life",
    "roadmap", "upcoming", "announced", "announcement",
]

# Patterns that need a regex rather than a substring.
_FRESH_EN_RE = [
    re.compile(r"\bin \d{4}\b"),                      # "in 2026"
    re.compile(r"\b(19|20)\d{2}\b"),                  # a bare year
    re.compile(r"\bversion\s+(is|of)\b"),
    re.compile(r"\bwhat'?s new\b"),
    re.compile(r"\bhow much (is|are|does|do)\b"),
]

_FRESH_ZH = [
    "最新", "最近", "而家", "宜家", "依家", "今日", "今天", "今晚", "昨日", "尋日",
    "聽日", "明天", "呢排", "近排", "今個星期", "今個月", "今年", "目前", "現時",
    "现在", "當前", "当前", "新版本", "新版", "版本係", "版本是", "邊個版本",
    "哪个版本", "哪個版本", "更新咗", "有冇更新", "有没有更新", "更新",
    "幾錢", "几钱", "價錢", "价格", "幾多錢", "股價", "股价", "匯率", "汇率",
    "新聞", "新闻", "頭條", "头条", "發生咩事", "发生什么", "發生咗咩",
    "天氣", "天气", "邊個贏", "谁赢", "誰贏", "剛出", "刚出", "啱啱推出",
    "推出咗", "released", "宣布", "公布", "邊間", "點樣買",
]

# --- timeless signals -------------------------------------------------------
# These describe a *kind of thinking*, not a fact lookup, and veto the web route.

# Note: deliberately NOT here - "what is", "what are", "係咩", "是什么". They open
# both definitions and lookups ("what is the latest release"), and a question with
# no freshness signal already routes local, so they buy nothing and cost accuracy.
_TIMELESS_EN = [
    "explain", "explanation", "how does",
    "how do i", "how to", "why does", "why is", "difference between",
    "compare", "pros and cons", "summarise", "summarize", "rewrite", "rephrase",
    "translate", "brainstorm", "give me ideas", "write a", "write me",
    "refactor", "debug", "fix this", "example of", "best practice",
    "in simple terms", "like i'm five", "step by step", "walk me through",
    "define", "definition of", "meaning of", "concept",
]

_TIMELESS_ZH = [
    "解釋", "解释", "點解", "点解", "為咩", "为什么",
    "點樣寫", "怎么写", "教我", "分別", "分别", "區別", "区别",
    "比較", "比较", "總結", "总结", "摘要", "改寫", "改写", "重寫", "重写",
    "翻譯", "翻译", "brainstorm", "諗吓", "举例", "舉例", "例子", "概念",
    "簡單啲講", "简单点说", "一步一步",
]


@dataclass
class RouteDecision:
    route: Route
    query: str          # the query with any /local or /web prefix removed
    reason: str
    forced: bool = False
    signals: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.signals is None:
            self.signals = []


def _has_word(haystack: str, needle: str) -> bool:
    """Word-boundary match for Latin needles; plain substring otherwise."""
    if needle.isascii():
        return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack) is not None
    return needle in haystack


def _matches(text: str, terms: list[str]) -> list[str]:
    return [t for t in terms if _has_word(text, t)]


def classify(query: str, mode: str = "auto") -> RouteDecision:
    """Return the route for this query. `mode` of local/web forces the result."""
    text = query.strip()

    # 1. Explicit prefix beats everything, including an explicit mode.
    override = _OVERRIDE_RE.match(text)
    if override:
        stripped = _OVERRIDE_RE.sub("", text, count=1).strip()
        route: Route = "web" if override.group(1).lower() == "web" else "local"
        return RouteDecision(
            route=route,
            query=stripped or text,
            reason=f"forced by /{override.group(1).lower()} prefix",
            forced=True,
        )

    # 2. An explicit mode from the API body.
    if mode in ("local", "web"):
        return RouteDecision(route=mode, query=text, reason=f"forced by mode={mode}", forced=True)

    lowered = text.lower()

    fresh = _matches(lowered, _FRESH_EN) + _matches(text, _FRESH_ZH)
    fresh += [p.pattern for p in _FRESH_EN_RE if p.search(lowered)]
    if not fresh:
        return RouteDecision("local", text, "no freshness signal")

    timeless = _matches(lowered, _TIMELESS_EN) + _matches(text, _TIMELESS_ZH)
    if timeless:
        # "explain the latest React feature" is a teaching request, not a lookup.
        return RouteDecision(
            "local",
            text,
            "freshness signal overridden by timeless intent",
            signals=fresh + timeless,
        )

    return RouteDecision("web", text, "freshness signal", signals=fresh)
