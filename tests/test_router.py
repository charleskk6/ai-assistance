"""Phase 1C: routing decisions. These are the spec's own examples plus the
edge cases that broke the first implementation."""

from __future__ import annotations

import pytest

from app.router.classifier import classify

# --- the acceptance cases from the spec ------------------------------------
SPEC_CASES = [
    ("Explain dependency injection", "local"),
    ("What is the latest Python release?", "web"),
    ("Current Node.js version", "web"),
    ("而家最新 Python version 係邊個？", "web"),
    ("解釋下 dependency injection", "local"),
    ("/local latest Python release", "local"),
    ("/web explain dependency injection", "web"),
]


@pytest.mark.parametrize("query,expected", SPEC_CASES)
def test_spec_cases(query, expected):
    assert classify(query).route == expected


# --- freshness, English -----------------------------------------------------
@pytest.mark.parametrize(
    "query",
    [
        "latest AI news",
        "what happened in the election",
        "how much is a MacBook Air",
        "Node 22 release notes",
        "current price of bitcoin",
        "who won the match today",
        "is that package deprecated",
        "what's new in TypeScript",
        "React roadmap 2026",
        "weather this week",
    ],
)
def test_english_freshness_routes_web(query):
    assert classify(query).route == "web"


# --- freshness, Cantonese / Chinese ----------------------------------------
@pytest.mark.parametrize(
    "query",
    [
        "今日有咩新聞？",
        "而家 iPhone 幾錢？",
        "最近 AI 有咩發展？",
        "Python 有冇更新？",
        "呢排港股點呀",
        "美金匯率而家幾多",
        "尋日發生咩事",
        "邊個版本係最新",
    ],
)
def test_cantonese_freshness_routes_web(query):
    assert classify(query).route == "web"


# --- timeless -------------------------------------------------------------
@pytest.mark.parametrize(
    "query",
    [
        "What is dependency injection?",
        "how does a hash map work",
        "difference between TCP and UDP",
        "rewrite this paragraph to be shorter",
        "translate this to English",
        "brainstorm names for a cat",
        "refactor this function",
        "點解要用 async",
        "教我點樣寫 decorator",
        "總結呢段文字",
        "TCP 同 UDP 有咩分別",
    ],
)
def test_timeless_routes_local(query):
    assert classify(query).route == "local"


def test_timeless_intent_vetoes_freshness_signal():
    """'explain the latest X' is teaching, not a lookup."""
    d = classify("explain the latest React features")
    assert d.route == "local"
    assert "overridden" in d.reason

    assert classify("解釋下最新嘅 React features").route == "local"


# --- overrides -------------------------------------------------------------
def test_override_strips_prefix_from_query():
    d = classify("/web explain dependency injection")
    assert d.route == "web" and d.forced
    assert d.query == "explain dependency injection"


def test_override_is_case_insensitive_and_beats_mode():
    d = classify("/LOCAL latest Python release", mode="web")
    assert d.route == "local" and d.forced


def test_mode_forces_route():
    assert classify("explain recursion", mode="web").route == "web"
    assert classify("latest Python release", mode="local").route == "local"
    assert classify("latest Python release", mode="auto").route == "web"


def test_override_alone_keeps_original_query():
    d = classify("/web")
    assert d.query == "/web"  # nothing left to ask; don't hand an empty string on


def test_word_boundary_avoids_false_positives():
    # "newsletter" contains "news"; "release" must not match inside "released"
    # incorrectly either - both are word-boundary matched.
    assert classify("how do I design a newsletter template").route == "local"
    assert classify("what does the increment operator do").route == "local"


def test_decision_reports_signals_for_logging():
    d = classify("latest Python version")
    assert d.route == "web"
    assert d.signals and "latest" in d.signals
