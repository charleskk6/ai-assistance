"""Phase 1E: fetching, extraction, chunking and ranking."""

from __future__ import annotations

import httpx
import respx

from app.retrieval.chunker import chunk_text
from app.retrieval.extractor import extract
from app.retrieval.fetcher import PageFetcher, is_public_http_url
from app.retrieval.ranker import rank, tokenize

URL = "https://example.com/page"


def fetcher(**kw) -> PageFetcher:
    kw.setdefault("check_public", False)  # example.com does not resolve in CI
    return PageFetcher(timeout_s=1, **kw)


# --- fetcher ---------------------------------------------------------------
@respx.mock
async def test_fetch_returns_html():
    respx.get(URL).mock(
        return_value=httpx.Response(200, html="<html><body>hi</body></html>")
    )
    page = await fetcher().fetch(URL)
    assert page is not None and "hi" in page.html


@respx.mock
async def test_bad_url_status_is_skipped_not_raised():
    respx.get(URL).mock(return_value=httpx.Response(404, text="nope"))
    assert await fetcher().fetch(URL) is None


@respx.mock
async def test_connection_error_is_skipped():
    respx.get(URL).mock(side_effect=httpx.ConnectError("dns"))
    assert await fetcher().fetch(URL) is None


@respx.mock
async def test_timeout_is_skipped():
    respx.get(URL).mock(side_effect=httpx.ReadTimeout("slow"))
    assert await fetcher().fetch(URL) is None


@respx.mock
async def test_invalid_content_type_is_skipped():
    respx.get(URL).mock(
        return_value=httpx.Response(
            200, content=b"%PDF-1.4", headers={"content-type": "application/pdf"}
        )
    )
    assert await fetcher().fetch(URL) is None


@respx.mock
async def test_oversized_declared_response_is_skipped():
    respx.get(URL).mock(
        return_value=httpx.Response(
            200, text="x", headers={"content-type": "text/html", "content-length": "99999999"}
        )
    )
    assert await fetcher(max_bytes=1000).fetch(URL) is None


@respx.mock
async def test_oversized_undeclared_body_is_cut_off():
    """No content-length header: the streaming cap has to catch it."""
    respx.get(URL).mock(
        return_value=httpx.Response(
            200, content=b"a" * 500_000, headers={"content-type": "text/html"}
        )
    )
    assert await fetcher(max_bytes=1000).fetch(URL) is None


@respx.mock
async def test_fetch_many_drops_failures_and_keeps_successes():
    respx.get("https://ok.com/a").mock(
        return_value=httpx.Response(200, html="<p>good</p>")
    )
    respx.get("https://bad.com/b").mock(side_effect=httpx.ConnectError("x"))
    respx.get("https://slow.com/c").mock(side_effect=httpx.ReadTimeout("x"))
    pages = await fetcher().fetch_many(
        ["https://ok.com/a", "https://bad.com/b", "https://slow.com/c"]
    )
    assert [p.url for p in pages] == ["https://ok.com/a"]


def test_ssrf_guard_rejects_private_and_non_http():
    assert is_public_http_url("http://127.0.0.1:8000/ask") is False
    assert is_public_http_url("http://localhost/x") is False
    assert is_public_http_url("file:///etc/passwd") is False
    assert is_public_http_url("ftp://example.com/x") is False
    assert is_public_http_url("http://169.254.169.254/latest/meta-data/") is False
    assert is_public_http_url("https://nonexistent.invalid/x") is False


# --- extractor -------------------------------------------------------------
def test_extract_removes_boilerplate_and_scripts():
    html = (
        "<html><head><title>Python Downloads</title></head><body>"
        "<nav>Skip to content</nav><script>var x=1;</script>"
        "<article><p>The current stable release is Python 3.13.1.</p></article>"
        "<footer>Accept all cookies</footer></body></html>"
    )
    result = extract(html, "https://python.org/downloads/")
    assert "Python 3.13.1" in result.text
    for noise in ("var x", "Skip to content", "Accept all cookies"):
        assert noise not in result.text


def test_extract_empty_page_yields_empty_text():
    assert extract("").text == ""
    assert extract("<html><body></body></html>").text == ""


def test_extract_falls_back_when_no_article_markup():
    html = "<html><title>T</title><body><div><div>Bare text with no article.</div></div></body></html>"
    result = extract(html)
    assert "Bare text" in result.text
    assert result.title == "T"


def test_extract_survives_malformed_html():
    result = extract("<html><body><p>unclosed <b>text")
    assert "unclosed" in result.text


# --- chunker ---------------------------------------------------------------
def test_chunker_respects_size_and_keeps_paragraphs():
    text = "\n\n".join(f"Paragraph {i} with some words in it." for i in range(20))
    chunks = chunk_text(text, title="T", url=URL, domain="example.com", size=120)
    assert len(chunks) > 1
    assert all(len(c.text) <= 130 for c in chunks)
    assert all(c.domain == "example.com" for c in chunks)
    assert [c.position for c in chunks] == list(range(len(chunks)))


def test_chunker_splits_an_oversized_paragraph_on_sentences():
    text = " ".join(f"Sentence number {i} here." for i in range(80))
    chunks = chunk_text(text, title="T", url=URL, domain="d.com", size=200, overlap=20)
    assert len(chunks) > 1
    assert all(len(c.text) <= 220 for c in chunks)


def test_chunker_handles_empty_and_whitespace():
    assert chunk_text("", title="T", url=URL, domain="d") == []
    assert chunk_text("   \n\n  ", title="T", url=URL, domain="d") == []


# --- ranker ----------------------------------------------------------------
def test_tokenizer_produces_cjk_bigrams():
    tokens = tokenize("最新版本")
    assert "最新" in tokens and "版本" in tokens and "最" in tokens


def test_tokenizer_keeps_versioned_terms_intact():
    assert "python3.13" in tokenize("Python3.13")
    assert "c++" in tokenize("c++ is fast") or "c" in tokenize("c++ is fast")


def test_ranker_puts_relevant_chunk_first_and_drops_irrelevant():
    chunks = chunk_text(
        "Python 3.13.1 is the current stable release.\n\n"
        "The weather in Paris is mild today.\n\n"
        "Older Python versions are archived.",
        title="Downloads", url="https://python.org/d", domain="python.org", size=60,
    )
    ev = rank("latest stable Python version", chunks, top_k=3, budget_chars=1000)
    assert "3.13.1" in ev[0].text
    assert not any("Paris" in e.text for e in ev)  # zero-score chunks excluded


def test_ranker_works_on_cantonese_query():
    chunks = chunk_text(
        "Python 最新穩定版本係 3.13.1。\n\n巴黎今日天氣和暖。",
        title="下載", url="https://python.org/d", domain="python.org", size=40,
    )
    ev = rank("而家最新 Python version 係邊個", chunks, top_k=2, budget_chars=1000)
    assert ev and "3.13.1" in ev[0].text


def test_ranker_respects_budget_and_top_k():
    chunks = chunk_text(
        "\n\n".join(f"Python release note {i} about the version." for i in range(30)),
        title="T", url="https://a.com/x", domain="a.com", size=60,
    )
    assert len(rank("python version", chunks, top_k=2, budget_chars=10_000)) <= 2
    ev = rank("python version", chunks, top_k=20, budget_chars=150)
    assert sum(len(e.text) for e in ev) <= 150 + 60


def test_ranker_limits_chunks_from_one_domain():
    chunks = []
    for rank_i, domain in enumerate(["a.com", "b.com"]):
        chunks += chunk_text(
            "\n\n".join(f"Python version release note {i}." for i in range(8)),
            title="T", url=f"https://{domain}/x", domain=domain, size=50, doc_rank=rank_i,
        )
    ev = rank("python version release", chunks, top_k=8, budget_chars=10_000)
    domains = [e.domain for e in ev]
    assert domains.count("a.com") <= 3 and "b.com" in domains


def test_ranker_on_empty_input():
    assert rank("anything", [], top_k=3, budget_chars=100) == []
