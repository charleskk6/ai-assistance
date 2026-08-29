"""Phase 1D: search providers. Every external call is mocked - the test suite
never touches the network."""

from __future__ import annotations

import httpx
import pytest
import respx

from app.config import Settings
from app.search.base import SearchError
from app.search.provider import (
    BraveProvider,
    DuckDuckGoProvider,
    SearxngProvider,
    TavilyProvider,
    build_search_provider,
)

DDG_HTML = """
<html><body>
<div class="result results_links">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.python.org%2Fdownloads%2F&amp;rut=x">
    Download <b>Python</b></a>
  <a class="result__snippet" href="#">The current stable release is Python 3.13.1.</a>
</div>
<div class="result results_links">
  <a class="result__a" href="https://docs.python.org/3/whatsnew/">What's New</a>
  <a class="result__snippet" href="#">Changes in recent releases.</a>
</div>
<div class="result results_links">
  <a class="result__a" href="https://www.python.org/downloads/">Duplicate URL</a>
</div>
</body></html>
"""


def ddg() -> DuckDuckGoProvider:
    return DuckDuckGoProvider(timeout_s=5, user_agent="test")


@respx.mock
async def test_duckduckgo_parses_unwraps_and_dedupes():
    respx.post("https://html.duckduckgo.com/html/").mock(
        return_value=httpx.Response(200, text=DDG_HTML)
    )
    results = await ddg().search("latest python", limit=5)

    assert len(results) == 2  # third result is a duplicate URL
    first = results[0]
    assert first.url == "https://www.python.org/downloads/"  # redirect unwrapped
    assert first.title == "Download Python"                   # tags stripped
    assert "3.13.1" in first.snippet
    assert first.domain == "python.org"                       # www. removed


@respx.mock
async def test_duckduckgo_respects_limit():
    respx.post("https://html.duckduckgo.com/html/").mock(
        return_value=httpx.Response(200, text=DDG_HTML)
    )
    assert len(await ddg().search("q", limit=1)) == 1


@respx.mock
async def test_duckduckgo_network_failure_raises_search_error():
    respx.post("https://html.duckduckgo.com/html/").mock(
        side_effect=httpx.ConnectError("no network")
    )
    with pytest.raises(SearchError) as exc:
        await ddg().search("q", limit=3)
    assert exc.value.code == "search_unavailable"


@respx.mock
async def test_duckduckgo_empty_page_returns_no_results():
    respx.post("https://html.duckduckgo.com/html/").mock(
        return_value=httpx.Response(200, text="<html><body>nothing</body></html>")
    )
    assert await ddg().search("q", limit=3) == []


@respx.mock
async def test_brave_normalises_results():
    respx.get("https://api.search.brave.com/res/v1/web/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "web": {
                    "results": [
                        {
                            "title": "Python 3.13",
                            "url": "https://python.org/downloads/",
                            "description": "Latest <strong>stable</strong>",
                            "page_age": "2026-01-02",
                        },
                        {"title": "bad", "url": "javascript:void(0)"},
                    ]
                }
            },
        )
    )
    results = await BraveProvider("key", 5).search("q", limit=5)
    assert len(results) == 1  # non-http result dropped
    assert results[0].snippet == "Latest stable"  # HTML stripped
    assert results[0].published == "2026-01-02"


async def test_brave_without_key_fails_loudly():
    with pytest.raises(SearchError) as exc:
        BraveProvider("", 5)
    assert exc.value.code == "search_misconfigured"


@respx.mock
async def test_tavily_normalises_results():
    respx.post("https://api.tavily.com/search").mock(
        return_value=httpx.Response(
            200,
            json={"results": [{"title": "T", "url": "https://a.com/x", "content": "c"}]},
        )
    )
    results = await TavilyProvider("key", 5).search("q", limit=3)
    assert results[0].domain == "a.com"


@respx.mock
async def test_searxng_normalises_results():
    respx.get("http://127.0.0.1:8080/search").mock(
        return_value=httpx.Response(
            200, json={"results": [{"title": "S", "url": "https://b.org/y", "content": "c"}]}
        )
    )
    results = await SearxngProvider("http://127.0.0.1:8080", 5).search("q", limit=3)
    assert results[0].domain == "b.org"


@pytest.mark.parametrize(
    "name,cls",
    [
        ("duckduckgo", DuckDuckGoProvider),
        ("brave", BraveProvider),
        ("tavily", TavilyProvider),
        ("searxng", SearxngProvider),
    ],
)
def test_factory_builds_each_provider(name, cls):
    s = Settings(
        _env_file=None,
        local_assistant_token="t",
        search_provider=name,
        brave_api_key="k",
        tavily_api_key="k",
    )
    assert isinstance(build_search_provider(s), cls)
