"""Concrete search providers.

DuckDuckGo is the default because it needs no account and costs nothing. Brave,
Tavily and SearXNG are drop-in alternatives when DDG rate-limits you; set
SEARCH_PROVIDER and the matching key.
"""

from __future__ import annotations

import html
import re
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from app.config import Settings
from app.search.base import SearchError, SearchProvider, SearchResult

_TAG_RE = re.compile(r"<[^>]+>")


def _clean(text: str) -> str:
    return html.unescape(_TAG_RE.sub("", text)).strip()


class DuckDuckGoProvider(SearchProvider):
    """Scrapes DDG's no-JavaScript HTML endpoint. No API key, no cost."""

    name = "duckduckgo"
    ENDPOINT = "https://html.duckduckgo.com/html/"

    _RESULT_RE = re.compile(
        r'<a[^>]+class="result__a"[^>]+href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>'
        r'(?P<rest>.*?)(?=<a[^>]+class="result__a"|</body>)',
        re.DOTALL | re.IGNORECASE,
    )
    _SNIPPET_RE = re.compile(
        r'class="result__snippet"[^>]*>(?P<snippet>.*?)</a>', re.DOTALL | re.IGNORECASE
    )

    def __init__(self, timeout_s: float, user_agent: str) -> None:
        self._client = httpx.AsyncClient(
            timeout=timeout_s,
            follow_redirects=True,
            headers={"User-Agent": user_agent, "Accept-Language": "en,zh-HK;q=0.8"},
        )

    @staticmethod
    def _unwrap(url: str) -> str:
        """DDG wraps results as //duckduckgo.com/l/?uddg=<encoded>."""
        if "duckduckgo.com/l/" in url or url.startswith("//duckduckgo.com/l/"):
            qs = parse_qs(urlparse(url if "//" not in url[:2] else "https:" + url).query)
            if qs.get("uddg"):
                return unquote(qs["uddg"][0])
        if url.startswith("//"):
            return "https:" + url
        return url

    async def search(self, query: str, limit: int) -> list[SearchResult]:
        try:
            resp = await self._client.post(self.ENDPOINT, data={"q": query, "kl": "wt-wt"})
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise SearchError("search_unavailable", f"DuckDuckGo request failed: {exc}") from exc

        results: list[SearchResult] = []
        seen: set[str] = set()
        for match in self._RESULT_RE.finditer(resp.text):
            url = self._unwrap(html.unescape(match.group("url")))
            if not url.startswith("http") or url in seen:
                continue
            seen.add(url)
            snippet_match = self._SNIPPET_RE.search(match.group("rest"))
            results.append(
                SearchResult(
                    title=_clean(match.group("title")),
                    url=url,
                    snippet=_clean(snippet_match.group("snippet")) if snippet_match else "",
                )
            )
            if len(results) >= limit:
                break
        return results

    async def aclose(self) -> None:
        await self._client.aclose()


class BraveProvider(SearchProvider):
    """Brave Search API. Free tier is ~2000 queries/month with a card on file."""

    name = "brave"
    ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str, timeout_s: float) -> None:
        if not api_key:
            raise SearchError("search_misconfigured", "BRAVE_API_KEY is not set.")
        self._client = httpx.AsyncClient(
            timeout=timeout_s,
            headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
        )

    async def search(self, query: str, limit: int) -> list[SearchResult]:
        try:
            resp = await self._client.get(
                self.ENDPOINT, params={"q": query, "count": limit}
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise SearchError("search_unavailable", f"Brave request failed: {exc}") from exc
        items = (resp.json().get("web") or {}).get("results", [])
        return [
            SearchResult(
                title=_clean(it.get("title", "")),
                url=it.get("url", ""),
                snippet=_clean(it.get("description", "")),
                published=it.get("page_age") or it.get("age"),
            )
            for it in items
            if it.get("url", "").startswith("http")
        ][:limit]

    async def aclose(self) -> None:
        await self._client.aclose()


class TavilyProvider(SearchProvider):
    """Tavily. Returns pre-extracted content, which we reuse to skip a fetch."""

    name = "tavily"
    ENDPOINT = "https://api.tavily.com/search"

    def __init__(self, api_key: str, timeout_s: float) -> None:
        if not api_key:
            raise SearchError("search_misconfigured", "TAVILY_API_KEY is not set.")
        self._api_key = api_key
        self._client = httpx.AsyncClient(timeout=timeout_s)

    async def search(self, query: str, limit: int) -> list[SearchResult]:
        try:
            resp = await self._client.post(
                self.ENDPOINT,
                json={
                    "api_key": self._api_key,
                    "query": query,
                    "max_results": limit,
                    "search_depth": "basic",
                },
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise SearchError("search_unavailable", f"Tavily request failed: {exc}") from exc
        return [
            SearchResult(
                title=_clean(it.get("title", "")),
                url=it.get("url", ""),
                snippet=_clean(it.get("content", "")),
                published=it.get("published_date"),
            )
            for it in resp.json().get("results", [])
            if it.get("url", "").startswith("http")
        ][:limit]

    async def aclose(self) -> None:
        await self._client.aclose()


class SearxngProvider(SearchProvider):
    """Self-hosted SearXNG. Best privacy, but you have to run the container."""

    name = "searxng"

    def __init__(self, base_url: str, timeout_s: float) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout_s)

    async def search(self, query: str, limit: int) -> list[SearchResult]:
        try:
            resp = await self._client.get(
                f"{self.base_url}/search",
                params={"q": query, "format": "json", "safesearch": 0},
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise SearchError("search_unavailable", f"SearXNG request failed: {exc}") from exc
        return [
            SearchResult(
                title=_clean(it.get("title", "")),
                url=it.get("url", ""),
                snippet=_clean(it.get("content", "")),
                published=it.get("publishedDate"),
            )
            for it in resp.json().get("results", [])
            if it.get("url", "").startswith("http")
        ][:limit]

    async def aclose(self) -> None:
        await self._client.aclose()


def build_search_provider(settings: Settings) -> SearchProvider:
    if settings.search_provider == "duckduckgo":
        return DuckDuckGoProvider(settings.search_timeout_s, settings.fetch_user_agent)
    if settings.search_provider == "brave":
        return BraveProvider(settings.brave_api_key, settings.search_timeout_s)
    if settings.search_provider == "tavily":
        return TavilyProvider(settings.tavily_api_key, settings.search_timeout_s)
    if settings.search_provider == "searxng":
        return SearxngProvider(settings.searxng_url, settings.search_timeout_s)
    raise ValueError(f"Unsupported SEARCH_PROVIDER: {settings.search_provider}")
