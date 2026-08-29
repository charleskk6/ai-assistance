"""Bounded, defensive page fetching.

This is not a crawler: it downloads exactly the pages the search provider named,
never follows links out of them, and gives up quickly. Every limit here exists to
protect a fanless laptop from a hostile or merely careless web server.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from app.utils.logging import get_logger

log = get_logger("assistant.fetch")

_ALLOWED_CONTENT = ("text/html", "application/xhtml", "text/plain", "application/xml")


@dataclass
class FetchedPage:
    url: str
    html: str
    content_type: str


def is_public_http_url(url: str) -> bool:
    """Reject non-HTTP schemes and anything pointing at a private address.

    A search result is attacker-influenceable input, so this closes the obvious
    SSRF hole: a result URL resolving to localhost or the LAN could otherwise make
    the assistant fetch from the router's admin page or from itself.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    host = parsed.hostname
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            return False
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return False
    return True


class PageFetcher:
    def __init__(
        self,
        *,
        timeout_s: float = 8.0,
        max_bytes: int = 2_000_000,
        concurrency: int = 4,
        user_agent: str = "LocalAssistant/0.1",
        check_public: bool = True,
    ) -> None:
        self.max_bytes = max_bytes
        self.check_public = check_public
        self._sem = asyncio.Semaphore(concurrency)
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_s, connect=4.0),
            follow_redirects=True,
            max_redirects=3,
            headers={
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9",
                "Accept-Language": "en,zh-HK;q=0.8,zh;q=0.7",
            },
        )

    async def fetch(self, url: str) -> FetchedPage | None:
        """Return the page, or None for anything we should quietly skip."""
        if self.check_public and not is_public_http_url(url):
            log.debug("skip non-public url %s", url)
            return None

        async with self._sem:
            try:
                async with self._client.stream("GET", url) as resp:
                    if resp.status_code >= 400:
                        log.debug("skip %s: HTTP %d", url, resp.status_code)
                        return None

                    ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                    if ctype and not any(ctype.startswith(a) for a in _ALLOWED_CONTENT):
                        log.debug("skip %s: content-type %s", url, ctype)
                        return None

                    declared = resp.headers.get("content-length")
                    if declared and declared.isdigit() and int(declared) > self.max_bytes:
                        log.debug("skip %s: declared %s bytes", url, declared)
                        return None

                    # Stream so an undeclared 500MB body cannot exhaust memory.
                    chunks: list[bytes] = []
                    total = 0
                    async for chunk in resp.aiter_bytes(65536):
                        total += len(chunk)
                        if total > self.max_bytes:
                            log.debug("skip %s: exceeded %d bytes", url, self.max_bytes)
                            return None
                        chunks.append(chunk)

                    body = b"".join(chunks)
                    encoding = resp.encoding or "utf-8"
            except (httpx.HTTPError, ssl_error_types()) as exc:
                log.debug("skip %s: %s", url, exc)
                return None

        try:
            text = body.decode(encoding, errors="replace")
        except (LookupError, UnicodeDecodeError):
            text = body.decode("utf-8", errors="replace")
        return FetchedPage(url=str(url), html=text, content_type=ctype or "text/html")

    async def fetch_many(self, urls: list[str]) -> list[FetchedPage]:
        """Fetch concurrently; failures are dropped, not raised."""
        results = await asyncio.gather(
            *(self.fetch(u) for u in urls), return_exceptions=True
        )
        pages: list[FetchedPage] = []
        for url, res in zip(urls, results):
            if isinstance(res, BaseException):
                log.debug("fetch error %s: %s", url, res)
            elif res is not None:
                pages.append(res)
        return pages

    async def aclose(self) -> None:
        await self._client.aclose()


def ssl_error_types():  # pragma: no cover - trivial
    import ssl

    return ssl.SSLError
