"""Search provider abstraction.

The LLM never talks to a search engine directly. Everything goes through
`SearchProvider.search()` so the engine can be swapped in one env var.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from urllib.parse import urlparse


class SearchError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""
    published: str | None = None

    @property
    def domain(self) -> str:
        return (urlparse(self.url).hostname or "").removeprefix("www.")


class SearchProvider(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    async def search(self, query: str, limit: int) -> list[SearchResult]:
        """Return up to `limit` normalised results, best first."""

    async def aclose(self) -> None:  # pragma: no cover - default no-op
        return None
