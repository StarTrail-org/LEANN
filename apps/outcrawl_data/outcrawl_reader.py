"""Read source-attributed web documents using the Outcrawl search and scrape API."""

import asyncio
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)
API_URL = "https://api.outcrawl.ai/v1"
USER_AGENT = "LEANN-Outcrawl/0.1 (https://github.com/StarTrail-org/LEANN)"
ERRORS = {401: "invalid API key", 402: "out of credits", 429: "rate limited"}


def _check(response: httpx.Response, what: str) -> dict[str, Any]:
    """Return the JSON answer, raising with Outcrawl's status for a failed call."""
    if response.status_code != 200:
        reason = ERRORS.get(response.status_code, response.text[:200])
        raise RuntimeError(f"Outcrawl {what} failed (HTTP {response.status_code}): {reason}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"Outcrawl {what} returned an invalid response")
    return payload


class OutcrawlReader:
    """Search the web and optionally scrape result pages for local LEANN ingestion.

    This reader sends the search query and selected URLs to a remote service,
    with the API key from ``api_key`` or the ``OUTCRAWL_API_KEY`` environment
    variable. It does not load dotenv files.
    """

    def __init__(self, api_key: str | None = None, timeout: float = 90.0):
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.api_key = (api_key or os.environ.get("OUTCRAWL_API_KEY", "")).strip()
        if not self.api_key:
            raise ValueError("An Outcrawl API key is required (set OUTCRAWL_API_KEY)")
        self.timeout = timeout

    async def read_web_data(
        self, query: str, max_results: int = 5, scrape: bool = False
    ) -> list[dict[str, Any]]:
        """Return text/metadata documents for 1-20 search results.

        With scrape enabled, each selected URL is scraped to Markdown. A page
        that cannot be scraped keeps its search snippet and emits a warning.
        Search and transport failures propagate; the reader does not retry.
        The timeout bounds the entire operation.
        """
        if not query.strip():
            raise ValueError("A nonempty search query is required")
        if not 1 <= max_results <= 20:
            raise ValueError("max_results must be between 1 and 20")
        return await asyncio.wait_for(self._read(query, max_results, scrape), timeout=self.timeout)

    async def _read(self, query: str, max_results: int, scrape: bool) -> list[dict[str, Any]]:
        headers = {"Authorization": f"Bearer {self.api_key}", "User-Agent": USER_AGENT}
        async with httpx.AsyncClient(headers=headers, timeout=self.timeout) as client:
            search = _check(
                await client.post(f"{API_URL}/search", json={"query": query, "limit": max_results}),
                "search",
            )
            hits = search.get("hits")
            if not isinstance(hits, list):
                raise RuntimeError("Outcrawl search response is missing its hits list")
            selected = {}
            for hit in hits:
                if isinstance(hit, dict) and hit.get("url"):
                    selected.setdefault(hit["url"], hit)
                if len(selected) >= max_results:
                    break
            pages: dict[str, dict[str, Any]] = {}
            if scrape and selected:
                answers = await asyncio.gather(
                    *(self._scrape(client, url) for url in selected), return_exceptions=True
                )
                for url, answer in zip(selected, answers):
                    if isinstance(answer, Exception):
                        logger.warning(
                            "Scrape failed for %s (%s); keeping its snippet", url, answer
                        )
                    else:
                        pages[url] = answer
        documents = []
        for url, hit in selected.items():
            page = pages.get(url, {})
            text = (page.get("markdown") or "").strip() or (hit.get("snippet") or "").strip()
            if text:
                title = (page.get("metadata") or {}).get("title") or hit.get("title") or url
                documents.append(
                    {"text": text, "metadata": {"url": url, "title": title, "source": "outcrawl"}}
                )
        return documents

    @staticmethod
    async def _scrape(client: httpx.AsyncClient, url: str) -> dict[str, Any]:
        body = {"url": url, "formats": ["markdown"], "onlyMainContent": True}
        return _check(await client.post(f"{API_URL}/scrape", json=body), "scrape")
