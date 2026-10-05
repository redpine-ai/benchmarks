"""The three retrievers this benchmark compares: Redpine Science, Tavily (web search),
and Exa. Each exposes one async method, `search(query, max_results) -> list[Result]`,
called directly and once per query -- no agent loop, no reformulation, no tool-use
framework. That is a deliberate methodological choice, not a missing feature: it asks
"given one query, which system's own ranking surfaces the correct document, and at
what rank", which a single retrieval call answers more validly than an agent that can
retry, rephrase, or fall back to another tool. See the package README for the fuller
argument.

Tavily and Exa are configured to match exa-labs/benchmarks' own reference
implementation, not just its README -- checked against the actual searcher source
(shared/shared/searchers/{tavily,exa}.py), since the two disagree on real defaults:

- Tavily's reference searcher sets search_depth="advanced". Its own API default is
  "basic" (shallower page parsing) if you don't set it explicitly.
- Exa's reference searcher for this exact benchmark is instantiated as
  ExaSearcher(category="publication", include_text=True) -- not left unrestricted,
  despite the README's prose suggesting otherwise. category is Exa's own built-in
  content-type classifier (research paper/preprint/journal article vs. news,
  consumer-health explainers, personal sites), not a domain allowlist.
- Exa's request uses query-scoped highlights (highlights: {"query": query}), not a bare
  highlights: true, which asks Exa for *a* representative snippet rather than the
  one relevant to this specific query.
- Exa's request also pins type="auto" explicitly. Exa's /search endpoint's `type`
  enum is instant/fast/auto/deep-lite/deep/deep-reasoning; the deep* tiers run a
  multi-step "research process" rather than a single retrieval pass, at a different
  price tier. Pinning this avoids ever silently resolving into that tier.
- Tavily's web search excludes github.com (see _EXCLUDED_DOMAINS below): the query
  set's own source, exa-labs/benchmarks, hosts its query-to-gold-DOI mapping
  directly in its public GitHub repo, so an unrestricted search could in principle
  find the answer key rather than doing any real retrieval.

Neither Tavily nor Exa returns a structured DOI in its response -- matching.py's
title-containment fallback exists specifically because of that.
"""

from __future__ import annotations

import httpx
from pydantic import BaseModel


class Result(BaseModel):
    """One retrieved item, normalized across all three systems. doi is populated
    only for Redpine (the only one of the three with a structured field for it);
    every other system's matching falls through to title comparison."""

    title: str | None = None
    doi: str | None = None
    url: str | None = None
    snippet: str = ""


TOP_K = 10


class RedpineRetriever:
    """Wraps Redpine Science's search API (POST /api/v1/search/query on
    api.redpine.ai). `collection` accepts a comma-separated list -- the API takes a
    `collections` array (max 5), so a single collection is just a 1-item array;
    this keeps a single-collection and multi-collection setup on the same code path.
    """

    provider_id = "redpine"

    def __init__(self, client: httpx.AsyncClient, api_key: str, collection: str,
                 max_results: int = TOP_K):
        self._client = client
        self._api_key = api_key
        self._collections = [c.strip() for c in collection.split(",") if c.strip()]
        if not self._collections or len(self._collections) > 5:
            raise ValueError(
                f"RedpineRetriever requires 1-5 collections, got {len(self._collections)} "
                f"from collection={collection!r}")
        self._max_results = max_results

    async def search(self, query: str, max_results: int | None = None) -> list[Result]:
        n = max_results or self._max_results
        resp = await self._client.post(
            "/api/v1/search/query",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"collections": self._collections, "query": query, "limit": n},
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        out = []
        for r in results[:n]:
            meta = r.get("metadata") or {}
            out.append(Result(title=meta.get("title"), doi=meta.get("doi"),
                              url=r.get("url"), snippet=r.get("text") or ""))
        return out


# exa-labs/benchmarks (the public source of this query set) hosts the query-to-
# gold-DOI mapping directly in its own GitHub repo. An unrestricted web search
# could in principle "find" a query's answer key there rather than doing any real
# retrieval -- this is the one domain excluded, everything else is open web.
_EXCLUDED_DOMAINS = ["github.com"]


class TavilyRetriever:
    provider_id = "tavily"

    def __init__(self, client: httpx.AsyncClient, api_key: str, max_results: int = TOP_K):
        self._client = client
        self._api_key = api_key
        self._max_results = max_results

    async def search(self, query: str, max_results: int | None = None) -> list[Result]:
        n = max_results or self._max_results
        resp = await self._client.post(
            "/search",
            json={"api_key": self._api_key, "query": query, "max_results": n,
                 "search_depth": "advanced", "exclude_domains": _EXCLUDED_DOMAINS},
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        return [Result(title=r.get("title"), doi=None, url=r.get("url"),
                       snippet=r.get("content", "")) for r in results]


class ExaRetriever:
    provider_id = "exa"

    def __init__(self, client: httpx.AsyncClient, api_key: str, max_results: int = TOP_K):
        self._client = client
        self._api_key = api_key
        self._max_results = max_results

    async def search(self, query: str, max_results: int | None = None) -> list[Result]:
        n = max_results or self._max_results
        resp = await self._client.post(
            "/search",
            headers={"x-api-key": self._api_key},
            json={"query": query, "numResults": n, "type": "auto",
                 "category": "publication",
                 "contents": {"highlights": {"query": query}, "text": True}},
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        return [Result(title=r.get("title"), doi=None, url=r.get("url"),
                       snippet="\n".join(h["text"] if isinstance(h, dict) else h
                                          for h in (r.get("highlights") or [])) or (r.get("text") or ""))
                for r in results]
