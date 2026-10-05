"""The two search backends, as plain HTTP calls that return raw results.

Both raise on failure; the renderer's dispatch turns a failure into the text the
model reads. Redpine Science is Redpine's public search API; web search is
Tavily at advanced depth with no generated answer and no raw page content.
"""
from dataclasses import dataclass

import httpx

REDPINE_QUERY_CHARS = 1000   # the API rejects longer queries
REDPINE_MAX_LIMIT = 30       # server-side cap on results per call
TAVILY_URL = "https://api.tavily.com/search"
TAVILY_SEARCH_DEPTH = "advanced"
TAVILY_QUERY_CHARS = 400
TAVILY_MAX_RESULTS = 10
REDPINE_TIMEOUT_S = 30
TAVILY_TIMEOUT_S = 60


@dataclass(frozen=True)
class RedpineConfig:
    url: str
    key: str
    collection: str = "Redpine Science"


def _post_results(post, url, key, body, timeout):
    resp = post(url, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=timeout)
    resp.raise_for_status()
    return resp.json().get("results", []) or []


def redpine_search(cfg: RedpineConfig, query: str, *, limit: int, filters: dict | None = None,
                   post=httpx.post) -> list[dict]:
    body = {"collections": [cfg.collection], "query": query[:REDPINE_QUERY_CHARS],
            "limit": max(1, min(int(limit), REDPINE_MAX_LIMIT))}
    if filters is not None:
        body["filters"] = filters
    return _post_results(post, f"{cfg.url}/api/v1/search/query", cfg.key, body, REDPINE_TIMEOUT_S)


def tavily_search(key: str, query: str, *, max_results: int, post=httpx.post) -> list[dict]:
    body = {"query": query[:TAVILY_QUERY_CHARS], "search_depth": TAVILY_SEARCH_DEPTH,
            "max_results": max(1, min(int(max_results), TAVILY_MAX_RESULTS)),
            "include_answer": False, "include_raw_content": False}
    return _post_results(post, TAVILY_URL, key, body, TAVILY_TIMEOUT_S)
