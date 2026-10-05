import httpx
import pytest

from harness.clients import RedpineConfig, redpine_search, tavily_search

CFG = RedpineConfig(url="https://api.example", key="k", collection="Redpine Science")


class FakePost:
    def __init__(self, status=200, payload=None, exc=None):
        self.status, self.payload, self.exc, self.calls = status, payload, exc, []

    def __call__(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        if self.exc:
            raise self.exc
        return httpx.Response(self.status, json=self.payload, request=httpx.Request("POST", url))


def test_redpine_request_shape_and_clamp():
    post = FakePost(payload={"results": [{"id": "a", "text": "t", "metadata": {"title": "T"}}]})
    out = redpine_search(CFG, "q" * 1200, limit=99, post=post)
    assert post.calls[0]["url"] == "https://api.example/api/v1/search/query"
    assert post.calls[0]["headers"] == {"Authorization": "Bearer k"}
    assert post.calls[0]["json"] == {"collections": ["Redpine Science"], "query": "q" * 1000, "limit": 30}
    assert out == [{"id": "a", "text": "t", "metadata": {"title": "T"}}]


def test_redpine_filters_passed_through():
    post = FakePost(payload={"results": []})
    redpine_search(CFG, "q", limit=10, filters={"journal": "Nature"}, post=post)
    assert post.calls[0]["json"]["filters"] == {"journal": "Nature"}


def test_redpine_raises_on_http_error():
    with pytest.raises(httpx.HTTPStatusError):
        redpine_search(CFG, "q", limit=10, post=FakePost(status=500, payload={}))


def test_tavily_request_shape_and_clamp():
    post = FakePost(payload={"results": [{"title": "P", "url": "https://x.org/a", "content": "c"}]})
    out = tavily_search("tk", "w" * 500, max_results=50, post=post)
    assert post.calls[0]["url"] == "https://api.tavily.com/search"
    assert post.calls[0]["headers"] == {"Authorization": "Bearer tk"}
    assert post.calls[0]["json"] == {"query": "w" * 400, "search_depth": "advanced", "max_results": 10,
                                     "include_answer": False, "include_raw_content": False}
    assert out[0]["url"] == "https://x.org/a"


def test_tavily_raises_on_transport_error():
    with pytest.raises(httpx.ConnectError):
        tavily_search("tk", "q", max_results=10, post=FakePost(exc=httpx.ConnectError("boom")))
