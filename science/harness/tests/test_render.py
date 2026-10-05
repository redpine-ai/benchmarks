import httpx

from harness.clients import RedpineConfig
from harness.config import load_config
from harness.render import RENDERERS, make_dispatch

CONNECT = [{"id": "c1", "text": "A" * 5000, "metadata": {"title": "T1", "journal": "JAMA", "publication_date": "2021-03-01", "doi": "10.1/x"}},
           {"id": "c2", "text": "second", "metadata": {"title": "T2", "pmid": "42"}},
           {"id": None, "text": "bare", "metadata": {}}]
WEB = [{"title": "Page", "url": "https://www.example.org/a", "content": "B" * 5000},
       {"title": "", "url": "https://other.net/b", "content": "c"}]


def test_indexed_connect_numbers_from_state_and_caps():
    r = RENDERERS["indexed"]
    state = r.new_state()
    first = r.connect(CONNECT[:2], state, 4000)
    assert first == "[0] " + "A" * 4000 + "\n\n[1] second"
    assert r.connect(CONNECT[:1], state, 4000).startswith("[2] ")


def test_indexed_web_numbers_from_one_with_domain():
    r = RENDERERS["indexed"]
    assert r.web(WEB, 4000) == "[1] Page (example.org)\n" + "B" * 4000 + "\n\n[2] (untitled) (other.net)\nc"


def test_indexed_empty_and_error_text():
    r = RENDERERS["indexed"]
    assert r.empty == "No results found."
    assert r.error("search_connect", RuntimeError("x")) == "search_connect error: RuntimeError: x"


def test_handles_connect_venue_fallbacks():
    r = RENDERERS["handles"]
    out = r.connect(CONNECT, r.new_state(), 4000)
    parts = out.split("\n\n")
    assert parts[0] == "[S:c1] T1 (JAMA 2021)\n" + "A" * 4000
    assert parts[1] == "[S:c2] T2 (PMID 42)\nsecond"


def test_handles_connect_minimal_result():
    r = RENDERERS["handles"]
    assert r.connect(CONNECT[2:], r.new_state(), 4000) == "[S:?]\nbare"


def test_handles_web_uses_url_and_domain():
    r = RENDERERS["handles"]
    assert r.web(WEB[:1], 4000) == "[S:https://www.example.org/a] Page (www.example.org)\n" + "B" * 4000


def test_handles_empty_and_error_text():
    r = RENDERERS["handles"]
    assert r.empty == "(no results)"
    assert r.error("search_web", RuntimeError("x")) == "ERROR: x"


class FakePost:
    def __init__(self, payload):
        self.payload, self.calls = payload, []

    def __call__(self, url, headers=None, json=None, timeout=None):
        self.calls.append(json)
        return httpx.Response(200, json=self.payload, request=httpx.Request("POST", url))


Q2 = '''
model = "claude-opus-5"
provider = "anthropic"
max_tokens = 8192
thinking = "adaptive"
budget = 20
top_k = 10
passage_chars = 4000
renderer = "handles"
final_text = "last_turn"
system_template = "s {tool_sentence}"
user_template = "{question}"
budget_exhausted = "b"
final_nudge = "f"
arm_order = ["connect"]
[tools.search_connect]
kind = "redpine"
description = "d"
input_schema = { type = "object" }
[tools.search_web]
kind = "web"
description = "d"
input_schema = { type = "object" }
[arms.connect]
tools = ["search_connect", "search_web"]
tool_sentence = "t"
'''


def cfg(tmp_path, text=Q2):
    p = tmp_path / "c.toml"
    p.write_text(text)
    return load_config(p)


def test_dispatch_clamps_limits_and_passes_filters(tmp_path):
    post = FakePost({"results": CONNECT[:1]})
    d = make_dispatch(cfg(tmp_path), RedpineConfig("https://api.example", "k", "C"), "tk", post=post)
    text, meta = d("search_connect", {"query": "q", "limit": 99, "filters": {"journal": "JAMA"}})
    assert post.calls[0]["limit"] == 30 and post.calls[0]["filters"] == {"journal": "JAMA"}
    assert text.startswith("[S:c1] T1") and meta == [{"id": "c1", "title": "T1", "journal": "JAMA",
                                                      "publication_date": "2021-03-01", "doi": "10.1/x"}]
    d("search_web", {"query": "q", "max_results": 50})
    assert post.calls[1]["max_results"] == 10


def test_dispatch_defaults_to_top_k_and_reports_errors(tmp_path):
    post = FakePost({"results": []})
    d = make_dispatch(cfg(tmp_path), RedpineConfig("https://api.example", "k", "C"), "tk", post=post)
    assert d("search_connect", {"query": "q"}) == ("(no results)", [])
    assert post.calls[0]["limit"] == 10

    def boom(url, headers=None, json=None, timeout=None):
        raise httpx.ConnectError("down")

    d2 = make_dispatch(cfg(tmp_path), RedpineConfig("https://api.example", "k", "C"), "tk", post=boom)
    assert d2("search_web", {"query": "q"}) == ("ERROR: down", [])


def test_dispatch_unconfigured_tool(tmp_path):
    d = make_dispatch(cfg(tmp_path), None, None)
    assert d("search_connect", {"query": "q"}) == ("unknown tool: search_connect", [])
    assert d("nope", {}) == ("unknown tool: nope", [])
