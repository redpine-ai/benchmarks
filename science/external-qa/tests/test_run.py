import json

import httpx
import pytest

import src.run
from harness.config import load_config
from src.data import UPSTREAM_URL, load_claims
from src.run import CONFIG, judge, main, parse_args

ROWS = [
    {"input": "Claim A.", "answer": "true", "gold_ctx": [{"title": "t", "text": "x"}]},
    {"input": "Claim B.", "answer": "false", "gold_ctx": []},
    {"input": "", "answer": "true"},
]


def test_load_claims_downloads_once_and_caches(tmp_path):
    calls = []

    def get(url, timeout=None, follow_redirects=None):
        calls.append(url)
        body = "\n".join(json.dumps(r) for r in ROWS)
        return httpx.Response(200, text=body, request=httpx.Request("GET", url))

    path = tmp_path / "scifact_test.jsonl"
    claims = load_claims(path, get=get)
    assert calls == [UPSTREAM_URL]
    assert claims == [{"id": "scifact_0", "question": "Claim A.", "gold": "true"},
                      {"id": "scifact_1", "question": "Claim B.", "gold": "false"}]
    assert load_claims(path, get=get) == claims and len(calls) == 1


def test_parse_args_defaults():
    cfg = load_config(CONFIG)
    a = parse_args([], cfg)
    assert a.arms == list(cfg.arm_order) and a.limit is None and a.provider == "anthropic"
    assert a.workers == 4 and a.out == "runs"


def test_main_stops_before_download_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(src.run, "load_dotenv", lambda *a, **k: None)

    def no_download(*a, **k):
        raise AssertionError("downloaded")

    monkeypatch.setattr(src.run, "load_claims", no_download)
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
        main(["--limit", "1"])


def test_judge():
    assert judge({"gold": "true"}, "a", "Answer: true") == {"official_correct": True}
