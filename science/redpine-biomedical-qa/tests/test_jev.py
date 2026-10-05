import json
from types import SimpleNamespace as NS

import pytest

from src.jev import CRITERIA, ENDPOINT, INSTRUCTIONS, MODEL, VERDICTS, build_request, jev_answer, labels_of
from tests.test_judge import FRAILTY


def test_build_request_is_one_choice_question_per_claim():
    req = build_request("Q?", "scope text", ["claim one", "claim two"], "answer text")
    assert req["model"] == MODEL and ENDPOINT == "https://openrouter.ai/api/alpha/decisions"
    assert req["state"] == "QUESTION:\nQ?\n\nSCOPE OF THE REQUIRED CLAIMS:\nscope text\n\nCANDIDATE ANSWER:\nanswer text"
    assert list(req["questions"]) == ["c1", "c2"]
    q = req["questions"]["c1"]
    assert q["type"] == "choice" and q["criteria"] == CRITERIA and set(CRITERIA) == set(VERDICTS)
    assert q["instructions"] == INSTRUCTIONS.format(claim="claim one") and q["instructions"].endswith("REQUIRED CLAIM: claim one")


def test_build_request_fallbacks():
    req = build_request("", "", ["c"], "a")
    assert "QUESTION:\n(not provided)\n" in req["state"] and "SCOPE OF THE REQUIRED CLAIMS:\n(none stated)\n" in req["state"]


def test_labels_of_reads_each_choice_in_order():
    resp = {"answers": {"c1": {"choice": "stated", "probabilities": {}, "confidence": 0.9},
                        "c2": {"choice": "not_addressed"}}}
    assert labels_of(resp, 2) == ["stated", "not_addressed"]
    with pytest.raises(ValueError, match="no choice for c3"):
        labels_of(resp, 3)
    with pytest.raises(ValueError, match="no answers"):
        labels_of({}, 1)


def fake_post_factory(labels):
    calls = []

    def post(payload):
        calls.append(payload)
        return {"answers": {f"c{i}": {"choice": lab, "probabilities": {lab: 0.9}, "confidence": 0.9}
                            for i, lab in enumerate(labels, 1)}}

    return post, calls


def test_jev_answer_one_request_same_shape_as_sonnet():
    post, calls = fake_post_factory(["stated", "stated", "not_addressed"])
    out = jev_answer(post, "q", FRAILTY, "ans")
    assert len(calls) == 1 and list(calls[0]["questions"]) == ["c1", "c2", "c3"]
    assert [v["verdict"] for v in out["verdicts"]] == ["stated", "stated", "not_addressed"]
    assert all(v["reason"] is None for v in out["verdicts"])
    assert set(out) == {"claims", "verdicts", "coverage", "contradiction_rate", "correctness"}
    assert round(out["coverage"], 2) == 0.67
    assert "probabilities" not in json.dumps(out) and "confidence" not in json.dumps(out)


def test_jev_answer_rejects_unknown_label():
    post, _ = fake_post_factory(["maybe", "stated", "stated"])
    with pytest.raises(ValueError, match="unknown verdict"):
        jev_answer(post, "q", FRAILTY, "ans")


def test_http_post_retries_429_and_transport_errors(monkeypatch):
    import httpx

    from src.jev import http_post

    calls = []
    sleeps = []

    def fake_post(url, json, timeout, headers):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("boom")
        if len(calls) == 2:
            return NS(status_code=429)
        return NS(status_code=200, json=lambda: {"answers": {}})

    monkeypatch.setattr(httpx, "post", fake_post)
    result = http_post("k", sleep=lambda s: sleeps.append(s))({"x": 1})
    assert result == {"answers": {}}
    assert len(calls) == 3 and sleeps == [1.0, 2.0]


def test_http_post_does_not_retry_4xx(monkeypatch):
    import httpx

    from src.jev import http_post

    calls = []

    def fake_post(url, json, timeout, headers):
        calls.append(1)
        return NS(status_code=400)

    monkeypatch.setattr(httpx, "post", fake_post)
    with pytest.raises(ValueError, match="http 400"):
        http_post("k", sleep=lambda s: None)({"x": 1})
    assert len(calls) == 1
