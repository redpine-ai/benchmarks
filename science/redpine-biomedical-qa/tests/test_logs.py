"""The shipped run log: its shape, its join with the verdicts, and what it must not carry."""
import json
from collections import Counter
from pathlib import Path

from src.data import load_questions

TRACK = Path(__file__).resolve().parents[1]
LOG = json.loads((TRACK / "logs" / "expert_qa_20260915.json").read_text())
VERDICTS = [json.loads(l) for l in (TRACK / "results" / "verdicts.jsonl").read_text().splitlines()]
ROW_REQUIRED = {"id", "answers", "stripped_spans", "traces", "excluded"}
ROW_ALLOWED = ROW_REQUIRED | {"track", "question", "gold", "doi", "access"}
SEARCH_KEYS = {"tool", "step", "query", "n_results", "budget_refused"}
BANNED_KEYS = {"text", "content", "passage", "passages", "results", "agent_retrieved_context", "agent_trajectory"}


def test_header():
    assert LOG["arms"] == ["websearch", "connect"]
    assert LOG["model"] == "claude-opus-5" and LOG["tool_budget"] == 20
    assert LOG["provider"] and LOG["served_model_id"]
    assert LOG["pool_size"] == 179 and LOG["n_complete"] == 179


def test_rows_cover_every_question_once():
    ids = [r["id"] for r in LOG["rows"]]
    assert sorted(ids) == sorted(q["id"] for q in load_questions()) and len(set(ids)) == 179


def test_row_shape():
    for r in LOG["rows"]:
        assert ROW_REQUIRED <= set(r) <= ROW_ALLOWED, r["id"]
        assert set(r["answers"]) == set(r["stripped_spans"]) == set(r["traces"]) == {"websearch", "connect"}
        for arm in ("websearch", "connect"):
            assert isinstance(r["answers"][arm], str) and r["answers"][arm].strip()
            assert isinstance(r["stripped_spans"][arm], int) and r["stripped_spans"][arm] >= 0
            for s in r["traces"][arm]["searches"]:
                assert set(s) == SEARCH_KEYS
        assert r["excluded"] is None


def test_log_joins_verdicts_one_to_one():
    log_pairs = {(r["id"], a) for r in LOG["rows"] for a in LOG["arms"]}
    verdict_pairs = {(v["id"], v["arm"]) for v in VERDICTS}
    assert log_pairs == verdict_pairs
    assert set(Counter((v["id"], v["arm"]) for v in VERDICTS).values()) == {3}
    assert Counter(v["judge"] for v in VERDICTS) == {"sonnet5": 358, "jev1": 358, "jev2": 358}


def _walk(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield "key", k
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def test_no_banned_key_anywhere():
    for kind, k in _walk(LOG):
        assert not (kind == "key" and k in BANNED_KEYS), k


def test_stripping_happened_somewhere():
    assert sum(r["stripped_spans"][a] for r in LOG["rows"] for a in LOG["arms"]) > 0
    assert any("[passage text removed]" in r["answers"][a] for r in LOG["rows"] for a in LOG["arms"])
