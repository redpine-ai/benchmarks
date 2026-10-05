import json
from types import SimpleNamespace as NS

import pytest

from src.judge import (MAX_ATTEMPTS, MAX_TOKENS, PROMPT, SCHEMA_SUFFIX, SYSTEM_PROMPT, VERDICTS,
                       build_prompt, judge_answer, parse_json_reply, required_claims, score_verdicts,
                       scope_of, validate_verdicts)

FRAILTY = ("A strong answer conveys: Greater frailty burden is associated with a substantially higher "
           "incidence of stroke than the lowest frailty category; Stroke risk rises in a graded, "
           "dose-response fashion as the frailty index increases; Frailer patients already carried "
           "higher conventional stroke risk scores at baseline, so part of the excess stroke risk may "
           "reflect confounding by these factors. Certainty: strong but imprecise. Scope: atrial "
           "fibrillation coexisting with heart failure with preserved ejection fraction in older adults, "
           "outpatient settings. An answer is wrong if it directs anticoagulation by frailty category.")


def test_required_claims_and_scope():
    claims = required_claims(FRAILTY)
    assert len(claims) == 3 and claims[0].startswith("Greater frailty burden")
    assert scope_of(FRAILTY) == ("atrial fibrillation coexisting with heart failure with preserved ejection "
                                 "fraction in older adults, outpatient settings")


def test_required_claims_ignores_blank_segments():
    assert required_claims("A strong answer conveys: a; ; b;. Certainty: c. Scope: s.") == ["a", "b"]


def test_required_claims_rejects_non_template():
    with pytest.raises(ValueError, match="template"):
        required_claims("Just an answer.")


def test_prompt_without_scope():
    p = build_prompt("Q?", "A strong answer conveys: a; b. Certainty: c.", "ans")
    assert "Scope of the required claims:\n(none stated)\n" in p
    assert "Required claims:\n1. a\n2. b\n" in p and "({n_claims})" not in p
    assert "MUST equal the number of required claims (2)." in p and p.endswith(SCHEMA_SUFFIX)


def test_prompt_is_the_metric_prompt():
    assert PROMPT.startswith("You are assessing a candidate answer to a biomedical question against the required claims")
    assert "Question:\n{question}\n\nScope of the required claims:\n{scope}\n\nRequired claims:\n{claims}\n\nCandidate answer:\n{actual_output}" in PROMPT
    assert VERDICTS == ("stated", "contradicted", "not_addressed")
    assert SYSTEM_PROMPT == "You are an impartial evaluation judge. When asked for JSON, reply with JSON only."
    assert MAX_TOKENS == 8192


def test_validate_verdicts_passes_well_formed_list():
    assert validate_verdicts([{"verdict": "stated", "reason": "r1"}, {"verdict": "not_addressed", "reason": "r2"}], 2) == [
        {"verdict": "stated", "reason": "r1"}, {"verdict": "not_addressed", "reason": "r2"}]


def test_validate_verdicts_count_mismatch_raises():
    with pytest.raises(ValueError, match="2 verdicts for 3"):
        validate_verdicts([{"verdict": "stated", "reason": ""}] * 2, 3)


def test_validate_verdicts_rejects_unknown_label_and_shape():
    with pytest.raises(ValueError, match="unknown verdict"):
        validate_verdicts([{"verdict": "maybe", "reason": ""}], 1)
    with pytest.raises(ValueError, match="0 verdicts for 1"):
        validate_verdicts(None, 1)


def test_parse_json_reply_strips_fences():
    body = {"verdicts": [{"verdict": "stated", "reason": "r"}]}
    assert parse_json_reply(json.dumps(body)) == body
    assert parse_json_reply("```json\n" + json.dumps(body) + "\n```") == body
    assert parse_json_reply("Here you go:\n" + json.dumps(body) + "\nDone.") == body
    with pytest.raises(ValueError, match="no JSON object"):
        parse_json_reply("no braces here")


def test_score_frailty_example():
    s = score_verdicts([{"verdict": "stated"}, {"verdict": "stated"}, {"verdict": "not_addressed"}])
    assert round(s["coverage"], 2) == 0.67 and s["contradiction_rate"] == 0 and round(s["correctness"], 2) == 0.80


def test_score_edge_cases():
    assert score_verdicts([{"verdict": "stated"}] * 3) == {"coverage": 1.0, "contradiction_rate": 0.0, "correctness": 1.0}
    s = score_verdicts([{"verdict": "contradicted"}, {"verdict": "stated"}, {"verdict": "stated"}])
    assert round(s["contradiction_rate"], 2) == 0.33
    assert score_verdicts([{"verdict": "contradicted"}] * 2)["correctness"] == 0.0


class FakeClient:
    """Returns each reply text in turn; the last one repeats."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []
        self.messages = NS(create=self.create)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        text = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return NS(content=[NS(type="text", text=text)], stop_reason="end_turn")


GOOD = json.dumps({"verdicts": [{"verdict": "stated", "reason": "a"}, {"verdict": "stated", "reason": "b"},
                                {"verdict": "not_addressed", "reason": "c"}]})


def test_judge_answer_sends_the_pinned_request():
    c = FakeClient(GOOD)
    out = judge_answer(c, "How does frailty relate to stroke?", FRAILTY, "Frailty raises stroke risk in a graded way.")
    k = c.calls[0]
    assert k["model"] == "claude-sonnet-5" and k["max_tokens"] == 8192
    assert k["system"] == [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]
    assert k["messages"] == [{"role": "user", "content": build_prompt("How does frailty relate to stroke?", FRAILTY,
                                                                      "Frailty raises stroke risk in a graded way.")}]
    assert not {"tools", "tool_choice", "temperature", "thinking"} & set(k)
    assert len(out["claims"]) == 3 and out["verdicts"][2]["verdict"] == "not_addressed"
    assert out["verdicts"][0]["claim"].startswith("Greater frailty burden")
    assert round(out["coverage"], 2) == 0.67 and round(out["correctness"], 2) == 0.80


def test_judge_answer_retries_on_wrong_count():
    short = json.dumps({"verdicts": [{"verdict": "stated", "reason": "a"}]})
    c = FakeClient(short, GOOD)
    out = judge_answer(c, "q", FRAILTY, "a")
    assert len(c.calls) == 2 and len(out["verdicts"]) == 3


def test_judge_answer_raises_after_attempts():
    c = FakeClient("not json at all")
    with pytest.raises(ValueError, match=f"after {MAX_ATTEMPTS} attempts"):
        judge_answer(c, "q", FRAILTY, "a")
    assert len(c.calls) == MAX_ATTEMPTS


from src.judge import judge_log, write_rows  # noqa: E402
import src.judge  # noqa: E402

LOG = {"arms": ["websearch", "connect"], "rows": [
    {"id": "q1", "question": "Q1?", "gold": FRAILTY,
     "answers": {"websearch": "Frailty raises stroke risk.", "connect": "Graded risk with frailty."},
     "traces": {"websearch": {"searches": []}, "connect": {"searches": []}}, "excluded": None},
    {"id": "q2", "question": "Q2?", "gold": FRAILTY,
     "answers": {"websearch": "", "connect": "x"},
     "traces": {"websearch": {"searches": []}, "connect": {"searches": []}}, "excluded": "refusal"},
]}


def fake_judge(question, gold, answer):
    claims = required_claims(gold)
    v = [{"claim": c, "verdict": "stated", "reason": "ok"} for c in claims]
    return {"claims": claims, "verdicts": v, **score_verdicts(v)}


def test_judge_log_rows_have_the_verdicts_shape():
    rows = judge_log(fake_judge, LOG, "sonnet5")
    assert [(r["id"], r["arm"], r["judge"]) for r in rows] == [("q1", "websearch", "sonnet5"), ("q1", "connect", "sonnet5")]
    r = rows[0]
    assert set(r) == {"id", "arm", "judge", "verdicts", "coverage", "contradiction_rate", "correctness"}
    assert [v["claim_index"] for v in r["verdicts"]] == [1, 2, 3]
    assert set(r["verdicts"][0]) == {"claim_index", "verdict", "reason"}
    assert r["coverage"] == 1.0 and r["correctness"] == 1.0


def test_judge_log_skips_excluded_and_empty():
    seen = []

    def spy(question, gold, answer):
        seen.append((question, answer))
        return fake_judge(question, gold, answer)

    log = {"arms": ["websearch"], "rows": [
        {"id": "a", "question": "A?", "gold": FRAILTY, "answers": {"websearch": "   "}, "traces": {}, "excluded": None},
        {"id": "b", "question": "B?", "gold": FRAILTY, "answers": {"websearch": "fine"}, "traces": {}, "excluded": "arm_error"},
        {"id": "c", "question": "C?", "gold": FRAILTY, "answers": {"websearch": "judged"}, "traces": {}, "excluded": None}]}
    assert [r["id"] for r in judge_log(spy, log, "t")] == ["c"]
    assert seen == [("C?", "judged")]


def test_judge_log_takes_gold_from_questions():
    log = {"arms": ["websearch"], "rows": [{"id": "q9", "answers": {"websearch": "ans"}, "traces": {}, "excluded": None}]}
    rows = judge_log(fake_judge, log, "t", questions=[{"id": "q9", "question": "Q9?", "gold": FRAILTY}])
    assert rows[0]["id"] == "q9" and len(rows[0]["verdicts"]) == 3
    with pytest.raises(ValueError, match="q9"):
        judge_log(fake_judge, log, "t")


def test_write_rows_is_jsonl(tmp_path):
    p = write_rows(judge_log(fake_judge, LOG, "t"), tmp_path / "out" / "v.jsonl")
    lines = p.read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["id"] == "q1"


def test_main_judges_a_log_file(tmp_path, monkeypatch):
    log_path = tmp_path / "log.json"
    log_path.write_text(json.dumps(LOG))
    monkeypatch.setattr(src.judge, "_load_dotenv", lambda: None)
    monkeypatch.setattr(src.judge, "_build_client", lambda provider, model: (FakeClient(GOOD), model))
    out = tmp_path / "v.jsonl"
    src.judge.main([str(log_path), "--judge-tag", "sonnet5", "--out", str(out)])
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 2 and rows[0]["judge"] == "sonnet5" and rows[0]["verdicts"][2]["verdict"] == "not_addressed"


def test_main_stops_without_key(tmp_path, monkeypatch):
    log_path = tmp_path / "log.json"
    log_path.write_text(json.dumps(LOG))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(src.judge, "_load_dotenv", lambda: None)
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
        src.judge.main([str(log_path)])


def test_main_names_a_missing_log(tmp_path, monkeypatch):
    monkeypatch.setattr(src.judge, "_load_dotenv", lambda: None)
    monkeypatch.setattr(src.judge, "_build_client", lambda provider, model: (FakeClient(GOOD), model))
    with pytest.raises(SystemExit, match="nope.json"):
        src.judge.main([str(tmp_path / "nope.json")])


def test_main_rejects_a_log_without_arms(tmp_path, monkeypatch):
    log_path = tmp_path / "log.json"
    log_path.write_text(json.dumps({"rows": []}))
    monkeypatch.setattr(src.judge, "_load_dotenv", lambda: None)
    monkeypatch.setattr(src.judge, "_build_client", lambda provider, model: (FakeClient(GOOD), model))
    with pytest.raises(SystemExit, match="'arms'"):
        src.judge.main([str(log_path)])


def test_prompt_hash_is_pinned_to_the_published_run():
    import hashlib
    h = hashlib.sha256(build_prompt("Q", FRAILTY, "A").encode()).hexdigest()
    assert h == "43572ccadbdc4d6726c3afac5cc2ba5f13a9b5cc5a705a3b573b5330c87dd4ea"


def test_build_client_ignores_bedrock_model_id(monkeypatch):
    monkeypatch.setenv("BEDROCK_MODEL_ID", "eu.anthropic.claude-opus-5")
    recorded_env = {}

    def fake_build_client(provider, model, env):
        recorded_env.update(env)
        return object(), env.get("BEDROCK_MODEL_ID") or "mapped"

    monkeypatch.setattr("harness.runner.build_client", fake_build_client)
    _, served = src.judge._build_client("bedrock", "claude-sonnet-5")
    assert served == "mapped"
    assert "BEDROCK_MODEL_ID" not in recorded_env
