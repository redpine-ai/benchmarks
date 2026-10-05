import json
from types import SimpleNamespace as NS

import pytest

from harness.config import load_config
from harness.runner import check_credentials, run_arms, score, write_log

Q1 = '''
model = "claude-sonnet-5"
provider = "anthropic"
max_tokens = 8192
thinking = "adaptive"
budget = 16
top_k = 10
passage_chars = 4000
renderer = "indexed"
final_text = "all_turns"
system_template = "s {tool_sentence}"
user_template = "{question}"
budget_exhausted = "b"
final_nudge = "f"
arm_order = ["closed_book", "redpine_forced", "web_forced"]
[tools.search_connect]
kind = "redpine"
description = "dc"
input_schema = { type = "object" }
[tools.web_search]
kind = "web"
description = "dw"
input_schema = { type = "object" }
[arms.closed_book]
tools = []
tool_sentence = "none"
[arms.redpine_forced]
tools = ["search_connect"]
force = "search_connect"
tool_sentence = "one"
[arms.web_forced]
tools = ["web_search"]
force = "web_search"
tool_sentence = "web"
'''


@pytest.fixture
def cfg(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text(Q1)
    return load_config(p)


def trace(stop="end_turn", error=None):
    return {"stop_reason": stop, "error": error, "steps_used": 1, "tool_calls": 0,
            "budget_refusals": 0, "forced_final": False, "searches": []}


def test_score_uses_complete_rows_only():
    arms = ["a", "b"]
    rows = [
        {"judge": {"a": {"official_correct": True}, "b": {"official_correct": True}},
         "traces": {"a": trace(), "b": trace()}, "answers": {"a": "x", "b": "y"}},
        {"judge": {"a": {"official_correct": False}, "b": {"official_correct": True}},
         "traces": {"a": trace(), "b": trace()}, "answers": {"a": "x", "b": "y"}},
        {"judge": {"a": {"official_correct": True}, "b": {"official_correct": False}},
         "traces": {"a": trace(), "b": trace(error="boom")}, "answers": {"a": "x", "b": ""}},
        {"judge": {"a": {"official_correct": False}, "b": {"official_correct": False}},
         "traces": {"a": trace(stop="refusal"), "b": trace()}, "answers": {"a": "", "b": "y"}},
    ]
    scores, excluded = score(rows, arms)
    assert scores == {"a": 50.0, "b": 100.0} and excluded == {"arm_error": 1, "refusal": 1}


def test_score_without_judge():
    rows = [{"traces": {"a": trace()}, "answers": {"a": "x"}}]
    assert score(rows, ["a"]) == (None, {"arm_error": 0, "refusal": 0})


def test_check_credentials_messages(cfg):
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
        check_credentials(cfg, ["closed_book"], "anthropic", {})
    with pytest.raises(SystemExit, match="REDPINE_API_KEY"):
        check_credentials(cfg, ["redpine_forced"], "anthropic", {"ANTHROPIC_API_KEY": "a"})
    with pytest.raises(SystemExit, match="TAVILY_API_KEY"):
        check_credentials(cfg, ["web_forced"], "anthropic", {"ANTHROPIC_API_KEY": "a"})
    redpine, web = check_credentials(cfg, ["closed_book"], "anthropic", {"ANTHROPIC_API_KEY": "a"})
    assert redpine is None and web is None
    redpine, web = check_credentials(cfg, ["redpine_forced", "web_forced"], "anthropic",
                                     {"ANTHROPIC_API_KEY": "a", "REDPINE_API_KEY": "r", "TAVILY_API_KEY": "t"})
    assert redpine.key == "r" and redpine.url == "https://api.redpine.ai" and web == "t"


class FakeClient:
    def __init__(self, raise_on_tools=False):
        self.messages = NS(create=self.create)
        self.raise_on_tools = raise_on_tools

    def create(self, **kwargs):
        if kwargs.get("tools"):
            if self.raise_on_tools:
                raise RuntimeError("boom")
            if not any(m["role"] == "assistant" for m in kwargs["messages"]):
                return NS(stop_reason="tool_use", content=[
                    NS(type="tool_use", id="i", name=kwargs["tools"][0]["name"], input={"query": "q"})])
        return NS(stop_reason="end_turn", content=[NS(type="text", text="Answer: true [0]")])


def judge(item, arm, text):
    return {"official_correct": item["gold"] in text}


def test_run_arms_log_shape(cfg, tmp_path):
    items = [{"id": "x0", "question": "A.", "gold": "true"}, {"id": "x1", "question": "B.", "gold": "false"}]
    seen = []

    def dispatch_factory(arm):
        def dispatch(name, tool_input):
            seen.append((arm, name))
            return "[0] p", [{"title": "t"}]
        return dispatch

    log = run_arms(items, cfg, client=FakeClient(), provider="anthropic", served_model_id="claude-sonnet-5",
                   arms=["closed_book", "redpine_forced"], dispatch_factory=dispatch_factory, judge=judge,
                   workers=2, benchmark="B")
    assert log["benchmark"] == "B" and log["arms"] == ["closed_book", "redpine_forced"]
    assert log["tool_budget"] == 16 and log["tool_sentences"]["redpine_forced"] == "one"
    assert log["renderer"] == "indexed" and log["config_path"] == f"{tmp_path.name}/c.toml"
    assert log["pool_size"] == 2 and log["n_complete"] == 2
    assert log["scores"] == {"closed_book": 50.0, "redpine_forced": 50.0}
    row = {r["id"]: r for r in log["rows"]}["x1"]
    assert row["gold"] == "false" and row["excluded"] is None
    assert row["judge"] == {"closed_book": {"official_correct": False}, "redpine_forced": {"official_correct": False}}
    assert row["answers"]["redpine_forced"] == "Answer: true [0]"
    assert set(row["traces"]["redpine_forced"]["searches"][0]) == {"tool", "step", "query", "n_results", "chars", "budget_refused"}
    assert sorted(set(seen)) == [("redpine_forced", "search_connect")]
    path = write_log(log, tmp_path / "runs", "scifact")
    assert path.name.startswith("scifact_") and json.loads(path.read_text())["scores"] == log["scores"]


def test_run_arms_without_judge_stores_answers_only(cfg):
    items = [{"id": "x0", "question": "A."}]
    log = run_arms(items, cfg, client=FakeClient(), provider="anthropic", served_model_id="m",
                   arms=["closed_book"], dispatch_factory=lambda arm: None, judge=None, workers=1, benchmark="B")
    assert "judge" not in log["rows"][0] and log["scores"] is None and log["n_complete"] == 1


def test_arm_exception_becomes_arm_error(cfg):
    items = [{"id": "x0", "question": "A.", "gold": "true"}]
    log = run_arms(items, cfg, client=FakeClient(raise_on_tools=True), provider="anthropic", served_model_id="m",
                   arms=["closed_book", "redpine_forced"], dispatch_factory=lambda arm: None, judge=judge,
                   workers=1, benchmark="B")
    row = log["rows"][0]
    assert row["traces"]["redpine_forced"]["error"] == "RuntimeError: boom" and row["answers"]["redpine_forced"] == ""
    assert log["excluded"] == {"arm_error": 1, "refusal": 0} and log["n_complete"] == 0
    assert log["scores"] == {"closed_book": None, "redpine_forced": None}


def test_bedrock_bearer_token_needs_no_boto3(cfg):
    assert check_credentials(cfg, ["closed_book"], "bedrock", {"AWS_BEARER_TOKEN_BEDROCK": "t"}) == (None, None)


def test_bedrock_without_token_or_boto3_exits(cfg, monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_boto3(name, *a, **k):
        if name == "boto3":
            raise ImportError
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_boto3)
    with pytest.raises(SystemExit, match="AWS_BEARER_TOKEN_BEDROCK"):
        check_credentials(cfg, ["closed_book"], "bedrock", {})


def test_run_arms_sends_the_served_model_id(cfg):
    seen = []

    class Client:
        def __init__(self):
            self.messages = NS(create=self.create)

        def create(self, **kwargs):
            seen.append(kwargs["model"])
            return NS(stop_reason="end_turn", content=[NS(type="text", text="Answer: true")])

    run_arms([{"id": "x", "question": "A.", "gold": "true"}], cfg, client=Client(), provider="bedrock",
             served_model_id="eu.anthropic.claude-sonnet-5", arms=["closed_book"], dispatch_factory=lambda a: None,
             judge=None, workers=1, benchmark="B")
    assert seen == ["eu.anthropic.claude-sonnet-5"]
