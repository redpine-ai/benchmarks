from types import SimpleNamespace as NS

from harness.agent import ask, empty_trace
from harness.config import load_config

BASE = '''
model = "m"
provider = "anthropic"
max_tokens = 8192
thinking = "adaptive"
budget = 16
top_k = 10
passage_chars = 4000
renderer = "indexed"
final_text = "{final_text}"
system_template = "s {tool_sentence}"
user_template = "p {question}"
budget_exhausted = "Tool budget exhausted: no tool calls remain. Answer now from what you already have."
final_nudge = "You are out of tool calls. Give your final answer now, based on what you have found so far."
arm_order = ["closed_book", "forced", "free"]
[tools.search_connect]
kind = "redpine"
description = "d"
input_schema = { type = "object" }
[arms.closed_book]
tools = []
tool_sentence = "none"
[arms.forced]
tools = ["search_connect"]
force = "search_connect"
tool_sentence = "one"
[arms.free]
tools = ["search_connect"]
tool_sentence = "one"
'''


def cfg(tmp_path, final_text="all_turns", budget=16):
    p = tmp_path / "c.toml"
    p.write_text(BASE.replace("{final_text}", final_text).replace("budget = 16", f"budget = {budget}"))
    return load_config(p)


def text(t):
    return NS(stop_reason="end_turn", content=[NS(type="text", text=t)])


def tool_use(*queries, text_before=None):
    blocks = ([NS(type="text", text=text_before)] if text_before else []) + [
        NS(type="tool_use", id=f"id{i}", name="search_connect", input={"query": q})
        for i, q in enumerate(queries)]
    return NS(stop_reason="tool_use", content=blocks)


class FakeClient:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []
        self.messages = NS(create=self.create)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def dispatch_ok(name, tool_input):
    return f"[0] passage for {tool_input['query']}", [{"title": "t"}]


def test_closed_book_is_one_plain_call(tmp_path):
    c = FakeClient([text("Answer: true")])
    out, trace = ask(c, cfg(tmp_path), "closed_book", "q", dispatch_ok)
    k = c.calls[0]
    assert out == "Answer: true" and "tools" not in k and "tool_choice" not in k
    assert k["system"] == "s none" and k["messages"] == [{"role": "user", "content": "p q"}]
    assert k["thinking"] == {"type": "adaptive"} and k["max_tokens"] == 8192 and k["model"] == "m"
    assert trace["steps_used"] == 1 and trace["tool_calls"] == 0 and trace["forced_final"] is False


def test_forced_first_call_then_free(tmp_path):
    c = FakeClient([tool_use("q1"), tool_use("q2"), text("Answer: false")])
    out, trace = ask(c, cfg(tmp_path), "forced", "q", dispatch_ok)
    assert c.calls[0]["tool_choice"] == {"type": "tool", "name": "search_connect"}
    assert "tool_choice" not in c.calls[1]
    assert out == "Answer: false"
    assert trace["tool_calls"] == 2 and trace["steps_used"] == 3
    assert trace["searches"][0] == {"tool": "search_connect", "step": 1, "query": "q1", "n_results": 1,
                                    "chars": len("[0] passage for q1"), "budget_refused": False}
    fed = c.calls[1]["messages"][-1]
    assert fed == {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "id0",
                                                "content": "[0] passage for q1"}]}


def test_free_arm_never_forces(tmp_path):
    c = FakeClient([tool_use("q1"), text("done")])
    ask(c, cfg(tmp_path), "free", "q", dispatch_ok)
    assert "tool_choice" not in c.calls[0]


def test_parallel_tool_use_blocks_each_charged(tmp_path):
    c = FakeClient([tool_use("a", "b"), text("Answer: true")])
    _, trace = ask(c, cfg(tmp_path), "free", "q", dispatch_ok)
    assert trace["tool_calls"] == 2 and len(c.calls[1]["messages"][-1]["content"]) == 2


def test_budget_exhausted_forces_final(tmp_path):
    calls = []

    def dispatch(name, tool_input):
        calls.append(tool_input["query"])
        return "r", []

    c = FakeClient([tool_use("q0"), tool_use("q1"), text("Answer: true")])
    out, trace = ask(c, cfg(tmp_path, budget=2), "free", "q", dispatch)
    assert calls == ["q0", "q1"] and out == "Answer: true" and trace["forced_final"] is True
    assert c.calls[-1]["tool_choice"] == {"type": "none"}
    assert c.calls[-1]["messages"][-1] == {"role": "user", "content": cfg(tmp_path).final_nudge}


def test_over_budget_block_gets_exhausted_text(tmp_path):
    c = FakeClient([tool_use("a", "b", "c"), text("Answer: true")])
    _, trace = ask(c, cfg(tmp_path, budget=2), "free", "q", dispatch_ok)
    fed = c.calls[1]["messages"][-1]["content"]
    assert fed[2]["content"] == cfg(tmp_path).budget_exhausted
    assert trace["tool_calls"] == 2 and trace["budget_refusals"] == 1 and trace["searches"][2]["budget_refused"]


def test_all_turns_policy_joins_every_turn(tmp_path):
    c = FakeClient([tool_use("q1", text_before="Looking."), text("Answer: true")])
    out, _ = ask(c, cfg(tmp_path, "all_turns"), "free", "q", dispatch_ok)
    assert out == "Looking.\nAnswer: true"


def test_last_turn_policy_keeps_only_the_final_turn(tmp_path):
    c = FakeClient([tool_use("q1", text_before="Looking."), text("Answer: true")])
    out, _ = ask(c, cfg(tmp_path, "last_turn"), "free", "q", dispatch_ok)
    assert out == "Answer: true"


def test_last_turn_policy_uses_forced_final(tmp_path):
    c = FakeClient([tool_use("q0", text_before="Looking."), text("Final.")])
    out, trace = ask(c, cfg(tmp_path, "last_turn", budget=1), "free", "q", dispatch_ok)
    assert out == "Final." and trace["forced_final"] is True


def test_all_turns_forced_final_replaces_earlier_text(tmp_path):
    c = FakeClient([tool_use("q0", text_before="Looking."), text("Final.")])
    out, trace = ask(c, cfg(tmp_path, "all_turns", budget=1), "free", "q", dispatch_ok)
    assert out == "Final." and trace["forced_final"] is True


def test_all_turns_keeps_earlier_text_when_forced_turn_is_empty(tmp_path):
    c = FakeClient([tool_use("q0", text_before="Looking."), NS(stop_reason="end_turn", content=[])])
    out, _ = ask(c, cfg(tmp_path, "all_turns", budget=1), "free", "q", dispatch_ok)
    assert out == "Looking."


def test_last_turn_empty_when_final_turn_textless(tmp_path):
    c = FakeClient([tool_use("q0", text_before="Looking."), NS(stop_reason="end_turn", content=[])])
    out, _ = ask(c, cfg(tmp_path, "last_turn"), "free", "q", dispatch_ok)
    assert out == ""


def test_refusal_is_recorded(tmp_path):
    c = FakeClient([NS(stop_reason="refusal", content=[])])
    out, trace = ask(c, cfg(tmp_path), "closed_book", "q", dispatch_ok)
    assert out == "" and trace["stop_reason"] == "refusal"


def test_empty_trace():
    t = empty_trace("RuntimeError: boom")
    assert t == {"stop_reason": None, "error": "RuntimeError: boom", "steps_used": 0, "tool_calls": 0,
                 "budget_refusals": 0, "forced_final": False, "searches": []}


def test_model_override_reaches_the_client(tmp_path):
    c = FakeClient([text("Answer: true")])
    ask(c, cfg(tmp_path), "closed_book", "q", dispatch_ok, model="eu.anthropic.claude-sonnet-5")
    assert c.calls[0]["model"] == "eu.anthropic.claude-sonnet-5"
