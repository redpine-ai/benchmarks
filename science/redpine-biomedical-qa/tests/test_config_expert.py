from pathlib import Path

from harness.config import load_config

CFG = load_config(Path(__file__).resolve().parents[1] / "configs" / "expert_qa.toml")


def test_settings():
    assert CFG.path == "configs/expert_qa.toml" and "/Users/" not in CFG.path
    assert CFG.model == "claude-opus-5" and CFG.budget == 20 and CFG.max_tokens == 8192
    assert CFG.thinking == "adaptive" and CFG.renderer == "handles" and CFG.final_text == "last_turn"
    assert CFG.arm_order == ["websearch", "connect"]
    assert CFG.arms["websearch"].force is None and CFG.arms["connect"].force is None
    assert CFG.arms["connect"].tools == ("search_connect", "search_web")


def test_system_prompt_shape():
    s = CFG.system("connect")
    assert s.startswith("You are a biomedical research assistant. Answer the user's question with "
                        "tool-grounded evidence and inline citations.\n\nTools available to you:\n- search_connect:")
    assert "\n- search_web: public web search; returns an extract of each page with its URL.\n\nHOW TO SEARCH:\n" in s
    assert "You have a budget of 20 tool calls for the whole question -- the same budget regardless of which tools you have." in s
    assert s.endswith("State your conclusion in the first sentence, then the supporting evidence. Be concise. "
                      "Do not restate the question. Do not pad.")
    w = CFG.system("websearch")
    assert "search_connect" not in w and w.replace(CFG.arms["websearch"].tool_sentence, "") == \
        s.replace(CFG.arms["connect"].tool_sentence, "")


def test_user_prompt_and_tools():
    assert CFG.prompt("Q?") == ("Question: Q?\n\nBegin by calling at least one search tool to gather evidence. "
                                "After you have enough grounded evidence, give a concise final answer with "
                                "inline citations. Every claim must be backed by a citation drawn from actual "
                                "tool output -- never fabricate identifiers.")
    schemas = {t["name"]: t for t in CFG.tool_schemas("connect")}
    assert set(schemas["search_connect"]["input_schema"]["properties"]) == {"query", "limit", "filters"}
    assert schemas["search_connect"]["input_schema"]["properties"]["limit"]["maximum"] == 30
    assert schemas["search_web"]["input_schema"]["properties"]["max_results"]["maximum"] == 10
    assert CFG.final_nudge.startswith("You have reached the search step cap.")
