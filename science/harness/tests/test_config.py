from pathlib import Path

import pytest

from harness.config import load_config

MINIMAL = '''
model = "claude-sonnet-5"
provider = "anthropic"
max_tokens = 8192
thinking = "adaptive"
budget = 3
top_k = 10
passage_chars = 4000
renderer = "indexed"
final_text = "all_turns"
system_template = "Sys. {tool_sentence} Budget {budget}."
user_template = "Q: {question}"
budget_exhausted = "no more"
final_nudge = "answer now"
arm_order = ["none", "one"]

[tools.search_connect]
kind = "redpine"
description = "d1"
input_schema = { type = "object", properties = { query = { type = "string" } }, required = ["query"] }

[arms.none]
tools = []
tool_sentence = "No tools."

[arms.one]
tools = ["search_connect"]
force = "search_connect"
tool_sentence = "You have search_connect, {budget} calls."
'''


def write(tmp_path, text):
    p = tmp_path / "c.toml"
    p.write_text(text)
    return p


def test_loads_and_renders(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL))
    assert cfg.model == "claude-sonnet-5" and cfg.budget == 3 and cfg.renderer == "indexed"
    assert cfg.arm_order == ["none", "one"]
    assert cfg.arms["one"].force == "search_connect" and cfg.arms["none"].force is None
    assert cfg.system("one") == "Sys. You have search_connect, 3 calls. Budget 3."
    assert cfg.prompt("x?") == "Q: x?"
    assert cfg.tool_schemas("none") == []
    assert cfg.tool_schemas("one") == [{"name": "search_connect", "description": "d1",
                                        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}},
                                                         "required": ["query"]}}]
    assert cfg.thinking_param() == {"type": "adaptive"}
    assert cfg.tools["search_connect"].kind == "redpine"


def test_header_keys(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL))
    h = cfg.header()
    assert h["config_path"] == f"{tmp_path.name}/c.toml"
    assert h["model"] == "claude-sonnet-5" and h["tool_budget"] == 3 and h["max_tokens"] == 8192
    assert h["thinking"] == {"type": "adaptive"} and h["redpine_top_k"] == 10 and h["web_top_k"] == 10
    assert h["passage_char_limit"] == 4000 and h["web_search_backend"] == "tavily/advanced (client-side)"
    assert h["tool_sentences"] == {"none": "No tools.", "one": "You have search_connect, {budget} calls."}
    assert h["tool_descriptions"] == {"search_connect": "d1"}
    assert h["system_template"].startswith("Sys.") and h["user_template"] == "Q: {question}"
    assert h["renderer"] == "indexed" and h["final_text"] == "all_turns"


def test_unknown_tool_in_arm(tmp_path):
    bad = MINIMAL.replace('tools = ["search_connect"]', 'tools = ["nope"]')
    with pytest.raises(ValueError, match="arms.one.tools: unknown tool 'nope'"):
        load_config(write(tmp_path, bad))


def test_unknown_renderer(tmp_path):
    with pytest.raises(ValueError, match="renderer"):
        load_config(write(tmp_path, MINIMAL.replace('renderer = "indexed"', 'renderer = "fancy"')))


def test_force_must_be_an_arm_tool(tmp_path):
    bad = MINIMAL.replace('force = "search_connect"', 'force = "search_web"')
    with pytest.raises(ValueError, match="arms.one.force"):
        load_config(write(tmp_path, bad))


def test_arm_order_must_match_arms(tmp_path):
    with pytest.raises(ValueError, match="arm_order"):
        load_config(write(tmp_path, MINIMAL.replace('arm_order = ["none", "one"]', 'arm_order = ["none"]')))


def test_missing_field_named(tmp_path):
    with pytest.raises(ValueError, match="user_template"):
        load_config(write(tmp_path, MINIMAL.replace('user_template = "Q: {question}"\n', "")))
