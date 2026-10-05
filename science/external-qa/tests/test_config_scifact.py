"""The config must reproduce the published log headers exactly."""
import json
from pathlib import Path

from harness.config import load_config

TRACK = Path(__file__).resolve().parents[1]
CFG = load_config(TRACK / "configs" / "scifact.toml")
LOGS = sorted((TRACK / "logs").glob("scifact_*.json"))

HEADER_KEYS = ("model", "tool_budget", "max_tokens", "thinking", "redpine_top_k", "web_top_k",
               "passage_char_limit", "web_search_backend", "tool_sentences", "tool_descriptions")


def test_config_path_is_relative():
    assert CFG.path == "configs/scifact.toml" and "/Users/" not in CFG.path


def test_config_matches_every_published_header():
    assert len(LOGS) == 5
    header = CFG.header()
    for path in LOGS:
        log = json.loads(path.read_text())
        for key in HEADER_KEYS:
            assert header[key] == log[key], (path.name, key)
        assert log["arms"] == CFG.arm_order


def test_arms_differ_in_exactly_the_tool_clause():
    systems = {arm: CFG.system(arm) for arm in CFG.arm_order}
    for arm, system in systems.items():
        sentence = CFG.arms[arm].tool_sentence.replace("{budget}", "16")
        assert sentence in system
        assert system.replace(sentence, "{x}") == systems["closed_book"].replace(
            CFG.arms["closed_book"].tool_sentence, "{x}")


def test_forced_tools_and_prompt():
    assert CFG.arms["closed_book"].force is None and CFG.tool_schemas("closed_book") == []
    assert CFG.arms["web_forced"].force == "web_search"
    assert CFG.arms["redpine_forced"].force == "search_connect"
    assert CFG.arms["redpine_first"].force == "search_connect"
    assert [t["name"] for t in CFG.tool_schemas("redpine_first")] == ["search_connect", "web_search"]
    assert CFG.prompt("X is Y.") == "Claim: X is Y.\n\nStart with: Answer: <true or false> -- then explain briefly."
    assert CFG.budget_exhausted == ("Tool budget exhausted: no tool calls remain. Answer now from what "
                                    "you already have.")


def test_prompt_and_tool_strings_are_pinned():
    assert CFG.system_template == (
        "You are verifying a scientific claim against the biomedical literature (ScholarQABench "
        "SciFact). Decide whether the claim is supported (true) or refuted (false) by the evidence. "
        "{tool_sentence} When a retrieved passage supports a statement you make, cite it inline with "
        "its bracketed number exactly as shown, for example [0] or [2]. Passage numbering starts at 0. "
        "Cite only numbers that appear in the passages you were given. Start your response with a line "
        "in exactly this form: Answer: <true or false> -- then explain your reasoning afterward. Put "
        "only the single word true or false on the Answer line.")
    assert CFG.final_nudge == ("You are out of tool calls. Give your final answer now, based on what "
                               "you have found so far.")
    assert CFG.budget_exhausted == ("Tool budget exhausted: no tool calls remain. Answer now from what "
                                    "you already have.")
    assert CFG.user_template == "Claim: {question}\n\nStart with: Answer: <true or false> -- then explain briefly."
    schema = {"type": "object", "properties": {"query": {"type": "string", "description": "The search query."}},
             "required": ["query"]}
    assert CFG.tools["web_search"].input_schema == schema
    assert CFG.tools["search_connect"].input_schema == schema
