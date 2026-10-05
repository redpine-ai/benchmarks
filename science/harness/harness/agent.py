"""One question through one arm: a tool-use loop with a fixed budget.

The first turn of an arm with a forced tool is pinned to that tool with
`tool_choice`; after that the model decides. Every `tool_use` block is charged
against one budget shared by all tools; a block past the budget receives the
config's exhausted message instead of a result. If the model is still calling
tools when the turns run out, one text-only turn is forced. An arm with no
tools is a single plain call.

The scored text follows the config's `final_text` policy: `all_turns` joins the
text of every turn with newlines (the SciFact runs); `last_turn` keeps only the
final turn's text (the expert-question runs).
"""
from .config import RunConfig


def _text(resp, sep: str) -> str:
    return sep.join(b.text for b in (resp.content or []) if getattr(b, "type", "") == "text")


def empty_trace(error: str) -> dict:
    return {"stop_reason": None, "error": error, "steps_used": 0, "tool_calls": 0,
            "budget_refusals": 0, "forced_final": False, "searches": []}


def ask(client, cfg: RunConfig, arm: str, question: str, dispatch,
        model: str | None = None) -> tuple[str, dict]:
    """`model` is the id the provider serves (a Bedrock inference profile, for example);
    it defaults to the config's model name, which the Anthropic API accepts as is."""
    tools = cfg.tool_schemas(arm)
    force = cfg.arms[arm].force
    sep = "\n" if cfg.final_text == "all_turns" else ""
    messages = [{"role": "user", "content": cfg.prompt(question)}]
    kwargs = {"model": model or cfg.model, "max_tokens": cfg.max_tokens, "system": cfg.system(arm),
              "thinking": cfg.thinking_param()}
    searches, text_parts = [], []
    last_turn_text = ""
    steps = calls = refusals = 0
    answered = forced_final = False
    last_stop = None

    def trace() -> dict:
        return {"stop_reason": last_stop, "error": None, "steps_used": steps, "tool_calls": calls,
                "budget_refusals": refusals, "forced_final": forced_final, "searches": searches}

    if not tools:
        resp = client.messages.create(messages=list(messages), **kwargs)
        steps, last_stop = 1, resp.stop_reason
        return _text(resp, sep), trace()

    kwargs["tools"] = tools
    for step in range(cfg.budget):
        steps += 1
        turn_kwargs = kwargs
        if force and step == 0:
            turn_kwargs = {**kwargs, "tool_choice": {"type": "tool", "name": force}}
        resp = client.messages.create(messages=list(messages), **turn_kwargs)
        last_stop = resp.stop_reason
        last_turn_text = _text(resp, sep)
        if last_turn_text:
            text_parts.append(last_turn_text)
        if resp.stop_reason != "tool_use":
            answered = True
            break
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for block in resp.content:
            if getattr(block, "type", "") != "tool_use":
                continue
            tool_input = block.input or {}
            refused = calls >= cfg.budget
            if refused:
                refusals += 1
                out, meta = cfg.budget_exhausted, []
            else:
                calls += 1
                out, meta = dispatch(block.name, tool_input)
            searches.append({"tool": block.name, "step": steps, "query": tool_input.get("query", ""),
                             "n_results": len(meta), "chars": len(out), "budget_refused": refused})
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": out})
        messages.append({"role": "user", "content": results})

    if not answered:
        forced_final = True
        messages.append({"role": "user", "content": cfg.final_nudge})
        resp = client.messages.create(messages=list(messages), **{**kwargs, "tool_choice": {"type": "none"}})
        last_stop = resp.stop_reason
        last_turn_text = _text(resp, sep)

    if cfg.final_text == "last_turn":
        return last_turn_text, trace()
    joined = "\n".join(text_parts)
    if forced_final:
        return (last_turn_text or joined), trace()
    return joined, trace()
