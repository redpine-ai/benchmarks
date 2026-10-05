"""A run config: everything that differs between the two answer-quality tracks.

Loaded from TOML. Prompt text, tool descriptions, budgets and model settings are
data here so a reader can see one track's whole setup in one file, and diff
two files to see how the tracks differ. Behaviour (the loop, the clients,
the two renderers) stays in code and is selected by name.
"""
import tomllib
from dataclasses import dataclass
from pathlib import Path

RENDERERS = ("indexed", "handles")
FINAL_TEXT = ("all_turns", "last_turn")
THINKING = ("adaptive",)
KINDS = ("redpine", "web")
WEB_SEARCH_BACKEND = "tavily/advanced (client-side)"

_REQUIRED = ("model", "provider", "max_tokens", "thinking", "budget", "top_k", "passage_chars",
             "renderer", "final_text", "system_template", "user_template", "budget_exhausted",
             "final_nudge", "arm_order", "tools", "arms")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    kind: str
    description: str
    input_schema: dict


@dataclass(frozen=True)
class ArmSpec:
    name: str
    tools: tuple[str, ...]
    force: str | None
    tool_sentence: str


@dataclass(frozen=True)
class RunConfig:
    model: str
    provider: str
    max_tokens: int
    thinking: str
    budget: int
    top_k: int
    passage_chars: int
    renderer: str
    final_text: str
    system_template: str
    user_template: str
    budget_exhausted: str
    final_nudge: str
    tools: dict[str, ToolSpec]
    arms: dict[str, ArmSpec]
    arm_order: list[str]
    path: str

    def system(self, arm: str) -> str:
        sentence = self.arms[arm].tool_sentence.replace("{budget}", str(self.budget))
        return (self.system_template.replace("{tool_sentence}", sentence)
                .replace("{budget}", str(self.budget)))

    def prompt(self, question: str) -> str:
        return self.user_template.replace("{question}", question)

    def tool_schemas(self, arm: str) -> list[dict]:
        return [{"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in (self.tools[n] for n in self.arms[arm].tools)]

    def thinking_param(self) -> dict:
        return {"type": self.thinking}

    def header(self) -> dict:
        return {
            "model": self.model, "tool_budget": self.budget, "max_tokens": self.max_tokens,
            "thinking": self.thinking_param(), "redpine_top_k": self.top_k, "web_top_k": self.top_k,
            "passage_char_limit": self.passage_chars, "web_search_backend": WEB_SEARCH_BACKEND,
            "tool_sentences": {a: self.arms[a].tool_sentence for a in self.arm_order},
            "tool_descriptions": {n: t.description for n, t in self.tools.items()},
            "system_template": self.system_template, "user_template": self.user_template,
            "renderer": self.renderer, "final_text": self.final_text, "config_path": self.path,
        }


def _check(cond: bool, message: str) -> None:
    if not cond:
        raise ValueError(message)


def load_config(path: Path) -> RunConfig:
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    for key in _REQUIRED:
        _check(key in raw, f"config {path}: missing field '{key}'")
    _check(raw["renderer"] in RENDERERS, f"renderer must be one of {RENDERERS}, got {raw['renderer']!r}")
    _check(raw["final_text"] in FINAL_TEXT, f"final_text must be one of {FINAL_TEXT}")
    _check(raw["thinking"] in THINKING, f"thinking must be one of {THINKING}")
    _check(raw["budget"] > 0, "budget must be positive")
    tools = {}
    for name, t in raw["tools"].items():
        _check(t.get("kind") in KINDS, f"tools.{name}.kind must be one of {KINDS}")
        tools[name] = ToolSpec(name, t["kind"], t["description"], t["input_schema"])
    arms = {}
    for name, a in raw["arms"].items():
        for tool in a["tools"]:
            _check(tool in tools, f"arms.{name}.tools: unknown tool '{tool}'")
        force = a.get("force")
        _check(force is None or force in a["tools"], f"arms.{name}.force must name one of the arm's tools")
        arms[name] = ArmSpec(name, tuple(a["tools"]), force, a["tool_sentence"])
    _check(sorted(raw["arm_order"]) == sorted(arms), "arm_order must list every arm exactly once")
    return RunConfig(
        model=raw["model"], provider=raw["provider"], max_tokens=int(raw["max_tokens"]),
        thinking=raw["thinking"], budget=int(raw["budget"]), top_k=int(raw["top_k"]),
        passage_chars=int(raw["passage_chars"]), renderer=raw["renderer"], final_text=raw["final_text"],
        system_template=raw["system_template"], user_template=raw["user_template"],
        budget_exhausted=raw["budget_exhausted"], final_nudge=raw["final_nudge"],
        tools=tools, arms=arms, arm_order=list(raw["arm_order"]),
        path=str(Path(*Path(path).parts[-2:])),
    )
