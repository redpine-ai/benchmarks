"""Run a set of questions through the configured arms and write one log.

Credentials are checked before anything is downloaded or called. A row is
excluded from every arm's score when any arm errored or the model refused with
no text; the counts are recorded under `excluded`. Logs store every answer and
a per-arm trace of tool calls (tool, query, result count, characters), never
the tool results (an answer may quote them).
"""
import datetime as dt
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Mapping

from . import agent
from .clients import RedpineConfig
from .config import RunConfig

BEDROCK_MODEL_IDS = {"claude-sonnet-5": "eu.anthropic.claude-sonnet-5",
                     "claude-opus-5": "eu.anthropic.claude-opus-5"}
REDPINE_API_URL = "https://api.redpine.ai"
ANTHROPIC_TIMEOUT_S = 180
ANTHROPIC_MAX_RETRIES = 3


def check_credentials(cfg: RunConfig, arms: list[str], provider: str,
                      env: Mapping[str, str]) -> tuple[RedpineConfig | None, str | None]:
    if provider == "anthropic" and not (env.get("ANTHROPIC_API_KEY") or env.get("ANTHROPIC_AUTH_TOKEN")):
        sys.exit("ANTHROPIC_API_KEY is required (or use --provider bedrock)")
    if provider == "bedrock" and not env.get("AWS_BEARER_TOKEN_BEDROCK"):
        try:
            import boto3  # noqa: F401
        except ImportError:
            sys.exit("--provider bedrock needs AWS_BEARER_TOKEN_BEDROCK, or AWS credentials with "
                     "the bedrock extra: uv sync --extra bedrock")
    kinds = {cfg.tools[t].kind for a in arms for t in cfg.arms[a].tools}
    redpine = web = None
    if "redpine" in kinds:
        key = env.get("REDPINE_API_KEY")
        if not key:
            sys.exit("REDPINE_API_KEY is required for the Redpine Science arms (sign up at https://app.redpine.ai)")
        redpine = RedpineConfig(url=env.get("REDPINE_API_URL", REDPINE_API_URL), key=key,
                                collection=env.get("REDPINE_COLLECTION", "Redpine Science"))
    if "web" in kinds:
        web = env.get("TAVILY_API_KEY")
        if not web:
            sys.exit("TAVILY_API_KEY is required for the web search arms")
    return redpine, web


def build_client(provider: str, model: str, env: Mapping[str, str]):
    import anthropic
    if provider == "bedrock":
        served = env.get("BEDROCK_MODEL_ID") or BEDROCK_MODEL_IDS.get(model, model)
        region = env.get("AWS_REGION", "eu-west-1")
        return anthropic.AnthropicBedrock(aws_region=region, timeout=ANTHROPIC_TIMEOUT_S,
                                          max_retries=ANTHROPIC_MAX_RETRIES), served
    return anthropic.Anthropic(timeout=ANTHROPIC_TIMEOUT_S, max_retries=ANTHROPIC_MAX_RETRIES), model


def _refused(trace: dict, answer: str) -> bool:
    return trace.get("stop_reason") == "refusal" and not (answer or "").strip()


def _incomplete_reason(row: dict, arms: list[str]) -> str | None:
    for arm in arms:
        if row["traces"][arm].get("error"):
            return "arm_error"
        if _refused(row["traces"][arm], row["answers"].get(arm, "")):
            return "refusal"
    return None


def score(rows: list[dict], arms: list[str]) -> tuple[dict | None, dict]:
    excluded = {"arm_error": 0, "refusal": 0}
    kept = []
    for row in rows:
        reason = _incomplete_reason(row, arms)
        if reason:
            excluded[reason] += 1
        else:
            kept.append(row)
    if not rows or "judge" not in rows[0]:
        return None, excluded
    scores = {}
    for arm in arms:
        correct = sum(1 for r in kept if r["judge"][arm]["official_correct"])
        scores[arm] = round(100 * correct / len(kept), 1) if kept else None
    return scores, excluded


def run_arms(items: list[dict], cfg: RunConfig, *, client, provider: str, served_model_id: str,
             arms: list[str], dispatch_factory: Callable[[str], Callable], judge, workers: int,
             benchmark: str, redpine_collection: str = "Redpine Science") -> dict:
    def one(item):
        answers, traces, judged = {}, {}, {}
        for arm in arms:
            try:
                text, trace = agent.ask(client, cfg, arm, item["question"], dispatch_factory(arm),
                                        model=served_model_id)
            except Exception as e:  # noqa: BLE001  (boundary: any API failure becomes an arm error)
                text, trace = "", agent.empty_trace(f"{type(e).__name__}: {e}")
            answers[arm], traces[arm] = text, trace
            if judge is not None:
                judged[arm] = judge(item, arm, text)
        row = {**item, "answers": answers}
        if judge is not None:
            row["judge"] = judged
        row["traces"] = traces
        row["excluded"] = _incomplete_reason(row, arms)
        return row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(one, items))
    scores, excluded = score(rows, arms)
    return {
        "benchmark": benchmark,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "provider": provider, "served_model_id": served_model_id,
        "redpine_collection": redpine_collection, "arms": list(arms),
        **cfg.header(),
        "pool_size": len(items), "limit": len(items),
        "n_complete": len(items) - sum(excluded.values()), "excluded": excluded,
        "scores": scores, "rows": rows,
    }


def write_log(log: dict, out_dir: Path, prefix: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{prefix}_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(log, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def report(log: dict, path: Path) -> None:
    print(f"n={log['n_complete']} of {log['pool_size']}  excluded={log['excluded']}")
    for arm, s in (log["scores"] or {}).items():
        print(f"  {arm:15s} {s}")
    print("wrote", path)
    if log["pool_size"] and log["n_complete"] == 0:
        print(f"warning: every item was excluded, excluded={log['excluded']}", file=sys.stderr)
        sys.exit(1)
