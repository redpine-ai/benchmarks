"""The claim judge: one verdict per required claim, three scores from the verdicts.

A fixed model reads the question, the scope from the gold and the required
claims, and says of each claim whether the answer states it, contradicts it, or
does not address it. Claim coverage is the share stated, the contradiction rate
the share contradicted, and correctness their harmonic combination
2C(1-K) / (C + (1-K)), the report's Equation 1. The claim list is parsed from the
gold's template, never judged by the model, so the verdict count is checkable.

The request is the one that produced the published verdicts: the system prompt
below, the user prompt with the verdict schema appended as text, max_tokens 8192,
no temperature and no thinking parameter. The reply is parsed as JSON; a verdict
count that does not equal the claim count is retried, up to MAX_ATTEMPTS.

    uv run python -m src.judge logs/expert_qa_20260915.json --provider bedrock

judges every answer in a log and writes verdict rows; see judging.md.
"""
import json
import re

VERDICTS = ("stated", "contradicted", "not_addressed")
MAX_TOKENS = 8192
MAX_ATTEMPTS = 3
SYSTEM_PROMPT = "You are an impartial evaluation judge. When asked for JSON, reply with JSON only."
SCHEMA_SUFFIX = ('\n\nReply with ONLY a JSON object that validates against this JSON Schema (same field '
    'names, no extra keys):\n{"$defs": {"_ClaimVerdict": {"properties": {"verdict": {"enum": ["stated", '
    '"contradicted", "not_addressed"], "title": "Verdict", "type": "string"}, "reason": {"title": "Reason", '
    '"type": "string"}}, "required": ["verdict", "reason"], "title": "_ClaimVerdict", "type": "object"}}, '
    '"properties": {"verdicts": {"items": {"$ref": "#/$defs/_ClaimVerdict"}, "title": "Verdicts", "type": '
    '"array"}}, "required": ["verdicts"], "title": "_ClaimVerdicts", "type": "object"}')

_CLAIM_LIST = re.compile(r"A strong answer conveys:\s*(.*?)\.\s*Certainty:", re.S)
_SCOPE = re.compile(r"Scope:\s*(.*?)\.?\s*(?:An answer is wrong|$)", re.S)

PROMPT = (
    "You are assessing a candidate answer to a biomedical question against the required claims "
    "of an expert-verified gold answer. Judge each required claim independently.\n\n"
    "Label each required claim with exactly one of:\n"
    "stated -- the candidate answer conveys the claim, including its direction of effect and "
    "any comparison or ranking the claim makes.\n"
    "contradicted -- the candidate answer asserts the opposite of the claim: a reversed "
    "direction of effect, a reversed ranking, or the claim applied to a population, condition "
    "or care setting outside the stated scope without qualification.\n"
    "not_addressed -- the candidate answer is silent on the claim, or addresses a different "
    "claim instead. Silence is not contradiction.\n\n"
    "Judge at the semantic level. A claim is stated if the answer conveys the same meaning in "
    "different words, in a different order, split across several sentences, or with more "
    "specific detail than the claim carries. Do not require lexical overlap.\n\n"
    "These answers are not tied to specific papers, so a claim counts as stated even when the "
    "answer cites no source and names no study. Attribution is not part of this judgement.\n\n"
    "Require the direction of effect. If a claim says a factor raises a risk and the answer "
    "says it lowers that risk, that is contradicted. If the answer says only that the two are "
    "related, that is not_addressed.\n\n"
    "Do not penalize content that goes beyond the required claims. Do not judge whether a "
    "claim is true; judge whether the answer states it or contradicts it. Do not reward "
    "length, fluency, hedging, or formatting.\n\n"
    "Question:\n{question}\n\n"
    "Scope of the required claims:\n{scope}\n\n"
    "Required claims:\n{claims}\n\n"
    "Candidate answer:\n{actual_output}\n\n"
    "Return one verdict per required claim, in the same order. The number of verdicts MUST "
    "equal the number of required claims ({n_claims})."
)


def required_claims(gold: str) -> list[str]:
    m = _CLAIM_LIST.search(gold or "")
    if not m:
        raise ValueError("gold does not follow the template: no 'A strong answer conveys: ... . "
                         "Certainty:' section to take required claims from")
    claims = [c.strip() for c in m.group(1).split(";") if c.strip()]
    if not claims:
        raise ValueError("gold's required-claim list is empty")
    return claims


def scope_of(gold: str) -> str:
    m = _SCOPE.search(gold or "")
    return m.group(1).strip() if m else ""


def build_prompt(question: str, gold: str, answer: str) -> str:
    claims = required_claims(gold)
    return PROMPT.format(
        question=question or "(not provided)", scope=scope_of(gold) or "(none stated)",
        claims="\n".join(f"{i}. {c}" for i, c in enumerate(claims, 1)),
        actual_output=answer, n_claims=len(claims)) + SCHEMA_SUFFIX


def validate_verdicts(verdicts, n: int) -> list[dict]:
    if not isinstance(verdicts, list) or len(verdicts) != n:
        got = len(verdicts) if isinstance(verdicts, list) else 0
        raise ValueError(f"judge returned {got} verdicts for {n} required claims")
    for v in verdicts:
        if not isinstance(v, dict) or v.get("verdict") not in VERDICTS:
            raise ValueError(f"unknown verdict {v.get('verdict') if isinstance(v, dict) else v!r}")
    return [{"verdict": v["verdict"], "reason": v.get("reason", "")} for v in verdicts]


def score_verdicts(verdicts: list[dict]) -> dict:
    n = len(verdicts)
    coverage = sum(v["verdict"] == "stated" for v in verdicts) / n
    contradiction = sum(v["verdict"] == "contradicted" for v in verdicts) / n
    denominator = coverage + (1 - contradiction)
    correctness = 2 * coverage * (1 - contradiction) / denominator if denominator else 0.0
    return {"coverage": coverage, "contradiction_rate": contradiction, "correctness": correctness}


_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")


def parse_json_reply(text: str) -> dict:
    """The first JSON object in a reply, with code fences and surrounding prose removed."""
    body = _FENCE.sub("", text or "")
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("judge reply carries no JSON object")
    try:
        return json.loads(body[start:end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"judge reply is not valid JSON: {e.msg}") from e


def _reply_text(resp) -> str:
    return "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")


def judge_answer(client, question: str, gold: str, answer: str, model: str = "claude-sonnet-5",
                 attempts: int = MAX_ATTEMPTS) -> dict:
    claims = required_claims(gold)
    prompt = build_prompt(question, gold, answer)
    last = None
    for _ in range(attempts):
        resp = client.messages.create(
            model=model, max_tokens=MAX_TOKENS,
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt}])
        try:
            verdicts = validate_verdicts(parse_json_reply(_reply_text(resp)).get("verdicts"), len(claims))
        except ValueError as e:
            last = e
            continue
        return {"claims": claims, "verdicts": [{"claim": c, **v} for c, v in zip(claims, verdicts)],
                **score_verdicts(verdicts)}
    raise ValueError(f"judge failed after {attempts} attempts: {last}")


def judge_log(judge_fn, log: dict, judge_tag: str, questions=()) -> list[dict]:
    """One verdict row per (question, arm) in a harness log. Rows marked excluded and
    empty answers are skipped. The gold comes from the row, else from `questions` by id."""
    golds = {q["id"]: q for q in questions}
    out = []
    for row in log["rows"]:
        source = row if "gold" in row else golds.get(row["id"])
        if source is None:
            raise ValueError(f"{row['id']}: no gold in the log row or in the question set")
        for arm in log["arms"]:
            answer = (row.get("answers") or {}).get(arm) or ""
            if row.get("excluded") or not answer.strip():
                continue
            j = judge_fn(source["question"], source["gold"], answer)
            out.append({"id": row["id"], "arm": arm, "judge": judge_tag,
                        "verdicts": [{"claim_index": i, "verdict": v["verdict"], "reason": v["reason"]}
                                     for i, v in enumerate(j["verdicts"], 1)],
                        "coverage": j["coverage"], "contradiction_rate": j["contradiction_rate"],
                        "correctness": j["correctness"]})
    return out


def write_rows(rows: list[dict], path):
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def _load_dotenv():
    from pathlib import Path
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _build_client(provider: str, model: str):
    import os
    from harness import runner
    if provider == "anthropic" and not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise SystemExit("ANTHROPIC_API_KEY is required (or use --provider bedrock)")
    env = {k: v for k, v in os.environ.items() if k != "BEDROCK_MODEL_ID"}
    return runner.build_client(provider, model, env)


def main(argv=None):
    import argparse
    import sys
    from pathlib import Path
    p = argparse.ArgumentParser(description="Judge every answer in a run log, one verdict per required claim.")
    p.add_argument("log", help="a log written by src.run (or one in the same shape)")
    p.add_argument("--provider", default="anthropic", choices=["anthropic", "bedrock"])
    p.add_argument("--model", default="claude-sonnet-5")
    p.add_argument("--judge-tag", default="sonnet5", help="the judge column written to each row")
    p.add_argument("--out", default=None, help="output JSONL; default results/verdicts.<judge-tag>.jsonl")
    p.add_argument("--limit", type=int, default=None, help="first N log rows")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    _load_dotenv()
    client, served = _build_client(args.provider, args.model)
    print(f"judge {served} via {args.provider}")
    try:
        text = Path(args.log).read_text(encoding="utf-8")
    except OSError as e:
        sys.exit(f"cannot read log {args.log}: {e.strerror or e}")
    try:
        log = json.loads(text)
    except json.JSONDecodeError as e:
        sys.exit(f"{args.log} is not valid JSON: {e.msg}")
    if not isinstance(log, dict) or "rows" not in log or "arms" not in log:
        sys.exit(f"{args.log}: a run log needs 'arms' and 'rows' at the top level")
    if args.limit:
        log = {**log, "rows": log["rows"][:args.limit]}
    from .data import load_questions
    rows = judge_log(lambda q, g, a: judge_answer(client, q, g, a, model=served), log, args.judge_tag,
                     load_questions())
    out = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "results" / f"verdicts.{args.judge_tag}.jsonl"
    write_rows(rows, out)
    for arm in log["arms"]:
        sub = [r for r in rows if r["arm"] == arm]
        if sub:
            print(f"{arm:9} n {len(sub)} coverage {sum(r['coverage'] for r in sub) / len(sub):.3f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
