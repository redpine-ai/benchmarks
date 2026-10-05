"""The second judge: TypeSafe Jev through OpenRouter's decisions endpoint.

One request per answer, carrying one choice question per required claim, each with
the three verdict labels as its choices and the claim-coverage prompt's definitions as
its instructions. The answers are assembled into the same dict shape as
src.judge.judge_answer, with reasons null because Jev records none. The probabilities
and confidences in the reply are read for the label only and never stored.

    uv run python -m src.jev logs/expert_qa_20260915.json --judge-tag jev1
"""
import json
import os
import sys
import time
from pathlib import Path

from .judge import VERDICTS, judge_log, required_claims, score_verdicts, scope_of, validate_verdicts, write_rows

ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.13"
TIMEOUT_S = 120

CRITERIA = {
    "stated": ("the candidate answer conveys the claim, including its direction of effect and any "
               "comparison or ranking the claim makes"),
    "contradicted": ("the candidate answer asserts the opposite of the claim: a reversed direction of "
                     "effect, a reversed ranking, or the claim applied to a population, condition or "
                     "care setting outside the stated scope without qualification"),
    "not_addressed": ("the candidate answer is silent on the claim, or addresses a different claim "
                      "instead. Silence is not contradiction"),
}

INSTRUCTIONS = (
    "Label the REQUIRED CLAIM below against the CANDIDATE ANSWER in the state. Judge this claim "
    "independently of the other claims.\n\n"
    "Judge at the semantic level. The claim is stated if the answer conveys the same meaning in "
    "different words, in a different order, split across several sentences, or with more specific "
    "detail than the claim carries. Do not require lexical overlap. A claim counts as stated even "
    "when the answer cites no source and names no study; attribution is not part of this judgement. "
    "Require the direction of effect: if the claim says a factor raises a risk and the answer says it "
    "lowers that risk, that is contradicted; if the answer says only that the two are related, that "
    "is not_addressed. Silence is not contradiction. Do not penalize content beyond the required "
    "claims, do not judge whether the claim is true, and do not reward length, fluency, hedging or "
    "formatting.\n\n"
    "REQUIRED CLAIM: {claim}"
)


def build_request(question: str, scope: str, claims: list[str], answer: str, model: str = MODEL) -> dict:
    state = (f"QUESTION:\n{question or '(not provided)'}\n\n"
             f"SCOPE OF THE REQUIRED CLAIMS:\n{scope or '(none stated)'}\n\n"
             f"CANDIDATE ANSWER:\n{answer}")
    questions = {f"c{i}": {"type": "choice", "instructions": INSTRUCTIONS.format(claim=c),
                           "criteria": dict(CRITERIA)}
                 for i, c in enumerate(claims, 1)}
    return {"model": model, "state": state, "questions": questions}


def labels_of(response: dict, n_claims: int) -> list[str]:
    """One label per claim from a decisions reply; a missing answer is an error, never a default."""
    answers = response.get("answers") if isinstance(response, dict) else None
    if not isinstance(answers, dict):
        raise ValueError("Jev reply carries no answers")
    out = []
    for i in range(1, n_claims + 1):
        a = answers.get(f"c{i}")
        if not isinstance(a, dict) or "choice" not in a:
            raise ValueError(f"Jev reply has no choice for c{i}")
        out.append(a["choice"])
    return out


def jev_answer(post, question: str, gold: str, answer: str, model: str = MODEL) -> dict:
    claims = required_claims(gold)
    reply = post(build_request(question, scope_of(gold), claims, answer, model))
    raw = [{"verdict": label, "reason": None} for label in labels_of(reply, len(claims))]
    verdicts = [{"verdict": v["verdict"], "reason": None} for v in validate_verdicts(raw, len(claims))]
    return {"claims": claims, "verdicts": [{"claim": c, **v} for c, v in zip(claims, verdicts)],
            **score_verdicts(verdicts)}


RETRIES = 6

def http_post(key: str, sleep=time.sleep):
    import httpx

    def post(payload: dict) -> dict:
        last = None
        for attempt in range(RETRIES):
            try:
                r = httpx.post(ENDPOINT, json=payload, timeout=TIMEOUT_S,
                               headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            except httpx.TransportError as e:
                last = f"{type(e).__name__}: {e}"
            else:
                if r.status_code == 200:
                    return r.json()
                last = f"http {r.status_code}"
                if r.status_code != 429 and r.status_code < 500:
                    raise ValueError(f"Jev request failed: {last}")
            delay = 2.0 ** attempt
            sleep(60.0 if delay > 60.0 else delay)
        raise ValueError(f"Jev request failed after {RETRIES} attempts: {last}")

    return post


def main(argv=None):
    import argparse
    from dotenv import load_dotenv
    p = argparse.ArgumentParser(description="Judge every answer in a run log with Jev, one request per answer.")
    p.add_argument("log")
    p.add_argument("--judge-tag", default="jev1")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--out", default=None)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY is required for the Jev judge")
    try:
        log = json.loads(Path(args.log).read_text(encoding="utf-8"))
    except OSError as e:
        sys.exit(f"cannot read log {args.log}: {e.strerror or e}")
    except json.JSONDecodeError as e:
        sys.exit(f"{args.log} is not valid JSON: {e.msg}")
    if not isinstance(log, dict) or "rows" not in log or "arms" not in log:
        sys.exit(f"{args.log}: a run log needs 'arms' and 'rows' at the top level")
    if args.limit:
        log = {**log, "rows": log["rows"][:args.limit]}
    from .data import load_questions
    post = http_post(key)
    rows = judge_log(lambda q, g, a: jev_answer(post, q, g, a, model=args.model), log, args.judge_tag, load_questions())
    out = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "results" / f"verdicts.{args.judge_tag}.jsonl"
    write_rows(rows, out)
    print("wrote", out, len(rows), "rows")


if __name__ == "__main__":
    main()
