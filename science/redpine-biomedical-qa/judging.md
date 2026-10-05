# Judging

A second, fixed model, Claude Sonnet 5 (`claude-sonnet-5`; `eu.anthropic.claude-sonnet-5` on
Amazon Bedrock), judges every answer the same way for both arms. It reads the question, the
population and setting from the gold, and the required claims, and says of each claim
whether the answer states it, contradicts it, or does not address it. `src/judge.py` holds
the prompt verbatim and the arithmetic.

```bash
uv run python -m src.judge logs/expert_qa_20260915.json --provider bedrock --judge-tag sonnet5
uv run python -m src.judge runs/<my log>.json --provider anthropic --model claude-sonnet-5
uv run python -m src.jev logs/expert_qa_20260915.json --judge-tag jev1
```

Each command writes one JSONL row per question and arm in the shape of
`results/verdicts.jsonl`. Output goes to `results/verdicts.<judge-tag>.jsonl` unless
`--out` names another file.

## The request

The request is the one that produced the published verdicts.

- System prompt, sent with ephemeral cache control:
  `You are an impartial evaluation judge. When asked for JSON, reply with JSON only.`
- User prompt: `PROMPT` in `src/judge.py`, filled with the question, the scope, the
  numbered required claims and the answer, followed by the verdict schema as text
  (`SCHEMA_SUFFIX`). `uv run python -c "import json; from src.judge import build_prompt; q = json.loads(open('data/questions.jsonl').readline()); print(build_prompt(q['question'], q['gold'], 'A'))"`
  prints one.
- `max_tokens` 8192. No temperature, no thinking parameter: the model's defaults.
- The reply is parsed as JSON (code fences and surrounding prose are dropped). It must hold
  `verdicts`, a list with exactly one `{verdict, reason}` per required claim, each
  `verdict` one of `stated`, `contradicted`, `not_addressed`. A reply that fails this is
  retried, up to three attempts; if the third also fails, the command stops with an error
  and writes no rows, so no verdict is ever invented for an answer.

Porting the judge to another model means writing one function that sends those two
strings (the system prompt and the user prompt) and returns the reply text, then calling
`parse_json_reply`, `validate_verdicts` and `score_verdicts` from `src/judge.py`.
`src/jev.py` shows the other route: a judge with its own request shape that returns one
label per claim and reuses `validate_verdicts` and `score_verdicts`.

The required claims are parsed from the gold, not chosen by the judge: everything after
"A strong answer conveys:" up to "Certainty:", split on semicolons.

## The scores

Three scores follow. With n required claims, of which the answer states s and contradicts c:

- claim coverage C = s / n, the primary score;
- contradiction rate K = c / n;
- correctness = 2 C (1 - K) / (C + (1 - K)), the harmonic mean of the share of claims stated
  and the share not contradicted.

An answer that states every claim and contradicts none scores 1; one that states half the
claims and contradicts the rest scores 0.5.

Worked example, from the report. A cardiology question about frailty burden and stroke risk
has three required claims. One web-search answer states the first two and does not address
the third: coverage 2/3 = 0.67, contradiction rate 0, correctness 2 x 0.67 x 1 / (0.67 + 1)
= 0.80.

## The second judge

TypeSafe Jev 1.13, reached through OpenRouter, scored the same stored answers with one
choice question per required claim, the same three labels and the same definitions; its
scores are reported beside Sonnet 5's as a robustness check, never as the primary result.
Jev judged every answer twice. Its reported scores average the two passes per question; its
verdict labels ship for both passes, as `jev1` and `jev2` in `results/verdicts.jsonl`, and
pass 1 is the one compared with Sonnet 5. Jev also returns per-label probabilities and a
confidence; they are not published because no reported number uses them. It records no
reason, so its reasons are null.

## What the judges saw

The published verdicts judged each answer as written, citations included. The answers in
`logs/` have quoted passage text replaced by `[passage text removed]`, so rejudging them can
differ.
