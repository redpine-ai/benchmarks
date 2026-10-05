# External QA benchmark: ScholarQABench SciFact

Redpine Science against no retrieval and against web search, full agentic answer, on the
SciFact subtask of ScholarQABench. This directory holds the result the report states, the
five run logs behind it, the exact arm definitions, and a runner that reproduces the setup.

## Result

Claude Sonnet 5, 208 claims, mean over five runs:

| arm | claims answered correctly |
| --- | ---: |
| No retrieval | 87.6% |
| Web search | 93.1% |
| Redpine Science | 94.4% |
| Redpine Science then web | 93.9% |

An agent with Redpine Science answers 94.4 percent of the claims correctly, against 87.6
percent for the same agent with no retrieval: 6.8 percentage points more, or 7.8 percent
more claims right, averaged over the five runs and positive in each of them.

A claim is excluded from every arm of a run when the model refused it with no answer in
at least one arm, or when an arm call failed after retries, for example on provider
throttling. Each run scores 196 to 199 claims. Per-run values and exclusion counts are in
[`results/scifact_runs.md`](results/scifact_runs.md); the spread across runs is run-to-run
variation, not sampling error within a run.

Every arm scores above 87 percent. The no-retrieval score is the floor a retrieval system
starts from, and on this benchmark it leaves 12 points to gain; Redpine Science closes over
half of them. A benchmark on which the model already answers nine claims in ten from memory can
show that retrieval helps, but not by how much, and it says nothing about whether the
retrieved papers were the right ones. The report's other three evaluations take those
questions up.

## Why this benchmark

Most public biomedical benchmarks cannot show a retrieval effect: models answer most
questions from memory, and the question text has been public for years. ScholarQABench is a
recent suite of literature tasks, one of the use cases Redpine Science serves. Its SciFact
subtask holds 208 biomedical claims, each with a label saying whether the published
literature supports or refutes it. The agent reads the claim, searches if it has a tool, and
states a verdict.

The name is written in full because the subtask is ScholarQABench's packaging of SciFact
into 208 binary claims; scores are not comparable with the original SciFact release.

## Setup

Four arms, same model, same loop, same system prompt; the prompt differs in one clause, the
list of tools. Web search is Tavily. In each arm with a tool the first call is forced to that
arm's primary tool, so no arm answers from memory alone; after that the agent decides,
within a budget of 16 tool calls. The whole setup is
[`configs/scifact.toml`](configs/scifact.toml); [`arms.md`](arms.md) walks through it against
the log header.

## Scoring

The suite's own rule: strip citation markers, lowercase, and mark the answer correct if the
gold label is a substring of it. No judge model is involved. `src/grade.py` re-derives the
rule from the benchmark's `scripts/citation_correctness_eval.py`. The scored text is the
agent's output across all its turns for the claim, as in the logged runs.

## What is here

- `logs/`: the five run logs. Each keeps the run header, per-arm scores, and per claim the
  claim text, gold label, whether each arm was correct, and the exclusion reason if any.
  Model answers and retrieved text are not included.
- `results/scifact_runs.md`: the five-run table.
- `arms.md`: arm definitions, system prompt, tool descriptions, settings.
- `src/`: the data loader, the scoring rule and the run command. The loop itself is the
  harness in `../harness/`, driven by `configs/scifact.toml`.
- `configs/scifact.toml`: model, budget, prompt text, tool descriptions and schemas, result
  format.
- `tests/`: offline tests, including one that recomputes every score in `logs/` from its rows.

## Running it

```bash
cd external-qa
uv sync                  # add --extra bedrock for --provider bedrock
uv run pytest
cp .env.example .env     # fill in the keys below
uv run python -m src.run --limit 2
uv run python -m src.run
```

| Variable | Needed for | Where to get it |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | the answering model | https://console.anthropic.com; or `--provider bedrock` with AWS credentials (`AWS_REGION` defaults to `eu-west-1`); `BEDROCK_MODEL_ID` overrides the Bedrock model id |
| `AWS_BEARER_TOKEN_BEDROCK` | `--provider bedrock` | an Amazon Bedrock API key from the AWS console; with it set, no AWS login or `--extra bedrock` is needed |
| `REDPINE_API_KEY` | the Redpine Science arms | https://app.redpine.ai |
| `TAVILY_API_KEY` | the web search arms | https://tavily.com |

A missing key stops the run with a one-line message before anything is downloaded or
called. The claims are downloaded from the ScholarQABench repository on first run. A
full run is 208 claims times four arms. Logs land in `runs/`; each has the same row
fields as `logs/` plus the model's answers and a per-arm trace of tool calls, and the
same header fields plus `system_template`, `user_template`, `renderer`, `final_text` and
`config_path`. To compare a rerun with the published runs, read `scores` from your log
and the five in `logs/`; `results/scifact_runs.md` has the published table.

Redpine Science's retrieval stack changes over time, so a rerun measures the stack on its
own date; the logs carry theirs.

## Data and licences

The claims and gold labels in `logs/` are redistributed from ScholarQABench under ODC-BY,
with attribution to ScholarQABench (Asai et al., 2026) and SciFact (Wadden et al., 2020).
SciFact releases its claims and annotations under CC BY 4.0; the 208 claims are from its dev
split. `src/grade.py` re-derives `compute_match` and `remove_citations` from
ScholarQABench's `scripts/citation_correctness_eval.py`, which is MIT (copyright 2024 Akari
Asai); the notice is reproduced in `src/grade.py`. Code in this directory is MIT.
