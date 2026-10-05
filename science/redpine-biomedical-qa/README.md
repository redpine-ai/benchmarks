# Redpine Biomedical QA

Redpine Science against web search, full agentic answer, on a literature-grounded
question set built from single papers and checked by domain experts. This directory
holds the question set, the runner and its setup, the run log with every model answer
(retrieved passage text removed), the judge code, and the evidence behind the results
below: the per-question judge verdicts and the statistics recomputed from them. The
experts' claim labels are not published.

## Result

All 179 questions, both arms, judged per required claim. Sonnet 5 is the primary judge,
fixed before any expert label was collected; Jev (TypeSafe, via OpenRouter, mean of two
passes) is a second judge over the same answers. Differences are Redpine Science with web
search as fallback minus web search only, paired per question, 95% t intervals.

| Score | Judge | Web search only | + Redpine Science | Difference |
| --- | --- | --- | --- | --- |
| Claim coverage | Sonnet 5 | 0.702 | 0.801 | +0.100 [+0.058, +0.142] |
| | Jev | 0.696 | 0.791 | +0.095 [+0.056, +0.134] |
| Contradiction rate | Sonnet 5 | 0.069 | 0.038 | -0.031 [-0.055, -0.008] |
| | Jev | 0.107 | 0.078 | -0.029 [-0.056, -0.002] |
| Correctness | Sonnet 5 | 0.759 | 0.847 | +0.088 [+0.049, +0.127] |
| | Jev | 0.744 | 0.828 | +0.084 [+0.048, +0.121] |

Per question under Sonnet 5, the Redpine Science arm covers more claims on 62 questions,
fewer on 22, the same on 95. The two judges agree on 89.9% (Redpine Science arm) and 87.0%
(web search arm) of claim verdicts, Krippendorff's alpha 0.70 and 0.72.

**Expert check.** The report also has three domain experts label a blinded sample with the
same three verdicts and compares each judge with them. The experts' labels and every statistic
computed from them are not published, so that comparison is not recomputed here; see the report.

Recomputed from `results/verdicts.jsonl` by `tests/test_verdicts.py` (the score table, the
62 / 22 / 95 split and the per-track differences, through `summary.json`) and
`tests/test_agreement.py` (judge agreement 89.9% and 87.0%, alpha 0.70 and 0.72, through
`agreement.json`). The one-sentence definition of the
agreement statistic is in `results/agreement.json` under `definitions`; Jev means pass 1 in
the comparison with Sonnet 5.

Rejudging `logs/` with `src.judge` reproduces the published verdicts up to two things: the
judges scored the answers before passage text was removed, and the model is not
deterministic.

## What it measures

Each question is written from one paper in the Redpine Science full-text corpus and
carries a gold: two to four required claims, a certainty sentence, the population and
setting the claims hold for, and one over-claim an answer must not make. Claude Opus 5
answers every question in two arms with a budget of 20 tool calls: web search only, and
web search plus Redpine Science. Unlike the external QA track, no first call is forced:
the user message asks for at least one search, and the agent chooses when to use each
tool. A fixed judge scores each answer per claim; see Judging.

## Dataset

`data/questions.jsonl` holds the 179 questions with their golds, one JSON object per line;
[`data/README.md`](data/README.md) describes the fields, the gold grammar and a one-call
loader. The set is what the report calls the expert-validated question set.

## Setup

Everything the model sees is in [`configs/expert_qa.toml`](configs/expert_qa.toml): the
system prompt, the per-arm tool list, the tool descriptions and schemas, the budget, and
the result format. Web search is Tavily. Redpine Science search accepts an optional
metadata filter. Results are shown as `[S:<id>] Title (venue)` followed by the passage,
and the answer scored is the text of the agent's final turn.

### Adding an arm

An arm is an `[arms.<name>]` table in the TOML: its tool list and its tool sentence. An
arm with `tools = []` runs the model with no retrieval. Model, provider, budget and prompt
are set once per TOML. Copy the file, add the arm to `arm_order`, and run `uv run python -m src.run --config
my/arms.toml --arms <name>`. A new search backend is code, not config: one client function
in `harness/harness/clients.py`, one dispatch branch in `harness/harness/render.py`, and
its kind in `harness/harness/config.py`.

## Judging

A fixed model judges every answer per required claim; [`judging.md`](judging.md) has the
prompt, the request, the parsing rule, the three scores and the contract for porting the
judge to another model. `src/judge.py` implements the Sonnet 5 judge and the command that
judges a log; `src/jev.py` is the second judge, TypeSafe Jev through OpenRouter. The judge
is not wired into the runner: `src/run.py` stores answers, and judging is a separate step.

## What is here

- `logs/expert_qa_20260915.json`: the run behind the published numbers, both arms, every
  answer with quoted passage text removed and the number of removed spans per answer, the
  tool-call trace per arm, and the exclusions.
- `results/verdicts.jsonl`: per question, arm and judge (`sonnet5`, `jev1`, `jev2`), the
  verdict per required claim and the three scores.
- `results/summary.json`: the score table, the per-question split and the per-track
  differences.
- `results/agreement.json`: the judge agreement above, with its definition.
- `src/judge.py`, `src/jev.py`: the two judges and the command that turns a log into
  verdict rows.
- `scripts/summarize.py` and `scripts/agreement.py`: recompute `summary.json` and
  `agreement.json` from `verdicts.jsonl`.
- `scripts/guard.py`: the check that refuses to write any output containing a listed
  identifier; the tests run it over every file here when `Q2_FORBIDDEN` names the list.
  The list is private, so that one test skips with a warning on a fresh clone; every
  other test runs.

The public files were produced from the private evaluation data by a script that is not
published. It removed reviewer names and free-text notes, and it replaced, in every
answer and in every Sonnet 5 reason, each run of eight or more consecutive words that
also appears in the body of a passage retrieved for that answer with `[passage text
removed]`; passage header lines were left out of the comparison, so cited handles,
titles, venues, URLs and figures stay. About nine percent of the answer text was
removed; the count per answer is `stripped_spans` in the log. Sonnet 5's one-sentence
reason ships with each verdict, clipped at 400 characters; two reasons are withheld and
read "reason withheld". Jev records no reason, so its reasons are null.

## Running it

```bash
cd redpine-biomedical-qa
uv sync                  # add --extra bedrock for --provider bedrock
uv run pytest
cp .env.example .env     # fill in the keys below
uv run python -m src.run --limit 2
uv run python -m src.run
uv run python -m src.judge runs/<log>.json --provider bedrock --judge-tag sonnet5
```

| Variable | Needed for | Where to get it |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | the answering model and the Sonnet judge | https://console.anthropic.com; or `--provider bedrock` with AWS credentials (`AWS_REGION` defaults to `eu-west-1`); `BEDROCK_MODEL_ID` overrides the Bedrock model id |
| `AWS_BEARER_TOKEN_BEDROCK` | `--provider bedrock` | an Amazon Bedrock API key from the AWS console; with it set, no AWS login or `--extra bedrock` is needed |
| `REDPINE_API_KEY` | the Redpine Science arm | https://app.redpine.ai |
| `TAVILY_API_KEY` | both arms (web search) | https://tavily.com |
| `OPENROUTER_API_KEY` | the Jev judge | https://openrouter.ai |

A missing key stops the run with a one-line message before anything is called. A full
run is every question in `data/questions.jsonl` times two arms. Logs land in `runs/`
with the resolved setup in the header, the model's answers, and a per-arm trace of tool
calls (tool, query, result count, characters), never the tool results themselves; an
answer can still quote a passage. The log of the published run is in `logs/`, with
passage text quoted in answers removed.

## Data and licences

The questions and golds were written by Redpine and reviewed by domain experts. The
answers in `logs/` are Claude Opus 5 output with retrieved passage text removed. Jev's
verdict labels are Redpine's to publish: TypeSafe's Master Customer Agreement assigns
Output to the customer and places no restriction on publishing or benchmarking it.
OpenRouter's terms defer to the model provider's terms for output. Code in this
directory is MIT.
