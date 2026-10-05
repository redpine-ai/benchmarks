# Redpine Biomedical Retrieval Evaluation

Redpine Science vs. PubMed, blinded PhD-level domain-expert relevance judging (0-4 scale,
Precision@K, DCG@K). Judging data, scoring code, and results for the report's Redpine retrieval
evaluation.

## Result

n=115: 90 questions judged by one expert and 25 by two.

| System | Precision@5 (score ≥2) | Mean DCG@5 | Mean relevance (0-4) |
|--------|------------------------:|-----------:|----------------------:|
| Redpine Science | 75.2% | 7.262 | 2.43 |
| PubMed | 39.8% | 4.019 | 1.38 |

Precision@5 counts a score of 2 or higher as relevant; Consensus's published precision metric
counts 3 or higher.

Recomputed from [`data/answer_key.json`](data/answer_key.json) and
[`data/ratings.json`](data/ratings.json) by
[`tests/test_reproduces_published_numbers.py`](tests/test_reproduces_published_numbers.py).

## Why blinded expert judging

The sibling [external-retrieval](../external-retrieval/) track checks whether a
system surfaces one known gold paper. Most real research questions have no single right answer,
so judging relevance takes domain knowledge, not a DOI match. Blinded PhD-level judges score each
result 0-4, single-shot (no agent, and no reformulation beyond the PubMed relaxation described
under Building a new package).

## Dataset

`data/questions.json` holds 115 questions: 90 written by domain experts in cardiology, neurology
and rheumatology, and 25 real-world questions across other fields, judged by two experts for
inter-rater agreement. The 90 are single-judge, 30 assigned to each domain's expert
(cardiology's and rheumatology's 30 are each tagged that `domain`; 9 of
neurology's 30 carry adjacent tags (immunology 4, critical care 2, microbiology 2, pharmacology 1)).
25 more are shared, scored by both the cardiology and neurology experts, for measuring
inter-rater agreement between them (quadratic-weighted Cohen's κ = 0.59 over 246 paired ratings, recomputed by
`tests/test_agreement.py`).

All 115 are scored, in one `data/ratings.json` (a row's `judge` field marks the expert, not the
filename). Rheumatology's expert scored the 30 single-judge questions only, not
the shared 25, so the inter-rater agreement comparison stays two-judge (cardiology vs.
neurology), not three-way. This does not affect the Precision@5/DCG@5/mean relevance numbers
above, which average the two judges'
scores on each shared item and then pool every scored item.

Judges are identified by role (`cardiology_expert`, `neurology_expert`, `rheumatology_expert`),
never by name. A role tells you which field's expert gave a rating, and nothing else: no
expert is named in this repository or in the report, and the report's acknowledgement does
not pair names with fields, so a role never resolves to a person.
The judges saw each result's title and snippet, with no URL, DOI, or publication date: those
format differently between Redpine's structured results and a scraped PubMed page and would give
the system away. `package.json` publishes the titles only; the snippet text is PubMed and Redpine
Science content and is not redistributed. Full field
descriptions: [`data/README.md`](data/README.md).

## Scoring

[`src/score_ratings.py`](src/score_ratings.py) joins ratings back to the hidden answer key, then
computes Precision@K, DCG@K, and mean relevance by rank per system. See its docstring for the
exact formulas and edge cases, including how a zero-result query is scored (DCG=0, not dropped,
since PubMed returning nothing is itself part of what this measures).

## What is here

- `data/questions.json`, `data/answer_key.json`, `data/package.json`: the question set, the
  hidden system/rank mapping, and the blinded package a judge saw, titles only.
- `data/ratings.json`: every judge's 0-4 scores.
- `results/`: the stats `score_ratings.py` writes (summary, per-rank CSV, raw scores).
- `src/score_ratings.py`: the scoring code.
- `src/build_package.py`: builds a new blinded package from live Redpine/PubMed calls (see
  Building a new package); everything else here scores an existing one and needs no API access.
- `tests/`: scoring-logic tests and the published-number reproduction test.

## Running it

```bash
cd redpine-biomedical-retrieval
uv sync
uv run pytest
uv run python -m src.score_ratings data/answer_key.json data/ratings.json
```

No API keys needed. The table above reproduces entirely from checked-in data. To score a
new judging round, pass `--out-dir runs/my_round` so `results/` keeps the published numbers.

### Building a new package

```bash
cp .env.example .env          # REDPINE_API_KEY; PubMed needs no key
uv run python -m src.build_package data/questions.json runs/my_package --limit 5
```

Writes `runs/my_package/package_my_package.json` (what a judge sees) and
`answer_key_my_package.json` (never shown to a judge), both stamped with one `package_id`.
Each question gets one direct search per system, top 5, title and snippet only; left/right and
display order are randomized per question, and both sides are cut to the same count. PubMed
gets the question as written, and only when that returns fewer than 8 hits (the top 5 plus 3 spares) does it retry with
stopwords removed and then fewer trailing words, down to two; the relaxation only ever adds
PubMed results, so it cannot disadvantage PubMed. New ratings need new expert
judging: a rebuilt package draws a fresh randomization, so `data/ratings.json` does not apply to it.

## Data and licences

The 90 single-judge questions are hand-written by domain experts for this evaluation; the 25
shared questions are real-world questions. Result titles are
real PubMed and Redpine Science retrieval output; the snippets the judges read are not published. Code in
this directory is MIT.
