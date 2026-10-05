# External retrieval benchmark: Exa publication retrieval

Redpine Science vs. Tavily vs. Exa, single-shot retrieval (one call per system per
query, no agent, no reformulation), against Exa's own publication-retrieval
benchmark. This directory holds the dataset, the runner, and the results behind
the report's external retrieval track.

## Result

n=668; see [Dataset](#dataset) for which 668 and why:

| System | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 |
|--------|---------:|---------:|----------:|----:|--------:|
| Redpine Science | 69.8% | 80.5% | 83.1% | 0.744 | 0.765 |
| Exa | 62.9% | 76.3% | 79.3% | 0.688 | 0.714 |
| Tavily | 48.2% | 57.5% | 60.9% | 0.524 | 0.545 |

Recomputed directly from [`results/gold_doc_results.json`](results/gold_doc_results.json)
by [`tests/test_reproduces_published_numbers.py`](tests/test_reproduces_published_numbers.py).

Exa's own published recall is on the full 1,866-query set with no stated cutoff, so it is
not comparable with the subset here.

## Why single-shot, not agentic

The obvious way to test "does Redpine Science retrieve the right paper" is inside
an agent loop, the way a real user would experience it. That setup has a real
confound: an agent with multiple tools can fall back from one to another, so a
correct answer might have come from a different tool than the one being measured,
with no way to tell which. Calling each system's search API directly, once, with
no reformulation, removes that ambiguity: each system answers with only its own
ranking. It is a classical IR question: given one query, which system's own
ranking surfaces the correct document, and at what rank. Single-shot retrieval
answers it.

## Dataset

[Exa's own publication-retrieval benchmark](https://github.com/exa-labs/benchmarks)
asks 1,866 questions, each with one gold paper. A query only tests retrieval
quality if the gold paper is something Redpine Science could in principle find:
scoring against a paper outside Redpine's corpus measures corpus coverage, not
retrieval. The filter keeps only those queries, and all three systems are scored on
the same set, so the numbers here are not comparable to full-benchmark numbers
published elsewhere.

`data/publication_search_in_corpus.jsonl` is the subset confirmed present in
Redpine's corpus by a direct per-DOI membership check against Redpine's search
API (an exact-match filter query per DOI, independent of any query text or
ranking), not inferred from any other signal. 671 queries passed that check, selected
before any system was run. Three of them failed on a transient API error for one system
during the run and are excluded rather than scored as misses, leaving the 668 in this file,
on which all three systems are scored.

Schema matches exa-labs/benchmarks' own `publication_search.jsonl`.

## Setup

Tavily and Exa are configured to match exa-labs/benchmarks' reference
implementation as its searcher source runs, not as its README describes it; the
two disagree in two places (Tavily's default search depth, and whether Exa's
searcher restricts to the `publication` category). See [`src/retrievers.py`](src/retrievers.py)'s module
docstring for the specifics and why each one matters.

## Scoring

DOI-exact match where available (only Redpine Science's API returns a structured
DOI; Tavily and Exa never do), else normalized-title containment against the gold
paper's title. See [`src/matching.py`](src/matching.py) for the exact rule and why
containment, not exact equality.

## What is here

- `data/publication_search_in_corpus.jsonl`: the 668-query dataset.
- `results/gold_doc_results.json`: per-query rank for all three systems from the
  run behind the published table, per query, with every candidate's title, DOI,
  URL and the DOIs extracted from its snippet -- enough to re-match from the file.
  The snippet text itself is not shipped: Redpine Science snippets are chunks of
  licensed full text, and Exa and Tavily snippets are publisher-page extracts.
- `src/retrievers.py`: the three retrievers, one method each, `search(query,
  max_results) -> list[Result]`.
- `src/matching.py`: DOI/title matching and the Recall/MRR/nDCG math.
- `src/benchmark.py`: the run command.
- `tests/`: matching-logic unit tests, plus the test that recomputes the published
  numbers from `results/gold_doc_results.json`.

## Running it

```bash
cd external-retrieval
uv sync
uv run pytest
cp .env.example .env     # fill in the keys below
uv run python -m src.benchmark --limit 20 --out /tmp/smoke.json   # smoke test
uv run python -m src.benchmark                                    # full 668-query run
```

| Variable | Needed for | Where to get it |
| --- | --- | --- |
| `REDPINE_API_KEY` | the Redpine Science arm | https://app.redpine.ai |
| `REDPINE_API_URL` | the Redpine API base (default `https://api.redpine.ai`) | optional |
| `REDPINE_COLLECTION` | which Redpine collection to search (default `Redpine Science`) | optional |
| `TAVILY_API_KEY` | the Tavily arm | https://tavily.com |
| `EXA_API_KEY` | the Exa arm | https://exa.ai |

A missing key stops the run with a one-line message before anything is called. A
full run makes roughly 2,000 API calls total (668 queries times 3 systems) and
takes on the order of ten minutes at the default concurrency.

Redpine Science's retrieval stack changes over time, so a rerun measures the
stack on its own date; `results/gold_doc_results.json` carries the date its
searches ran (`run_date`) and, in `rescored_note`, the date the ranks were last
recomputed from the stored candidates after a matching fix.

## Data and licences

The query set is derived from [exa-labs/benchmarks](https://github.com/exa-labs/benchmarks)'
publication-retrieval benchmark (MIT, Copyright (c) 2025 Exa Labs; the notice is in
[`data/LICENSE-exa-labs`](data/LICENSE-exa-labs)), filtered to the subset confirmed present in
Redpine's corpus. The supporting quotes from the gold papers are not redistributed. Code in
this directory is MIT.
