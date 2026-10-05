# Redpine Science Benchmarks

Open benchmarks for evaluating Redpine Science against general web search and
purpose-built retrieval engines, across external and Redpine question sets,
on both full agentic answers and retrieval alone.

Companion to [the preprint](paper/) and [the launch post](https://www.redpine.ai/insights/redpine-science-outperforms-web-search-and-public-benchmarks).

## Tracks

| Track | Questions | Measures | Description |
|-------|-----------|----------|-------------|
| [External QA benchmark](external-qa/) | 208 claims | Agentic answer | ScholarQABench SciFact, four arms, Claude Sonnet 5, mean of five runs |
| [Redpine Biomedical QA](redpine-biomedical-qa/) | 179 | Agentic answer | Literature-grounded questions from single papers, checked by domain experts; Claude Opus 5, two arms |
| [External retrieval benchmark](external-retrieval/) | 668 | Retrieval only | Exa's publication-retrieval benchmark, DOI-confirmed subset, run the way Exa's own team runs it |
| [Redpine Biomedical Retrieval Evaluation](redpine-biomedical-retrieval/) | 115 | Retrieval only | Blinded PhD-level domain-expert relevance panel vs. PubMed, following Consensus's published methodology |

## Results

Each track's own README has the full results table, methodology, and caveats. Headline
number per track:

- **External QA benchmark:** Redpine Science answers 94.4% of ScholarQABench SciFact claims
  correctly, vs. 87.6% with no retrieval and 93.1% with web search (208 claims, 196 to 199
  scored per run, mean of five runs). [external-qa/](external-qa/)
- **Redpine Biomedical QA:** an agent with Redpine Science states 80.1% of the required
  claims vs. 70.2% with web search only, +0.100 [+0.058, +0.142] paired, n=179, under the
  primary judge; a second judge gives +0.095. The question set was reviewed by domain
  experts; the report compares the judges with expert labels, which are not published.
  [redpine-biomedical-qa/](redpine-biomedical-qa/)
- **External retrieval benchmark:** Redpine Science 83.1% Recall@10, vs. 79.3% (Exa) and 60.9%
  (Tavily web search), n=668. [external-retrieval/](external-retrieval/)
- **Redpine Biomedical Retrieval Evaluation:** Redpine Science 75.2% Precision@5 vs. 39.8% (PubMed), n=115,
  90 questions judged by one expert and 25 by two. [redpine-biomedical-retrieval/](redpine-biomedical-retrieval/)

## Quick start

Each answer-quality track is its own package with its own README, config and
`.env.example`. The two retrieval tracks reproduce their published numbers from checked-in
data with no API keys needed; `external-retrieval` can also rerun live against Redpine/
Tavily/Exa, and `redpine-biomedical-retrieval` can rerun its scoring step and build a new blinded
judging package from live Redpine/PubMed calls (`src/build_package.py`). The two answer-quality tracks share one
harness in `harness/`; a track's whole setup (model, budget, prompt, tools) is one TOML file
under its `configs/`.

```bash
git clone https://github.com/redpine-ai/benchmarks.git
cd benchmarks/science/external-qa
uv sync
uv run pytest
cp .env.example .env     # fill in the keys the track README lists
uv run python -m src.run --limit 2
```

Each answer-quality track stops with a one-line message naming the missing key or
dependency before it downloads or calls anything.

## Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- API keys only for live reruns: Redpine Science (https://app.redpine.ai) for every
  track's Redpine arm; Tavily for web search; Exa for `external-retrieval`; an Anthropic
  API key or AWS Bedrock access for the answer-quality tracks

## Status

Each track's README gives its headline numbers and names the test that recomputes them
from the files it ships.

## License

MIT
