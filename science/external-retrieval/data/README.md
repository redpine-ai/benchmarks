# Dataset

`publication_search_in_corpus.jsonl` holds 668 queries, one JSON object per line, each
with a single gold paper.

| Field | Meaning |
| --- | --- |
| `query_id` | id from the source benchmark |
| `text` | the query |
| `track` / `bucket` | the source benchmark's own categorization |
| `gold_paper.doi` | the gold paper's DOI |
| `gold_paper.title` | the gold paper's title |
| `metadata.gold_answer` | the expected answer, where the source benchmark provides one |
| `tags` | source benchmark tags |

This dataset is based on [exa-labs/benchmarks](https://github.com/exa-labs/benchmarks)' own
publication-retrieval benchmark (1,866 queries). See their
[blog post](https://exa.ai/blog/publications-search) for the full methodology behind it.
Filtered to the subset of queries whose gold paper is confirmed present in Redpine Science's
corpus: retrieval quality can only be measured on a paper a system could in principle find, and
this restriction applies the same standard to Tavily and Exa as to Redpine (see the package
README's Dataset section for why).

"Confirmed present" means a direct per-DOI membership check against Redpine's search API,
an exact-match filter query per DOI, independent of any query text or search ranking, not
an assumption or a label carried over from elsewhere. Every DOI in this file was checked this
way individually.

671 of the 1,866 queries passed that check, selected before any system was run. Three failed
on a transient API error for one system during the run and are excluded rather than scored as
misses; this file holds the remaining 668, each with results from all three systems in the run
behind `../results/gold_doc_results.json`.
