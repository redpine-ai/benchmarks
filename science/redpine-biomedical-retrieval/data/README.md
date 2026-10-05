# Dataset

Four files, all from a real judging round, anonymized for public release.

| File | What it is |
| --- | --- |
| `questions.json` | 115 questions: `id`, `query`, `domain`, `subspecialty`, `judging_pool` (`single_judge` or `multi_judge`) |
| `package.json` | The blinded judging package a domain expert saw. Per question, two unlabeled result sets (`left`/`right`), each result's `title`; the snippets the judge also read are not published. No URL, DOI, or publication date shown: those format differently enough between Redpine's structured results and a scraped PubMed page to be a tell on their own |
| `answer_key.json` | The hidden mapping `package.json` was built from: which system (`redpine` or `pubmed`) was `left`/`right` per question, and each side's true retrieval rank behind its shuffled display position |
| `ratings.json` | All three domain experts' 0-4 relevance scores in one flat list, by display position, for every result they judged. One row is `{judge, query_id, side, rank, score}`; `judge` tells rows from different experts apart, not which file they're in |

**Anonymization:** judges are identified by role (`cardiology_expert`, `neurology_expert`,
`rheumatology_expert`), never by name. `package_id`/`run_name` are provenance stamps that let
`score_ratings.py` verify a ratings file was scored against the package it claims; the shipped
answer key and ratings predate the stamp, `score_ratings.py` reports this and scores them
unchanged. Neither is identifying information.

**115 questions, all scored.** `judging_pool` splits them 90 `single_judge` (30 assigned to each
of cardiology's, neurology's, and rheumatology's experts; cardiology's and rheumatology's 30 are
each tagged that `domain`, while 9 of neurology's carry
adjacent tags: immunology 4, critical care 2, microbiology 2, pharmacology 1) plus 25 `multi_judge` (a shared
pool, for measuring inter-rater agreement, κ = 0.59 over 246 pairs; see the track README).
Cardiology's and neurology's experts both scored the shared 25; rheumatology's expert scored only
the 30 single-judge questions, so the agreement comparison is two-judge, not three-way. This
does not affect Precision@5/DCG@5/mean relevance, which average the two judges'
scores on each shared item and then pool every scored item.

**Scale:** 0-4 relevance (0 = completely irrelevant, 4 = the perfect answer is in the chunk),
following Consensus's published methodology. Precision here counts a score of 2 or higher as
relevant; Consensus's published precision metric counts 3 or higher. Top 5 results per system
per question (`JUDGE_TOP_K=5`, against Consensus's top-10, for labor-cost reasons: every result
pair here is read and judged by hand, not scored automatically).

**Comparator:** PubMed. This track measures Redpine Science against what a researcher
does today, typing a question into PubMed unassisted, including PubMed's own natural-language
search failures, rather than an idealized isolation against another AI-search product's
undisclosed query processing.
