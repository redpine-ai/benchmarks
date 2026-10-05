"""Gold-document matching and the closed-form IR metrics this benchmark reports.

Matching is DOI-or-title, not DOI-only: Tavily and Exa never return a structured
DOI field (see retrievers.py), so DOI-only matching would silently force their
recall to 0% regardless of whether they actually found the right paper. DOI
matching isn't limited to that structured field either: a result's URL or
snippet often embeds a DOI even with no clean field for it (a doi.org link, a
citation string in the text) -- extracting it is the same technique
exa-labs/benchmarks' own current grader (benchmarks/graders/paper.py) uses, and
for the same reason. _DOI_RE below matches their regex exactly, so a DOI this
benchmark can find is one their own reference grader would find too.

Title matching is normalized-containment (lowercase, HTML tags and punctuation
stripped, whitespace collapsed, then checked either direction as a substring),
not exact equality. Real publisher/aggregator page titles routinely carry a
venue affix a database's own title field never does -- Nature appends
" | Scientific Reports", Frontiers prepends "Frontiers | ", PMC mirrors append
" - PMC". Exact-equality matching silently misses every one of these,
undercounting web-search recall for reasons that have nothing to do with
whether the system found the right paper. Containment survives the affix while
still requiring the full title to be present verbatim, which keeps
false-positive risk low for anything but a very short title (see
MIN_TITLE_LEN). HTML tags are stripped whole, not just their angle brackets --
a gold title with un-cleaned markup (e.g. "CD4<sup>+</sup> T-cell") otherwise
leaves the tag name itself as residue ("sup sup") that silently breaks
containment against an identical, cleanly-formatted candidate title. This is a
deliberate departure from exa-labs/benchmarks' own title-matching rule,
word-level Jaccard similarity >= 0.70, not an attempt to replicate it exactly
-- containment is stricter in some cases, looser in none.

With exactly one relevant document per query, the standard IR metrics collapse to
simple closed forms:
  - Recall@k: fraction of queries where the gold document appears anywhere in the
    top k results.
  - MRR: mean of 1/rank if found, 0 if not.
  - nDCG: DCG = 1/log2(rank+1) if found else 0; ideal DCG (gold doc at rank 1) is
    1/log2(2) = 1, so nDCG = DCG directly, no separate normalization step.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
_TITLE_NORM_RE = re.compile(r"[^a-z0-9\s]")
# Below this length, substring containment risks a coincidental match (a short
# generic phrase could sit inside an unrelated longer title) -- fall back to exact
# equality only for titles this short.
MIN_TITLE_LEN = 25

# Matches exa-labs/benchmarks' own benchmarks/graders/paper.py exactly -- same
# pattern, same reason: a DOI embedded in a URL or in body text, not just a
# structured field.
_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s,;\"'>\]]+", re.IGNORECASE)


def normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    d = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if d.startswith(prefix):
            d = d[len(prefix):]
    return d.rstrip(".,;:)")


def _extract_dois(*texts: str | None) -> set[str]:
    """Scan arbitrary text (a URL, a snippet) for embedded DOIs. Tavily and Exa
    never populate Result.doi, but a doi.org link or an in-text citation often
    carries one anyway -- this is how a result without a structured DOI field
    can still match on DOI, not just on title."""
    dois: set[str] = set()
    for text in texts:
        if not text:
            continue
        for match in _DOI_RE.finditer(text):
            dois.add(normalize_doi(match.group(0)))
    return dois


def normalize_title(title: str | None) -> str | None:
    """Lowercase, strip HTML tags and punctuation, collapse whitespace.

    HTML tags are stripped WHOLE (angle brackets and tag name together) before
    the general punctuation strip, not left to it: stripping only the angle
    brackets leaves the tag name itself as a bare word (e.g. "<sup>+</sup>"
    degrading to "sup sup" instead of disappearing), which pollutes a gold
    title that happens to carry un-cleaned markup and silently breaks
    containment against an otherwise-identical, cleanly-formatted candidate
    title. Found live in this dataset: one gold title keeps a literal
    "<sup>+</sup>" around a superscripted plus sign.
    """
    if not title:
        return None
    t = _HTML_TAG_RE.sub(" ", title.lower())
    t = _TITLE_NORM_RE.sub(" ", t)
    t = " ".join(t.split())
    return t or None


def _titles_match(gold_norm: str | None, candidate_norm: str | None) -> bool:
    if not gold_norm or not candidate_norm:
        return False
    if len(gold_norm) < MIN_TITLE_LEN or len(candidate_norm) < MIN_TITLE_LEN:
        return gold_norm == candidate_norm
    return gold_norm in candidate_norm or candidate_norm in gold_norm


def _field(r, name: str):
    """A candidate is a retrievers.Result in a live run and a plain dict when read
    back from results/gold_doc_results.json; both must re-match the same way."""
    return r.get(name) if isinstance(r, dict) else getattr(r, name, None)


def find_rank(gold_doi: str | None, gold_title: str | None, results: list) -> int | None:
    """results: anything with .doi, .title, .url, .snippet attributes
    (retrievers.py's Result), or stored rows carrying .snippet_dois instead of the
    snippet text (results/gold_doc_results.json ships the DOIs extracted from each
    snippet, never the snippet itself). Returns the 1-based rank of the first
    match, or None if not found. Priority order matches exa-labs/benchmarks' own
    current grader: DOI (structured field, or extracted from the result's
    URL/snippet), then title."""
    gold_doi_norm = normalize_doi(gold_doi)
    gold_title_norm = normalize_title(gold_title)
    for i, r in enumerate(results):
        if gold_doi_norm:
            if normalize_doi(_field(r, "doi")) == gold_doi_norm:
                return i + 1
            extracted = _extract_dois(_field(r, "url"), _field(r, "snippet"))
            extracted |= set(_field(r, "snippet_dois") or ())
            if gold_doi_norm in extracted:
                return i + 1
        if _titles_match(gold_title_norm, normalize_title(_field(r, "title"))):
            return i + 1
    return None


def _dcg(rank: int | None) -> float:
    return 1.0 / math.log2(rank + 1) if rank is not None else 0.0


def summarize(ranks: list[int | None]) -> dict:
    """ranks: one entry per query, the rank find_rank() returned (or None)."""
    n = len(ranks)
    found = [r for r in ranks if r is not None]
    return {
        "n": n,
        "n_found": len(found),
        "recall": len(found) / n if n else 0.0,
        "mrr": sum(1.0 / r for r in found) / n if n else 0.0,
        "ndcg": sum(_dcg(r) for r in ranks) / n if n else 0.0,
        "rank_distribution": dict(sorted(Counter(found).items())),
    }


def print_summary(name: str, ranks: list[int | None]) -> None:
    s = summarize(ranks)
    print(f"--- {name} (n={s['n']}) ---")
    print(f"  Recall: {100 * s['recall']:.1f}%  ({s['n_found']}/{s['n']} found)")
    print(f"  MRR:    {s['mrr']:.3f}")
    print(f"  nDCG:   {s['ndcg']:.3f}")
    if s["rank_distribution"]:
        print(f"  rank distribution: {s['rank_distribution']}")
    print()
