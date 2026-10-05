"""Recomputes Recall/MRR/nDCG from results/gold_doc_results.json and checks they
match what the README publishes. If this fails, either the results file or the
README drifted -- fix whichever is wrong, don't loosen the tolerance."""

import json
from pathlib import Path

import pytest

from src.matching import summarize

RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "gold_doc_results.json"

# Matches the README's Results table exactly.
PUBLISHED = {
    "redpine": {"n": 668, "recall_at_1": 0.698, "recall_at_5": 0.805, "recall": 0.831, "mrr": 0.744, "ndcg": 0.765},
    "exa": {"n": 668, "recall_at_1": 0.629, "recall_at_5": 0.763, "recall": 0.793, "mrr": 0.688, "ndcg": 0.714},
    "tavily": {"n": 668, "recall_at_1": 0.482, "recall_at_5": 0.575, "recall": 0.609, "mrr": 0.524, "ndcg": 0.545},
}


@pytest.fixture(scope="module")
def rows():
    return json.loads(RESULTS_PATH.read_text())["results"]


@pytest.mark.parametrize("system", ["redpine", "exa", "tavily"])
def test_reproduces_published_numbers(rows, system):
    ranks = [r[f"{system}_rank"] for r in rows]
    s = summarize(ranks)
    expected = PUBLISHED[system]
    assert s["n"] == expected["n"]
    # Rounded to 3 places on both sides -- the README quotes recall to one decimal
    # percent and MRR/nDCG to three decimals, so this is the real precision, not
    # an arbitrarily loose tolerance.
    assert round(s["recall"], 3) == expected["recall"]
    assert round(s["mrr"], 3) == expected["mrr"]
    assert round(s["ndcg"], 3) == expected["ndcg"]
    # Recall@1 and Recall@5, the report's other two columns, from the same ranks.
    for k in (1, 5):
        at_k = sum(1 for r in ranks if r is not None and r <= k) / len(ranks)
        assert round(at_k, 3) == expected[f"recall_at_{k}"]


def test_every_row_has_all_three_arms(rows):
    # The published n=668 assumes complete data for all three systems on every
    # row -- if this ever fails, the "one canonical n" promise in the README
    # is broken and needs fixing before anything else.
    for r in rows:
        assert "redpine_rank" in r and "tavily_rank" in r and "exa_rank" in r


def test_no_duplicate_query_ids(rows):
    ids = [r["query_id"] for r in rows]
    assert len(ids) == len(set(ids))


def test_stored_candidates_rematch_to_the_stored_ranks(rows):
    """The shipped file carries title, doi, url and snippet_dois per candidate; re-running
    the matcher over those dicts must give back every stored rank."""
    import json as _json
    from src.matching import find_rank
    data_path = RESULTS_PATH.parents[1] / "data" / "publication_search_in_corpus.jsonl"
    titles = {}
    for line in data_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = _json.loads(line)
            titles[rec["query_id"]] = rec["gold_paper"]["title"]
    for row in rows:
        for system in ("redpine", "exa", "tavily"):
            rank = find_rank(row["gold_doi"], titles[row["query_id"]], row[f"{system}_results"])
            assert rank == row[f"{system}_rank"], (row["query_id"], system)
