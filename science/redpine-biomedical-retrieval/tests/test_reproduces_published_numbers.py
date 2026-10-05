"""Recomputes Precision@5/DCG@5/mean relevance from the checked-in data/answer_key.json
+ data/ratings.json and checks they match what the README publishes. If this fails,
either the data files or the README drifted -- fix whichever is wrong, don't
loosen the tolerance."""

import json
from pathlib import Path

import pytest

from src.score_ratings import compute_stats, _load_answer_key, _load_ratings_file

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Matches the README's Results table exactly. All three domain experts scored,
# n=115. Precision@5 uses RELEVANT_THRESHOLD=2 (see score_ratings.py), a looser
# cutoff than the 3-or-4 threshold this report's scale is otherwise matched to.
PUBLISHED = {
    "redpine": {"n_scored": 572, "precision": 0.752, "mean_dcg": 7.262, "mean_relevance": 2.43},
    "pubmed": {"n_scored": 571, "precision": 0.398, "mean_dcg": 4.019, "mean_relevance": 1.38},
}


@pytest.fixture(scope="module")
def stats():
    answer_key = _load_answer_key(DATA_DIR / "answer_key.json")["rows"]
    ratings_rows = _load_ratings_file(DATA_DIR / "ratings.json")["rows"]
    return compute_stats(answer_key, ratings_rows)


@pytest.mark.parametrize("system", ["redpine", "pubmed"])
def test_reproduces_published_numbers(stats, system):
    s = stats["systems"][system]
    expected = PUBLISHED[system]
    assert s["n_scored"] == expected["n_scored"]
    # Rounded to the same precision the README quotes -- one decimal percent for
    # precision, three decimals for DCG, two for mean relevance.
    assert round(s["precision"], 3) == expected["precision"]
    assert round(s["mean_dcg"], 3) == expected["mean_dcg"]
    assert round(s["mean_relevance"], 2) == expected["mean_relevance"]


def test_all_three_domain_experts_scored(stats):
    # 115 total questions, all scored: 90 single-judge (30 each across cardiology,
    # neurology, rheumatology) + 25 shared multi-judge. Every question in the
    # answer key has ratings from its assigned judge(s).
    assert stats["n_queries"] == 115
    assert stats["judges"] == ["cardiology_expert", "neurology_expert", "rheumatology_expert"]
    questions = json.loads((DATA_DIR / "questions.json").read_text())
    # The 25-question multi-judge pool spans several other domains beyond the
    # three single-judge ones (real-world questions, not domain-authored) -- so
    # this checks the three are present, not that they're the only domains.
    assert {"cardiology", "neurology", "rheumatology"} <= {q["domain"] for q in questions}


def test_rheumatology_expert_did_not_score_the_shared_pool():
    # Cardiology's and neurology's experts both scored the 25-question shared
    # multi-judge pool (for inter-rater agreement between the two of them);
    # rheumatology's expert scored only their own 30 single-judge questions, not
    # the shared 25 -- a real asymmetry in this panel, not a gap to silently
    # fill in later. Locks in what data/README.md and the track README both
    # state in prose, so it can't drift out of sync with the actual data.
    ratings = json.loads((DATA_DIR / "ratings.json").read_text())
    questions = {q["id"]: q for q in json.loads((DATA_DIR / "questions.json").read_text())}
    rheum_qids = {r["query_id"] for r in ratings if r["judge"] == "rheumatology_expert"}
    assert len(rheum_qids) == 30
    assert {questions[q]["judging_pool"] for q in rheum_qids} == {"single_judge"}
    assert {questions[q]["domain"] for q in rheum_qids} == {"rheumatology"}


def test_ratings_file_is_one_file_with_all_judges(stats):
    # All experts' rows live in a single data/ratings.json, not split per domain,
    # so a new judge is one file update, not a new file plus a code change.
    ratings = json.loads((DATA_DIR / "ratings.json").read_text())
    assert isinstance(ratings, list)
    assert len(ratings) == 1390  # 546 cardiology + 544 neurology + 300 rheumatology rows


def test_no_duplicate_query_ids():
    questions = json.loads((DATA_DIR / "questions.json").read_text())
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids))


def test_answer_key_uses_redpine_not_connect():
    # The systems are named redpine and pubmed, matching external-retrieval's
    # redpine_rank field; guard against a regression.
    answer_key = json.loads((DATA_DIR / "answer_key.json").read_text())
    systems = {row[k] for row in answer_key["answers"] for k in ("left_system", "right_system")}
    assert systems == {"redpine", "pubmed"}
