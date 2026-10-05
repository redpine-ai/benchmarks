"""Tests for score_ratings.py's scoring math. compute_stats is the pure,
file-I/O-free core, tested here in isolation from main()'s file-reading/CSV/
summary-writing glue.

answer_key fixtures carry {side}_true_ranks since a rating row's "rank" is a
shuffled DISPLAY position, not a true retrieval rank -- compute_stats must
resolve it through this mapping. Most fixtures use the identity mapping
([1, 2, 3, ...]) since they're testing something else; the one test that
specifically exercises a non-identity mapping is named for it.
"""

import math

import pytest

from src.score_ratings import RELEVANT_THRESHOLD, _dcg, compute_stats


# ---------- _dcg ----------

def test_dcg_known_value():
    # sum(score / log2(rank + 1)): 2/log2(2) + 1/log2(3) = 2.0 + 0.6309...
    assert _dcg({1: 2, 2: 1}) == pytest.approx(2.0 + 1 / math.log2(3))


def test_dcg_empty_is_zero():
    assert _dcg({}) == 0.0


def test_dcg_single_rank_one_divides_by_log2_2_which_is_one():
    assert _dcg({1: 2}) == pytest.approx(2.0)  # log2(2) == 1, so score/1 == score


# ---------- compute_stats ----------

def _row(qid, side, rank, score):
    return {"query_id": qid, "side": side, "rank": rank, "score": score}


def _key(left_system, right_system, k=5, left_true_ranks=None, right_true_ranks=None):
    return {
        "left_system": left_system, "right_system": right_system,
        "left_true_ranks": left_true_ranks if left_true_ranks is not None else list(range(1, k + 1)),
        "right_true_ranks": right_true_ranks if right_true_ranks is not None else list(range(1, k + 1)),
    }


def test_relevant_threshold_is_two():
    assert RELEVANT_THRESHOLD == 2


def test_precision_counts_scores_at_or_above_threshold():
    answer_key = {"q1": _key("redpine", "pubmed", k=2)}
    ratings = [
        {"judge": "j", **_row("q1", "left", 1, 2)},   # relevant (threshold is 2)
        {"judge": "j", **_row("q1", "left", 2, 1)},   # not relevant
        {"judge": "j", **_row("q1", "right", 1, 4)},  # relevant
        {"judge": "j", **_row("q1", "right", 2, 0)},  # not relevant
    ]
    stats = compute_stats(answer_key, ratings)
    assert stats["systems"]["redpine"]["precision"] == pytest.approx(0.5)
    assert stats["systems"]["pubmed"]["precision"] == pytest.approx(0.5)


def test_compute_stats_resolves_display_position_to_true_rank_via_answer_key():
    # Display position 1 actually shows true rank 3, and vice versa.
    answer_key = {"q1": _key("redpine", "pubmed", left_true_ranks=[3, 1, 2],
                             right_true_ranks=[1, 2, 3])}
    ratings = [{"judge": "j", **_row("q1", "left", 1, 4)}]  # display pos 1 -> true rank 3
    stats = compute_stats(answer_key, ratings)
    by_rank = stats["systems"]["redpine"]["by_rank"]
    assert 3 in by_rank and by_rank[3]["mean"] == 4
    assert 1 not in by_rank


def test_none_score_dropped_not_treated_as_zero():
    answer_key = {"q1": _key("redpine", "pubmed", k=1)}
    ratings = [{"judge": "j", **_row("q1", "left", 1, None)}]
    stats = compute_stats(answer_key, ratings)
    assert stats["systems"]["redpine"]["n_scored"] == 0
    assert stats["systems"]["redpine"]["mean_relevance"] is None


def test_zero_result_side_contributes_dcg_zero_but_excluded_from_precision():
    # left side genuinely returned nothing (empty true_ranks); right side scored normally.
    answer_key = {"q1": _key("redpine", "pubmed", left_true_ranks=[], right_true_ranks=[1])}
    ratings = [{"judge": "j", **_row("q1", "right", 1, 4)}]
    stats = compute_stats(answer_key, ratings)
    redpine = stats["systems"]["redpine"]
    assert redpine["zero_result_questions"] == 1
    assert redpine["n_scored"] == 0  # item-level: nothing to average
    assert redpine["mean_relevance"] is None
    assert redpine["mean_dcg"] == pytest.approx(0.0)  # query-level: counts as a real DCG=0


def test_unjudged_but_nonempty_side_excluded_entirely_not_treated_as_zero():
    # left side has real items (true_ranks non-empty) but no ratings rows yet --
    # "not judged" must NOT be conflated with "genuinely returned nothing."
    answer_key = {"q1": _key("redpine", "pubmed", left_true_ranks=[1, 2], right_true_ranks=[1])}
    ratings = [{"judge": "j", **_row("q1", "right", 1, 4)}]
    stats = compute_stats(answer_key, ratings)
    redpine = stats["systems"]["redpine"]
    assert redpine["zero_result_questions"] == 0
    assert redpine["n_scored"] == 0
    assert redpine["mean_dcg"] is None  # no per-query DCG contributed at all


def test_multiple_judges_averaged_per_item_before_aggregating():
    answer_key = {"q1": _key("redpine", "pubmed", k=1)}
    ratings = [
        {"judge": "a", **_row("q1", "left", 1, 2)},
        {"judge": "b", **_row("q1", "left", 1, 4)},
    ]
    stats = compute_stats(answer_key, ratings)
    assert stats["systems"]["redpine"]["mean_relevance"] == pytest.approx(3.0)
    assert stats["judges"] == ["a", "b"]


def test_system_with_no_scored_items_still_appears_with_none_fields():
    answer_key = {"q1": _key("redpine", "pubmed", k=1)}
    stats = compute_stats(answer_key, [])
    for system in ("redpine", "pubmed"):
        s = stats["systems"][system]
        assert s["n_scored"] == 0
        assert s["mean_relevance"] is None
        assert s["precision"] is None
        assert s["mean_dcg"] is None
