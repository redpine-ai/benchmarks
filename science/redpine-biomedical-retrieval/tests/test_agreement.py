"""Inter-rater agreement on the 25-question shared pool: quadratic-weighted Cohen's kappa between
the cardiology and neurology experts over every result both scored, recomputed from the shipped
ratings. The report quotes kappa = 0.59, n = 246."""
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
LEVELS = 5  # the 0-4 relevance scale


def shared_pool_pairs():
    pools = {q["id"]: q["judging_pool"] for q in json.loads((DATA / "questions.json").read_text())}
    scores = {}
    for r in json.loads((DATA / "ratings.json").read_text()):
        if pools[r["query_id"]] == "multi_judge" and r["score"] is not None:
            scores.setdefault((r["query_id"], r["side"], r["rank"]), {})[r["judge"]] = r["score"]
    return [(s["cardiology_expert"], s["neurology_expert"]) for s in scores.values() if len(s) == 2]


def quadratic_weighted_kappa(pairs, levels=LEVELS):
    n = len(pairs)
    observed = [[0] * levels for _ in range(levels)]
    for a, b in pairs:
        observed[a][b] += 1
    rows = [sum(row) for row in observed]
    cols = [sum(observed[i][j] for i in range(levels)) for j in range(levels)]
    weight = lambda i, j: (i - j) ** 2 / (levels - 1) ** 2  # noqa: E731
    disagreement = sum(weight(i, j) * observed[i][j] for i in range(levels) for j in range(levels))
    expected = sum(weight(i, j) * rows[i] * cols[j] / n for i in range(levels) for j in range(levels))
    return 1 - disagreement / expected


def test_kappa_toy_cases():
    assert quadratic_weighted_kappa([(0, 0), (4, 4), (2, 2)]) == 1.0
    assert quadratic_weighted_kappa([(0, 4), (4, 0)]) < 0


def test_shared_pool_kappa_matches_the_report():
    pairs = shared_pool_pairs()
    assert len(pairs) == 246
    assert round(quadratic_weighted_kappa(pairs), 3) == 0.586
