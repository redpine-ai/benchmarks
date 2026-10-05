import json
import re
from collections import Counter
from pathlib import Path

TRACK = Path(__file__).resolve().parents[1]
QUESTIONS = TRACK / "data" / "questions.jsonl"
FIELDS = ["id", "track", "question", "gold", "doi", "access"]
NO_WRONG_IF = {"steered/10.1177/15230864251401591"}
FORBIDDEN = re.compile(r"@|/Users/|\"quote\"", re.I)


def rows():
    return [json.loads(l) for l in QUESTIONS.read_text().splitlines() if l.strip()]


def test_shape_and_counts():
    r = rows()
    assert len(r) == 179
    assert all(list(x) == FIELDS for x in r)
    assert Counter(x["track"] for x in r) == {"cardiology": 79, "neuroimmunology": 60, "sepsis": 20, "immunology": 20}
    assert Counter(x["access"] for x in r) == {"open": 105, "paywalled": 74}
    assert len({x["id"] for x in r}) == 179


def test_gold_template_and_doi():
    for x in rows():
        g = x["gold"]
        assert g.startswith("A strong answer conveys:") and "Certainty:" in g and "Scope:" in g
        assert ("An answer is wrong if" in g) or x["id"] in NO_WRONG_IF
        assert x["id"] == "steered/" + x["doi"] and x["doi"].startswith("10.")
        assert x["question"].strip() and x["gold"].strip()


def test_no_private_material():
    text = QUESTIONS.read_text()
    assert not FORBIDDEN.search(text)


def test_claim_counts():
    from src.judge import required_claims
    counts = Counter(len(required_claims(x["gold"])) for x in rows())
    assert counts == {2: 2, 3: 79, 4: 98}
    assert sum(k * v for k, v in counts.items()) == 633
