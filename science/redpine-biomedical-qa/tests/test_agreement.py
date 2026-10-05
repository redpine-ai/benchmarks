"""The judge agreement the README and the report print, recomputed from verdicts.jsonl.
Every value is pinned to the precision it is printed at."""
import json
from pathlib import Path

import pytest


from scripts.agreement import DEFINITIONS, compute, load_rows  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
A = compute(load_rows())


def r3(x):
    return round(x, 3)


def test_agreement_json_is_compute_output():
    assert json.loads((RESULTS / "agreement.json").read_text()) == json.loads(json.dumps(A))


def test_every_statistic_has_a_definition():
    stats = set(A) - {"definitions"}
    assert set(DEFINITIONS) == stats
    assert all(d.endswith(".") and d.count(". ") == 0 for d in DEFINITIONS.values())


def test_judge_agreement():
    j = A["judge_agreement"]
    assert j["connect"]["n_claims"] == j["websearch"]["n_claims"] == 633
    assert (r3(j["connect"]["agreement"]), r3(j["websearch"]["agreement"])) == (0.899, 0.870)
    assert (r3(j["connect"]["alpha"]), r3(j["websearch"]["alpha"])) == (0.702, 0.717)
