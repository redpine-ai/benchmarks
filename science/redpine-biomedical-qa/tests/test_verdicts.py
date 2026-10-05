"""The shipped judge verdicts and the summary statistics recomputed from them."""
import json
from pathlib import Path

import pytest


from scripts.summarize import f1, summarize  # noqa: E402
from src.data import load_questions  # noqa: E402
from src.judge import required_claims  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
QUESTIONS = load_questions()
VERDICTS = [json.loads(line) for line in (RESULTS / "verdicts.jsonl").read_text().splitlines()]
SUMMARY = json.loads((RESULTS / "summary.json").read_text())
TRACKS = {q["id"]: q["track"] for q in QUESTIONS}

ARMS = ("websearch", "connect")
JUDGES = ("sonnet5", "jev1", "jev2")

# README: (websearch, connect, diff, ci low, ci high) per score and judge
TABLE = {
    ("sonnet5", "coverage"): (0.702, 0.801, 0.100, 0.058, 0.142),
    ("jev", "coverage"): (0.696, 0.791, 0.095, 0.056, 0.134),
    ("sonnet5", "contradiction_rate"): (0.069, 0.038, -0.031, -0.055, -0.008),
    ("jev", "contradiction_rate"): (0.107, 0.078, -0.029, -0.056, -0.002),
    ("sonnet5", "correctness"): (0.759, 0.847, 0.088, 0.049, 0.127),
    ("jev", "correctness"): (0.744, 0.828, 0.084, 0.048, 0.121),
}
# Report: per-track coverage difference, (n, Sonnet 5 diff, Jev diff); intervals pinned as computed
TRACK_TABLE = {"cardiology": (79, 0.075, 0.052), "neuroimmunology": (60, 0.125, 0.114),
               "sepsis": (20, 0.188, 0.221), "immunology": (20, 0.033, 0.081)}
TRACK_CI = {("sonnet5", "cardiology"): (0.014, 0.136), ("sonnet5", "neuroimmunology"): (0.054, 0.196),
            ("sonnet5", "sepsis"): (0.002, 0.373), ("sonnet5", "immunology"): (-0.072, 0.139),
            ("jev", "cardiology"): (-0.001, 0.106), ("jev", "neuroimmunology"): (0.042, 0.186),
            ("jev", "sepsis"): (0.073, 0.369), ("jev", "immunology"): (-0.022, 0.184)}


def test_row_count_and_order():
    assert len(VERDICTS) == 179 * 2 * 3
    expected = [(q["id"], a, j) for q in QUESTIONS for a in ARMS for j in JUDGES]
    assert [(r["id"], r["arm"], r["judge"]) for r in VERDICTS] == expected


def test_row_fields_are_exact():
    for r in VERDICTS:
        assert set(r) == {"id", "arm", "judge", "verdicts", "coverage", "contradiction_rate", "correctness"}
        for v in r["verdicts"]:
            assert set(v) == {"claim_index", "verdict", "reason"}
            assert v["verdict"] in {"stated", "not_addressed", "contradicted"}


def test_claim_count_matches_gold():
    n = {q["id"]: len(required_claims(q["gold"])) for q in QUESTIONS}
    for r in VERDICTS:
        assert [v["claim_index"] for v in r["verdicts"]] == list(range(1, n[r["id"]] + 1))


def test_reasons_are_short_and_clean():
    for r in VERDICTS:
        for v in r["verdicts"]:
            s = v["reason"]
            if r["judge"] != "sonnet5":
                assert s is None
                continue
            assert isinstance(s, str) and s.strip() and len(s) <= 400
            assert not any(bad in s for bad in ("http", "@", "/Users/"))


def test_correctness_is_f1():
    for r in VERDICTS:
        assert r["correctness"] == pytest.approx(f1(r["coverage"], 1 - r["contradiction_rate"]))
    assert f1(0.0, 0.0) == 0.0


def test_summary_file_is_what_summarize_computes():
    assert json.loads(json.dumps(summarize(VERDICTS, TRACKS))) == SUMMARY


@pytest.mark.parametrize("key", TABLE)
def test_readme_table(key):
    judge, score = key
    s = SUMMARY["judges"][judge]["scores"][score]
    got = (s["websearch"], s["connect"], s["diff"], *s["ci"])
    assert tuple(round(x, 3) for x in got) == TABLE[key]


def test_counts_and_split():
    assert SUMMARY["n_questions"] == 179
    assert SUMMARY["n_claims_per_arm"] == 633
    assert SUMMARY["judges"]["sonnet5"]["coverage_split"] == {"more": 62, "fewer": 22, "same": 95}


@pytest.mark.parametrize("track", TRACK_TABLE)
def test_per_track(track):
    n, sonnet, jev = TRACK_TABLE[track]
    for judge, want in (("sonnet5", sonnet), ("jev", jev)):
        t = SUMMARY["judges"][judge]["tracks"][track]
        assert t["n"] == n
        assert round(t["diff"], 3) == want
        assert tuple(round(x, 3) for x in t["ci"]) == TRACK_CI[(judge, track)]
