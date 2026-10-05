import json
from pathlib import Path

LOGS = sorted((Path(__file__).resolve().parents[1] / "logs").glob("scifact_*.json"))
EXPECTED = {
    "scifact_20260911_110150": (197, {"closed_book": 88.3, "web_forced": 92.4, "redpine_forced": 94.9, "redpine_first": 93.9}),
    "scifact_20260915_132619": (199, {"closed_book": 86.9, "web_forced": 93.5, "redpine_forced": 93.5, "redpine_first": 93.0}),
    "scifact_20260915_140004": (199, {"closed_book": 87.4, "web_forced": 94.0, "redpine_forced": 96.0, "redpine_first": 93.5}),
    "scifact_20260915_145823": (199, {"closed_book": 87.4, "web_forced": 93.5, "redpine_forced": 94.5, "redpine_first": 94.5}),
    "scifact_20260915_150156": (196, {"closed_book": 87.8, "web_forced": 92.3, "redpine_forced": 92.9, "redpine_first": 94.4}),
}


def test_five_logs_present():
    assert [p.stem for p in LOGS] == sorted(EXPECTED)


def test_scores_recompute_from_rows():
    for path in LOGS:
        log = json.loads(path.read_text())
        n_expected, scores_expected = EXPECTED[path.stem]
        kept = [r for r in log["rows"] if r["excluded"] is None]
        assert len(kept) == n_expected == log["n_complete"]
        for arm in log["arms"]:
            pct = round(100 * sum(r["judge"][arm]["official_correct"] for r in kept) / len(kept), 1)
            assert pct == scores_expected[arm] == log["scores"][arm], (path.stem, arm)


def test_header_constants():
    for path in LOGS:
        log = json.loads(path.read_text())
        assert log["model"] == "claude-sonnet-5" and log["served_model_id"] == "eu.anthropic.claude-sonnet-5"
        assert log["tool_budget"] == 16 and log["max_tokens"] == 8192
        assert log["thinking"] == {"type": "adaptive"}
        assert log["redpine_top_k"] == 10 and log["web_top_k"] == 10 and log["passage_char_limit"] == 4000
        assert log["web_search_backend"] == "tavily/advanced (client-side)"
        assert all(set(r) == {"id", "question", "gold", "excluded", "judge"} for r in log["rows"])


def test_every_log_uses_one_exclusion_rule():
    # A row is excluded from every arm when any arm failed to answer. No run is
    # special-cased.
    for path in LOGS:
        log = json.loads(path.read_text())
        assert "refusals_scored_as_incorrect" not in log
        counted = sum(log["excluded"].values())
        assert counted == len(log["rows"]) - log["n_complete"]
        assert set(log["excluded"]) == {"arm_error", "refusal"}
        assert all(r["excluded"] in (None, "arm_error", "refusal") for r in log["rows"])


def test_means_match_report():
    import statistics
    by_arm = {a: [] for a in ("closed_book", "web_forced", "redpine_forced", "redpine_first")}
    for n, scores in EXPECTED.values():
        for a, s in scores.items():
            by_arm[a].append(s)
    means = {a: round(statistics.mean(v), 1) for a, v in by_arm.items()}
    assert means == {"closed_book": 87.6, "web_forced": 93.1, "redpine_forced": 94.4, "redpine_first": 93.9}
