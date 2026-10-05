"""Recompute results/summary.json from results/verdicts.jsonl.

    uv run python scripts/summarize.py

Per judge (Sonnet 5; Jev, the mean of its two passes per question) and per score:
the mean per arm, the paired difference connect minus websearch and its 95% t
interval; the per-question split on coverage; the coverage difference per track.
Correctness is recomputed per question as the F1 of coverage and one minus the
contradiction rate (0 when both are 0), after averaging the Jev passes.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.guard import OUT, check_forbidden, load_forbidden  # noqa: E402
from src.data import load_questions  # noqa: E402

JUDGE_PASSES = {"sonnet5": ("sonnet5",), "jev": ("jev1", "jev2")}
SCORES = ("coverage", "contradiction_rate", "correctness")


def f1(c, k1):
    """Correctness: the harmonic mean of coverage and one minus the contradiction rate."""
    return 2 * c * k1 / (c + k1) if c + k1 else 0.0


def interval(d):
    d = np.asarray(d, dtype=float)
    m = float(d.mean())
    half = float(stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)))
    return {"n": len(d), "diff": m, "ci": [m - half, m + half]}


def per_question(rows, passes):
    """{arm: {id: {score: value}}} with the passes averaged per question."""
    acc = {}
    for r in rows:
        if r["judge"] in passes:
            acc.setdefault(r["arm"], {}).setdefault(r["id"], []).append(r)
    out = {}
    for arm, by_id in acc.items():
        out[arm] = {}
        for qid, rs in by_id.items():
            assert len(rs) == len(passes), (qid, arm)
            cov = float(np.mean([r["coverage"] for r in rs]))
            con = float(np.mean([r["contradiction_rate"] for r in rs]))
            out[arm][qid] = {"coverage": cov, "contradiction_rate": con, "correctness": f1(cov, 1 - con)}
    return out


def summarize(verdict_rows, tracks):
    ids = list(dict.fromkeys(r["id"] for r in verdict_rows))
    result = {"n_questions": len(ids),
              "n_claims_per_arm": {arm: sum(len(r["verdicts"]) for r in verdict_rows
                                            if r["arm"] == arm and r["judge"] == "sonnet5")
                                   for arm in ("websearch", "connect")},
              "judges": {}}
    counts = set(result["n_claims_per_arm"].values())
    assert len(counts) == 1, result["n_claims_per_arm"]
    result["n_claims_per_arm"] = counts.pop()
    for judge, passes in JUDGE_PASSES.items():
        q = per_question(verdict_rows, passes)
        w, c = q["websearch"], q["connect"]
        scores = {}
        for s in SCORES:
            d = [c[i][s] - w[i][s] for i in ids]
            scores[s] = {"websearch": float(np.mean([w[i][s] for i in ids])),
                         "connect": float(np.mean([c[i][s] for i in ids])),
                         **{k: v for k, v in interval(d).items() if k != "n"}}
        d = np.array([c[i]["coverage"] - w[i]["coverage"] for i in ids])
        by_track = {}
        for t in dict.fromkeys(tracks[i] for i in ids):
            by_track[t] = interval([c[i]["coverage"] - w[i]["coverage"] for i in ids if tracks[i] == t])
        result["judges"][judge] = {
            "passes": list(passes), "scores": scores,
            "coverage_split": {"more": int((d > 0).sum()), "fewer": int((d < 0).sum()), "same": int((d == 0).sum())},
            "tracks": by_track}
    return result


def main():
    rows = [json.loads(line) for line in (OUT / "verdicts.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = summarize(rows, {q["id"]: q["track"] for q in load_questions()})
    text = json.dumps(summary, indent=2) + "\n"
    check_forbidden(text.splitlines(), "summary.json", load_forbidden())
    (OUT / "summary.json").write_text(text, encoding="utf-8")
    for judge, j in summary["judges"].items():
        for s, v in j["scores"].items():
            print(f"{judge:7} {s:18} websearch {v['websearch']:.3f} connect {v['connect']:.3f} "
                  f"diff {v['diff']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]")
        print(f"{judge:7} coverage split {j['coverage_split']}")
        for t, v in j["tracks"].items():
            print(f"{judge:7}   {t:16} n {v['n']:3} diff {v['diff']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]")
    print(f"n_questions {summary['n_questions']}, n_claims_per_arm {summary['n_claims_per_arm']}")


if __name__ == "__main__":
    main()
