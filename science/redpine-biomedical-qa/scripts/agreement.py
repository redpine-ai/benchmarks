"""Recompute results/agreement.json from results/verdicts.jsonl.

    uv run python scripts/agreement.py

Judge against judge: Jev pass 1 against Sonnet 5 on every required claim, per arm. Every
statistic carries a one-sentence definition under "definitions".
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.guard import OUT, check_forbidden, load_forbidden  # noqa: E402

LABELS = ["contradicted", "not_addressed", "stated"]  # the three real labels; order only indexes

DEFINITIONS = {
    "judge_agreement": "Jev pass 1 against Sonnet 5 on every required claim of every answer in one arm: "
                       "the share of claims given the same verdict, and Krippendorff's alpha (nominal, "
                       "three labels) over the same claims.",
}


def alpha(m, k):
    """Krippendorff's alpha, nominal, for a units x raters matrix of level indices (nan missing)."""
    o = np.zeros((k, k))
    for row in m:
        v = row[~np.isnan(row)].astype(int)
        if len(v) < 2:
            continue
        c = np.bincount(v, minlength=k).astype(float)
        o += (np.outer(c, c) - np.diag(c)) / (len(v) - 1)
    n, marg = o.sum(), o.sum(axis=1)
    do = (o.sum() - np.trace(o)) / n
    de = (n * n - (marg ** 2).sum()) / (n * (n - 1))
    return float(1 - do / de)


def compute(verdict_rows):
    verdict = {(r["judge"], r["id"], r["arm"], v["claim_index"]): v["verdict"]
               for r in verdict_rows for v in r["verdicts"]}
    out = {"definitions": DEFINITIONS, "judge_agreement": {}}
    for arm in ("connect", "websearch"):
        pairs = [(verdict[("jev1", i, a, c)], verdict[("sonnet5", i, a, c)])
                 for (j, i, a, c) in verdict if j == "sonnet5" and a == arm]
        m = np.array([[LABELS.index(x), LABELS.index(y)] for x, y in pairs], float)
        out["judge_agreement"][arm] = {"n_claims": len(pairs),
                                       "agreement": float(np.mean([x == y for x, y in pairs])),
                                       "alpha": alpha(m, 3)}
    return out


def load_rows():
    return [json.loads(line) for line in (OUT / "verdicts.jsonl").read_text(encoding="utf-8").splitlines()]


def main():
    result = compute(load_rows())
    text = json.dumps(result, indent=2) + "\n"
    check_forbidden(text.splitlines(), "agreement.json", load_forbidden())
    (OUT / "agreement.json").write_text(text, encoding="utf-8")
    for arm, v in result["judge_agreement"].items():
        print(f"judge agreement {arm:9} n {v['n_claims']} agreement {v['agreement']:.3f} alpha {v['alpha']:.3f}")


if __name__ == "__main__":
    main()
