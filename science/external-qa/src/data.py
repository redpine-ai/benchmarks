"""ScholarQABench's SciFact subtask: 208 biomedical claims labelled true or false.

Downloaded from the benchmark's own repository on first use and cached under
data/ (gitignored). The data is released by its authors under ODC-BY; this
repository does not redistribute it. Rows are identified as `scifact_<i>` by
zero-based position in the file, which is how the published logs name them.
"""
import json
from pathlib import Path

import httpx

# Pinned to the upstream commit the logged runs used, so an upstream edit cannot change the benchmark.
UPSTREAM_URL = ("https://raw.githubusercontent.com/AkariAsai/ScholarQABench/95e6fc52b0a8/"
                "data/single_paper_tasks/scifact_test.jsonl")
DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "scifact_test.jsonl"


def load_claims(path: Path | None = None, *, get=httpx.get) -> list[dict]:
    path = path or DEFAULT_PATH
    if not path.exists():
        resp = get(UPSTREAM_URL, timeout=60, follow_redirects=True)
        resp.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(resp.text, encoding="utf-8")
    claims = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        claim, gold = (row.get("input") or "").strip(), (row.get("answer") or "").strip().lower()
        if claim and gold:
            claims.append({"id": f"scifact_{i}", "question": claim, "gold": gold})
    return claims
