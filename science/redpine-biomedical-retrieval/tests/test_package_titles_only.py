"""The shipped judging package carries result titles only: the snippet text the judges read is
PubMed and Redpine Science content and is not redistributed."""
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "data" / "package.json"


def test_every_result_is_title_only():
    package = json.loads(PACKAGE.read_text(encoding="utf-8"))
    results = [r for q in package["queries"] for side in ("left", "right") for r in q[side]]
    assert len(package["queries"]) == 115 and len(results) == 1144
    assert all(set(r) == {"title"} and r["title"].strip() for r in results)
