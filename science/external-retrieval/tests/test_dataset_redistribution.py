"""The shipped query set carries Exa's content under its MIT notice, and no verbatim text from
the gold papers."""
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"


def test_no_supporting_quotes_and_exa_notice_ships():
    rows = [json.loads(line) for line in (DATA / "publication_search_in_corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 668
    assert not any("supporting_quote" in r.get("metadata", {}) for r in rows)
    assert "Copyright (c) 2025 Exa Labs" in (DATA / "LICENSE-exa-labs").read_text(encoding="utf-8")


def test_dataset_matches_the_results_row_for_row():
    rows = [json.loads(line) for line in (DATA / "publication_search_in_corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    results = json.loads((DATA.parent / "results" / "gold_doc_results.json").read_text(encoding="utf-8"))["results"]
    assert {(r["query_id"], r["gold_paper"]["doi"]) for r in rows} == {(r["query_id"], r["gold_doi"]) for r in results}
