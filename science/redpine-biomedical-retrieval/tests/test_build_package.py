import asyncio

import pytest

from src.build_package import (
    PubMed, blind, check_unique_ids, clean_snippet, fetch_topk, normalize, parse_pubmed_xml,
)


def test_blind_answer_key_recovers_true_rank_and_hides_system():
    by_system = {s: [{"title": f"paper {s[0]}{i}", "snippet": "x.", "snippet_full": "x."}
                     for i in range(1, 6)] for s in ("redpine", "pubmed")}
    for _ in range(20):
        pkg, key = blind({"id": "q1", "query": "q"}, by_system)
        assert {key["left_system"], key["right_system"]} == {"redpine", "pubmed"}
        assert "redpine" not in str(pkg) and "pubmed" not in str(pkg)
        for side in ("left", "right"):
            system = key[f"{side}_system"]
            for item, rank in zip(pkg[side], key[f"{side}_true_ranks"]):
                assert item == by_system[system][rank - 1]


def test_sides_cut_to_equal_count():
    class Fake:
        def __init__(self, n):
            self.n = n

        async def search(self, query, n):
            return [{"title": f"a distinct paper title number {i}", "text": "Sentence."}
                    for i in range(self.n)]

    out = asyncio.run(fetch_topk({"redpine": Fake(8), "pubmed": Fake(2)}, "q"))
    assert len(out["redpine"]) == len(out["pubmed"]) == 2


def test_normalize_drops_empty_and_duplicate_titles_before_top_k():
    hits = [{"title": "Aortic valve sclerosis outcomes in adults", "text": "One."},
            {"title": "Aortic valve sclerosis outcomes in adults", "text": "Same paper, chunk two."},
            {"title": "Empty result with a long enough title", "text": ""},
            *[{"title": f"another genuinely different paper {i}", "text": "Two."} for i in range(5)]]
    out = normalize(hits, top_k=5)
    assert len(out) == 5
    assert [h["title"] for h in out].count("Aortic valve sclerosis outcomes in adults") == 1
    assert all(h["snippet"] for h in out)


def test_clean_snippet_strips_markdown_and_ends_on_a_sentence():
    text = "## Abstract\nFirst sentence here. Second one. Third one. Fourth one."
    assert clean_snippet(text) == "First sentence here. Second one. Third one."


def test_pubmed_relaxation_is_deterministic_and_stops_at_two_words():
    terms = PubMed(client=None).relaxations("What is the role of statins in aortic valve sclerosis?")
    assert terms == ["statins aortic valve sclerosis", "statins aortic valve", "statins aortic"]


def test_parse_pubmed_xml():
    xml = """<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>42</PMID><Article>
      <ArticleTitle>A <i>title</i></ArticleTitle><Abstract><AbstractText>Part one.</AbstractText>
      <AbstractText>Part two.</AbstractText></Abstract></Article></MedlineCitation></PubmedArticle>
      </PubmedArticleSet>"""
    assert parse_pubmed_xml(xml) == {"42": {"title": "A title", "text": "Part one.\nPart two."}}


def test_duplicate_ids_fail_before_any_call():
    with pytest.raises(ValueError, match="q1"):
        check_unique_ids([{"id": "q1"}, {"id": "q1"}])
