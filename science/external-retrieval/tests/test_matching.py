from src.matching import find_rank, normalize_doi, normalize_title, summarize
from src.retrievers import Result


def test_doi_match_wins_regardless_of_title():
    results = [Result(title="Unrelated Page Title", doi="10.1/x", snippet="")]
    assert find_rank("10.1/x", "The Actual Gold Title", results) == 1


def test_doi_extracted_from_url_when_no_structured_field():
    # Tavily and Exa never populate .doi -- a doi.org link in the URL should
    # still count as a DOI match, same as exa-labs/benchmarks' own grader.
    results = [Result(title="Unrelated page title", doi=None,
                      url="https://doi.org/10.1038/s41586-021-03819-2", snippet="")]
    assert find_rank("10.1038/s41586-021-03819-2", "The Actual Gold Title", results) == 1


def test_doi_extracted_from_snippet_text():
    results = [Result(title="Unrelated page title", doi=None, url="https://example.com/page",
                      snippet="Cited as doi:10.1016/j.cell.2020.01.001 in the references.")]
    assert find_rank("10.1016/j.cell.2020.01.001", "The Actual Gold Title", results) == 1


def test_doi_extraction_does_not_false_positive_on_unrelated_doi():
    results = [Result(title="Unrelated page title", doi=None,
                      url="https://doi.org/10.1038/s41586-021-00000-0", snippet="")]
    assert find_rank("10.1038/s41586-021-03819-2", "The Actual Gold Title", results) is None


def test_doi_normalizes_prefix_and_case():
    assert normalize_doi("https://doi.org/10.1/X") == normalize_doi("10.1/x")


def test_title_containment_survives_venue_affix():
    gold = "A Sufficiently Long Paper Title About Something Specific"
    candidate = gold + " | Nature Scientific Reports"
    assert find_rank(None, gold, [Result(title=candidate, snippet="")]) == 1


def test_short_titles_require_exact_match():
    # Below MIN_TITLE_LEN, containment is disabled -- a short generic phrase
    # could otherwise sit inside an unrelated longer title.
    assert find_rank(None, "Short Title", [Result(title="A Short Title About Cats")]) is None
    assert find_rank(None, "Short Title", [Result(title="Short Title")]) == 1


def test_rank_is_one_indexed_and_first_match_wins():
    results = [Result(title="Wrong"), Result(title="A Sufficiently Long Correct Gold Title Here")]
    assert find_rank(None, "A Sufficiently Long Correct Gold Title Here", results) == 2


def test_not_found_returns_none():
    assert find_rank("10.1/x", "Gold Title", [Result(title="Nothing to do with it")]) is None


def test_normalize_title_empty_and_none():
    assert normalize_title(None) is None
    assert normalize_title("   ") is None


def test_normalize_title_strips_html_tags_whole_not_just_brackets():
    # Found live in this dataset: a gold title keeps literal "<sup>+</sup>"
    # around a superscript. Stripping only the angle brackets would leave
    # "sup sup" as residue, silently breaking containment against a clean
    # candidate title that says the same thing without markup.
    messy = normalize_title("PI3K promotes CD4<sup>+</sup> T-cell interactions")
    clean = normalize_title("PI3K promotes CD4 T-cell interactions")
    assert messy == clean


def test_title_match_survives_html_residue_in_gold_title():
    gold = "PI3K promotes CD4<sup>+</sup> T-cell interactions with antigen-presenting cells"
    candidate = "PI3K promotes CD4 T-cell interactions with antigen-presenting cells - PMC"
    assert find_rank(None, gold, [Result(title=candidate)]) == 1


def test_summarize_closed_form_metrics():
    # Two found (ranks 1 and 2), one miss.
    s = summarize([1, 2, None])
    assert s["n"] == 3
    assert s["n_found"] == 2
    assert abs(s["recall"] - 2 / 3) < 1e-9
    assert abs(s["mrr"] - (1 / 1 + 1 / 2) / 3) < 1e-9
    # nDCG at rank 1 is 1/log2(2) = 1.0, contributing its full weight.
    assert s["rank_distribution"] == {1: 1, 2: 1}


def test_doi_from_stored_snippet_dois_matches_like_snippet_text():
    from types import SimpleNamespace
    gold = "10.1016/j.cell.2020.01.001"
    stored = SimpleNamespace(title="Unrelated", doi=None, url=None,
                             snippet_dois=[gold])
    assert find_rank(gold, "Some Gold Title", [stored]) == 1
