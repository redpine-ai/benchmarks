from src.benchmark import _stored_result
from src.retrievers import Result


def test_stored_result_keeps_matching_relevant_fields_and_never_the_snippet_text():
    r = Result(title="A Title", doi="10.1/x", url="https://example.com",
               snippet="Licensed text citing doi:10.1016/j.cell.2020.01.001 here.")
    stored = _stored_result(r)
    assert stored == {"title": "A Title", "doi": "10.1/x", "url": "https://example.com",
                      "snippet_dois": ["10.1016/j.cell.2020.01.001"]}
    assert "snippet" not in stored


def test_stored_result_handles_empty_snippet():
    r = Result(title="A Title")
    assert _stored_result(r)["snippet_dois"] == []
