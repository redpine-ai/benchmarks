from src.grade import LABELS, compute_match, remove_citations


def test_labels():
    assert LABELS == ("true", "false")


def test_remove_citations_strips_bracketed_numbers():
    assert remove_citations("Supported [0] by evidence [1, 2].") == "Supported by evidence."


def test_match_is_case_insensitive_and_ignores_citations():
    assert compute_match("true", "Answer: TRUE [3]\n\nThe claim holds.")


def test_mismatch():
    assert not compute_match("false", "Answer: true\n\nThe claim holds.")


def test_official_rule_is_substring_containment():
    # The benchmark's own rule is substring containment of the gold label in the
    # lowercased answer, so "untrue" contains "true". Documented, not corrected.
    assert compute_match("true", "Answer: false. The claim is untrue.")
