import pytest

from scripts import guard


def test_load_forbidden_reads_one_word_per_line(tmp_path, monkeypatch):
    lst = tmp_path / "forbidden.txt"
    lst.write_text("Zyxwort\n\n  Quobble \n")
    monkeypatch.setenv("Q2_FORBIDDEN", str(lst))
    assert guard.load_forbidden() == ["Zyxwort", "Quobble"]


def test_load_forbidden_empty_when_unset(monkeypatch):
    monkeypatch.delenv("Q2_FORBIDDEN", raising=False)
    assert guard.load_forbidden() == []


def test_forbidden_pattern_is_whole_word_case_insensitive():
    p = guard.forbidden_pattern(["Zyxwort"])
    assert p.search("said ZYXWORT.") and not p.search("a zyxworth b")


def test_check_forbidden_names_line_not_value():
    guard.check_forbidden(["a zyxworth b"], "x.jsonl", ["Zyxwort"])
    with pytest.raises(SystemExit) as e:
        guard.check_forbidden(["fine", "said ZYXWORT."], "x.jsonl", ["Zyxwort"])
    assert "x.jsonl line 2" in str(e.value) and "zyxwort" not in str(e.value).lower()


def test_check_forbidden_no_words_is_noop():
    guard.check_forbidden(["anything"], "x", [])
