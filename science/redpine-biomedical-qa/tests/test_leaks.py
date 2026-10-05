"""Nothing identifying leaves the private data: structural checks on every shipped result,
every tracked file under scripts/, the README and judging.md, and, when Q2_FORBIDDEN names
the list, a whole-word check of every listed identifier over every file of the track."""
import json
import os
import re
import subprocess
import warnings
from pathlib import Path

import pytest

from scripts import guard

TRACK = Path(__file__).resolve().parents[1]
RESULT_FILES = sorted(p for p in (TRACK / "results").rglob("*") if p.is_file())
SCRIPT_FILES = sorted(TRACK / f for f in subprocess.run(
    ["git", "ls-files", "--cached", "--others", "--exclude-standard", "scripts"],
    cwd=TRACK, capture_output=True, text=True, check=True).stdout.split())
README = TRACK / "README.md"
JUDGING = TRACK / "judging.md"
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")
BANNED_KEYS = {"note", "quote", "span", "position", "slot", "duplicate_of", "exportedAt", "reviewer",
               "rater", "confidence", "probabilities"}
LOG_FILES = sorted(p for p in (TRACK / "logs").glob("*.json") if p.is_file())
LOG_BANNED_KEYS = BANNED_KEYS | {"text", "content", "passage", "passages", "results",
                                 "agent_retrieved_context", "agent_trajectory"}
# The README's setup table links where to get each API key; any other URL is a leak.
README_URLS = {"https://console.anthropic.com", "https://app.redpine.ai", "https://tavily.com", "https://openrouter.ai"}
MAX_CHARS = 400


def _rel(p):
    return str(p.relative_to(TRACK))


def test_there_is_something_to_check():
    names = {p.name for p in RESULT_FILES}
    assert {"verdicts.jsonl", "summary.json", "agreement.json"} <= names
    assert {"guard.py", "summarize.py", "agreement.py"} <= {p.name for p in SCRIPT_FILES}
    assert LOG_FILES, "no log under logs/"


@pytest.mark.parametrize("path", RESULT_FILES + SCRIPT_FILES + [README, JUDGING], ids=_rel)
def test_no_email_path_or_long_line(path):
    for n, line in enumerate(path.read_text().splitlines(), 1):
        assert not EMAIL.search(line), f"{_rel(path)} line {n}: email"
        assert "/Users/" not in line, f"{_rel(path)} line {n}: local path"
        assert len(line) <= MAX_CHARS or path.suffix == ".jsonl", f"{_rel(path)} line {n}: long line"
        urls = set(re.findall(r"https?://[\w.-]+(?:/[\w./-]*)?", line))
        if path == README:
            assert urls <= README_URLS, f"{_rel(path)} line {n}: unexpected URL"
        else:
            assert "http" not in line, f"{_rel(path)} line {n}: http"


def _walk(obj):
    """Yield every key and every string value in a parsed JSON document."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield "key", k
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)
    elif isinstance(obj, str):
        yield "str", obj


@pytest.mark.parametrize("path", [p for p in RESULT_FILES if p.suffix in (".json", ".jsonl")], ids=_rel)
def test_no_banned_key_or_long_string(path):
    text = path.read_text()
    docs = [json.loads(line) for line in text.splitlines()] if path.suffix == ".jsonl" else [json.loads(text)]
    for n, doc in enumerate(docs, 1):
        for kind, s in _walk(doc):
            assert not (kind == "key" and s in BANNED_KEYS), f"{_rel(path)} doc {n}: banned key {s}"
            assert len(s) <= MAX_CHARS, f"{_rel(path)} doc {n}: string over {MAX_CHARS} characters"


EMAIL_FILENAME_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "gif", "html", "htm", "docx", "xlsx"}


@pytest.mark.parametrize("path", LOG_FILES, ids=_rel)
def test_log_has_no_banned_key_email_or_path(path):
    log = json.loads(path.read_text())
    answer_strings = {s for row in log["rows"] for s in (row.get("answers") or {}).values()}
    for kind, s in _walk(log):
        assert not (kind == "key" and s in LOG_BANNED_KEYS), f"{_rel(path)}: banned key {s}"
        if kind == "str":
            if s in answer_strings:
                # a cited filename with an @ is not an address
                assert all(m.group(0).rsplit(".", 1)[-1].lower() in EMAIL_FILENAME_EXTENSIONS
                           for m in EMAIL.finditer(s)), f"{_rel(path)}: email"
            else:
                assert not EMAIL.search(s), f"{_rel(path)}: email"
            assert "/Users/" not in s, f"{_rel(path)}: local path"
    # answers may be long and may cite URLs; every other string keeps the 400-character rule
    for row in log["rows"]:
        for key, value in row.items():
            if key in ("answers", "question", "gold"):
                continue
            for kind, s in _walk(value):
                assert not (kind == "str" and len(s) > MAX_CHARS), f"{_rel(path)} {row['id']}: long string under {key}"


def forbidden_words():
    """The list named by Q2_FORBIDDEN, or None with a visible warning when it is not set."""
    if not os.environ.get("Q2_FORBIDDEN"):
        warnings.warn("Q2_FORBIDDEN not set: the forbidden-identifier check is skipped; "
                      "the structural leak checks still run", UserWarning)
        return None
    return guard.load_forbidden()


def test_leak_check_without_list_warns(monkeypatch):
    monkeypatch.delenv("Q2_FORBIDDEN", raising=False)
    with pytest.warns(UserWarning, match="Q2_FORBIDDEN not set"):
        assert forbidden_words() is None


def _log_hits(path, words):
    """Scan one log file for forbidden identifiers, as a JSON walk rather than a line scan.

    Every key and every string is checked case-insensitively, except the values under
    rows[*].answers and the query values under rows[*].traces[*].searches[*].query: those are
    free model- or user-written text where a listed identifier may also be a common lowercase
    word, so they are checked only against a capitalized, case-sensitive form of each word (a
    name, not a common word).

    A hit string never carries anything read from the file: not the matched word, not a key
    name, not a row id, not an arm name. Locators name positions only: rows by index
    (rows[i]), keys as "a header key" or "a key under <path>", and values by a field path
    built from fixed schema names (answers, traces, searches, query), with the arm replaced by
    its index in log["arms"] (arms[k]) so an arm string can never appear either.
    """
    doc = json.loads(path.read_text())
    normal = guard.forbidden_pattern(words)
    capital = re.compile(r"(?<!\w)(?:" + "|".join(w[0].upper() + re.escape(w[1:]) for w in words) + r")(?!\w)")
    try:
        rel = _rel(path)  # relative to TRACK for a real log; the bare name for a test's tmp_path
    except ValueError:
        rel = path.name
    arms = doc.get("arms", [])
    hits = []

    def arm_slot(arm):
        return arms.index(arm) if arm in arms else "?"

    def key_hit(where):
        hits.append(f"{rel}: a header key" if where is None else f"{rel}: a key under {where}")

    def value_hit(where):
        hits.append(f"{rel}: {where}")

    def scan_generic(obj, where):
        """Case-insensitive scan of an arbitrary nested structure; reports the position only,
        never the key name or the matched text."""
        for kind, s in _walk(obj):
            if kind == "key" and normal.search(s):
                key_hit(where)
            elif kind == "str" and normal.search(s):
                value_hit(where)

    for k, v in doc.items():
        if normal.search(k):
            key_hit(None)
        if k != "rows":
            scan_generic(v, "header")

    for i, row in enumerate(doc.get("rows", [])):
        row_where = f"rows[{i}]"
        for key, value in row.items():
            if normal.search(key):
                key_hit(row_where)
            if key == "answers":
                for arm, ans in (value or {}).items():
                    if normal.search(arm):
                        key_hit(f"{row_where}.answers")
                    if capital.search(str(ans)):
                        value_hit(f"{row_where}.answers.arms[{arm_slot(arm)}]")
            elif key == "traces":
                for arm, trace in (value or {}).items():
                    if normal.search(arm):
                        key_hit(f"{row_where}.traces")
                    arm_where = f"{row_where}.traces.arms[{arm_slot(arm)}]"
                    for tk, tv in (trace or {}).items():
                        if normal.search(tk):
                            key_hit(arm_where)
                        if tk == "searches":
                            for j, s in enumerate(tv or []):
                                for sk, sv in s.items():
                                    if normal.search(sk):
                                        key_hit(f"{arm_where}.searches[{j}]")
                                    if sk == "query":
                                        if capital.search(str(sv)):
                                            value_hit(f"{arm_where}.searches[{j}].query")
                                    elif isinstance(sv, str) and normal.search(sv):
                                        value_hit(f"{arm_where}.searches[{j}]")
                        else:
                            scan_generic(tv, arm_where)
            else:
                scan_generic(value, row_where)
    return hits


def test_log_hits_routes_patterns(tmp_path):
    """A synthetic log with a made-up word, exercising every branch of _log_hits: a header
    key equal to the word, a lowercase occurrence in an answer and in a query (neither should
    count), a capitalized occurrence in an answer (should count), and a lowercase occurrence
    in a question (should count) — and never the word itself in any returned hit string."""
    log = {
        "arms": ["websearch", "connect"],
        "Zyxwort": "unrelated header value",
        "rows": [{
            "id": "q1",
            "question": "Does zyxwort raise risk?",
            "answers": {
                "websearch": "The answer mentions zyxwort in passing.",
                "connect": "The answer mentions Zyxwort in passing.",
            },
            "traces": {
                "websearch": {"searches": [{"tool": "web", "step": 1, "query": "zyxwort trial",
                                            "n_results": 3, "budget_refused": False}]},
                "connect": {"searches": []},
            },
        }],
    }
    path = tmp_path / "log.json"
    path.write_text(json.dumps(log))
    hits = _log_hits(path, ["Zyxwort"])
    assert len(hits) == 3  # header key, question (lowercase), connect answer (capitalized)
    assert not any("zyxwort" in h.lower() for h in hits)


def test_no_forbidden_identifier_in_track():
    words = forbidden_words()
    if words is None:
        pytest.skip("Q2_FORBIDDEN not set")
    assert words, "Q2_FORBIDDEN names an empty list"
    files = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "."],
                           cwd=TRACK, capture_output=True, text=True, check=True).stdout.split()
    assert "results/summary.json" in files and "results/verdicts.jsonl" in files
    assert any(f.startswith("logs/") for f in files)
    pattern = guard.forbidden_pattern(words)
    hits = []
    for i, f in enumerate(files):
        if f.startswith("logs/"):
            hits += _log_hits(TRACK / f, words)
        else:
            # positions only: a path could itself contain a listed word
            hits += [f"files[{i}] line {n}" for n, line in enumerate((TRACK / f).read_text(errors="replace").splitlines(), 1)
                     if pattern.search(line)]
    hits += [f"filename files[{i}]" for i, f in enumerate(files) if pattern.search(f)]
    assert hits == []  # file and line only, never the identifier
