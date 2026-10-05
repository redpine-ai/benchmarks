"""The forbidden-identifier gate shared by the scripts and the tests.

Q2_FORBIDDEN names a file, outside the repository, with one forbidden identifier per
line. Nothing here ever prints a listed identifier: a hit is reported by file and line.
"""
import os
import re
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "results"


def load_forbidden():
    path = os.environ.get("Q2_FORBIDDEN")
    if not path:
        return []
    return [w.strip() for w in Path(path).read_text(encoding="utf-8").splitlines() if w.strip()]


def forbidden_pattern(words):
    return re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, words)) + r")(?!\w)", re.IGNORECASE)


def check_forbidden(lines, where, words):
    """Exit naming the file and line of the first hit, never the matched identifier."""
    if not words:
        return
    pattern = forbidden_pattern(words)
    for n, line in enumerate(lines, 1):
        if pattern.search(line):
            sys.exit(f"forbidden identifier in {where} line {n}")
