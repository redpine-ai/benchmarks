"""The expert-validated question set, one JSON object per line.

Fields: id, track, question, gold, doi, access; see data/README.md.
"""
import json
import sys
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "questions.jsonl"


def load_questions(path: Path = DEFAULT_PATH) -> list[dict]:
    if not path.exists():
        sys.exit(f"question set not found at {path}; see README.md (Data)")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for r in rows:
        if not {"id", "question"} <= set(r):
            sys.exit(f"{path}: every row needs 'id' and 'question'")
    return rows
