"""Scoring for ScholarQABench's SciFact subtask.

The benchmark's own rule, re-derived from the source's
scripts/citation_correctness_eval.py (github.com/AkariAsai/ScholarQABench,
functions remove_citations and compute_match): strip bracketed citation
markers from both sides, lowercase both, and test whether the gold label is a
substring of the answer. No judge model is involved. Gold is "true" or "false".

remove_citations and compute_match are derived from that file, which is
released under the MIT License, Copyright (c) 2024 Akari Asai:

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import re

LABELS = ("true", "false")

_CITATION = re.compile(r"\[\d+(?:,\s*\d+)*\]")


def remove_citations(text: str) -> str:
    text = _CITATION.sub("", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text.replace(" .", ".").replace(" ,", ",")


def compute_match(gold: str, answer: str) -> bool:
    return remove_citations(gold).lower() in remove_citations(answer or "").lower()
