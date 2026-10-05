"""Builds a blinded judging package from live Redpine Science and PubMed calls.

For every question: one direct search call per system (no agent, no reformulation),
top 5 each, cleaned to title + snippet, then blinded and randomized twice per question:

  1. Which system is `left` vs `right`.
  2. The display order within each side, so a judge can't read retrieval rank off the
     page. The true rank behind each display position goes only into the answer key;
     score_ratings.py resolves it back before computing DCG/precision.

Two files come out, both stamped with the same `package_id` so score_ratings.py can
refuse a ratings file scored against a different build:

  - package_<run>.json: what the judge sees (no system names, URLs, DOIs or dates:
    those format differently between the two systems and would give them away).
  - answer_key_<run>.json: which system is which side, and each side's true ranks.
    Never share it with a judge.

Both sides are cut to the same item count per question, since a side that is always
shorter is itself a tell.

PubMed gets the question exactly as written. When that returns fewer than 8 hits (the top 5 plus 3 spares)
(PubMed's own search often returns nothing for a full natural-language sentence),
it retries with stopwords removed and then with progressively fewer trailing
words, down to two, keeping whichever attempt found the most results. The
relaxation is deterministic, so anyone rerunning this gets the same queries.

Run:
  uv run python -m src.build_package data/questions.json runs/my_package

Needs REDPINE_API_KEY. PubMed (NCBI E-utilities) needs no key. Output is written
after every question, so a crash keeps the questions already done.
"""

import asyncio
import json
import os
import random
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

ARMS = ["redpine", "pubmed"]
TOP_K = 5
OVERFETCH = 3  # spare candidates so dropping empty or duplicate results doesn't shorten a side
CONCURRENCY = 6
PUBMED_PACING_SECONDS = 0.4  # NCBI allows 3 requests/second without an API key

# ---- snippet cleaning ------------------------------------------------------------

SNIPPET_MAX_CHARS, SNIPPET_MAX_SENTENCES = 450, 3
SNIPPET_FULL_MAX_CHARS, SNIPPET_FULL_MAX_SENTENCES = 1100, 8  # the UI's "show more" text

_MD_HEADER_RE = re.compile(r"^#{1,6}\s*.*$", re.MULTILINE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def clean_snippet(text: str, max_chars: int = SNIPPET_MAX_CHARS,
                  max_sentences: int = SNIPPET_MAX_SENTENCES) -> str:
    """Strips markdown headers (Redpine's chunks carry them, PubMed abstracts don't,
    so they'd identify the system) and cuts at a sentence boundary on both sides,
    so neither side reliably ends in a mid-sentence '...'."""
    text = " ".join(_MD_HEADER_RE.sub(" ", text or "").split())
    if not text:
        return ""
    kept, length = [], 0
    for s in _SENTENCE_SPLIT_RE.split(text):
        if kept and (len(kept) >= max_sentences or length + len(s) + 1 > max_chars):
            break
        kept.append(s)
        length += len(s) + 1
    result = " ".join(kept)
    if result[-1] not in ".!?":  # no sentence end within budget: word-boundary cut
        cut = result[:max_chars]
        space = cut.rfind(" ")
        if space > max_chars * 0.6:
            cut = cut[:space]
        result = cut.rstrip(".,;: ") + "..."
    return result


TITLE_DEDUPE_MIN_LEN = 20


def title_key(title: str | None) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", "", (title or "").lower()).split())


def is_duplicate_title(key: str, seen: list[str]) -> bool:
    """Exact or prefix match in either direction, so a truncated title still matches
    its full form. Short keys never match: too likely a coincidence."""
    if len(key) < TITLE_DEDUPE_MIN_LEN:
        return False
    return any(key.startswith(s) or s.startswith(key)
               for s in seen if len(s) >= TITLE_DEDUPE_MIN_LEN)


def normalize(hits: list[dict], top_k: int = TOP_K) -> list[dict]:
    """Drops empty and duplicate-title hits BEFORE truncating to top_k, so the
    over-fetched spares fill the gaps and rank N means the N-th real result on
    both sides. Redpine returns chunks, so one paper can appear several times."""
    out, seen = [], []
    for h in hits:
        snippet = clean_snippet(h["text"])
        key = title_key(h["title"])
        if not snippet or not (h.get("title") or "").strip() or is_duplicate_title(key, seen):
            continue
        if key:
            seen.append(key)
        out.append({"title": h["title"], "snippet": snippet,
                    "snippet_full": clean_snippet(h["text"], SNIPPET_FULL_MAX_CHARS,
                                                  SNIPPET_FULL_MAX_SENTENCES)})
    return out[:top_k]


# ---- retrievers ------------------------------------------------------------------

_TRANSIENT = {408, 409, 429, 500, 502, 503, 504}


async def _retry(fn, attempts: int = 4):
    for i in range(attempts):
        try:
            return await fn()
        except httpx.HTTPStatusError as e:
            if i == attempts - 1 or e.response.status_code not in _TRANSIENT:
                raise
        except httpx.TransportError:
            if i == attempts - 1:
                raise
        await asyncio.sleep(min(8.0, 0.5 * 2 ** i))


class Redpine:
    def __init__(self, client: httpx.AsyncClient, api_key: str, collection: str):
        self._client, self._api_key = client, api_key
        self._collections = [c.strip() for c in collection.split(",") if c.strip()]

    async def search(self, query: str, n: int) -> list[dict]:
        resp = await self._client.post(
            "/api/v1/search/query", headers={"Authorization": f"Bearer {self._api_key}"},
            json={"collections": self._collections, "query": query, "limit": n})
        resp.raise_for_status()
        return [{"title": (r.get("metadata") or {}).get("title"), "text": r.get("text") or ""}
                for r in resp.json().get("results", [])[:n]]


class PubMed:
    _STOPWORDS = frozenset("""a an the in with of to for and or as by from on at is
        are was were be been being can could should would do does did will shall
        may might must has have had undergoing what which how why who whom this
        these those it its rather than compared versus vs true key specific
        significant reliable improvement benefit role provide provided performed
        continue continued receive receiving contribute contributes correlate
        correlates predict predicting associated determinant patients patient
        early clinical guideline-directed medical therapy""".split())
    _MIN_WORDS = 2  # thinner than this returns off-topic term collisions

    def __init__(self, client: httpx.AsyncClient):
        self._client = client
        self._lock = asyncio.Lock()  # one question at a time keeps us under NCBI's limit

    def relaxations(self, query: str) -> list[str]:
        """Fallback queries, most specific first: stopwords removed, then one
        trailing word fewer each step, down to _MIN_WORDS."""
        words = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9()/.'-]*", query.rstrip("?"))
                 if w.lower() not in self._STOPWORDS]
        return [" ".join(words[:k]) for k in range(len(words), self._MIN_WORDS - 1, -1)]

    async def search(self, query: str, n: int) -> list[dict]:
        async with self._lock:
            best = await self._search_term(query, n)
            for term in self.relaxations(query):
                if len(best) >= n:
                    break
                candidate = await self._search_term(term, n)
                if len(candidate) > len(best):
                    best = candidate
            return best

    async def _search_term(self, term: str, n: int) -> list[dict]:
        await asyncio.sleep(PUBMED_PACING_SECONDS)
        resp = await _retry(lambda: self._get("/entrez/eutils/esearch.fcgi", {
            "db": "pubmed", "term": term, "retmax": n, "retmode": "json", "sort": "relevance"}))
        ids = resp.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        await asyncio.sleep(PUBMED_PACING_SECONDS)
        resp = await _retry(lambda: self._get("/entrez/eutils/efetch.fcgi", {
            "db": "pubmed", "id": ",".join(ids), "rettype": "abstract", "retmode": "xml"}))
        by_pmid = parse_pubmed_xml(resp.text)
        return [by_pmid[p] for p in ids if p in by_pmid]  # keep esearch's relevance order

    async def _get(self, path: str, params: dict) -> httpx.Response:
        resp = await self._client.get(path, params=params)
        resp.raise_for_status()
        return resp


def parse_pubmed_xml(xml_text: str) -> dict[str, dict]:
    import xml.etree.ElementTree as ET
    out = {}
    for article in ET.fromstring(xml_text).findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID")
        if not pmid:
            continue
        title_el = article.find(".//ArticleTitle")
        out[pmid] = {
            "title": "".join(title_el.itertext()) if title_el is not None else None,
            "text": "\n".join("".join(el.itertext())
                              for el in article.findall(".//Abstract/AbstractText")),
        }
    return out


async def fetch_topk(retrievers: dict, query: str) -> dict[str, list[dict]]:
    n = TOP_K + OVERFETCH
    raw = await asyncio.gather(*(_retry(lambda r=r: r.search(query, n))
                                 for r in retrievers.values()))
    results = {arm: normalize(hits) for arm, hits in zip(retrievers, raw)}
    k = min(len(v) for v in results.values())  # equal counts: a shorter side is a tell
    return {arm: v[:k] for arm, v in results.items()}


# ---- packaging -------------------------------------------------------------------

def shuffle_for_display(items: list[dict]) -> tuple[list[dict], list[int]]:
    """Returns (items in random display order, true 1-based rank of each)."""
    order = random.sample(range(len(items)), len(items))
    return [items[i] for i in order], [i + 1 for i in order]


def blind(query_item: dict, by_system: dict[str, list[dict]]) -> tuple[dict, dict]:
    left, right = random.sample(ARMS, 2)
    left_view, left_ranks = shuffle_for_display(by_system[left])
    right_view, right_ranks = shuffle_for_display(by_system[right])
    package_row = {"id": query_item["id"], "query": query_item["query"],
                   "left": left_view, "right": right_view}
    key_row = {"id": query_item["id"], "left_system": left, "right_system": right,
               "left_true_ranks": left_ranks, "right_true_ranks": right_ranks}
    return package_row, key_row


def check_unique_ids(queries: list[dict]) -> None:
    """Ratings and the answer key join on id; a duplicate would silently merge two
    questions' scores. Fails before any API call is spent."""
    ids = [q["id"] for q in queries]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"Duplicate question id(s): {dupes}")


async def main(queries_path: Path, out_dir: Path, limit: int | None = None) -> None:
    load_dotenv()
    if not os.environ.get("REDPINE_API_KEY"):
        sys.exit("Missing REDPINE_API_KEY (see .env.example).")
    queries = json.loads(queries_path.read_text(encoding="utf-8"))[:limit]
    check_unique_ids(queries)
    out_dir.mkdir(parents=True, exist_ok=True)
    run = out_dir.name
    meta = {"package_id": uuid.uuid4().hex[:12], "run_name": run,
            "source_file": str(queries_path),
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds")
                        .replace("+00:00", "Z")}
    package_path = out_dir / f"package_{run}.json"
    key_path = out_dir / f"answer_key_{run}.json"
    errors_path = out_dir / f"errors_{run}.json"

    retrievers = {
        "redpine": Redpine(
            httpx.AsyncClient(base_url=os.environ.get("REDPINE_API_URL", "https://api.redpine.ai"),
                              timeout=60),
            os.environ["REDPINE_API_KEY"], os.environ.get("REDPINE_COLLECTION", "Redpine Science")),
        "pubmed": PubMed(httpx.AsyncClient(base_url="https://eutils.ncbi.nlm.nih.gov", timeout=60)),
    }
    sem = asyncio.Semaphore(CONCURRENCY)

    async def run_one(q: dict):
        async with sem:
            try:
                return q, await fetch_topk(retrievers, q["query"])
            except Exception as e:
                return q, f"{type(e).__name__}: {e}"

    package_rows, key_rows, errors = [], [], []
    tasks = [asyncio.create_task(run_one(q)) for q in queries]
    for done, coro in enumerate(asyncio.as_completed(tasks), 1):
        q, result = await coro
        if isinstance(result, str):
            errors.append({"id": q["id"], "query": q["query"], "error": result})
            print(f"[{done}/{len(queries)}] ERROR {q['id']}: {result}", file=sys.stderr)
        else:
            package_row, key_row = blind(q, result)
            package_rows.append(package_row)
            key_rows.append(key_row)
            print(f"[{done}/{len(queries)}] ok {q['id']} ({len(package_row['left'])} per side)")
        package_path.write_text(json.dumps({**meta, "queries": package_rows}, indent=2,
                                           ensure_ascii=False))
        key_path.write_text(json.dumps({**meta, "answers": key_rows}, indent=2,
                                       ensure_ascii=False))
        if errors:
            errors_path.write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nDone: {len(package_rows)} ok, {len(errors)} errors.")
    print(f"Judging package: {package_path} (package_id={meta['package_id']})")
    print(f"Answer key (do not share with judges): {key_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Build a blinded Redpine vs. PubMed judging package.")
    p.add_argument("questions", type=Path, help="e.g. data/questions.json")
    p.add_argument("out_dir", type=Path, help="e.g. runs/my_package (its name becomes run_name)")
    p.add_argument("--limit", type=int, default=None, help="first N questions only")
    a = p.parse_args()
    asyncio.run(main(a.questions, a.out_dir, a.limit))
