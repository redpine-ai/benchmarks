"""Single-shot retrieval benchmark: Redpine Science vs. Tavily vs. Exa, one call per
system per query, no agent loop, no reformulation. See the package README for what
this measures and why single-shot, not agentic.

Run:
    uv run python -m src.benchmark --limit 50 --out /tmp/smoke.json   # smoke test
    uv run python -m src.benchmark                                    # full 668-query run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from datetime import date
from pathlib import Path

import httpx
from dotenv import load_dotenv

from src.matching import _extract_dois, find_rank, print_summary
from src.retrievers import ExaRetriever, RedpineRetriever, Result, TavilyRetriever

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "publication_search_in_corpus.jsonl"
RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "gold_doc_results.json"
TOP_K = 10
# Kept low deliberately: Redpine Science is a meta-collection that merges several
# member collections per query, which costs more server-side per request than a
# single-collection search. A real run at higher concurrency threw sporadic 500s
# under load that succeeded immediately on retry -- backend load under
# concurrency, not a bad query -- so this stays conservative rather than tuned for
# raw speed.
CONCURRENCY = 3


async def _search_call(name: str, retriever, query: str) -> list[Result]:
    return await retriever.search(query, max_results=TOP_K)


async def _with_retry(fn, attempts: int = 4, base_delay: float = 0.5, max_delay: float = 8.0):
    for i in range(attempts):
        try:
            return await fn()
        except httpx.HTTPStatusError as e:
            transient = e.response.status_code in (408, 409, 429, 500, 502, 503, 504)
            if i == attempts - 1 or not transient:
                raise
            await asyncio.sleep(min(max_delay, base_delay * (2**i)))
        except (httpx.TimeoutException, httpx.ConnectError):
            if i == attempts - 1:
                raise
            await asyncio.sleep(min(max_delay, base_delay * (2**i)))


def _stored_result(r) -> dict:
    """What gets persisted per candidate, alongside the rank. Keeping title/doi/url
    (not just the final rank) is what lets a matching-logic change -- a new regex,
    a looser threshold, a bug fix -- be checked against this exact paid-for API
    response later, instead of re-spending real credits just to see what a code
    change would have done. The snippet text itself is never stored: Redpine
    Science snippets are chunks of licensed full text, and Exa and Tavily snippets
    are publisher-page extracts. Only the DOIs the matcher could extract from the
    snippet are kept, which is all the matcher ever used it for."""
    return {"title": r.title, "doi": r.doi, "url": r.url,
            "snippet_dois": sorted(_extract_dois(r.snippet))}


async def _run_one(record: dict, retrievers: dict, sem: asyncio.Semaphore) -> dict:
    """Each arm's search is independent: one arm erroring doesn't drop the other
    arms' otherwise-successful results for the same query. A row's f"{name}_rank"
    key is simply absent for whichever arm(s) errored (not set to None -- that
    would misrepresent "errored" as "searched and found nothing")."""
    async with sem:
        query = record["text"]
        gold_doi = record["gold_paper"]["doi"]
        gold_title = record["gold_paper"].get("title")
        names = list(retrievers.keys())
        raw_results = await asyncio.gather(
            *[_with_retry(lambda n=name: _search_call(n, retrievers[n], query)) for name in names],
            return_exceptions=True,
        )
        out = {"query_id": record["query_id"], "query": query, "gold_doi": gold_doi}
        errors: dict[str, str] = {}
        for name, raw in zip(names, raw_results):
            if isinstance(raw, BaseException):
                errors[name] = f"{type(raw).__name__}: {raw}"
                continue
            out[f"{name}_rank"] = find_rank(gold_doi, gold_title, raw)
            out[f"{name}_results"] = [_stored_result(r) for r in raw]
        if errors:
            out["errors"] = errors
        return out


async def main(limit: int | None, seed: int, out: str | None = None) -> None:
    load_dotenv()
    # Defaults to the canonical results file so a full run still publishes it as before;
    # pass --out for a smoke test so it doesn't overwrite the checked-in published numbers.
    results_path = Path(out) if out else RESULTS_PATH
    records = [json.loads(l) for l in DATA_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    if limit is not None:
        random.seed(seed)
        records = random.sample(records, min(limit, len(records)))

    # REDPINE_COLLECTION is not in this list: it has a real default below, unlike the three
    # keys, which have no sensible default and must be supplied.
    missing = [k for k in ("REDPINE_API_KEY", "TAVILY_API_KEY",
                           "EXA_API_KEY") if not os.environ.get(k)]
    if missing:
        print(f"Missing required environment variable(s): {', '.join(missing)}. "
              f"Copy .env.example to .env and fill them in.", file=sys.stderr)
        sys.exit(1)

    retrievers = {
        "redpine": RedpineRetriever(
            httpx.AsyncClient(base_url=os.environ.get("REDPINE_API_URL", "https://api.redpine.ai"),
                              timeout=60),
            os.environ["REDPINE_API_KEY"],
            os.environ.get("REDPINE_COLLECTION", "Redpine Science"), max_results=TOP_K),
        "tavily": TavilyRetriever(
            httpx.AsyncClient(base_url="https://api.tavily.com", timeout=60),
            os.environ["TAVILY_API_KEY"], max_results=TOP_K),
        "exa": ExaRetriever(
            httpx.AsyncClient(base_url="https://api.exa.ai", timeout=60),
            os.environ["EXA_API_KEY"], max_results=TOP_K),
    }
    arms = list(retrievers.keys())

    sem = asyncio.Semaphore(CONCURRENCY)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    tasks = [asyncio.create_task(_run_one(r, retrievers, sem)) for r in records]
    done = 0
    for coro in asyncio.as_completed(tasks):
        result = await coro
        done += 1
        rows.append(result)
        ranks = " ".join(f"{name}={result[f'{name}_rank']}" if f"{name}_rank" in result
                         else f"{name}=<err>" for name in arms)
        if result.get("errors"):
            err_str = "; ".join(f"{name}={msg}" for name, msg in result["errors"].items())
            print(f"[{done}/{len(records)}] {result['query_id']}: {ranks}  ERRORS: {err_str}",
                  file=sys.stderr)
        else:
            print(f"[{done}/{len(records)}] {result['query_id']}: {ranks}")
        wrapped = {"run_date": str(date.today()), "dataset": str(DATA_PATH.relative_to(DATA_PATH.parents[1])),
                  "n": len(rows), "systems": arms, "results": rows}
        results_path.write_text(json.dumps(wrapped, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nWrote {len(rows)} results to {results_path}\n")
    for name in arms:
        ranks = [r[f"{name}_rank"] for r in rows if f"{name}_rank" in r]
        print_summary(name, ranks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Run on a random subset instead of the full dataset.")
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--out", type=str, default=None,
                        help="Write results here instead of results/gold_doc_results.json. "
                             "Use this for smoke tests (e.g. with --limit) so they don't "
                             "overwrite the checked-in, published results file.")
    args = parser.parse_args()
    if args.limit is not None and not args.out:
        parser.error("--limit needs --out: a partial run must not overwrite results/gold_doc_results.json")
    asyncio.run(main(args.limit, args.seed, args.out))
