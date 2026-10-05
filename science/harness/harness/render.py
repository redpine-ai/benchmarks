"""How search results are shown to the model, and the dispatch that runs a tool call.

Two renderers, one per answer-quality track, each owning every string the model sees from a
tool: the result layout, the empty-result text and the error text.

`indexed` (ScholarQABench SciFact): Redpine passages are numbered from 0 as
`[n] text`, numbering continuing across a question's searches; web results are
numbered from 1 per call as `[n] Title (domain)` then the extract.

`handles` (expert-validated questions): every result is `[S:<id>] Title (venue)`
then the text, where the id is Redpine's passage id or the page URL, and the
venue is journal and year, else PMID, else DOI, else the site domain.
"""
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlsplit

import httpx

from .clients import RedpineConfig, redpine_search, tavily_search
from .config import RunConfig


def _domain(url: str) -> str:
    try:
        return urlsplit(url).netloc
    except ValueError:
        return ""


def _indexed_connect(results, state, chars):
    lines = [f"[{state['next'] + i}] {(r.get('text') or '')[:chars]}" for i, r in enumerate(results)]
    state["next"] += len(results)
    return "\n\n".join(lines)


def _indexed_web(results, chars):
    lines = []
    for n, r in enumerate(results, 1):
        title = (r.get("title") or "").strip() or "(untitled)"
        dom = _domain(r.get("url") or "").removeprefix("www.")
        lines.append(f"[{n}] {title}" + (f" ({dom})" if dom else "") + f"\n{(r.get('content') or '')[:chars]}")
    return "\n\n".join(lines)


def _venue(meta: dict, url: str) -> str:
    year = str(meta.get("publication_date") or "")[:4]
    prov = " ".join(str(x) for x in (meta.get("journal"), year) if x)
    if prov:
        return prov
    if meta.get("pmid"):
        return f"PMID {meta['pmid']}"
    if meta.get("doi"):
        return f"doi:{meta['doi']}"
    return _domain(url)


def _handles(items, chars):
    parts = []
    for cid, title, meta, url, text in items:
        head = f"[S:{cid or url or '?'}]"
        if title:
            head += f" {title}"
        venue = _venue(meta, url)
        if venue:
            head += f" ({venue})"
        parts.append(f"{head}\n{(text or '')[:chars]}")
    return "\n\n".join(parts)


def _handles_connect(results, state, chars):
    return _handles([(r.get("id"), (r.get("metadata") or {}).get("title"), r.get("metadata") or {},
                      "", r.get("text")) for r in results], chars)


def _handles_web(results, chars):
    return _handles([(None, r.get("title"), {}, r.get("url") or "", r.get("content")) for r in results], chars)


@dataclass(frozen=True)
class Renderer:
    connect: Callable[[list, dict, int], str]
    web: Callable[[list, int], str]
    empty: str
    error: Callable[[str, Exception], str]

    @staticmethod
    def new_state() -> dict:
        return {"next": 0}


RENDERERS = {
    "indexed": Renderer(_indexed_connect, _indexed_web, "No results found.",
                        lambda tool, e: f"{tool} error: {type(e).__name__}: {e}"),
    "handles": Renderer(_handles_connect, _handles_web, "(no results)",
                        lambda tool, e: f"ERROR: {e}"),
}


def _connect_meta(r: dict) -> dict:
    return {"id": r.get("id"), **(r.get("metadata") or {})}


def _web_meta(r: dict) -> dict:
    return {"url": r.get("url"), "title": r.get("title"), "published_date": r.get("published_date")}


def make_dispatch(cfg: RunConfig, redpine: RedpineConfig | None, web_key: str | None, *, post=httpx.post):
    """A dispatch for ONE arm of ONE question: `dispatch(tool_name, tool_input) -> (text, meta)`."""
    renderer = RENDERERS[cfg.renderer]
    state = renderer.new_state()

    def dispatch(name: str, tool_input: dict) -> tuple[str, list[dict]]:
        tool_input = tool_input or {}
        query = tool_input.get("query", "") or ""
        spec = cfg.tools.get(name)
        try:
            if spec is not None and spec.kind == "redpine" and redpine is not None:
                results = redpine_search(redpine, query, limit=tool_input.get("limit") or cfg.top_k,
                                         filters=tool_input.get("filters"), post=post)
                if not results:
                    return renderer.empty, []
                return renderer.connect(results, state, cfg.passage_chars), [_connect_meta(r) for r in results]
            if spec is not None and spec.kind == "web" and web_key:
                results = tavily_search(web_key, query, max_results=tool_input.get("max_results") or cfg.top_k,
                                        post=post)
                if not results:
                    return renderer.empty, []
                return renderer.web(results, cfg.passage_chars), [_web_meta(r) for r in results]
        except Exception as e:  # noqa: BLE001  (boundary: HTTP failures reach the model as text)
            return renderer.error(name, e), []
        return f"unknown tool: {name}", []

    return dispatch
