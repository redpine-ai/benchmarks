"""Joins one or more judges' exported ratings (per-item 0-4 scores, rank preserved)
back with the hidden answer_key.json to reconstruct which system each item actually
came from, then computes the same three statistics Consensus's published
methodology reports, on their own native 0-4 scale, at this track's own top-5
(JUDGE_TOP_K=5, against Consensus's top-10, for labor-cost reasons: every result
pair is read and judged by hand):

  1. Precision@K -- fraction of judged items scored >= RELEVANT_THRESHOLD (2),
     per system, where K is however many ranks were actually judged (read from
     the data itself, not hardcoded). Consensus's published precision metric
     counts 3 or higher; this track counts 2 or higher. Not
     to be confused with the textbook IR "Average Precision" (rank-weighted) --
     this is the simpler "fraction relevant-or-better" style of precision.
  2. DCG@K -- discounted cumulative gain per query, averaged across queries,
     per system. Standard (non-exponential) formula: sum(score / log2(rank + 1)),
     rank 1-based.
  3. Average relevance by rank -- mean score at each rank position (1..K),
     per system, so a rank-position plot can be built directly from the CSV
     this script writes.

Systems are read from the answer key, not hardcoded -- this track's comparison
arm has changed once already (Exa -> PubMed) with zero changes needed here.

Display position vs. true rank: build_package.py shuffles each side's display
order per query so a judge can't read retrieval rank off the page. A rating
row's "rank" field is therefore the display position the judge saw, not the
true retrieval rank -- compute_stats resolves it through the answer key's
`{side}_true_ranks` before using it as a DCG/precision key.

zero_result_questions: a genuine zero-item side (the retriever found nothing
for this query) is a real, complete data point, distinct from "not yet
judged" -- it counts toward zero_result_questions and contributes an honest
DCG=0 for that query, rather than silently vanishing from every stat the way
an early version of this script let it. mean_relevance/precision are
item-level and can't include a zero-item side directly, so
zero_result_questions is reported as its own explicit count per system,
always alongside those two, so a reader can't mistake "n=1, mean=3.5" for
general performance without also seeing "and returned nothing on 4 other
questions."

Multiple judges: all judges' rows live in ONE data/ratings.json (each row
carries its own "judge" field, so it's still a flat, judge-attributable list,
not that the distinction is lost) rather than one file per domain expert, so a
new judge is a data update, not a new file plus a new positional argument.
Still accepts multiple ratings-file arguments on the command line (any number,
including one), for callers scoring against a package with several separate
export files instead; scores for the same (query_id, side, rank) are averaged
across judges before computing the stats above either way. In this panel only
the 25-question shared pool has two judges; the other 90 questions have one.
This track has ratings from all three domain experts (cardiology, neurology,
rheumatology), n=115.

Writes into results/ (not alongside answer_key.json -- see RESULTS_DIR):
relevance_by_rank.csv (per-rank means, for a rank-position plot), summary.json
(the headline Precision@K / DCG@K numbers per system), and raw_scores.json
(per-item/per-query scores, for error bars).

Run: uv run python -m src.score_ratings data/answer_key.json data/ratings.json
"""

import csv
import json
import math
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

RELEVANT_THRESHOLD = 2  # score >= this counts as "relevant" for precision; Consensus's
                        # published precision metric counts 3 or higher.


def _dcg(scores_by_rank: dict[int, float]) -> float:
    return sum(score / math.log2(rank + 1) for rank, score in scores_by_rank.items())


def compute_stats(answer_key: dict[str, dict], ratings_rows: list[dict]) -> dict:
    """Pure computation, no file I/O.

    answer_key: id -> {left_system, right_system, left_true_ranks, right_true_ranks}
    (already loaded/parsed). `{side}_true_ranks[i]` is the true retrieval rank of
    the item a judge saw at DISPLAY position i+1 on that side.

    ratings_rows: flat list of {judge, query_id, side, rank, score}. score may
    be None for an unscored item, dropped here rather than treated as a real 0.

    Returns {judges, n_queries, systems: {system: {n_scored, mean_relevance,
    precision, mean_dcg, k, zero_result_questions, by_rank: {rank: {mean, n}}}}}.
    A system with zero scored items still appears in `systems`, with
    n_scored=0 and the other fields None.
    """
    raw: dict[tuple, list[float]] = {}
    judges: set[str] = set()
    for row in ratings_rows:
        if row["score"] is None:
            continue
        qid, side, position = row["query_id"], row["side"], row["rank"]
        true_rank = answer_key[qid][f"{side}_true_ranks"][position - 1]
        key = (qid, side, true_rank)
        raw.setdefault(key, []).append(row["score"])
        judges.add(row["judge"])

    by_query: dict[str, dict[str, dict[int, float]]] = {}  # qid -> side -> rank -> avg score
    for (qid, side, rank), vals in raw.items():
        by_query.setdefault(qid, {}).setdefault(side, {})[rank] = sum(vals) / len(vals)

    systems = sorted({key_row[f"{side}_system"]
                      for key_row in answer_key.values() for side in ("left", "right")})
    per_system: dict[str, dict[int, list[float]]] = {s: {} for s in systems}
    per_query_dcg: dict[str, list[float]] = {s: [] for s in systems}
    zero_result_questions: dict[str, int] = {s: 0 for s in systems}

    for qid, key_row in answer_key.items():
        sides = by_query.get(qid, {})
        for side in ("left", "right"):
            system = key_row[f"{side}_system"]
            true_ranks = key_row[f"{side}_true_ranks"]
            if not true_ranks:
                zero_result_questions[system] += 1
                per_query_dcg[system].append(0.0)
                continue
            scores_by_rank = sides.get(side, {})
            if not scores_by_rank:
                continue  # has items, just not judged yet -- exclude, not a failure
            for rank, score in scores_by_rank.items():
                per_system[system].setdefault(rank, []).append(score)
            per_query_dcg[system].append(_dcg(scores_by_rank))

    result = {"judges": sorted(judges), "n_queries": len(answer_key), "systems": {}}
    for system in systems:
        by_rank = {rank: {"mean": sum(scores) / len(scores), "n": len(scores)}
                   for rank, scores in per_system[system].items()}
        all_scores = [s for scores in per_system[system].values() for s in scores]
        result["systems"][system] = {
            "n_scored": len(all_scores),
            "mean_relevance": (sum(all_scores) / len(all_scores)) if all_scores else None,
            "precision": (sum(1 for s in all_scores if s >= RELEVANT_THRESHOLD) / len(all_scores))
                         if all_scores else None,
            "mean_dcg": (sum(per_query_dcg[system]) / len(per_query_dcg[system]))
                        if per_query_dcg[system] else None,
            "k": max(by_rank) if by_rank else 0,
            "zero_result_questions": zero_result_questions[system],
            "by_rank": by_rank,
            "raw": {
                "item_scores": all_scores,
                "per_query_dcg": per_query_dcg[system],
            },
        }
    return result


def _load_answer_key(path: Path) -> dict:
    """Handles the build-provenance envelope ({package_id, run_name, answers: [...]})
    written by build_package.py, and the older bare-list format. An old-format
    file parses fine with package_id=None -- scoring still works exactly the
    same, it just can't be automatically verified against a ratings file's
    declared package_id."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        rows = {row["id"]: row for row in data}
        return {"rows": rows, "package_id": None, "run_name": None}
    rows = {row["id"]: row for row in data["answers"]}
    return {"rows": rows, "package_id": data.get("package_id"), "run_name": data.get("run_name")}


def _load_ratings_file(path: Path) -> dict:
    """Handles two formats: the current envelope ({judge, package_id, run_name,
    disqualified: [...], rows: [...]}), and an older bare list of
    {judge, query_id, side, rank, score} rows (each already carrying its own
    "judge")."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        judges = {row["judge"] for row in data}
        judge = judges.pop() if len(judges) == 1 else None
        return {"judge": judge, "package_id": None, "run_name": None, "rows": data, "disqualified": []}
    rows = [{**row, "judge": data["judge"]} for row in data["rows"]]
    disqualified = [{**d, "judge": data["judge"]} for d in data.get("disqualified", [])]
    return {"judge": data.get("judge"), "package_id": data.get("package_id"),
            "run_name": data.get("run_name"), "rows": rows, "disqualified": disqualified}


def _check_package_match(ratings_path: Path, ratings_meta: dict, answer_key_meta: dict) -> None:
    """Raises loudly if a ratings file's declared package_id doesn't match the
    answer_key.json it's about to be scored against, rather than silently
    joining ratings to items the judge never actually saw. Skips the check
    (with a warning, not silently) when either side has no package_id --
    almost always an older ratings export predating this mechanism."""
    ratings_id, answer_key_id = ratings_meta["package_id"], answer_key_meta["package_id"]
    if ratings_id is None or answer_key_id is None:
        print(f"WARNING: {ratings_path} or its answer_key.json has no build-"
              f"provenance package_id (an older export) -- cannot verify these "
              f"were scored against the same build. Proceeding, but this is the "
              f"first place to suspect if the numbers look wrong.")
        return
    if ratings_id != answer_key_id:
        raise ValueError(
            f"{ratings_path} was scored against package '{ratings_meta.get('run_name')}' "
            f"(package_id={ratings_id}), but this answer_key.json is from package "
            f"'{answer_key_meta.get('run_name')}' (package_id={answer_key_id}) -- these "
            f"do not match. Find the ratings file that matches this answer_key, or "
            f"rebuild the package and re-collect ratings.")


def main(answer_key_path: Path, ratings_paths: list[Path], out_dir: Path = RESULTS_DIR):
    answer_key_meta = _load_answer_key(answer_key_path)
    answer_key = answer_key_meta["rows"]
    ratings_rows = []
    disqualified_rows = []
    for p in ratings_paths:
        ratings_meta = _load_ratings_file(p)
        _check_package_match(p, ratings_meta, answer_key_meta)
        ratings_rows.extend(ratings_meta["rows"])
        disqualified_rows.extend(ratings_meta["disqualified"])
    stats = compute_stats(answer_key, ratings_rows)

    print(f"Judges included: {stats['judges']}")
    print(f"Queries in answer key: {stats['n_queries']}")
    print()

    if disqualified_rows:
        print(f"Disqualified by judges ({len(disqualified_rows)}):")
        for d in disqualified_rows:
            print(f"  {d['query_id']} (by {d['judge']}): {d['reason']}")
        print()

    csv_rows = []
    summary_systems = {}
    for system, s in stats["systems"].items():
        zero_note = (f"  zero-result questions: {s['zero_result_questions']}/{stats['n_queries']}"
                    if s["zero_result_questions"] else None)
        if s["n_scored"] == 0:
            print(f"{system}: no scored items" +
                 (f" ({s['zero_result_questions']}/{stats['n_queries']} returned "
                  f"zero results)" if s["zero_result_questions"] else ""))
            summary_systems[system] = s
            continue
        print(f"--- {system} ---")
        print(f"  items scored: {s['n_scored']}")
        print(f"  mean relevance (0-4): {s['mean_relevance']:.2f}")
        print(f"  precision@{s['k']} (>= {RELEVANT_THRESHOLD}): {100 * s['precision']:.1f}%")
        print(f"  mean DCG@{s['k']} per query: {s['mean_dcg']:.3f}")
        if zero_note:
            print(zero_note + "  (counted as DCG=0 above; excluded from mean "
                 "relevance/precision, which are item-level)")
        print("  average relevance by rank:")
        for rank in sorted(s["by_rank"]):
            r = s["by_rank"][rank]
            print(f"    rank {rank}: {r['mean']:.2f}  (n={r['n']})")
            csv_rows.append({"system": system, "rank": rank, "mean_score": round(r["mean"], 4),
                             "n": r["n"]})
        print()
        summary_systems[system] = s

    # Written to results/, not next to answer_key_path -- that used to write into data/
    # on a plain rerun of the documented command, leaving stray duplicate output files
    # sitting next to the real input data instead of the checked-in results/ copies.
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "relevance_by_rank.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["system", "rank", "mean_score", "n"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Wrote per-rank data for plotting: {out_csv}")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "n_queries": stats["n_queries"],
        "judges": stats["judges"],
        "disqualified": disqualified_rows,
        "systems": {system: {"k": s["k"], "n_scored": s["n_scored"],
                              "mean_relevance": s["mean_relevance"],
                              "precision": s["precision"], "mean_dcg": s["mean_dcg"],
                              "zero_result_questions": s["zero_result_questions"]}
                    for system, s in summary_systems.items()},
    }, indent=2))
    print(f"Wrote headline stats: {summary_path}")

    raw_path = out_dir / "raw_scores.json"
    raw_path.write_text(json.dumps({
        "n_queries": stats["n_queries"],
        "judges": stats["judges"],
        "systems": {system: s["raw"] for system, s in summary_systems.items()},
    }, indent=2))
    print(f"Wrote raw per-item/per-query scores: {raw_path}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    out_dir = RESULTS_DIR
    if "--out-dir" in argv:
        i = argv.index("--out-dir")
        out_dir = Path(argv[i + 1])
        del argv[i:i + 2]
    if len(argv) < 2:
        print("usage: uv run python -m src.score_ratings "
              "<answer_key.json> <ratings1.json> [ratings2.json ...] [--out-dir DIR]\n"
              "Use --out-dir for a new judging round so results/ keeps the published numbers.",
              file=sys.stderr)
        sys.exit(1)
    main(Path(argv[0]), [Path(p) for p in argv[1:]], out_dir)
