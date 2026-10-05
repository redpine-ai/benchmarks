"""Run ScholarQABench SciFact through the four arms and write one slim log.

    uv run python -m src.run --limit 20
    uv run python -m src.run --provider bedrock --workers 8

The setup (model, budget, prompt, tools) is configs/scifact.toml. Environment:
ANTHROPIC_API_KEY (or --provider bedrock with AWS credentials), REDPINE_API_KEY,
TAVILY_API_KEY for the web arms; a .env beside this package is loaded.
"""
import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from harness import runner
from harness.config import load_config
from harness.render import make_dispatch

from .data import load_claims
from .grade import compute_match

TRACK = Path(__file__).resolve().parents[1]
CONFIG = TRACK / "configs" / "scifact.toml"
BENCHMARK = "ScholarQABench (SciFact)"


def judge(item: dict, arm: str, text: str) -> dict:
    return {"official_correct": compute_match(item["gold"], text)}


def parse_args(argv, cfg):
    p = argparse.ArgumentParser(description="Run ScholarQABench SciFact across the four arms.")
    p.add_argument("--arms", nargs="+", default=list(cfg.arm_order), choices=cfg.arm_order)
    p.add_argument("--limit", type=_positive_int, default=None, help="first N claims (default: all 208)")
    p.add_argument("--provider", default=cfg.provider, choices=["anthropic", "bedrock"])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="runs", help="directory for the log")
    return p.parse_args(argv)


def _positive_int(value: str) -> int:
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("--limit must be a positive integer")
    return n


def main(argv=None):
    load_dotenv(TRACK / ".env")
    cfg = load_config(CONFIG)
    args = parse_args(sys.argv[1:] if argv is None else argv, cfg)
    redpine, web_key = runner.check_credentials(cfg, args.arms, args.provider, os.environ)
    claims = load_claims()
    if args.limit:
        claims = claims[:args.limit]
    client, served = runner.build_client(args.provider, cfg.model, os.environ)
    log = runner.run_arms(
        claims, cfg, client=client, provider=args.provider, served_model_id=served, arms=args.arms,
        dispatch_factory=lambda arm: make_dispatch(cfg, redpine, web_key), judge=judge,
        workers=args.workers, benchmark=BENCHMARK,
        redpine_collection=redpine.collection if redpine else "Redpine Science")
    runner.report(log, runner.write_log(log, Path(args.out), "scifact"))


if __name__ == "__main__":
    main()
