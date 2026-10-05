"""Run the expert-validated questions through the two arms and write one slim log.

    uv run python -m src.run --limit 2
    uv run python -m src.run --config my/arms.toml --arms solo

The setup (model, budget, prompt, tools) is configs/expert_qa.toml. Environment:
ANTHROPIC_API_KEY (or --provider bedrock with AWS credentials), REDPINE_API_KEY,
TAVILY_API_KEY; a .env beside this package is loaded. No scoring: the log stores
answers and traces.
"""
import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from harness import runner
from harness.config import load_config
from harness.render import make_dispatch

from .data import load_questions

TRACK = Path(__file__).resolve().parents[1]
CONFIG = TRACK / "configs" / "expert_qa.toml"
BENCHMARK = "Expert-validated questions"


def config_path(argv) -> Path:
    """The --config value, resolved against the current directory; the shipped TOML by default."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(CONFIG))
    known, _ = pre.parse_known_args(argv)
    return Path(known.config).resolve()


def parse_args(argv, cfg):
    p = argparse.ArgumentParser(description="Run the expert-validated questions across the configured arms.")
    p.add_argument("--config", default=str(CONFIG), help="TOML with the model, budget, prompt, tools and arms")
    p.add_argument("--arms", nargs="+", default=list(cfg.arm_order), choices=cfg.arm_order)
    p.add_argument("--limit", type=_positive_int, default=None, help="first N questions")
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
    argv = sys.argv[1:] if argv is None else argv
    load_dotenv(TRACK / ".env")
    cfg = load_config(config_path(argv))
    args = parse_args(argv, cfg)
    redpine, web_key = runner.check_credentials(cfg, args.arms, args.provider, os.environ)
    questions = load_questions()
    if args.limit:
        questions = questions[:args.limit]
    client, served = runner.build_client(args.provider, cfg.model, os.environ)
    log = runner.run_arms(
        questions, cfg, client=client, provider=args.provider, served_model_id=served, arms=args.arms,
        dispatch_factory=lambda arm: make_dispatch(cfg, redpine, web_key), judge=None,
        workers=args.workers, benchmark=BENCHMARK,
        redpine_collection=redpine.collection if redpine else "Redpine Science")
    runner.report(log, runner.write_log(log, Path(args.out), "expert_qa"))


if __name__ == "__main__":
    main()
