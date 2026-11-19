"""Common-pool resource simulation, analysis and visualization."""

import argparse
import json
from pathlib import Path
import sys

from .config import CONDITIONS, MODELS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for policy in ("abm", "llm"):
        command = commands.add_parser(policy)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--seed", type=int, action="append", required=True)
        command.add_argument("--growth-rate", type=float, action="append")
        command.add_argument("--rounds", type=int, help="Number of updates; defaults: ABM 101, LLM 51")
        command.add_argument("--parameters", type=Path, help="JSON economic-parameter overrides")
        command.add_argument("--dry-run", action="store_true")
        if policy == "llm":
            command.add_argument("--model", choices=MODELS, required=True)
            command.add_argument("--altruism-ratio", type=float, action="append")
            command.add_argument("--condition", choices=CONDITIONS, action="append")
            command.add_argument("--request-model")
            command.add_argument("--max-output-tokens", type=int)
            command.add_argument("--env-file", type=Path, default=Path(".env"))
            command.add_argument("--max-requests", type=int, help="Required cumulative physical-request limit, not a dollar budget")
            command.add_argument("--concurrency", type=int, default=10)
            command.add_argument("--resume", action="store_true")
            command.add_argument("--retry-uncertain", action="store_true")
    analyze = commands.add_parser("analyze")
    analyze.add_argument("--data", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--ablation", action="store_true")
    analyze.add_argument("--trials", type=int, default=10)
    analyze.add_argument("--statistics", action="store_true")
    plot = commands.add_parser("plot")
    plot.add_argument("--results", type=Path, required=True)
    plot.add_argument("--output", type=Path, required=True)
    norms = commands.add_parser("norms")
    norms.add_argument("--data", type=Path, required=True)
    norms.add_argument("--output", type=Path, required=True)
    norms.add_argument("--encoder", type=Path, required=True)
    norms.add_argument("--trials", type=int, default=10)
    args = parser.parse_args(argv)
    try:
        if args.command in ("abm", "llm"):
            from .simulation.experiment import plan, run_abm, run_llm
            settings = plan(args.command, args.seed, args.growth_rate or (.2, .6),
                            getattr(args, "altruism_ratio", None) or (0., .5, 1.),
                            getattr(args, "condition", None) or ("all",), args.rounds,
                            getattr(args, "model", "gpt-4o"),
                            json.loads(args.parameters.read_text()) if args.parameters else None,
                            getattr(args, "request_model", None), getattr(args, "max_output_tokens", None))
            if args.dry_run:
                print(json.dumps(settings, indent=2))
                return 0
            receipt = (run_abm(settings, args.output) if args.command == "abm" else
                       run_llm(settings, args.output, args.max_requests, args.concurrency,
                               args.env_file, args.resume, args.retry_uncertain))
            if args.command == "abm":
                (args.output / "status.json").write_text(json.dumps(receipt, indent=2) + "\n")
        elif args.command == "analyze":
            from .analysis.tables import run
            receipt = run(args.data, args.output, args.ablation, args.trials, args.statistics)
        elif args.command == "norms":
            from .analysis.norms import run
            receipt = run(args.data, args.output, args.encoder, args.trials)
        else:
            from .visualization.figures import run
            receipt = run(args.results, args.output)
            (args.output / "figures.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({k: v for k, v in receipt.items() if k != "input_sha256"}, indent=2))
        return 0
    except (ValueError, OSError, KeyError) as error:
        print(f"Command failed: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Stopped; saved responses and checkpoints are retained.", file=sys.stderr)
        return 130
