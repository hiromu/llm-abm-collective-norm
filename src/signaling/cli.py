"""Job-market simulation, analysis and visualization."""

import argparse
import json
from pathlib import Path
import sys

from .config import ARMS, MARKETS, MODELS
from .data import verify


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/paper"))

    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify", help="Check paper inputs and numerical summaries")

    abm = commands.add_parser("abm", help="Run the continuous job-market ABM")
    abm.add_argument("--output", type=Path, default=Path("build/abm"))
    abm.add_argument("--market", action="append", help="Productivity pair, e.g. 40/10 (default: 40/10 and 10/10)")
    abm.add_argument("--seed", type=int, action="append", help="Repeat for multiple seeds (default: 42)")
    abm.add_argument("--rounds", type=int, default=90)
    abm.add_argument("--parameters", type=Path, help="JSON overrides for economic and initialization parameters")

    abm_plot = commands.add_parser("plot-abm", help="Plot previously exported ABM results")
    abm_plot.add_argument("--results", type=Path, default=Path("build/abm"))
    abm_plot.add_argument("--output", type=Path, default=Path("build/abm-figures"))

    live = commands.add_parser("llm", help="Run new applicant decisions through the model API")
    live.add_argument("--model", choices=MODELS, required=True)
    live.add_argument("--seed", type=int, action="append", required=True)
    live.add_argument("--arm", choices=ARMS, action="append")
    live.add_argument("--market", choices=MARKETS, action="append")
    live.add_argument("--rounds", type=int, default=100)
    live.add_argument("--output", type=Path, required=True)
    live.add_argument("--env-file", type=Path, default=Path(".env"))
    live.add_argument("--max-requests", type=int, help="Required for live runs; cumulative physical request limit")
    live.add_argument("--concurrency", type=int, default=16)
    live.add_argument("--timeout", type=float, default=180)
    live.add_argument("--request-model", help="Explicit API model identifier override")
    live.add_argument("--effort", choices=("off", "minimal", "low", "medium", "high", "xhigh", "max"))
    live.add_argument("--max-output-tokens", type=int)
    live.add_argument("--dry-run", action="store_true", help="Print the plan without requests or output files")
    live.add_argument("--resume", action="store_true")
    live.add_argument("--retry-uncertain", action="store_true",
                      help="Allow retrying unresolved requests that may already have been charged")

    monitor = commands.add_parser("llm-status", help="Read experiment progress without provider calls")
    monitor.add_argument("--results", type=Path, required=True)

    llm_plot = commands.add_parser("plot-llm", help="Plot a completed live experiment from its CSV exports")
    llm_plot.add_argument("--results", type=Path, required=True)
    llm_plot.add_argument("--output", type=Path, required=True)

    plot = commands.add_parser("reproduce", help="Render figures and tables from saved results")
    plot.add_argument("--output", type=Path, default=Path("build/paper"))
    plot.add_argument("--simulation-results", type=Path,
                      help="Complete simulation output containing trajectories.csv")
    plot.add_argument("--semantics", type=Path, help="Complete semantic refit directory")
    plot.add_argument("--statistics", type=Path,
                      help="Saved technology_tests.csv directory for the contrast figure")

    analysis = commands.add_parser("analyze", help="Encode statements and refit semantic analyses")
    analysis.add_argument("--output", type=Path, default=Path("build/semantics"))
    analysis.add_argument("--encoder", type=Path, required=True,
                          help="Pinned MiniLM tokenizer.json and onnx/model.onnx directory")

    tests = commands.add_parser("stats", help="Paired seed tests of late technology use")
    tests.add_argument("--output", type=Path, default=Path("build/statistics"))

    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            receipt = {"inputs": verify(args.data)}
        elif args.command == "abm":
            from .simulation.abm import run
            receipt = run(args.output, args.market or MARKETS, args.seed or (42,),
                          args.rounds, args.parameters)
        elif args.command == "plot-abm":
            from .visualization.abm import run
            receipt = run(args.results, args.output)
        elif args.command == "llm":
            from .simulation.experiment import plan, run
            settings = plan(args.model, args.seed, args.arm or ARMS, args.market or MARKETS,
                            args.rounds, args.request_model, args.effort, args.max_output_tokens)
            receipt = settings if args.dry_run else run(
                settings, args.output, args.max_requests, args.concurrency, args.env_file,
                args.timeout, args.resume, args.retry_uncertain)
        elif args.command == "llm-status":
            from .simulation.experiment import inspect
            receipt = inspect(args.results)
        elif args.command == "plot-llm":
            from .visualization.experiment import run
            receipt = run(args.results, args.output)
        elif args.command == "reproduce":
            from .visualization.figures import run
            receipt = run(args.data, args.output, args.simulation_results, args.semantics, args.statistics)
        elif args.command == "analyze":
            from .analysis.semantics import run
            receipt = run(args.data, args.output, args.encoder)
        else:
            from .analysis.tables import technology_tests
            receipt = technology_tests(args.data, args.output)
        if getattr(args, "output", None) is not None and args.command != "llm":
            (args.output / "reproduction.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2))
    except (OSError, ValueError, KeyError) as error:
        print(f"Command failed: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Stopped; saved experiment responses are retained.", file=sys.stderr)
        return 130
    return 0
