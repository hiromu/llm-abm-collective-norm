"""Continuous job-market ABM runner."""

from dataclasses import asdict, replace
import json
import math

from ..config import MARKETS
from ..data import prepare_output, write_csv
from .economy import Market, Parameters


def run(output, markets=MARKETS, seeds=(42,), rounds=90, parameters=None):
    """Simulate and export each market/seed combination."""
    markets, seeds = tuple(dict.fromkeys(markets)), tuple(dict.fromkeys(seeds))
    if not markets or not seeds:
        raise ValueError("At least one market and seed are required")
    overrides = json.loads(parameters.read_text()) if parameters else {}
    if not isinstance(overrides, dict):
        raise ValueError("Parameters must be a JSON object")
    allowed = set(Parameters.__dataclass_fields__) - {"high_productivity", "low_productivity"}
    if set(overrides) - allowed:
        raise ValueError(f"Unknown parameters (productivity is set by --market): {sorted(set(overrides) - allowed)}")
    settings = {m: replace(Parameters.for_market(m), **overrides) for m in markets}
    cells = {(m, s): Market(p, s, policy="abm", rounds=rounds)
             for m, p in settings.items() for s in seeds}
    prepare_output(output, *([parameters] if parameters else []))
    rows, summaries = [], []
    tail = min(15, rounds)
    for (market, seed), model in cells.items():
        cell = [model.step() for _ in range(rounds)]
        identity = {"model": "abm", "arm": "social", "market": market, "seed": seed}
        rows.extend({**identity, **r} for r in cell)
        summaries.append({**identity, "first_round": rounds - tail + 1, "last_round": rounds,
                          **{key: math.fsum(r[key] for r in cell[-tail:]) / tail
                             for key in cell[0] if key != "round"}})
    write_csv(output / "economic_paths.csv", rows)
    write_csv(output / "late_means.csv", summaries)
    receipt = {"policy": "abm", "rounds": rounds, "seeds": seeds,
               "parameters": {m: asdict(p) for m, p in settings.items()},
               "cells": len(cells), "economic_rounds": len(rows),
               "late_window_rounds": tail}
    (output / "parameters.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
