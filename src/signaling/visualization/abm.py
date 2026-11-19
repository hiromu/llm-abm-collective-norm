"""Plots of exported job-market ABM results."""

from collections import defaultdict
import itertools
import json
import math

import numpy as np

from ..data import prepare_output, read_csv
from .figures import COLORS, STYLES, RATIO, configure_style, plt, save


def run(results, output):
    settings = json.loads((results / "parameters.json").read_text())
    rows = read_csv(results / "economic_paths.csv")
    seeds, rounds = settings["seeds"], settings["rounds"]
    markets = list(settings["parameters"])
    if settings["policy"] != "abm" or not markets or not seeds or rounds < 1:
        raise ValueError("An ABM result directory is required")
    expected = set(itertools.product(markets, seeds, range(1, rounds + 1)))
    keys = [(r["market"], int(r["seed"]), int(r["round"])) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("ABM plots require every requested market, seed and round exactly once")
    series = defaultdict(list)
    for row in rows:
        if row["model"] != "abm" or row["arm"] != "social":
            raise ValueError("Only ABM social-learning trajectories are supported")
        for role in ("HQ", "LQ", "Employer"):
            value = float(row[role])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Technology plots require nonempty quality groups and finite values in [0,1]")
            series[row["market"], role, int(row["round"])].append(value)
    prepare_output(output, results)
    family = configure_style(12)
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.4), layout="constrained")
    labels = {"HQ": "High quality", "LQ": "Low quality", "Employer": "Employers"}
    x = np.arange(1, rounds + 1)
    for ax, roles, title in zip(axes, (("HQ", "LQ"), ("Employer",)), ("Applicants", "Employers")):
        for market in markets:
            for role in roles:
                values = np.array([series[market, role, r] for r in x])
                label = f"{labels[role]} ({market})" if role != "Employer" else market
                ax.plot(x, values.mean(axis=1), color=COLORS[role], ls=STYLES.get(market, "-"),
                        lw=1.7, label=label)
                if len(seeds) > 1:
                    ax.fill_between(x, values.min(axis=1), values.max(axis=1),
                                    color=COLORS[role], alpha=.1, linewidth=0)
        ax.set(title=title, xlabel="Round", ylabel="Mean technology use", ylim=(-.02, 1.03),
               xlim=(0, rounds + 1), box_aspect=1 / RATIO)
        ax.grid(color="#eeeeee", linewidth=.7)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=10, loc="best")
    save(fig, output, "job_market_signaling_continuous")
    return {"font": family, "cells": settings["cells"], "rounds": rounds, "seeds": seeds,
            "source": str(results), "bands": "seed minimum and maximum" if len(seeds) > 1 else None}
