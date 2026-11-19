"""Plots of complete LLM experiment exports."""

from collections import defaultdict
import itertools
import json

import numpy as np

from ..data import prepare_output, read_csv
from .figures import COLORS, RATIO, configure_style, plt, save


def run(results, output):
    settings = json.loads((results / "plan.json").read_text())
    rows = read_csv(results / "economic_paths.csv")
    expected = set(itertools.product(settings["arms"], settings["markets"], settings["seeds"],
                                     range(1, settings["rounds"] + 1)))
    keys = [(r["arm"], r["market"], int(r["seed"]), int(r["round"])) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Plotting requires complete planned cells, rounds and seeds")
    series = defaultdict(list)
    for row in rows:
        if row["model"] != settings["model"]:
            raise ValueError("Model identifier differs from the experiment plan")
        for role in ("HQ", "LQ", "Employer"):
            value = float(row[role])
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Invalid technology-use value")
            series[row["arm"], row["market"], role, int(row["round"])].append(value)
    prepare_output(output, results)
    font = configure_style(12)
    fig, axes = plt.subplots(len(settings["markets"]), len(settings["arms"]), squeeze=False,
                             figsize=(7.2 * len(settings["arms"]), 4.8 * len(settings["markets"])),
                             layout="constrained")
    x = np.arange(1, settings["rounds"] + 1)
    for i, market in enumerate(settings["markets"]):
        for j, arm in enumerate(settings["arms"]):
            ax = axes[i, j]
            for role, label in (("HQ", "High quality"), ("LQ", "Low quality"), ("Employer", "Employers")):
                values = np.array([series[arm, market, role, r] for r in x])
                ax.plot(x, values.mean(axis=1), label=label, color=COLORS[role], lw=1.6)
                if len(settings["seeds"]) > 1:
                    ax.fill_between(x, values.min(axis=1), values.max(axis=1), color=COLORS[role], alpha=.1)
            ax.set(title=f"{settings['label']} · {arm} information · {market}",
                   xlabel="Round", ylabel="Mean technology use", ylim=(-.02, 1.03),
                   xlim=(1, settings["rounds"]), box_aspect=1 / RATIO)
            ax.grid(color="#eeeeee", linewidth=.7)
            ax.set_axisbelow(True)
            ax.legend(frameon=False, fontsize=10)
    save(fig, output, "technology")
    return {"source": str(results), "cells": settings["cells"], "font": font,
            "bands": "seed minimum and maximum", "economic_rounds": len(rows)}
