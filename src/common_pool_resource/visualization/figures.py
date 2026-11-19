"""Common-pool resource figures."""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap

from ..data import empty_output

COLORS = {0.: "#4c72b0", .5: "#dd8452", 1.: "#55a868"}
SOCIETIES = {0.: "Selfish", .5: "Mixed", 1.: "Altruistic"}
CONDITIONS = {"all": ("All", ""), "without_group_norm": ("Social learning only", "//"),
              "without_social_learning": ("Collective norms only", "\\\\"), "neither": ("Neither", "..")}


def color(ratio):
    return LinearSegmentedColormap.from_list("societies", list(COLORS.values()))(float(ratio))


def society(ratio):
    return SOCIETIES.get(ratio, f"Altruistic fraction {ratio:g}")


def style():
    available = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in ("Arial", "Liberation Sans", "DejaVu Sans") if f in available), "DejaVu Sans")
    plt.rcParams.update({"font.family": font, "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none"})
    return font


def save(fig, output, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output / f"{name}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_paths(paths, output, horizon, metrics, suffix=""):
    conditions = sorted(paths.condition.unique())
    if len(conditions) > 1:
        return [name for condition in conditions for name in
                plot_paths(paths[paths.condition == condition], output, horizon, metrics, "_" + condition)]
    models, rates = sorted(paths.model.unique()), sorted(paths.growth_rate.unique())
    files = []
    for metric, ylabel in metrics:
        fig, axes = plt.subplots(len(models), len(rates), squeeze=False,
                                 figsize=(6.5 * len(rates), 2.5 * len(models)), layout="constrained")
        for i, model in enumerate(models):
            for j, rate in enumerate(rates):
                ax = axes[i, j]
                for ratio in sorted(paths.altruism_ratio.unique()):
                    data = paths[(paths.model == model) & (paths.growth_rate == rate) &
                                 (paths.altruism_ratio == ratio) & (paths.metric == metric)].sort_values("Step")
                    x, y, se = data.Step.to_numpy(), data["mean"].to_numpy(), data["sem"].fillna(0).to_numpy()
                    ax.plot(x, y, color=color(ratio), label=society(ratio))
                    ax.fill_between(x, y-se, y+se, color=color(ratio), alpha=.15)
                ax.set(title=f"{model} · r = {rate:g}",
                       xlim=(0, max(1, horizon)))
                ax.grid(color="#eeeeee")
        fig.supylabel(ylabel)
        fig.supxlabel("Recorded Step")
        axes[0, 0].legend(frameon=False, fontsize=9)
        save(fig, output, metric + suffix)
        files.append(metric + suffix)
    return files


def run(results, output):
    empty_output(output, results)
    receipt = json.loads((results / "analysis.json").read_text())
    font = style()
    if (results / "norm_trajectories.csv").is_file():
        paths = pd.read_csv(results / "norm_trajectories.csv")
        labels = {"personal_similarity_zero_diagonal": "Personal-norm similarity\n(published definition)",
                  "personal_similarity_pairwise": "Personal-norm pairwise similarity",
                  "personal_group_alignment": "Personal–collective norm similarity",
                  "group_norm_stability": "Consecutive collective-norm similarity"}
        metrics = [(m, labels[m]) for m in labels if m in set(paths.metric)]
        files = plot_paths(paths, output, receipt["horizon_index"], metrics)
        return {"figures": files, "font": font,
                "error_bars": "one standard error across societies", "source": str(results.resolve())}
    means = pd.read_csv(results / "summary.csv")
    paths = pd.read_csv(results / "trajectories.csv")
    models = sorted(means.model.unique())
    rates = sorted(means.growth_rate.unique())
    conditions = [c for c in CONDITIONS if c in set(means.condition)]
    files = []
    for metric, ylabel in (("survival", "Survival time (recorded Step)"), ("harvest_per_agent", "Total harvest per agent")):
        if len(conditions) == 1:
            fig, axes = plt.subplots(1, len(rates), squeeze=False, figsize=(7.3 * len(rates), 4.7), layout="constrained")
            for ax, rate in zip(axes.flat, rates):
                ratios = sorted(means.altruism_ratio.unique())
                width = .72 / len(ratios)
                for j, ratio in enumerate(ratios):
                    values = means[(means.growth_rate == rate) & (means.altruism_ratio == ratio) & (means.metric == metric)].set_index("model").reindex(models)
                    ax.bar(np.arange(len(models)) + (j - (len(ratios)-1)/2) * width, values["mean"], width,
                           yerr=values["sem"].fillna(0), capsize=2, color=color(ratio), label=society(ratio))
                ax.set(xticks=np.arange(len(models)), xticklabels=models, ylabel=ylabel,
                       title=f"{ { .2: 'Harsh', .6: 'Rich' }.get(rate, 'Resource')} environment (r = {rate:g})")
                ax.tick_params(axis="x", labelrotation=35)
                ax.legend(frameon=False)
                ax.set_axisbelow(True)
                ax.grid(axis="y", color="#eeeeee")
        else:
            fig, axes = plt.subplots(len(models), len(rates), squeeze=False,
                                     figsize=(7.2 * len(rates), 3.2 * len(models)), layout="constrained")
            for i, model in enumerate(models):
                for j, rate in enumerate(rates):
                    ax = axes[i, j]
                    ratios = sorted(means.altruism_ratio.unique())
                    for k, condition in enumerate(conditions):
                        values = means[(means.model == model) & (means.growth_rate == rate) &
                                       (means.condition == condition) & (means.metric == metric)].set_index("altruism_ratio").reindex(ratios)
                        ax.bar(np.arange(len(ratios)) + (k - (len(conditions) - 1) / 2) * .18, values["mean"], .18,
                               yerr=values["sem"].fillna(0), capsize=2, color=[color(r) for r in ratios],
                               hatch=CONDITIONS[condition][1], edgecolor="#555555", linewidth=.4)
                    ax.set(xticks=np.arange(len(ratios)), xticklabels=[society(r) for r in ratios], ylabel=ylabel,
                           title=f"{model} · r = {rate:g}")
                    ax.grid(axis="y", color="#eeeeee")
                    ax.set_axisbelow(True)
            fig.legend(handles=[Patch(facecolor="white", edgecolor="#555555", hatch=CONDITIONS[c][1],
                                      label=CONDITIONS[c][0]) for c in conditions], loc="outside upper center", ncol=4, frameon=False)
        save(fig, output, metric)
        files.append(metric)
    if len(conditions) == 1:
        metrics = (("resource", "Remaining resource"), ("effort", "Mean fishing effort"),
                   ("efficiency", "Harvest efficiency"))
        files.extend(plot_paths(paths, output, receipt["horizon_index"], metrics))
    return {"figures": files, "font": font, "error_bars": "one standard error across societies",
            "source": str(results.resolve())}
