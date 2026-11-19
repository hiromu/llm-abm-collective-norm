"""Job-market trajectories, semantic analyses and seed contrasts."""

from collections import defaultdict
import hashlib
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image

from ..config import ARMS, DECISION_ROUNDS, LABELS, MARKETS, MODELS, QUALITIES, SEEDS
from ..analysis.tables import build_tables
from ..data import prepare_output, read_csv, trajectories, verify, verify_semantics, write_csv

COLORS = {"HQ": "#1f77b4", "LQ": "#ff7f0e", "Employer": "#603d89",
          "individual": "#777777", "social": "#603d89"}
STYLES = {"40/10": "-", "10/10": (0, (4.5, 2.5))}
RATIO = 1040 / 540


def configure_style(size=10.5):
    fonts = Path("/System/Library/Fonts/Supplemental")
    for name in ("Arial.ttf", "Arial Bold.ttf"):
        if (fonts / name).is_file():
            font_manager.fontManager.addfont(str(fonts / name))
    available = {f.name for f in font_manager.fontManager.ttflist}
    family = next((f for f in ("Arial", "Liberation Sans", "DejaVu Sans") if f in available))
    plt.rcParams.update({"font.family": family, "font.size": size, "text.color": "#242424",
                         "axes.labelcolor": "#333333", "xtick.color": "#333333",
                         "ytick.color": "#333333", "axes.edgecolor": "#555555",
                         "axes.linewidth": .65, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none"})
    return family


def save(fig, output, name):
    for extension in ("pdf", "png", "svg"):
        fig.savefig(output / f"{name}.{extension}", dpi=200, facecolor="white")
    plt.close(fig)


def market_bar_style(color, market):
    """Use solid high-skill and pale, hatched low-skill market bars."""
    if market not in MARKETS:
        raise ValueError(f"Unknown market: {market}")
    base = np.array(to_rgb(color))
    return {"facecolor": base if market == "40/10" else .3 * base + .7,
            "edgecolor": .8 * base, "hatch": None if market == "40/10" else "///"}


def pixel_text(fig, x, y, text, **kwargs):
    w, h = fig.get_size_inches() * fig.dpi
    return fig.text(x / w, 1 - y / h, text, va="top", **kwargs)


def pixel_legend(fig, x, y, text, color, style="-"):
    w, h = fig.get_size_inches() * fig.dpi
    fig.add_artist(Line2D([x / w, (x + 52) / w], [1 - y / h] * 2,
                          transform=fig.transFigure, color=color, lw=1.5, ls=style))
    pixel_text(fig, x + 70, y - 22, text, fontsize=14.4)


def technology_plot_axes(fig, left, top):
    w, h = fig.get_size_inches() * fig.dpi
    ax = fig.add_axes((left / w, (h - top - 540) / h, 1040 / w, 540 / h))
    ax.set(xlim=(-3.5, 103.5), ylim=(-.02, 1.03), xticks=(0, 20, 40, 60, 80, 100),
           yticks=np.arange(0, 1.01, .2))
    ax.set_yticklabels([f"{v:.1f}" for v in np.arange(0, 1.01, .2)])
    ax.tick_params(labelsize=13.68, length=0, pad=6)
    ax.grid(color="#eeeeee", linewidth=.6, alpha=.65)
    ax.set_axisbelow(True)
    ax.set_xlabel("Round", fontsize=14.4, labelpad=8)
    ax.set_ylabel("Mean technology use", fontsize=14.4, labelpad=9)
    return ax


def technology_axis(fig, left, top, curves):
    ax = technology_plot_axes(fig, left, top)
    for market, role, values in curves:
        ax.plot(range(1, 101), values, color=COLORS[role], lw=1.2, ls=STYLES[market],
                zorder=2 if role == "Employer" else 3)
    return ax


def abm_artwork_axis(fig, left, top, artwork, box):
    """Place curve artwork on matching plot axes."""
    pixels = np.array(artwork.crop(box).convert("RGBA"))
    rgb = pixels[:, :, :3].astype(int)
    # Make grayscale axis marks transparent while preserving colored curves.
    pixels[:, :, 3] = np.where(np.ptp(rgb, axis=2) > 4, 255, 0)
    pixels[340:, :, 3] = 0  # All supplied curves lie above this legend-only region.
    ax = technology_plot_axes(fig, left, top)
    ax.set(xlim=(0, 646), ylim=(455, 0), xticks=(22, 154, 286, 418, 550),
           xticklabels=(0, 20, 40, 60, 80), yticks=(441, 355, 270, 184, 99, 13),
           yticklabels=("0.0", "0.2", "0.4", "0.6", "0.8", "1.0"))
    ax.imshow(pixels, origin="upper", extent=(-.5, 645.5, 454.5, -.5),
              aspect="auto", interpolation="none", zorder=3)
    return ax


def series_table(data):
    grouped = defaultdict(list)
    for r in trajectories(data):
        grouped[r["model"], r["arm"], r["market"], r["role"]].append(r)
    return {key: [r["mean"] for r in sorted(rows, key=lambda x: x["round"])]
            for key, rows in grouped.items()}


def figure3(data, output, trajectory_data=None):
    configure_style(14.4)
    series = series_table(trajectory_data or data)
    fig = plt.figure(figsize=(12.8, 19.4), dpi=200, facecolor="white")
    with Image.open(data / "artwork/job_market_cycle_abm.png") as image:
        prefix = np.asarray(image.convert("RGBA"))
    fig.figimage(prefix[:1070], xo=0, yo=3880 - 1070, origin="upper",
                 resize=False, interpolation="none")
    fig.figimage(prefix[2060:], xo=0, yo=3880 - 2136, origin="upper",
                 resize=False, interpolation="none")
    pixel_text(fig, 48, 1095, "b.  Technology use in rule-based ABM", fontsize=18.72, weight="bold")
    pixel_legend(fig, 200, 1190, "Low skill (10/10)", "#666666", STYLES["10/10"])
    pixel_legend(fig, 800, 1190, "High skill (40/10)", "#666666")
    pixel_text(fig, 1472, 1168, "Initial technology use: U(0, 1)", fontsize=14.4)
    for left, label in ((200, "Applicants"), (1472, "Employers")):
        pixel_text(fig, left, 1250, label, fontsize=15.84, weight="bold")
    pixel_legend(fig, 200, 1330, "High quality", COLORS["HQ"])
    pixel_legend(fig, 720, 1330, "Low quality", COLORS["LQ"])
    abm_path = data.parent / "artwork/abm_uniform.png"
    with Image.open(abm_path) as artwork:
        if artwork.size != (1453, 589):
            raise ValueError("The supplied ABM artwork must retain its native 1453 x 589 size")
        abm_artwork_axis(fig, 200, 1380, artwork, (75, 69, 721, 524))
        abm_artwork_axis(fig, 1472, 1380, artwork, (797, 69, 1443, 524))
    for x, text, color, style in [(200, "Low skill (10/10)", "#666666", STYLES["10/10"]),
                                  (800, "High skill (40/10)", "#666666", "-"),
                                  (1472, "High quality", COLORS["HQ"], "-"),
                                  (2000, "Low quality", COLORS["LQ"], "-")]:
        pixel_legend(fig, x, 2190, text, color, style)
    pixel_text(fig, 200, 2250, "Social learning; 5 seeds; initial technology use: U(0, 1)",
               fontsize=13.68, color="#444444")
    for i, model in enumerate(MODELS):
        left, top = (200, 1472)[i % 2], (2390, 3170)[i // 2]
        pixel_text(fig, left, top - 70, LABELS[model], fontsize=15.84, weight="bold")
        technology_axis(fig, left, top, [(m, q, series[model, "social", m, q])
                                        for m in MARKETS for q in QUALITIES])
    save(fig, output, "figure3")
    return {"abm_artwork": str(abm_path), "sha256": hashlib.sha256(abm_path.read_bytes()).hexdigest(),
            "initialization": "U(0,1)", "abm_rounds": 90,
            "curves": "Supplied raster paths; no numerical digitization or resimulation"}


def welfare_bars(statistics, output):
    """Plot welfare levels and paired information effects."""
    family = configure_style(14)
    means = read_csv(statistics / "welfare_means.csv")
    contrasts = read_csv(statistics / "welfare_contrasts.csv")
    levels = {(r["model"], r["arm"], r["market"]): r for r in means}
    changes = {(r["model"], r["market"]): r for r in contrasts}
    if len(means) != 16 or set(levels) != {(m, a, k) for m in MODELS for a in ARMS for k in MARKETS}:
        raise ValueError("All sixteen welfare means are required")
    if len(contrasts) != 8 or set(changes) != {(m, k) for m in MODELS for k in MARKETS}:
        raise ValueError("All eight paired welfare contrasts are required")
    for record in means + contrasts:
        if int(record.get("n_seeds", record.get("n_pairs"))) != 5:
            raise ValueError("Every welfare estimate must use five society seeds")
        if not all(np.isfinite(float(record[k])) for k in ("mean", "ci_low", "ci_high")):
            raise ValueError("Non-finite welfare estimate")
        if not float(record["ci_low"]) <= float(record["mean"]) <= float(record["ci_high"]):
            raise ValueError("Welfare interval does not contain its mean")

    fig, axes = plt.subplots(2, 3, figsize=(19.2, 10.2), dpi=200)
    fig.subplots_adjust(left=.065, right=.99, bottom=.13, top=.85, wspace=.26, hspace=.57)
    labels = ["DeepSeek-4.1\nFlash", "GPT-6\nLuna", "GPT-OSS\n120B", "Qwen-3.5\n35B"]
    mixed_color = (np.array(to_rgb(COLORS["HQ"])) + np.array(to_rgb(COLORS["LQ"]))) / 2
    for col, heading in enumerate(("Individual information", "Social information", "Social - individual")):
        box = axes[0, col].get_position()
        fig.text((box.x0 + box.x1) / 2, .958, heading, ha="center", va="top",
                 fontsize=19, weight="bold")
    for row, market in enumerate(MARKETS):
        market_label = "High-skill market (40/10)" if row == 0 else "Low-skill market (10/10)"
        for col, ax in enumerate(axes[row]):
            records = ([levels[model, ARMS[col], market] for model in MODELS] if col < 2
                       else [changes[model, market] for model in MODELS])
            values = np.array([float(r["mean"]) for r in records])
            lower = np.array([float(r["ci_low"]) for r in records])
            upper = np.array([float(r["ci_high"]) for r in records])
            limits = ((0, 3200) if row == 0 else (0, 1050)) if col < 2 else (
                     (-140, 120) if row == 0 else (-30, 110))
            ticks = ((0, 800, 1600, 2400, 3200) if row == 0 else (0, 250, 500, 750, 1000)) if col < 2 else (
                    (-100, -50, 0, 50, 100) if row == 0 else (-25, 0, 25, 50, 75, 100))
            ax.set(xlim=(-.6, 3.6), ylim=limits, xticks=range(4), xticklabels=labels, yticks=ticks)
            ax.set_title(f"{chr(97 + row * 3 + col)}. {market_label}",
                         loc="left", fontsize=15, weight="bold", pad=16)
            ax.set_ylabel("Mean net welfare per round" if col < 2 else "Difference in net welfare per round",
                          fontsize=13.5, labelpad=9)
            ax.tick_params(length=0, labelsize=13, pad=6)
            ax.grid(axis="y", color="#ededed", lw=.7)
            ax.set_axisbelow(True)
            if col == 2:
                ax.axhline(0, color="#555555", lw=.8, zorder=2)
            ax.bar(range(4), values, width=.56, **market_bar_style(mixed_color, market),
                   linewidth=.85, zorder=3)
            ax.errorbar(range(4), values, yerr=np.stack((values - lower, upper - values)),
                        fmt="none", ecolor="#222222", capsize=4, elinewidth=1.2, zorder=4)
            padding = (limits[1] - limits[0]) * .025
            for x, value, lo, hi in zip(range(4), values, lower, upper):
                text_y = lo - padding if col == 2 and value < 0 else hi + padding
                if not limits[0] < text_y < limits[1]:
                    raise ValueError("A welfare interval or label exceeds the plot range")
                ax.text(x, text_y, f"{value:+.1f}" if col == 2 else f"{value:,.1f}", ha="center",
                        va="top" if col == 2 and value < 0 else "bottom", fontsize=12.5)
    handles = [Patch(**market_bar_style(mixed_color, market), label=label)
               for market, label in zip(MARKETS, ("High skill (40/10)", "Low skill (10/10)"))]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .025),
               ncol=2, frameon=False, fontsize=14, columnspacing=2.2)
    save(fig, output, "welfare_bars")
    return {"font": family, "panels": 6, "plotted_seed_points": 0,
            "error_bars": "pointwise 95% t intervals; paired in difference panels",
            "stars": "none; no hypothesis tests performed", "manuscript_updated": False}


def peer_information(data, output, trajectory_data=None):
    configure_style(15)
    series = series_table(trajectory_data or data)
    fig, axes = plt.subplots(4, 4, figsize=(14.4, 14.2), dpi=200)
    fig.subplots_adjust(left=.065, right=.975, bottom=.055, top=.85,
                        wspace=.17, hspace=.54)
    for arm_index, arm in enumerate(ARMS):
        first, last = (axes[0, i].get_position() for i in (2 * arm_index, 2 * arm_index + 1))
        fig.text((first.x0 + last.x1) / 2, .944, arm.title() + " information",
                 ha="center", weight="bold", fontsize=18)
        fig.add_artist(Line2D([first.x0, last.x1], [.931, .931],
                              transform=fig.transFigure, color="#aaaaaa", lw=.7))
    for col in range(4):
        box = axes[0, col].get_position()
        fig.text((box.x0 + box.x1) / 2, .907, "Applicants" if col % 2 == 0 else "Employers",
                 ha="center", fontsize=16, weight="bold")
    for row, model in enumerate(MODELS):
        box = axes[row, 0].get_position()
        fig.text(box.x0, box.y1 + .026, LABELS[model], weight="bold", fontsize=17)
        for col, ax in enumerate(axes[row]):
            arm = ARMS[col // 2]
            roles = QUALITIES if col % 2 == 0 else ("Employer",)
            ax.set(xlim=(1, 100), ylim=(0, 1.03), xticks=(1, 25, 50, 75, 100),
                   yticks=(0, .25, .5, .75, 1))
            ax.tick_params(length=0, labelsize=13.5, pad=4)
            ax.grid(color="#eeeeee", lw=.6)
            ax.set_axisbelow(True)
            if col == 0:
                ax.set_ylabel("Mean technology use", fontsize=14)
            else:
                ax.tick_params(labelleft=False)
            if row == 3:
                ax.set_xlabel("Round", fontsize=14.5)
            ax.text(0, 1.06, f"{chr(97 + 4 * row + col)}.", transform=ax.transAxes,
                    fontsize=13.5, weight="bold")
            for market in MARKETS:
                for role in roles:
                    ax.plot(range(1, 101), series[model, arm, market, role],
                            color=COLORS[role], ls=STYLES[market], lw=1.8)
    handles = [Line2D([], [], color=COLORS[q], label=name, lw=1.8)
               for q, name in zip((*QUALITIES, "Employer"), ("High quality", "Low quality", "Employers"))]
    handles += [Line2D([], [], color="#666666", ls=STYLES[m], label=name, lw=1.8)
                for m, name in zip(MARKETS, ("High skill (40/10)", "Low skill (10/10)"))]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.52, .995),
               ncol=5, frameon=False, fontsize=14, columnspacing=1.5, handlelength=2.1)
    save(fig, output, "figureS11_peer_information")


def technology_contrasts(statistics, output):
    """Plot paired seed contrasts in technology use."""
    configure_style(14)
    path = statistics / "technology_tests.csv"
    rows = read_csv(path)
    lookup = {(r["family"], r["model"], r["subgroup"]): r for r in rows}
    expected = {(family, model, group) for family in ("market", "quality", "peer")
                for model in MODELS for group in (QUALITIES if family == "market" else MARKETS)}
    if len(rows) != 24 or set(lookup) != expected:
        raise ValueError("The contrast figure requires all 24 planned comparisons")
    for r in rows:
        if int(r["n_pairs"]) != 5 or int(r["df"]) != 4:
            raise ValueError("Contrasts must use five society-seed pairs")
        fields = ("difference", "ci_low", "ci_high", "p_holm")
        if not all(np.isfinite(float(r[k])) for k in fields):
            raise ValueError("Non-finite statistical result")
        if not float(r["ci_low"]) <= float(r["difference"]) <= float(r["ci_high"]):
            raise ValueError("Confidence interval does not contain its point estimate")
        if not 0 <= float(r["p_holm"]) <= 1:
            raise ValueError("Invalid adjusted p value")
    fig, axes = plt.subplots(3, 1, figsize=(12.8, 11.2), dpi=200)
    fig.subplots_adjust(left=.105, right=.985, top=.91, bottom=.075, hspace=.65)
    headings = ["a. Market contrast: low skill - high skill (social)",
                "b. Quality contrast: high quality - low quality (social)",
                "c. Peer-information contrast: social - individual (both qualities)"]
    labels = ["DeepSeek-4.1\nFlash", "GPT-6\nLuna", "GPT-OSS\n120B", "Qwen-3.5\n35B"]
    mixed_color = (np.array(to_rgb(COLORS["HQ"])) + np.array(to_rgb(COLORS["LQ"]))) / 2
    for ax, family, heading in zip(axes, ("market", "quality", "peer"), headings):
        groups = QUALITIES if family == "market" else MARKETS
        ax.set_title(heading, loc="left", fontsize=15, weight="bold", pad=38)
        ax.axhline(0, color="#555555", lw=.9, zorder=2)
        ax.grid(axis="y", color="#ededed", lw=.7)
        ax.set_axisbelow(True)
        ax.set(xlim=(-.6, 3.6), xticks=range(4), xticklabels=labels,
               ylim=(-.35, .65), yticks=(-.2, 0, .2, .4, .6))
        ax.set_ylabel("Difference in mean\ntechnology use", fontsize=13)
        ax.tick_params(length=0, pad=5, labelsize=13)
        legend = []
        for group_index, group in enumerate(groups):
            selected = [lookup[family, model, group] for model in MODELS]
            mean = np.array([float(r["difference"]) for r in selected])
            lower = np.array([float(r["ci_low"]) for r in selected])
            upper = np.array([float(r["ci_high"]) for r in selected])
            position = np.arange(4) + (group_index - .5) * .34
            if family == "market":
                style = {"facecolor": COLORS[group], "edgecolor": COLORS[group]}
                name = ("High quality", "Low quality")[group_index]
            else:
                style = market_bar_style(mixed_color, group)
                name = ("High skill (40/10)", "Low skill (10/10)")[group_index]
            ax.bar(position, mean, width=.29, **style, linewidth=.9, zorder=3)
            ax.errorbar(position, mean, yerr=np.stack((mean - lower, upper - mean)),
                        fmt="none", ecolor="#222222", capsize=4, elinewidth=1.1, zorder=4)
            legend.append(Patch(**style, label=name))
            for x, m, lo, hi, r in zip(position, mean, lower, upper, selected):
                p = float(r["p_holm"])
                stars = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
                if stars:
                    ax.text(x, hi + .015 if m >= 0 else lo - .015, stars,
                            ha="center", va="bottom" if m >= 0 else "top", fontsize=13)
        ax.legend(handles=legend, loc="lower right", bbox_to_anchor=(1, 1.01),
                  frameon=False, ncol=2, fontsize=12, borderaxespad=0)
    save(fig, output, "figureS15_technology_contrasts")
    return {"comparisons": len(rows), "n_pairs": 5, "error_bars": "pointwise 95% t intervals",
            "stars": "Holm-adjusted p across all 24 tests", "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def technology_means(data, output):
    """Plot late subgroup means and pointwise t intervals across society seeds."""
    from scipy.stats import t

    configure_style(14)
    grouped = defaultdict(list)
    for row in trajectories(data):
        if row["role"] in QUALITIES and 86 <= row["round"] <= 100:
            grouped[row["model"], row["arm"], row["market"], row["role"]].append(row)
    critical = float(t.ppf(.975, len(SEEDS) - 1))
    summaries, seed_rows, lookup = [], [], {}
    for model in MODELS:
        for arm in ARMS:
            for market in MARKETS:
                for quality in QUALITIES:
                    rows = grouped[model, arm, market, quality]
                    if len(rows) != 15:
                        raise ValueError("Every subgroup needs all rounds 86--100")
                    values = np.array([math.fsum(r[f"seed_{seed}"] for r in rows) / 15
                                       for seed in SEEDS])
                    mean = math.fsum(values) / len(SEEDS)
                    half_width = critical * float(values.std(ddof=1)) / math.sqrt(len(SEEDS))
                    identity = {"model": model, "arm": arm, "market": market, "quality": quality}
                    summary = {**identity, "round_start": 86, "round_end": 100,
                               "n_seeds": len(SEEDS), "mean": mean,
                               "ci_low": mean - half_width, "ci_high": mean + half_width}
                    if not 0 <= summary["ci_low"] <= summary["ci_high"] <= 1:
                        raise ValueError("A confidence interval exceeds the plotted 0--1 range")
                    summaries.append(summary)
                    seed_rows.extend({**identity, "seed": seed, "mean": float(value)}
                                     for seed, value in zip(SEEDS, values))
                    lookup[model, arm, market, quality] = summary

    fig, axes = plt.subplots(2, 2, figsize=(16, 9.8), dpi=200, sharey=True)
    fig.subplots_adjust(left=.065, right=.99, bottom=.135, top=.86,
                        wspace=.14, hspace=.55)
    labels = ["DeepSeek-4.1\nFlash", "GPT-6\nLuna", "GPT-OSS\n120B", "Qwen-3.5\n35B"]
    for col, arm in enumerate(ARMS):
        box = axes[0, col].get_position()
        fig.text((box.x0 + box.x1) / 2, .955, arm.capitalize() + " information",
                 ha="center", va="top", fontsize=19, weight="bold")
    for row, market in enumerate(MARKETS):
        for col, arm in enumerate(ARMS):
            ax = axes[row, col]
            market_label = "High-skill market (40/10)" if row == 0 else "Low-skill market (10/10)"
            ax.set_title(f"{chr(97 + 2 * row + col)}. {market_label}",
                         loc="left", fontsize=16, weight="bold", pad=14)
            ax.set(xlim=(-.6, 3.6), ylim=(0, 1), xticks=range(4), xticklabels=labels,
                   yticks=np.arange(0, 1.01, .2))
            ax.tick_params(length=0, labelsize=13.5, pad=6)
            ax.grid(axis="y", color="#ededed", lw=.7)
            ax.set_axisbelow(True)
            if col == 0:
                ax.set_ylabel("Mean technology use", fontsize=14.5, labelpad=9)
            for quality_index, quality in enumerate(QUALITIES):
                positions = np.arange(4) + (quality_index - .5) * .36
                records = [lookup[model, arm, market, quality] for model in MODELS]
                means = np.array([record["mean"] for record in records])
                errors = np.array([record["ci_high"] - record["mean"] for record in records])
                ax.bar(positions, means, width=.32, **market_bar_style(COLORS[quality], market),
                       linewidth=.75, zorder=3)
                ax.errorbar(positions, means, yerr=errors, fmt="none", color="#202020",
                            capsize=4, elinewidth=1.2, zorder=5)
    handles = [Patch(facecolor=COLORS[q], label=name)
               for q, name in zip(QUALITIES, ("High quality", "Low quality"))]
    mixed_color = (np.array(to_rgb(COLORS["HQ"])) + np.array(to_rgb(COLORS["LQ"]))) / 2
    handles += [Patch(**market_bar_style(mixed_color, market), label=name)
                for market, name in zip(MARKETS, ("High skill (40/10)", "Low skill (10/10)"))]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.52, .025),
               ncol=4, frameon=False, fontsize=14.5, columnspacing=1.8)
    save(fig, output, "figureS14_technology_means")
    write_csv(output / "technology_subgroup_means.csv", summaries)
    write_csv(output / "technology_subgroup_seed_means.csv", seed_rows)
    return {"subgroups": len(summaries), "seed_means": len(seed_rows), "window": [86, 100],
            "n_seeds": len(SEEDS), "error_bars": "pointwise 95% t intervals for subgroup means",
            "critical_t": critical, "significance_tests": "none", "y_limits": [0, 1]}


def align_headings(fig, axes, headings):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    target = min(ax.yaxis.label.get_window_extent(renderer).x0 for ax in axes)
    for ax in axes:
        label = ax.yaxis.label
        left = label.get_window_extent(renderer).x0
        position = label.get_position()
        label.set_position((position[0] + target - left, position[1]))
    for text in headings:
        text.set_x(target / fig.bbox.width)


def decoding(data, output, semantics=None):
    configure_style()
    datasets = {t: read_csv((semantics or data / "semantics") / f"{t}_decoding.csv") for t in ("market", "quality")}
    scores = np.array([float(r["accuracy"]) * 100 for rows in datasets.values() for r in rows])
    lower = max(0, min(40, np.floor((scores.min() - 2) / 5) * 5))
    upper = min(100, max(75, np.ceil((scores.max() + 2) / 5) * 5))
    fig = plt.figure(figsize=(16.6, 6.55))
    left_axes = []
    for row, task in enumerate(("market", "quality")):
        for col, model in enumerate(MODELS):
            ax = fig.add_axes(( (1.12 + 3.9 * col) / 16.6, (3.88, 1.03)[row] / 6.55,
                               3.3 / 16.6, (3.3 / RATIO) / 6.55))
            ax.set(xlim=(1, 100), ylim=(lower, upper), xticks=(1, 20, 40, 60, 80, 100),
                   yticks=np.arange(np.ceil(lower / 10) * 10, upper + .01, 10), xlabel="Round")
            ax.axhline(50, color="#999999", ls=":", lw=.8)
            ax.grid(color="#eeeeee", lw=.5)
            if col == 0:
                ax.set_ylabel("Decoding accuracy (%)")
                left_axes.append(ax)
            else:
                ax.tick_params(labelleft=False)
            if row == 0:
                fig.text((1.12 + 3.9 * col + 1.65) / 16.6, 6.30 / 6.55, LABELS[model],
                         ha="center", weight="bold", fontsize=12.5)
            for arm in ARMS:
                for market in (MARKETS if task == "quality" else (None,)):
                    lookup = {(int(r["round"]), int(r["held_out_seed"])): float(r["accuracy"]) * 100
                              for r in datasets[task] if r["model"] == model and r["arm"] == arm
                              and (market is None or r["market"] == market)}
                    values = np.array([[lookup[t, s] for s in SEEDS] for t in DECISION_ROUNDS])
                    ax.fill_between(DECISION_ROUNDS, values.min(1), values.max(1), color=COLORS[arm], alpha=.075)
                    ax.plot(DECISION_ROUNDS, values.mean(1), color=COLORS[arm], lw=1.55,
                            ls=STYLES[market] if market else "-", marker="o" if market != "10/10" else "^", ms=2)
    headings = [fig.text(.03, y / 6.55, label, weight="bold", fontsize=12)
                for y, label in [(5.91, "a. Market decoding"), (3.22, "b. Quality decoding within each market")]]
    handles = [Line2D([], [], color=COLORS[a], label=a.title() + " information", lw=1.6) for a in ARMS]
    handles += [Line2D([], [], color="#555555", ls=STYLES[m], label=label, lw=1.6)
                for m, label in zip(MARKETS, ("High skill (40/10)", "Low skill (10/10)"))]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.51, .014), ncol=4, frameon=False, fontsize=10.8)
    align_headings(fig, left_axes, headings)
    save(fig, output, "figureS12_decoding")


def centroid_trajectories(data, output, semantics=None):
    configure_style()
    rows = read_csv((semantics or data / "semantics") / "centroid_projections.csv")
    grouped = defaultdict(list)
    for r in rows:
        if r["arm"] == "social":
            grouped[r["model"], r["market"], r["quality"], int(r["round"])].append([float(r["pc1"]), float(r["pc2"])])
    paths = {(model, m, q): np.array([np.mean(grouped[model, m, q, t], axis=0) for t in DECISION_ROUNDS])
             for model in MODELS for m in MARKETS for q in QUALITIES}
    fig = plt.figure(figsize=(16.6, 12.2))
    groups = [(m, q) for m in MARKETS for q in QUALITIES]
    left_axes, headings = [], []
    for col, model in enumerate(MODELS):
        fig.text((1.12 + 3.9 * col + 1.625) / 16.6, 11.95 / 12.2, LABELS[model],
                 weight="bold", fontsize=12.5, ha="center")
        all_points = np.concatenate([paths[model, m, q] for m, q in groups])
        lo, hi = all_points.min(0), all_points.max(0)
        padding = np.maximum((hi - lo) * .1, .004)
        for row, (market, quality) in enumerate(groups):
            ax = fig.add_axes(((1.12 + 3.9 * col) / 16.6, (9.05, 6.35, 3.65, .95)[row] / 12.2,
                              3.25 / 16.6, 2.05 / 12.2))
            ax.set(xlim=(lo[0] - padding[0], hi[0] + padding[0]),
                   ylim=(lo[1] - padding[1], hi[1] + padding[1]))
            for m, q in groups:
                if (m, q) != (market, quality):
                    xy = paths[model, m, q]
                    ax.plot(xy[:, 0], xy[:, 1], color="#b0b0b0", alpha=.18, lw=.6)
            xy = paths[model, market, quality]
            segments = np.stack([xy[:-1], xy[1:]], axis=1)
            base = np.array(to_rgb(COLORS[quality]))
            colors = [1 - intensity * (1 - base) for intensity in np.linspace(.2, 1, len(segments))]
            ax.add_collection(LineCollection(segments, colors=colors, linewidths=1.65))
            ax.scatter(*xy[0], s=34, facecolors="white", edgecolors=COLORS[quality], linewidths=1.2, zorder=4)
            ax.scatter(*xy[-1], s=42, color=COLORS[quality], edgecolors="white", linewidths=.7, zorder=4)
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.tick_params(labelsize=9)
            ax.grid(color="#f0f0f0", lw=.5)
            if col == 0:
                ax.set_ylabel("PC2")
                left_axes.append(ax)
            if row == 3:
                ax.set_xlabel("PC1")
    labels = ["High skill / high quality requirement", "High skill / low quality requirement",
              "Low skill / high quality requirement", "Low skill / low quality requirement"]
    for i, (y, label) in enumerate(zip((11.52, 8.62, 5.92, 3.22), labels)):
        headings.append(fig.text(.03, y / 12.2, f"{chr(97 + i)}. {label}", weight="bold", fontsize=12))
    handles = [Line2D([], [], color=COLORS[q], label=label, lw=1.7)
               for q, label in zip(QUALITIES, ("High quality requirement", "Low quality requirement"))]
    handles += [Line2D([], [], marker="o", color="#555555", mfc=fill, ls="", label=label)
                for fill, label in [("white", "Start (round 4)"), ("#555555", "End (round 100)")]]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.51, .012), ncol=4, frameon=False)
    align_headings(fig, left_axes, headings)
    save(fig, output, "figureS13_trajectories")


def build_figures(data, output, semantics=None, trajectory_data=None, statistics=None):
    family = configure_style()
    abm = figure3(data, output, trajectory_data)
    peer_information(data, output, trajectory_data)
    for render in (decoding, centroid_trajectories):
        render(data, output, semantics)
    means = technology_means(trajectory_data or data, output)
    result = {"figures": ["figure3", "figureS11_peer_information", "figureS12_decoding", "figureS13_trajectories",
                           "figureS14_technology_means"],
              "formats": ["pdf", "png", "svg"], "font": family, "means": means,
              "figure3_artwork": abm}
    if statistics:
        result["contrasts"] = technology_contrasts(statistics, output)
        result["figures"].append("figureS15_technology_contrasts")
    return result


def run(data, output, simulation_results=None, semantics=None, statistics=None):
    receipt = {"inputs": verify(data, scope="visualization")}
    trajectory_data = simulation_results or data
    rows = trajectories(trajectory_data)
    semantic_data = semantics or data / "semantics"
    if semantics:
        receipt["semantic_validation"] = verify_semantics(semantic_data)
    prepare_output(output, data, trajectory_data, semantic_data, *([statistics] if statistics else []))
    receipt["tables"] = build_tables(data, output, semantic_data, trajectory_data)
    receipt["figures"] = build_figures(data, output, semantic_data, trajectory_data, statistics)
    receipt["trajectory_input"] = {
        "path": str(trajectory_data / "trajectories.csv"), "rows": len(rows),
        "sha256": hashlib.sha256((trajectory_data / "trajectories.csv").read_bytes()).hexdigest()}
    receipt["semantic_results"] = str(semantic_data)
    return receipt
