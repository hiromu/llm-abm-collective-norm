"""Society-level summaries and paired seed contrasts."""

import math

from ..config import ARMS, MARKETS, MODELS, QUALITIES, SEEDS
from ..data import prepare_output, read_csv, trajectories, verify, write_csv


def build_tables(data, output, semantics=None, trajectory_data=None):
    rows = trajectories(trajectory_data or data)
    windows = {"early": (4, 9), "late": (86, 100)}
    seed_rows = []
    lookup = {}
    for model in MODELS:
        for arm in ARMS:
            for market in MARKETS:
                for role in (*QUALITIES, "Employer"):
                    for window, (lo, hi) in windows.items():
                        selected = [r for r in rows if (r["model"], r["arm"], r["market"], r["role"])
                                    == (model, arm, market, role) and lo <= r["round"] <= hi]
                        for seed in SEEDS:
                            value = math.fsum(r[f"seed_{seed}"] for r in selected) / len(selected)
                            lookup[model, arm, market, role, window, seed] = value
                            seed_rows.append({"model": model, "arm": arm, "market": market,
                                              "role": role, "window": window, "seed": seed,
                                              "mean_technology": value})
    write_csv(output / "technology_by_seed.csv", seed_rows)
    summary = []
    for model in MODELS:
        for arm in ARMS:
            for market in MARKETS:
                high = [lookup[model, arm, market, "HQ", "late", s] for s in SEEDS]
                low = [lookup[model, arm, market, "LQ", "late", s] for s in SEEDS]
                summary.append({"model": model, "arm": arm, "market": market,
                                "high_quality_mean": math.fsum(high) / 5,
                                "low_quality_mean": math.fsum(low) / 5,
                                "applicant_mean": (math.fsum(high) + math.fsum(low)) / 10,
                                "high_minus_low": math.fsum(a - b for a, b in zip(high, low)) / 5,
                                "positive_gap_seeds": sum(a > b for a, b in zip(high, low))})
    write_csv(output / "late_technology.csv", summary)
    decoding = []
    for task in ("market", "quality"):
        scores = read_csv((semantics or data / "semantics") / f"{task}_decoding.csv")
        keys = sorted({(r["model"], r["arm"], r.get("market", ""), int(r["round"])) for r in scores})
        for model, arm, market, round_ in keys:
            values = [float(r["accuracy"]) for r in scores
                      if (r["model"], r["arm"], r.get("market", ""), int(r["round"]))
                      == (model, arm, market, round_)]
            decoding.append({"task": task, "model": model, "arm": arm, "market": market,
                             "round": round_, "mean_accuracy": math.fsum(values) / len(values),
                             "seed_min": min(values), "seed_max": max(values), "folds": len(values)})
    write_csv(output / "decoding_accuracy.csv", decoding)
    return {"technology_seed_rows": len(seed_rows), "late_summary_rows": len(summary),
            "decoding_summary_rows": len(decoding)}


def technology_tests(data, output):
    """Exploratory paired seed contrasts in mean technology use, rounds 86--100."""
    import numpy as np
    from scipy import stats

    receipt = {"inputs": verify(data, scope="statistics")}
    rows = trajectories(data)
    levels = {}
    for model in MODELS:
        for arm in ARMS:
            for market in MARKETS:
                for quality in QUALITIES:
                    selected = [r for r in rows if (r["model"], r["arm"], r["market"], r["role"])
                                == (model, arm, market, quality) and 86 <= r["round"] <= 100]
                    for seed in SEEDS:
                        levels[model, arm, market, quality, seed] = math.fsum(
                            r[f"seed_{seed}"] for r in selected) / 15

    def values(model, arm, market, quality=None):
        return np.array([levels[model, arm, market, quality, s]
                         if quality is not None else math.fsum(
                             levels[model, arm, market, q, s] for q in QUALITIES) / 2
                         for s in SEEDS])

    comparisons = []
    for model in MODELS:
        for quality in QUALITIES:
            comparisons.append(("market", model, quality, "low-skill minus high-skill",
                                values(model, "social", "10/10", quality),
                                values(model, "social", "40/10", quality)))
        for market in MARKETS:
            comparisons.append(("quality", model, market, "high-quality minus low-quality",
                                values(model, "social", market, "HQ"),
                                values(model, "social", market, "LQ")))
            comparisons.append(("peer", model, market, "Social minus Individual, both qualities",
                                values(model, "social", market), values(model, "individual", market)))

    results, paired = [], []
    for family, model, subgroup, contrast, a, b in comparisons:
        identity = {"family": family, "model": model, "subgroup": subgroup, "contrast": contrast}
        d = a - b
        test = stats.ttest_rel(a, b, alternative="two-sided")
        ci = test.confidence_interval(confidence_level=.95)
        exact = stats.permutation_test((d,), np.mean, permutation_type="samples",
                                       n_resamples=np.inf, alternative="two-sided")
        results.append({**identity, "n_pairs": len(SEEDS), "df": int(test.df),
                        "first_mean": float(a.mean()), "second_mean": float(b.mean()),
                        "difference": float(d.mean()), "ci_low": float(ci.low),
                        "ci_high": float(ci.high), "t": float(test.statistic),
                        "p_raw": float(test.pvalue), "p_holm": 0.0,
                        "p_sign_flip": float(exact.pvalue),
                        "positive_seeds": int((d > 0).sum()), "negative_seeds": int((d < 0).sum())})
        paired.extend({**identity, "seed": s, "first": float(x), "second": float(y),
                       "difference": float(x-y)} for s, x, y in zip(SEEDS, a, b))
    adjusted = 0.0
    for rank, index in enumerate(sorted(range(len(results)), key=lambda i: results[i]["p_raw"])):
        adjusted = max(adjusted, min(1.0, (len(results)-rank) * results[index]["p_raw"]))
        results[index]["p_holm"] = adjusted
    prepare_output(output, data)
    write_csv(output / "technology_tests.csv", results)
    write_csv(output / "technology_paired_seeds.csv", paired)
    return {**receipt, "window": [86, 100], "independent_units": "five society seeds",
            "contrasts": len(results), "paired_rows": len(paired),
            "test": "two-sided paired t, df=4", "confidence_intervals": "95%, pointwise",
            "multiplicity": "Holm across all 24 contrasts",
            "sensitivity": "two-sided exact sign flips, all 32 patterns; unadjusted",
            "interpretation": "exploratory; no applicant- or round-level pseudoreplication"}


def welfare_tables(data, output):
    """Late welfare levels and paired information contrasts from saved outcomes."""
    from collections import defaultdict
    import hashlib
    import itertools
    import json
    import numpy as np
    from scipy.stats import t

    path = data / "economic_reference.csv"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = json.loads((data / "manifest.json").read_text())
    if digest != manifest["files"][path.name]:
        raise ValueError("Economic reference checksum mismatch")
    rows = read_csv(path)
    keys = [(r["model"], r["arm"], r["market"], int(r["seed"]), int(r["round"])) for r in rows]
    expected = set(itertools.product(MODELS, ARMS, MARKETS, SEEDS, range(1, 101)))
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Welfare analysis requires every complete 100-round cell")
    metrics = ("total_net", "output", "total_cost", "HQ_net", "LQ_net", "Employer_net",
               "HQ", "LQ", "Employer")
    grouped, max_error = defaultdict(list), 0.0
    for r in rows:
        for metric in metrics:
            r[metric] = float(r[metric])
            if not math.isfinite(r[metric]):
                raise ValueError("Non-finite economic outcome")
        cost = 100 * (1 - r["HQ"]) + 100 * (1 - r["LQ"]) + 40 * (1 - r["Employer"])
        errors = (r["total_net"] - (r["output"] - r["total_cost"]),
                  r["total_net"] - (100 * r["HQ_net"] + 100 * r["LQ_net"] + 20 * r["Employer_net"]),
                  r["total_cost"] - cost)
        max_error = max(max_error, *(abs(v) for v in errors))
        if r["market"] == "10/10" and (r["output"] != 1000 or int(r["hired"]) != 100):
            raise ValueError("Unexpected production or hiring in the low-skill market")
        if 86 <= int(r["round"]) <= 100:
            grouped[r["model"], r["arm"], r["market"], int(r["seed"])].append(r)
    if max_error > 1e-9 or any(len(group) != 15 for group in grouped.values()):
        raise ValueError("Welfare accounting or late-window completeness check failed")

    levels = {key: {metric: math.fsum(r[metric] for r in group) / 15 for metric in metrics}
              for key, group in grouped.items()}
    critical = float(t.ppf(.975, len(SEEDS) - 1))

    def interval(values):
        mean = math.fsum(values) / len(SEEDS)
        half = critical * float(np.std(values, ddof=1)) / math.sqrt(len(SEEDS))
        return {"mean": mean, "ci_low": mean - half, "ci_high": mean + half}

    means, by_seed, contrasts, paired = [], [], [], []
    for model in MODELS:
        for market in MARKETS:
            for arm in ARMS:
                identity = {"model": model, "arm": arm, "market": market}
                values = [levels[model, arm, market, seed] for seed in SEEDS]
                means.append({**identity, "round_start": 86, "round_end": 100,
                              "n_seeds": len(SEEDS), **interval([v["total_net"] for v in values]),
                              **{metric: math.fsum(v[metric] for v in values) / len(SEEDS)
                                 for metric in metrics if metric != "total_net"}})
                by_seed.extend({**identity, "seed": seed, **value} for seed, value in zip(SEEDS, values))
            identity = {"model": model, "market": market, "contrast": "social - individual"}
            differences = [{metric: levels[model, "social", market, seed][metric]
                            - levels[model, "individual", market, seed][metric] for metric in metrics}
                           for seed in SEEDS]
            d = [v["total_net"] for v in differences]
            contrasts.append({**identity, "round_start": 86, "round_end": 100,
                              "n_pairs": len(SEEDS), **interval(d),
                              "positive_seeds": sum(v > 0 for v in d),
                              "negative_seeds": sum(v < 0 for v in d),
                              **{metric: math.fsum(v[metric] for v in differences) / len(SEEDS)
                                 for metric in metrics if metric != "total_net"}})
            paired.extend({**identity, "seed": seed, **value}
                          for seed, value in zip(SEEDS, differences))
    prepare_output(output, data)
    for name, table in (("welfare_means", means), ("welfare_by_seed", by_seed),
                        ("welfare_contrasts", contrasts), ("welfare_paired_seeds", paired)):
        write_csv(output / f"{name}.csv", table)
    return {"source": str(path), "sha256": digest, "economic_rows": len(rows),
            "late_cells": len(by_seed), "means": len(means), "paired_contrasts": len(contrasts),
            "paired_seed_rows": len(paired), "window": [86, 100], "n_seeds": len(SEEDS),
            "quantity": "total output minus all applicant and employer costs per round",
            "units": "simulation payoff units, not USD",
            "error_bars": "pointwise 95% t intervals, df=4; paired for social - individual",
            "hypothesis_tests": "none", "max_accounting_error": max_error}
