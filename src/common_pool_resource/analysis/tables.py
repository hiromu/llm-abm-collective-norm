"""Society-level summaries of common-pool resource experiments."""

import ast
import hashlib
import itertools
import json
import math
import re

import numpy as np
import pandas as pd

from ..config import ABLATIONS, MODELS
from ..data import dumps, empty_output

GROUPS = ["model", "growth_rate", "altruism_ratio", "condition"]
SOCIETIES = {0.: "Selfish", .5: "Mixed", 1.: "Altruistic"}


def efficiency_benchmark(growth_rate, capacity, resource):
    """Harvest-efficiency denominator evaluated at the initial stock."""
    h, r, k = min(float(resource), float(capacity)), float(growth_rate) + 1, float(capacity)
    if r <= 0 or k <= 0 or h <= 0:
        return 0.
    root = ((1 + r) - math.sqrt(max(0., (1 + r)**2 - 4 * r * h / k))) * k / (2 * r)
    return min(h, max(0., h - root))


def condition(value):
    names = ast.literal_eval(value) if value else []
    if not isinstance(names, list) or any(not isinstance(n, str) for n in names):
        raise ValueError("Invalid ablation list")
    for name, removed in ABLATIONS.items():
        if set(names) == set(removed):
            return name
    raise ValueError("Unknown ablation")


def load(data, ablation=False, trials=10):
    """Load and validate society and agent records from CSV exports."""
    manifest = {}
    if (data / "plan.json").is_file():
        settings = json.loads((data / "plan.json").read_text())
        if json.loads((data / "status.json").read_text())["phase"] != "complete":
            raise ValueError("Analysis requires a complete experiment")
        models = pd.read_csv(data / "rounds.csv", keep_default_na=False)
        agents = pd.read_csv(data / "agents.csv", keep_default_na=False)
        keys = [*GROUPS, "seed", "Step"]
        shared = [k for k in agents if k in models and k not in keys]
        frame = agents.drop(columns=shared).merge(models, on=keys, validate="many_to_one")
        frame = frame.rename(columns={"index": "AgentID"})
        frame["file_id"] = frame["seed"].astype(str)
        frame["RunId"] = frame["seed"]
        expected_cells = {(settings["model"], c["parameters"]["growth_rate"], c["parameters"]["altruism_ratio"], c["condition"])
                          for c in settings["cells"]}
        horizon = settings["rounds"] - 1
        files = [data / name for name in ("plan.json", "status.json", "rounds.csv", "agents.csv")]
        convention = "aligned model and agent records; Step = physical round - 1"
    else:
        source = data / "ablation" if ablation else data
        files = sorted(source.glob("*_2025*.csv"))
        if not files:
            raise ValueError("No published fishing CSVs found")
        parts = []
        for path in files:
            match = re.fullmatch(r"(.+?)_(\d{8}_\d{6})(?:_.*)?", path.stem)
            if not match or match[1] not in MODELS:
                raise ValueError(f"Unexpected result filename: {path.name}")
            part = pd.read_csv(path, keep_default_na=False)
            part["model"], part["file_id"] = match[1], match[2]
            part["condition"] = part["ablation"].map(condition) if "ablation" in part else "all"
            parts.append(part)
        frame = pd.concat(parts, ignore_index=True)
        models = ("deepseek-r1", "gpt-4o", "qwen3-32b") if ablation else tuple(MODELS)
        conditions = tuple(ABLATIONS) if ablation else ("all",)
        expected_cells = set(itertools.product(models, (.2, .6), (0., .5, 1.), conditions))
        horizon = 50
        convention = "published Mesa Step indexing, model/agent offset and model-only Step 0 retained"
    required = {"Step", "AgentID", "RunId", "n_agents", "resource", "total_demand", "harvest", "effort", "payoff",
                "capacity", "init_resource", "growth_rate", "altruism_ratio"}
    if required - set(frame):
        raise ValueError(f"Missing fields: {sorted(required - set(frame))}")
    for name in required - {"RunId"}:
        raw = frame[name]
        converted = pd.to_numeric(raw.replace("", np.nan), errors="coerce")
        if ((raw != "") & converted.isna()).any() or np.isinf(converted).any():
            raise ValueError(f"Invalid numeric field: {name}")
        frame[name] = converted
    if frame[["Step", *GROUPS[1:3], "resource", "n_agents"]].isna().any().any():
        raise ValueError("Missing society-level values")
    if not (frame["Step"] == frame["Step"].astype(int)).all() or not frame["Step"].between(0, horizon).all():
        raise ValueError("Invalid step index")
    frame["Step"] = frame["Step"].astype(int)
    observed = set(frame[GROUPS].itertuples(index=False, name=None))
    if observed != expected_cells:
        raise ValueError("Incomplete or unexpected condition cohort")
    run_keys = [*GROUPS, "file_id", "RunId"]
    if frame.duplicated([*run_keys, "Step", "AgentID"]).any():
        raise ValueError("Duplicate society/step/agent records")
    for _, group in frame.groupby(run_keys, sort=False):
        if set(group.Step) != set(range(int(group.Step.max()) + 1)):
            raise ValueError("Missing steps within a society")
        for step, block in group.groupby("Step", sort=False):
            n = int(block.n_agents.iloc[0])
            if any(block[k].nunique(dropna=False) != 1 for k in ("resource", "total_demand", "n_agents")):
                raise ValueError("Inconsistent model records within a step")
            ids = set(block.AgentID.dropna())
            if ids != set(range(n)) and not (step == 0 and len(block) == 1 and not ids):
                raise ValueError("Incomplete agent cohort")
    counts = frame[run_keys].drop_duplicates().groupby(GROUPS).size()
    if trials is not None and not counts.eq(trials).all():
        raise ValueError(f"Each condition must contain {trials} independent societies; observed {sorted(counts.unique())}")
    for path in files:
        manifest[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    return frame, {"input_sha256": manifest, "convention": convention, "horizon_index": horizon,
                   "cells": len(expected_cells), "societies": int(counts.sum()), "rows": len(frame)}


def summaries(frame, horizon):
    run_keys = [*GROUPS, "file_id", "RunId"]
    by_step = frame.groupby([*run_keys, "Step"], sort=True)
    paths = by_step.agg(resource=("resource", "first"), total_demand=("total_demand", "first"),
                        harvest=("harvest", "sum"), effort=("effort", "mean"), payoff=("payoff", "mean"),
                        capacity=("capacity", "first"), init_resource=("init_resource", "first")).reset_index()
    paths["efficiency"] = paths.apply(lambda r: r.harvest / efficiency_benchmark(r.growth_rate, r.capacity, r.init_resource)
                                      if efficiency_benchmark(r.growth_rate, r.capacity, r.init_resource) else np.nan, axis=1)
    runs = paths.groupby(run_keys).agg(survival=("Step", "max"), steps_recorded=("Step", "size"),
                                      harvest_total=("harvest", "sum"), mean_efficiency=("efficiency", "mean")).reset_index()
    harvest = frame.groupby([*run_keys, "AgentID"]).harvest.sum().groupby(run_keys).mean().rename("harvest_per_agent")
    runs = runs.merge(harvest.reset_index(), on=run_keys, how="left", validate="one_to_one")
    runs["at_horizon"] = runs.survival.eq(horizon)
    runs["society"] = runs.altruism_ratio.map(lambda r: SOCIETIES.get(r, f"Altruistic fraction {r:g}"))
    long = runs.melt(id_vars=GROUPS + ["file_id", "RunId"], value_vars=["survival", "harvest_per_agent", "mean_efficiency"],
                     var_name="metric", value_name="value")
    means = long.groupby(GROUPS + ["metric"]).value.agg(["mean", "sem", "count"]).reset_index()
    paths_long = paths.melt(id_vars=run_keys + ["Step"], value_vars=["resource", "effort", "harvest", "payoff", "efficiency"],
                           var_name="metric", value_name="value")
    trajectories = paths_long.groupby(GROUPS + ["Step", "metric"]).value.agg(["mean", "sem", "count"]).reset_index()
    # Export agent-level standard errors separately from society-level errors.
    published = frame.groupby(GROUPS + ["Step"]).effort.agg(["mean", "sem", "count"]).reset_index()
    return runs, means, trajectories, paths, published


def statistics(runs, output):
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    tables, tukeys = [], []
    for rate, data in runs.groupby("growth_rate"):
        if data.model.nunique() < 2 or data.altruism_ratio.nunique() < 2:
            continue
        for metric in ("survival", "harvest_per_agent"):
            formula = f"{metric} ~ C(model) * C(society)"
            if data.condition.nunique() > 1:
                formula += " * C(condition)"
            fit = ols(formula, data=data.dropna(subset=[metric])).fit()
            if fit.df_resid <= 0:
                raise ValueError("Insufficient independent societies for ANOVA")
            table = sm.stats.anova_lm(fit, typ=2)
            table["eta_sq"] = table.sum_sq / table.sum_sq.sum()
            table["growth_rate"], table["metric"] = rate, metric
            tables.append(table.rename_axis("term").reset_index())
            result = pairwise_tukeyhsd(data[metric].dropna(), data.loc[data[metric].notna(), "model"])
            rows = pd.DataFrame(result._results_table.data[1:], columns=result._results_table.data[0])
            rows["growth_rate"], rows["metric"] = rate, metric
            tukeys.append(rows)
    if tables:
        pd.concat(tables).to_csv(output / "anova.csv", index=False)
        pd.concat(tukeys).to_csv(output / "tukey.csv", index=False)


def run(data, output, ablation=False, trials=10, tests=False):
    empty_output(output, data)
    frame, receipt = load(data, ablation, trials)
    runs, means, trajectories, paths, published = summaries(frame, receipt["horizon_index"])
    for name, table in (("societies", runs), ("summary", means), ("trajectories", trajectories),
                        ("society_paths", paths), ("effort_published", published)):
        table.to_csv(output / f"{name}.csv", index=False)
    if tests:
        statistics(runs, output)
    if "group_norm" in frame:
        frame[[*GROUPS, "file_id", "RunId", "Step", "group_norm"]].drop_duplicates().to_csv(output / "group_norms.csv", index=False)
    receipt.update(error_bars="standard errors across societies; effort_published retains agent-row SE",
                   efficiency="published denominator uses r = growth_rate + 1, evaluated at init_resource",
                   missing_after_collapse="not filled; trajectory counts are exported", ablation=ablation)
    (output / "analysis.json").write_text(dumps(receipt) + "\n")
    return receipt
