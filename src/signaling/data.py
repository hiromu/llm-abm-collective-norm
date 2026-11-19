"""Input validation for job-market analyses."""

import csv
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

from .config import ARMS, DECISION_ROUNDS, MARKETS, MODELS, QUALITIES, SEEDS


def prepare_output(output: Path, *inputs: Path):
    for data in inputs:
        if output.resolve().is_relative_to(data.resolve()) or data.resolve().is_relative_to(output.resolve()):
            raise ValueError("Output must be separate from its inputs")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError("Cannot write an empty result table")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_group(data: Path, model: str, arm: str, round_: int) -> list[dict]:
    if model not in MODELS or arm not in ARMS or round_ not in DECISION_ROUNDS:
        raise ValueError("Unknown corpus group")
    path = data / "corpus" / f"{model}__{arm}__r{round_:03d}.jsonl.gz"
    with gzip.open(path, "rt") as f:
        rows = [json.loads(line) for line in f]
    expected = set(itertools.product(MARKETS, SEEDS, range(200)))
    keys = [(r["market"], r["seed"], r["index"]) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError(f"Incomplete or duplicate corpus records in {path.name}")
    for r in rows:
        if (r["model"] != model or r["arm"] != arm or r["round"] != round_
                or r["quality"] not in QUALITIES or not isinstance(r["strategy"], str)
                or not r["strategy"].strip() or type(r["action"]) not in (int, float)
                or not math.isfinite(r["action"]) or not 0 <= r["action"] <= 1):
            raise ValueError(f"Invalid corpus record in {path.name}")
    return sorted(rows, key=lambda r: (r["market"], r["seed"], r["index"]))


def trajectories(data: Path) -> list[dict]:
    rows = read_csv(data / "trajectories.csv")
    expected = set(itertools.product(MODELS, MARKETS, ARMS, (*QUALITIES, "Employer"), range(1, 101)))
    keys = [(r["model"], r["market"], r["arm"], r["role"], int(r["round"])) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Trajectory table must contain every complete 100-round cell exactly once")
    for r in rows:
        r["round"] = int(r["round"])
        for k in ("mean", "seed_min", "seed_max", *(f"seed_{s}" for s in SEEDS)):
            r[k] = float(r[k])
            if not math.isfinite(r[k]) or not 0 <= r[k] <= 1 + 1e-14:
                raise ValueError(f"Invalid trajectory value: {k}")
        values = [r[f"seed_{s}"] for s in SEEDS]
        for k, actual in [("mean", math.fsum(values) / 5), ("seed_min", min(values)),
                          ("seed_max", max(values))]:
            if abs(r[k] - actual) > 1e-12:
                raise ValueError(f"Inconsistent trajectory aggregation: {k}")
    return rows


def verify(data: Path, scope="all") -> dict:
    """Check only the inputs required by the requested workflow."""
    if scope not in ("all", "visualization", "analysis", "statistics"):
        raise ValueError(f"Unknown verification scope: {scope}")
    manifest = json.loads((data / "manifest.json").read_text())
    required = {"settings.json"}
    if scope == "analysis":
        required.update(f"corpus/{m}__{a}__r{r:03d}.jsonl.gz"
                        for m in MODELS for a in ARMS for r in DECISION_ROUNDS)
    if scope in ("visualization", "statistics"):
        required.add("trajectories.csv")
    if scope == "visualization":
        required.update({"artwork/job_market_cycle_abm.png", "semantics/market_decoding.csv",
                         "semantics/quality_decoding.csv", "semantics/centroid_projections.csv"})
        required.update(f"semantics/{model}.npz" for model in MODELS)
    if scope == "analysis":
        required.add("encoder_files.json")
    if scope == "all":
        required = set(manifest["files"])
    for name in sorted(required):
        expected = manifest["files"][name]
        path = data / name
        if not path.resolve().is_relative_to(data.resolve()):
            raise ValueError("Manifest path escapes the data directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Input checksum mismatch: {name}")
    result = {"scope": scope, "input_files": len(required)}
    if scope in ("all", "visualization", "statistics"):
        result["trajectory_rows"] = len(trajectories(data))
    if scope == "all":
        result["economic_rounds"] = verify_economic_reference(data)
    if scope in ("all", "visualization"):
        result.update(verify_semantics(data / "semantics"))
    return result


def verify_economic_reference(data: Path) -> int:
    economic = read_csv(data / "economic_reference.csv")
    economic_keys = [(r["model"], r["arm"], r["market"], int(r["seed"]), int(r["round"]))
                     for r in economic]
    expected_economic = set(itertools.product(MODELS, ARMS, MARKETS, SEEDS, range(1, 101)))
    if len(economic_keys) != len(expected_economic) or set(economic_keys) != expected_economic:
        raise ValueError("Incomplete economic reference cohort")
    return len(economic)


def verify_semantics(semantics: Path) -> dict:
    folds = {}
    for task in ("market", "quality"):
        values = read_csv(semantics / f"{task}_decoding.csv")
        expected = set(itertools.product(MODELS, ARMS, DECISION_ROUNDS, SEEDS,
                                         MARKETS if task == "quality" else ("",)))
        keys = [(r["model"], r["arm"], int(r["round"]), int(r["held_out_seed"]),
                 r.get("market", "")) for r in values]
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError(f"Incomplete {task} cross-validation folds")
        for r in values:
            if not 0 <= float(r["accuracy"]) <= 1 or int(r["n_test"]) != (400 if task == "market" else 200):
                raise ValueError("Invalid decoding fold")
        folds[task] = len(values)
    projections = read_csv(semantics / "centroid_projections.csv")
    expected = set(itertools.product(MODELS, ARMS, DECISION_ROUNDS, MARKETS, SEEDS, QUALITIES))
    keys = [(r["model"], r["arm"], int(r["round"]), r["market"], int(r["seed"]), r["quality"])
            for r in projections]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Incomplete centroid projections")
    error = 0.0
    for model in MODELS:
        with np.load(semantics / f"{model}.npz") as z:
            # Match sklearn PCA.transform: projection, then projected-mean subtraction.
            xy = z["centroids"] @ z["components"].T - z["mean"][None, :] @ z["components"].T
            for r in (r for r in projections if r["model"] == model):
                error = max(error, float(np.max(np.abs(xy[int(r["vector_row"]) ] -
                                      [float(r["pc1"]), float(r["pc2"])]))))
    if error > 1e-7:
        raise ValueError(f"PCA projection mismatch: {error}")
    return {"decoding_folds": folds, "centroid_rows": len(projections),
            "max_projection_error": error}
