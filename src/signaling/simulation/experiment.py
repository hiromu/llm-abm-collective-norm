"""LLM applicant decisions with synchronized rounds and checkpoints."""

from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import closing
from dataclasses import asdict, replace
from datetime import datetime, timezone
import fcntl
import hashlib
import itertools
import json
import math
from pathlib import Path
import sqlite3

from ..config import ARMS, LABELS, MARKETS
from ..data import write_csv
from ..prompts import SYSTEM_PROMPT
from ..providers import APIError, Client, MODELS, ModelConfig, parse_decision, request_body
from .economy import Market, Parameters


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def now():
    return datetime.now(timezone.utc).isoformat()


def plan(model, seeds, arms=ARMS, markets=MARKETS, rounds=100, request_model=None,
         effort=None, max_tokens=None):
    if model not in MODELS or type(rounds) is not int or rounds < 4:
        raise ValueError("Choose a configured model and at least four rounds")
    seeds = tuple(seeds)
    if not seeds or any(type(s) is not int for s in seeds) or not arms or not markets:
        raise ValueError("Explicit seeds, information conditions and markets are required")
    seeds, arms, markets = sorted(set(seeds)), sorted(set(arms)), sorted(set(markets))
    if set(arms) - set(ARMS) or set(markets) - set(MARKETS):
        raise ValueError("Unknown information condition or market")
    config = MODELS[model]
    config = replace(config, model=request_model or config.model,
                     effort=effort if effort is not None else config.effort,
                     max_tokens=max_tokens if max_tokens is not None else config.max_tokens)
    if not isinstance(config.model, str) or not config.model.strip():
        raise ValueError("A request model identifier is required")
    if config.effort not in {"off", "minimal", "low", "medium", "high", "xhigh", "max"}:
        raise ValueError("Invalid reasoning effort")
    if type(config.max_tokens) is not int or config.max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    package = Path(__file__).resolve().parents[1]
    sources = ("simulation/experiment.py", "simulation/economy.py", "providers.py", "prompts.py", "config.py")
    return {"schema": 1, "model": model,
            "label": LABELS[model] if config.model == MODELS[model].model else config.model,
            "request": asdict(config), "system": SYSTEM_PROMPT,
            "seeds": seeds, "arms": arms, "markets": markets, "rounds": rounds,
            "parameters": {m: asdict(Parameters.for_market(m)) for m in markets},
            "cells": len(seeds) * len(arms) * len(markets),
            "judgments": len(seeds) * len(arms) * len(markets) * 200 * len(range(4, rounds + 1, 3)),
            "source_sha256": {name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in sources}}


def snapshot(world):
    return encoded({"round": world.round, "pending": world.pending, "rng": world.rng.getstate(),
                    "agents": [asdict(a) for a in world.agents]})


def restore(world, state):
    saved = json.loads(state)
    if len(saved["agents"]) != len(world.agents) or not 0 <= saved["round"] <= world.rounds:
        raise ValueError("Invalid saved market state")
    for agent, row in zip(world.agents, saved["agents"]):
        if agent.index != row["index"] or agent.quality != row["quality"] or set(row) != set(asdict(agent)):
            raise ValueError("Saved agent identity differs from the plan")
        agent.__dict__.update(row)
    version, values, gaussian = saved["rng"]
    world.rng.setstate((version, tuple(values), gaussian))
    world.round, world.pending = saved["round"], saved["pending"]


def open_ledger(path):
    db = sqlite3.connect(path)
    db.execute("PRAGMA synchronous=FULL")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS cells (id TEXT PRIMARY KEY, state TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS rounds (
            cell TEXT, round INTEGER, record TEXT NOT NULL, PRIMARY KEY(cell, round));
        CREATE TABLE IF NOT EXISTS decisions (
            cell TEXT, round INTEGER, agent INTEGER, record TEXT NOT NULL,
            PRIMARY KEY(cell, round, agent));
        CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY, cell TEXT, round INTEGER, agent INTEGER, request TEXT NOT NULL,
            status TEXT NOT NULL, raw TEXT, error TEXT, usage TEXT, started TEXT NOT NULL, finished TEXT);
        CREATE TABLE IF NOT EXISTS sessions (
            started TEXT, max_requests INTEGER, concurrency INTEGER, retry_uncertain INTEGER,
            accepted_before INTEGER NOT NULL);
    """)
    return db


def record_response(db, attempt, future, config, identity):
    raw, error, decision = "", None, None
    try:
        raw = future.result()
        decision = parse_decision(raw, config.provider)
        status = "accepted"
    except APIError as exc:
        raw, error = exc.raw, str(exc)
        status = "unknown" if exc.uncertain else "error"
    except (ValueError, KeyError, TypeError) as exc:
        status, error = "invalid", type(exc).__name__ + ": invalid final response"
    except Exception as exc:
        status, error = "unknown", type(exc).__name__ + ": request outcome unknown"
    try:
        usage = json.loads(raw).get("usage")
        if not isinstance(usage, dict):
            usage = None
        encoded(usage)
    except (ValueError, AttributeError):
        usage = None
    with db:
        db.execute("UPDATE attempts SET status=?,raw=?,error=?,usage=?,finished=? WHERE id=?",
                   (status, raw, error, encoded(usage), now(), attempt))
        if decision is not None:
            cell, round_, index, metadata = identity
            db.execute("INSERT INTO decisions VALUES (?,?,?,?)",
                       (cell, round_, index, encoded({**metadata, **decision})))
    return error


def decide(db, cell, world, metadata, settings, client, concurrency, max_requests,
           retry_uncertain, progress):
    round_ = world.round + 1
    saved = {i: json.loads(record) for i, record in db.execute(
        "SELECT agent,record FROM decisions WHERE cell=? AND round=?", (cell, round_))}
    uncertain = {i for (i,) in db.execute(
        "SELECT agent FROM attempts WHERE cell=? AND round=? AND status IN ('inflight','unknown')",
        (cell, round_))} - set(saved)
    if uncertain and not retry_uncertain:
        raise ValueError("Unresolved requests may have been charged; inspect the ledger before --retry-uncertain")
    missing = iter(i for i in range(len(world.applicants)) if i not in saved)
    config = ModelConfig(**settings["request"])
    sent = db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
    failure = None
    active = {}

    def submit(pool):
        nonlocal sent, failure
        while len(active) < concurrency and failure is None:
            index = next(missing, None)
            if index is None:
                break
            if sent >= max_requests:
                failure = "Request limit reached; saved answers are retained"
                break
            body = request_body(config, settings["system"], world.pending[index])
            with db:
                row = db.execute("INSERT INTO attempts (cell,round,agent,request,status,started) VALUES (?,?,?,?,?,?)",
                                 (cell, round_, index, encoded(body), "inflight", now()))
            attempt = row.lastrowid
            sent += 1
            request_id = hashlib.sha256(f"{cell}:{round_}:{index}:{attempt}".encode()).hexdigest()
            active[pool.submit(client.send, body, request_id)] = (
                attempt, (cell, round_, index, {**metadata, "round": round_, "index": index,
                                              "quality": world.applicants[index].quality}))

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        try:
            submit(pool)
            completed = 0
            while active:
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    attempt, identity = active.pop(future)
                    error = record_response(db, attempt, future, config, identity)
                    failure = failure or error
                    completed += 1
                if completed % 25 == 0 or not active:
                    progress(f"{cell}, round {round_}: {completed + len(saved)}/{len(world.applicants)} responses saved")
                submit(pool)
        except KeyboardInterrupt:
            # Preserve responses from already-sent requests before releasing the run lock.
            for future, (attempt, identity) in active.items():
                record_response(db, attempt, future, config, identity)
            raise
    if failure:
        raise ValueError(failure)
    answers = {i: json.loads(record)["technology_use"] for i, record in db.execute(
        "SELECT agent,record FROM decisions WHERE cell=? AND round=?", (cell, round_))}
    world.accept([answers[i] for i in range(len(world.applicants))])


def status(db, settings, accepted_before=None, failure=None):
    if accepted_before is None:
        session = db.execute("SELECT accepted_before FROM sessions ORDER BY rowid DESC LIMIT 1").fetchone()
        accepted_before = session[0] if session else 0
    attempts = dict(db.execute("SELECT status,COUNT(*) FROM attempts GROUP BY status"))
    accepted = db.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
    states = [json.loads(state) for (state,) in db.execute("SELECT state FROM cells")]
    completed = sum(state["round"] == settings["rounds"] for state in states)
    usage = [json.loads(u) for (u,) in db.execute("SELECT usage FROM attempts") if u]
    costs = [u["cost"] for u in usage if isinstance(u, dict) and type(u.get("cost")) in (int, float)
             and math.isfinite(u["cost"]) and u["cost"] >= 0]
    total = sum(attempts.values())
    return {"phase": ("report_pending" if failure else "complete") if completed == settings["cells"] else "stopped",
            "cells_completed": completed, "cells_planned": settings["cells"],
            "judgments": accepted, "judgments_planned": settings["judgments"],
            "new_judgments": accepted - accepted_before, "reused_judgments": accepted_before,
            "assigned_actions": 200 * len(states),
            "assigned_person_rounds": 200 * sum(min(3, state["round"]) for state in states),
            "economic_rounds": db.execute("SELECT COUNT(*) FROM rounds").fetchone()[0],
            "physical_requests": total, "attempt_status": attempts,
            "provider_reported_cost_usd": math.fsum(costs), "cost_unavailable_attempts": total - len(costs),
            "invoice_reconciled": False, "failure": failure}


def export(db, output, settings):
    rows = [json.loads(r) for (r,) in db.execute("SELECT record FROM rounds ORDER BY cell,round")]
    write_csv(output / "economic_paths.csv", rows)
    with (output / "decisions.jsonl").open("w") as f:
        for (record,) in db.execute("SELECT record FROM decisions ORDER BY cell,round,agent"):
            f.write(record + "\n")
    groups = defaultdict(dict)
    for row in rows:
        for role in ("HQ", "LQ", "Employer"):
            groups[row["arm"], row["market"], role, row["round"]][row["seed"]] = row[role]
    trajectories = []
    for (arm, market, role, round_), values in sorted(groups.items()):
        if set(values) != set(settings["seeds"]):
            raise ValueError("Incomplete seed cohort")
        trajectories.append({"model": settings["model"], "arm": arm, "market": market,
                             "role": role, "round": round_, "mean": math.fsum(values.values()) / len(values),
                             "seed_min": min(values.values()), "seed_max": max(values.values()),
                             **{f"seed_{s}": values[s] for s in settings["seeds"]}})
    write_csv(output / "trajectories.csv", trajectories)


def run(settings, output, max_requests, concurrency=16, env_file=Path(".env"),
        timeout=180, resume=False, retry_uncertain=False, client=None, progress=print):
    """Run one model; resume only the same scientific plan and source versions."""
    if type(max_requests) is not int or max_requests < 1:
        raise ValueError("An explicit positive request limit is required")
    if type(concurrency) is not int or not 1 <= concurrency <= 256:
        raise ValueError("concurrency must be between 1 and 256")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive")
    if resume and not (output / "plan.json").is_file():
        raise ValueError("No saved plan to resume")
    if not resume and output.exists() and any(output.iterdir()):
        raise ValueError("Output already exists; use --resume for the same experiment")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("This experiment is already running") from None
        plan_path = output / "plan.json"
        if plan_path.exists():
            if not resume:
                raise ValueError("Output already exists; use --resume for the same experiment")
            if json.loads(plan_path.read_text()) != settings:
                raise ValueError("Saved plan/source versions differ; do not mix experiments")
        else:
            plan_path.write_text(encoded(settings) + "\n")
        with closing(open_ledger(output / "ledger.sqlite")) as db:
            accepted_before = db.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
            if status(db, settings)["phase"] == "complete":
                if not all((output / p).is_file() for p in ("economic_paths.csv", "decisions.jsonl", "trajectories.csv")):
                    export(db, output, settings)
                receipt = status(db, settings)
                (output / "status.json").write_text(encoded(receipt) + "\n")
                return status(db, settings, accepted_before)
            with db:
                db.execute("INSERT INTO sessions VALUES (?,?,?,?,?)",
                           (now(), max_requests, concurrency, retry_uncertain, accepted_before))
            failure = None
            try:
                client = client or Client(ModelConfig(**settings["request"]), env_file, timeout)
                for arm, market, seed in itertools.product(settings["arms"], settings["markets"], settings["seeds"]):
                    cell = f"{arm}|{market}|{seed}"
                    world = Market(Parameters(**settings["parameters"][market]), seed, arm, rounds=settings["rounds"])
                    row = db.execute("SELECT state FROM cells WHERE id=?", (cell,)).fetchone()
                    if row:
                        restore(world, row[0])
                    else:
                        with db:
                            db.execute("INSERT INTO cells VALUES (?,?)", (cell, snapshot(world)))
                    metadata = {"model": settings["model"], "arm": arm, "market": market, "seed": seed}
                    while world.round < settings["rounds"]:
                        if world.pending is not None:
                            decide(db, cell, world, metadata, settings, client, concurrency,
                                   max_requests, retry_uncertain, progress)
                        result = world.step()
                        with db:
                            db.execute("INSERT INTO rounds VALUES (?,?,?)", (cell, world.round, encoded({**metadata, **result})))
                            db.execute("UPDATE cells SET state=? WHERE id=?", (snapshot(world), cell))
                    progress(f"Complete: {cell}, {settings['rounds']} economic rounds")
                export(db, output, settings)
            except (Exception, KeyboardInterrupt) as error:
                failure = str(error) if isinstance(error, ValueError) else type(error).__name__
                raise
            finally:
                (output / "status.json").write_text(encoded(status(db, settings, accepted_before, failure)) + "\n")
            return status(db, settings, accepted_before)


def inspect(output):
    """Read progress from local experiment records."""
    settings = json.loads((output / "plan.json").read_text())
    saved_status = output / "status.json"
    failure = json.loads(saved_status.read_text()).get("failure") if saved_status.is_file() else None
    with closing(sqlite3.connect((output / "ledger.sqlite").resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("BEGIN")
        receipt = status(db, settings, failure=failure)
    with (output / "run.lock").open() as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            receipt["phase"] = "running"
            receipt["failure"] = None
    return receipt
