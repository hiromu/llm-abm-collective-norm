"""Run complete societies with per-response persistence and stage checkpoints."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from dataclasses import asdict, replace
from datetime import datetime, timezone
import fcntl
import hashlib
import itertools
import json
from pathlib import Path
import sqlite3

from ..config import CONDITIONS, MODELS, Parameters
from ..data import dumps, empty_output, write_csv
from ..providers import Client, RequestError, configuration, content, request_body
from . import abm, llm


def plan(policy, seeds, growth_rates=(.2, .6), ratios=(0., .5, 1.), conditions=("all",),
         rounds=None, model="gpt-4o", overrides=None, request_model=None, max_tokens=None):
    if policy not in ("abm", "llm") or model not in MODELS:
        raise ValueError("Unknown policy or model")
    if overrides is not None and not isinstance(overrides, dict):
        raise ValueError("Parameter overrides must be a JSON object")
    if request_model is not None and (not isinstance(request_model, str) or not request_model.strip()):
        raise ValueError("Request model identifier must be a nonempty string")
    seeds = list(seeds)
    if not seeds or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
        raise ValueError("Supply explicit 32-bit society seeds")
    if set(conditions) - set(CONDITIONS) or not conditions or not growth_rates or not ratios:
        raise ValueError("Invalid or empty conditions")
    if max_tokens is not None and (type(max_tokens) is not int or max_tokens < 1):
        raise ValueError("Output token limit must be positive")
    rounds = rounds if rounds is not None else (101 if policy == "abm" else 51)
    if type(rounds) is not int or rounds < 1:
        raise ValueError("Round count must be positive")
    base = Parameters(**(overrides or {}))
    if policy == "abm":
        if tuple(conditions) != ("all",):
            raise ValueError("Language-information ablations apply only to LLM societies")
        ratios = (base.altruism_ratio,)
    cells = []
    for growth, ratio, condition, seed in itertools.product(sorted(set(growth_rates)), sorted(set(ratios)),
                                                          sorted(set(conditions)), sorted(set(seeds))):
        parameters = replace(base, growth_rate=growth, altruism_ratio=ratio)
        cells.append({"id": f"{growth:g}|{ratio:g}|{condition}|{seed}", "seed": seed,
                      "condition": condition, "parameters": asdict(parameters)})
    root = Path(__file__).resolve().parents[1]
    return {"schema": 1, "policy": policy, "model": model if policy == "llm" else "abm",
            "request": configuration(model, request_model, max_tokens) if policy == "llm" else None,
            "rounds": rounds, "cells": cells,
            "request_upper_bound": sum(c["parameters"]["n_agents"] * rounds *
                                       (3 if c["condition"] in ("without_group_norm", "neither") else 4)
                                       for c in cells) if policy == "llm" else 0,
            "sources": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(root.rglob("*.py"))}}


def metadata(settings, cell):
    return {"model": settings["model"], "seed": cell["seed"], "condition": cell["condition"],
            **cell["parameters"]}


def run_abm(settings, output):
    empty_output(output)
    (output / "plan.json").write_text(dumps(settings) + "\n")
    models, agents = [], []
    for cell in settings["cells"]:
        society = abm.Society(Parameters(**cell["parameters"]), cell["seed"], settings["rounds"])
        while society.running:
            row, people = society.step()
            meta = metadata(settings, cell)
            models.append({**meta, **row})
            agents.extend({**meta, **r} for r in people)
    write_csv(output / "rounds.csv", models)
    write_csv(output / "agents.csv", agents)
    return {"phase": "complete", "cells": len(settings["cells"]), "economic_rounds": len(models)}


def connect(path):
    db = sqlite3.connect(path)
    db.execute("PRAGMA synchronous=FULL")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS cells (id TEXT PRIMARY KEY, state TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS rounds (cell TEXT, step INTEGER, record TEXT, agents TEXT, PRIMARY KEY(cell,step));
        CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY, key TEXT, request TEXT, status TEXT, raw TEXT, content TEXT,
            usage TEXT, error TEXT, finished TEXT);
        CREATE INDEX IF NOT EXISTS attempts_key ON attempts(key);
    """)
    return db


def query(db, settings, cell, requests, client, max_requests, concurrency, retry_uncertain):
    keys = [cell + ":" + r["key"] for r in requests]
    bodies = {key: request_body(settings["request"], r["messages"]) for key, r in zip(keys, requests)}
    saved, uncertain = {}, set()
    for key in keys:
        prior = db.execute("SELECT request,status,content FROM attempts WHERE key=? ORDER BY id", (key,)).fetchall()
        for body, status, text in prior:
            if json.loads(body) != bodies[key]:
                raise ValueError("Saved request differs from the restored state")
            if status == "accepted":
                saved[key] = text
            elif status in ("inflight", "unknown"):
                uncertain.add(key)
    if uncertain - saved.keys() and not retry_uncertain:
        raise ValueError("Unresolved requests may have been billed; inspect before --retry-uncertain")
    missing = [key for key in keys if key not in saved]
    count = db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
    allowance = max_requests - count
    if missing and allowance <= 0:
        raise ValueError("Cumulative request limit reached")
    failure = None
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        active = {}
        for key in missing[:max(0, allowance)]:
            with db:
                attempt = db.execute("INSERT INTO attempts (key,request,status) VALUES (?,?,?)",
                                     (key, dumps(bodies[key]), "inflight")).lastrowid
            active[pool.submit(client.send, bodies[key])] = attempt

        def save(future):
            nonlocal failure
            raw, text, error, usage = "", None, None, None
            try:
                raw = future.result()
                text = content(raw, settings["request"]["provider"])
                status = "accepted"
            except RequestError as e:
                raw, error, status = e.raw, str(e), "unknown" if e.uncertain else "error"
            except (ValueError, TypeError, KeyError, AttributeError):
                status, error = "invalid", "Invalid or incomplete provider response"
            except Exception as e:
                status, error = "unknown", type(e).__name__
            try:
                usage = json.loads(raw).get("usage")
                dumps(usage)
            except (ValueError, TypeError, AttributeError):
                usage = None
            with db:
                db.execute("UPDATE attempts SET status=?,raw=?,content=?,usage=?,error=?,finished=? WHERE id=?",
                           (status, raw, text, dumps(usage), error, datetime.now(timezone.utc).isoformat(), active[future]))
            failure = failure or error

        try:
            for future in as_completed(active):
                save(future)
        except KeyboardInterrupt:
            for future, attempt in active.items():
                if db.execute("SELECT status FROM attempts WHERE id=?", (attempt,)).fetchone()[0] == "inflight":
                    save(future)
            raise
    if failure:
        raise ValueError(failure)
    selected = dict(db.execute(
        "SELECT key,content FROM attempts WHERE status='accepted' AND key IN ("
        + ",".join("?" for _ in keys) + ") ORDER BY finished,id", keys))
    if set(selected) != set(keys):
        raise ValueError("Cumulative request limit reached; partial stage saved")
    return {key.removeprefix(cell + ":"): text for key, text in selected.items()}


def progress(db, settings):
    states = [json.loads(s) for (s,) in db.execute("SELECT state FROM cells")]
    attempts = dict(db.execute("SELECT status,COUNT(*) FROM attempts GROUP BY status"))
    complete = sum(not s["running"] for s in states)
    return {"phase": "complete" if complete == len(settings["cells"]) else "stopped",
            "cells_completed": complete, "cells_planned": len(settings["cells"]),
            "economic_rounds": db.execute("SELECT COUNT(*) FROM rounds").fetchone()[0],
            "accepted_responses": attempts.get("accepted", 0), "physical_requests": sum(attempts.values()),
            "attempts": attempts, "invoice_reconciled": False}


def export(db, settings, output):
    cells = {c["id"]: c for c in settings["cells"]}
    rows, agents = [], []
    for cell, record, people in db.execute("SELECT cell,record,agents FROM rounds ORDER BY cell,step"):
        meta = metadata(settings, cells[cell])
        rows.append({**meta, **json.loads(record)})
        agents.extend({**meta, **r} for r in json.loads(people))
    write_csv(output / "rounds.csv", rows)
    write_csv(output / "agents.csv", agents)


def run_llm(settings, output, max_requests, concurrency=10, env_file=Path(".env"),
            resume=False, retry_uncertain=False, client=None):
    if type(max_requests) is not int or max_requests < 1 or not 1 <= concurrency <= 32:
        raise ValueError("A positive --max-requests and concurrency in [1,32] are required")
    if resume and not (output / "plan.json").is_file():
        raise ValueError("No saved experiment to resume")
    if not resume:
        empty_output(output)
    with (output / "run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Experiment already running") from None
        manifest = output / "plan.json"
        if manifest.exists() and not resume:
            raise ValueError("Experiment already exists; use --resume")
        if manifest.exists() and json.loads(manifest.read_text()) != settings:
            raise ValueError("Plan or sources changed; cannot resume this experiment")
        manifest.write_text(dumps(settings) + "\n")
        with closing(connect(output / "ledger.sqlite")) as db:
            initial = progress(db, settings)["accepted_responses"]
            failure = None
            try:
                if progress(db, settings)["phase"] != "complete":
                    client = client or Client(settings["request"], env_file)
                for cell in settings["cells"]:
                    found = db.execute("SELECT state FROM cells WHERE id=?", (cell["id"],)).fetchone()
                    world = (llm.Society.restore(json.loads(found[0])) if found else
                             llm.Society(Parameters(**cell["parameters"]), cell["seed"], settings["rounds"], cell["condition"]))
                    while world.running:
                        requests = world.requests()
                        with db:
                            db.execute("INSERT OR REPLACE INTO cells VALUES (?,?)", (cell["id"], dumps(world.state())))
                        answers = query(db, settings, cell["id"], requests, client, max_requests, concurrency, retry_uncertain)
                        result = world.accept(answers)
                        with db:
                            if result:
                                row, agents = result
                                db.execute("INSERT INTO rounds VALUES (?,?,?,?)", (cell["id"], row["Step"], dumps(row), dumps(agents)))
                            db.execute("UPDATE cells SET state=? WHERE id=?", (dumps(world.state()), cell["id"]))
                        if result:
                            print(f"{cell['id']}: round {world.round}/{settings['rounds']}, resource {world.resource:.3f}")
                export(db, settings, output)
            except (Exception, KeyboardInterrupt) as error:
                failure = str(error) if isinstance(error, ValueError) else type(error).__name__
                raise
            finally:
                receipt = progress(db, settings)
                receipt.update(new_responses=receipt["accepted_responses"] - initial,
                               reused_responses=initial, failure=failure)
                (output / "status.json").write_text(dumps(receipt) + "\n")
            return receipt
