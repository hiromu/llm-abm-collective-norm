"""CSV and JSON file helpers."""

import csv
import json
from pathlib import Path


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def empty_output(path, *inputs):
    path = Path(path)
    if any(path.resolve().is_relative_to(Path(p).resolve()) or Path(p).resolve().is_relative_to(path.resolve())
           for p in inputs):
        raise ValueError("Output must be separate from input data")
    if path.exists() and any(path.iterdir()):
        raise ValueError("Output directory must be empty")
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: dumps(v) if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()})
