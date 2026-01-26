from __future__ import annotations

import argparse
import json
from pathlib import Path

from jsonschema import validate

SCHEMA = {
    "type": "object",
    "required": ["category", "severity", "summary", "next_steps"],
    "properties": {
        "category": {"type": "string"},
        "severity": {"type": "string"},
        "summary": {"type": "string"},
        "next_steps": {"type": "array", "items": {"type": "string"}},
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=False, default="runs/latest/preds.jsonl")
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(
            f"Missing predictions file at {path}. Run your generation step first (e.g., write JSONL to this path).",
            flush=True,
        )
        raise SystemExit(1)

    ok, total = 0, 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                validate(instance=obj, schema=SCHEMA)
                ok += 1
            except Exception:
                pass
    if total == 0:
        print("No records to evaluate.", flush=True)
    else:
        print(f"JSON_schema_pass_rate={ok/total:.2%} ({ok}/{total})", flush=True)

if __name__ == "__main__":
    main()
