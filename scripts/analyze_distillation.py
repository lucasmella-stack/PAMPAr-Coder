"""Análisis rápido de datos de distillation."""
import json
from pathlib import Path

data_dir = Path("data/distillation")
for f in sorted(data_dir.glob("*.jsonl")):
    count = 0
    first = None
    for line in open(f, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        count += 1
        if first is None:
            first = json.loads(line)
    keys = list(first.keys()) if first else []
    sample = ""
    if first:
        for k in ["text", "instruction", "input", "output", "response"]:
            if k in first:
                sample = str(first[k])[:120]
                break
    print(f"{f.name}: {count} lineas | keys={keys}")
    print(f"  Sample: {sample}")
    print()
