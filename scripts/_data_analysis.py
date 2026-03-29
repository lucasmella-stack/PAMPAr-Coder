"""Analyze training data size and diversity."""

import json
from pathlib import Path

sft_files = list(Path("data").rglob("*.jsonl"))
print("=== ARCHIVOS JSONL ===")
total_lines = 0
for f in sorted(sft_files):
    lines = sum(1 for _ in open(f, encoding="utf-8"))
    total_lines += lines
    with open(f, encoding="utf-8") as fh:
        first = json.loads(fh.readline())
    keys = list(first.keys())[:5]
    rel = str(f)
    print(f"  {rel:55s} {lines:6d} lines  keys={keys}")

print(f"\n  TOTAL: {total_lines} lines across {len(sft_files)} files")

# Check main training files
main_files = [
    "data/distillation/distillation_data.jsonl",
    "data/code/train.jsonl",
    "data/code/train_massive.jsonl",
]
print("\n=== MAIN TRAINING FILES ===")
for p in main_files:
    path = Path(p)
    if path.exists():
        lines = sum(1 for _ in open(path, encoding="utf-8"))
        prompts = set()
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                key = d.get("prompt", d.get("instruction", d.get("input", "")))[:100]
                prompts.add(key)
        print(f"  {p}: {lines} total, {len(prompts)} unique prompts")

# Check what the ghidra_trainer actually loads
print("\n=== GHIDRA TRAINER DATA ===")
master = Path("data/master_sft.jsonl")
if master.exists():
    lines = sum(1 for _ in open(master, encoding="utf-8"))
    prompts = set()
    with open(master, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            key = d.get("prompt", d.get("instruction", d.get("input", "")))[:100]
            prompts.add(key)
    print(f"  master_sft.jsonl: {lines} total, {len(prompts)} unique prompts")

    # Sample diversity: check average output length
    lengths = []
    with open(master, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            out = d.get("output", d.get("response", d.get("completion", "")))
            lengths.append(len(out))
    if lengths:
        lengths.sort()
        print(
            f"  Output lengths: min={lengths[0]}, median={lengths[len(lengths) // 2]}, max={lengths[-1]}, mean={sum(lengths) / len(lengths):.0f}"
        )
else:
    print("  master_sft.jsonl NOT FOUND")
    # Look for what data file the trainer uses
    import re

    trainer = Path("scripts/ghidra_trainer.py")
    if trainer.exists():
        text = trainer.read_text(encoding="utf-8")
        matches = re.findall(r'["\'](data/[^"\']+\.jsonl)["\']', text)
        print(f"  Trainer references: {matches}")
