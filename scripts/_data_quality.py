"""Analyze master_sft.jsonl quality."""

import json

with open("data/master_sft.jsonl", encoding="utf-8") as f:
    lines = [json.loads(l) for l in f]

print(f"Total: {len(lines)}")
print(f"Keys: {list(lines[0].keys())}")
print(f"First 200 chars: {lines[0]['text'][:200]}")
print()

lengths = [len(l["text"]) for l in lines]
lengths.sort()
print(
    f"Text lengths: min={lengths[0]}, p25={lengths[len(lengths) // 4]}, median={lengths[len(lengths) // 2]}, p75={lengths[3 * len(lengths) // 4]}, max={lengths[-1]}"
)
print(f"Mean: {sum(lengths) / len(lengths):.0f} chars")

unique_starts = set(l["text"][:100] for l in lines)
print(f"Unique text starts (first 100ch): {len(unique_starts)}")

markers = ["### Solution:", "### Response:", "### Protocolo:", "### Scan:"]
for m in markers:
    count = sum(1 for l in lines if m in l["text"])
    print(f"  Contains '{m}': {count}")

# Check which data file ghidra_trainer uses
import re

trainer_text = open("scripts/ghidra_trainer.py", encoding="utf-8").read()
data_refs = re.findall(r'["\']([^"\']*\.jsonl)["\']', trainer_text)
print(f"\nghidra_trainer.py references: {data_refs}")

# Default data path in argparse
default_match = re.search(r'--data.*?default\s*=\s*["\']([^"\']+)["\']', trainer_text)
if default_match:
    print(f"Default --data: {default_match.group(1)}")
