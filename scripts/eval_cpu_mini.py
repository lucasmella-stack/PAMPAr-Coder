#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Eval HumanEval-Mini en CPU (mientras training usa la GPU).
3 problemas ligeros para validar que el modelo genera código funcional.
"""
import sys
import os
import json
import time
import torch
import multiprocessing

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pampar.coder.v2.config import PRESET_1_5B
from pampar.coder.v2.modelo import PampaRCoderV2
import sentencepiece as spm


PROBLEMS = [
    {
        "task_id": "HumanEval/0",
        "prompt": (
            "from typing import List\n\n\n"
            "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
            '    """ Check if in given list of numbers, are any two numbers closer to each other than\n'
            "    given threshold.\n"
            "    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n"
            "    False\n"
            "    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n"
            "    True\n"
            '    """\n'
        ),
        "test": (
            "def check(candidate):\n"
            "    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n"
            "    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n"
        ),
        "entry_point": "has_close_elements",
    },
    {
        "task_id": "HumanEval/2",
        "prompt": (
            "def truncate_number(number: float) -> float:\n"
            '    """ Given a positive floating point number, it can be decomposed into\n'
            "    an integer part (largest integer smaller than given number) and decimals\n"
            "    (leftover part always smaller than 1).\n"
            "    Return the decimal part of the number.\n"
            "    >>> truncate_number(3.5)\n"
            "    0.5\n"
            '    """\n'
        ),
        "test": (
            "def check(candidate):\n"
            "    assert candidate(3.5) == 0.5\n"
            "    assert abs(candidate(1.33) - 0.33) < 1e-6\n"
            "    assert abs(candidate(123.456) - 0.456) < 1e-6\n"
        ),
        "entry_point": "truncate_number",
    },
    {
        "task_id": "HumanEval/4",
        "prompt": (
            "from typing import List\n\n\n"
            "def mean_absolute_deviation(numbers: List[float]) -> float:\n"
            '    """ For a given list of input numbers, calculate Mean Absolute Deviation\n'
            "    around the mean of this dataset.\n"
            "    MAD = average | x - x_mean |\n"
            "    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])\n"
            "    1.0\n"
            '    """\n'
        ),
        "test": (
            "def check(candidate):\n"
            "    assert abs(candidate([1.0, 2.0, 3.0]) - 2.0/3.0) < 1e-6\n"
            "    assert abs(candidate([1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-6\n"
        ),
        "entry_point": "mean_absolute_deviation",
    },
]


def execute_safe(code, test, entry_point, timeout=10.0):
    """Execute code + tests in isolated process."""
    full = code + "\n" + test + f"\ncheck({entry_point})\n"
    mgr = multiprocessing.Manager()
    result = mgr.dict({"passed": False, "error": None})

    def run(r):
        try:
            exec(full, {})
            r["passed"] = True
        except Exception as e:
            r["error"] = f"{type(e).__name__}: {str(e)[:200]}"

    p = multiprocessing.Process(target=run, args=(result,))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join(1)
        return {"passed": False, "error": "TimeoutError"}
    return dict(result)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/workspace/checkpoints/step_2500.pt")
    parser.add_argument("--tokenizer", default="/workspace/PAMPAr-Coder/data/tokenizer/pampar_48k.model")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    print("=" * 60)
    print("  PAMPAr-Coder 1.5B — HumanEval Mini (CPU)")
    print("=" * 60)

    # Load model on CPU
    print("\nLoading model on CPU...")
    t0 = time.time()
    model = PampaRCoderV2(PRESET_1_5B)

    print("  Loading checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    gs = ckpt.get("global_step", "?")
    vl = ckpt.get("val_loss", "?")
    del ckpt
    print(f"  Model loaded in {time.time()-t0:.0f}s (global_step={gs}, val_loss={vl})")

    # Tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(args.tokenizer)
    model.registrar_tokenizer(tokenizer)

    # Evaluate
    print(f"\nEvaluating {len(PROBLEMS)} problems...\n")
    passed = 0
    total = len(PROBLEMS)

    for prob in PROBLEMS:
        tid = prob["task_id"]
        prompt = prob["prompt"]

        tokens = tokenizer.Encode(prompt)
        if len(tokens) > 200:
            tokens = tokens[-200:]
        input_ids = torch.tensor([tokens], dtype=torch.long)

        t1 = time.time()
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_tokens=args.max_tokens,
                temperature=0.0,
                top_k=1,
            )
        gen_time = time.time() - t1

        gen_tokens = out[0, len(tokens):].tolist()
        completion = tokenizer.Decode(gen_tokens)

        # Truncate at stop tokens
        for stop in ["\nclass ", "\ndef ", "\n#", "\nif __name__", "\nprint("]:
            idx = completion.find(stop)
            if idx != -1:
                completion = completion[:idx]

        full_code = prompt + completion
        result = execute_safe(full_code, prob["test"], prob["entry_point"])

        status = "PASS" if result["passed"] else "FAIL"
        if result["passed"]:
            passed += 1

        print(f"  {tid}: {status} ({gen_time:.1f}s)")
        comp_oneline = completion[:150].replace("\n", " | ")
        print(f"    Code: {comp_oneline}")
        if result.get("error"):
            print(f"    Error: {result['error'][:120]}")
        print()

    print("=" * 60)
    print(f"  Results: {passed}/{total} passed ({passed/total*100:.0f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
