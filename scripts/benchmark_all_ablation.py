#!/usr/bin/env python3
# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2025-2026 Lucas Ricardo Mella Chillemi
"""
benchmark_all_ablation.py — Benchmark de inferencia para los 4 experimentos de ablación.

Ejecuta:
  1. HumanEval pass@1 (164 problemas de código)
  2. Custom eval (16 problemas propios, 5 niveles de dificultad)
  3. Throughput (tokens/sec de generación)
  4. Early exit stats (profundidad media de salida, solo PAMPAr)

Uso:
  python -X utf8 scripts/benchmark_all_ablation.py
  python -X utf8 scripts/benchmark_all_ablation.py --experiments pampar_v3 vanilla_gpt
  python -X utf8 scripts/benchmark_all_ablation.py --skip-humaneval
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# ── Metadata ─────────────────────────────────────────────────────────────────

EXPERIMENTS = ["pampar_v3", "no_llaves", "single_stream", "vanilla_gpt"]
PAMPAR_VARIANTS = {"pampar_v3", "no_llaves", "single_stream"}


# ═════════════════════════════════════════════════════════════════════════════
# Model loading
# ═════════════════════════════════════════════════════════════════════════════


def load_experiment(
    experiment: str,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, Any]:
    """Load model and tokenizer for an experiment.

    Returns:
        (model, tokenizer)
    """
    import sentencepiece as spm

    # Tokenizer (shared across all experiments)
    tok_candidates = [
        ROOT / "data" / "tokenizer" / "pampar_48k.model",
        ROOT / "pampar_48k.model",
        Path("pampar_48k.model"),
    ]
    tokenizer_path = next((p for p in tok_candidates if p.exists()), None)
    if tokenizer_path is None:
        raise FileNotFoundError(f"Tokenizer not found in: {tok_candidates}")

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(str(tokenizer_path))

    if experiment in PAMPAR_VARIANTS:
        from pampar.inference import load_model

        model, _cfg = load_model(
            checkpoint_path, device, with_tokenizer=False, verbose=True
        )
        model.registrar_tokenizer(tokenizer)
    else:
        from vanilla_gpt import VanillaGPT, VanillaGPTConfig

        cfg = VanillaGPTConfig()
        model = VanillaGPT(cfg).to(device)
        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        state = ckpt.get("modelo", ckpt.get("model", ckpt))
        model.load_state_dict(state)
        model.eval()

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded {experiment}: {params:.1f}M params on {device}")
    return model, tokenizer


# ═════════════════════════════════════════════════════════════════════════════
# Generation
# ═════════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    max_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.95,
    stop_sequences: list[str] | None = None,
    collect_exit_levels: bool = False,
    use_early_exit: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Generate text by completing a prompt with nucleus sampling.

    Returns:
        (generated_text, info_dict)
    """
    if stop_sequences is None:
        stop_sequences = ["\n\ndef ", "\n\nclass ", "\n\n\n"]

    ids = tokenizer.Encode(prompt)
    generated = list(ids)
    exit_levels: list[int] = []

    is_pampar = hasattr(model, "registrar_tokenizer")

    for _ in range(max_tokens):
        ctx = torch.tensor([generated[-512:]], dtype=torch.long, device=device)

        if is_pampar:
            logits, _, info = model(ctx, use_early_exit=use_early_exit)
            if collect_exit_levels:
                exit_levels.append(info.get("exit_nivel", -1))
        else:
            logits, _, info = model(ctx)

        next_logits = logits[0, -1]

        if temperature <= 0.0:
            next_token = int(next_logits.argmax())
        else:
            next_logits = next_logits / temperature
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[mask] = float("-inf")
            probs = F.softmax(sorted_logits, dim=-1)
            idx = int(torch.multinomial(probs, 1))
            next_token = int(sorted_indices[idx])

        generated.append(next_token)
        decoded = tokenizer.Decode(generated[len(ids) :])

        for stop in stop_sequences:
            if stop in decoded:
                decoded = decoded[: decoded.index(stop)]
                n_tokens = len(generated) - len(ids)
                return decoded, {"n_tokens": n_tokens, "exit_levels": exit_levels}

    n_tokens = len(generated) - len(ids)
    return tokenizer.Decode(generated[len(ids) :]), {
        "n_tokens": n_tokens,
        "exit_levels": exit_levels,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Safe execution with timeout
# ═════════════════════════════════════════════════════════════════════════════


class _TimeoutError(Exception):
    pass


def _timeout_handler(_signum: int, _frame: Any) -> None:
    raise _TimeoutError("Timeout")


def execute_with_timeout(code: str, timeout_sec: int = 10) -> tuple[bool, str]:
    """Execute Python code with timeout. Returns (passed, error_msg)."""
    if sys.platform == "win32":
        result: dict[str, Any] = {"passed": False, "error": "timeout"}

        def _run() -> None:
            try:
                ns: dict[str, Any] = {}
                exec(compile(code, "<eval>", "exec"), ns)  # noqa: S102
                result["passed"] = True
                result["error"] = ""
            except AssertionError as e:
                result["error"] = f"AssertionError: {e}"
            except Exception as e:
                result["error"] = f"{type(e).__name__}: {e}"

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=timeout_sec)
        if t.is_alive():
            return False, "timeout"
        return result["passed"], result["error"]

    old = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_sec)
    try:
        ns: dict[str, Any] = {}
        exec(compile(code, "<eval>", "exec"), ns)  # noqa: S102
        signal.alarm(0)
        return True, ""
    except _TimeoutError:
        return False, "timeout"
    except AssertionError as e:
        signal.alarm(0)
        return False, f"AssertionError: {e}"
    except Exception as e:
        signal.alarm(0)
        return False, f"{type(e).__name__}: {e}"
    finally:
        signal.signal(signal.SIGALRM, old)


def execute_and_verify(
    code: str, verify_fn: Any, timeout_sec: int = 10
) -> tuple[bool, str]:
    """Execute code and run verification function on result namespace."""
    if sys.platform == "win32":
        result: dict[str, Any] = {"passed": False, "error": "timeout"}

        def _run() -> None:
            try:
                ns: dict[str, Any] = {}
                exec(compile(code, "<eval>", "exec"), ns)  # noqa: S102
                if verify_fn(ns):
                    result["passed"] = True
                    result["error"] = ""
                else:
                    result["error"] = "verification failed"
            except Exception as e:
                result["error"] = f"{type(e).__name__}: {e}"

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=timeout_sec)
        if t.is_alive():
            return False, "timeout"
        return result["passed"], result["error"]

    old = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_sec)
    try:
        ns: dict[str, Any] = {}
        exec(compile(code, "<eval>", "exec"), ns)  # noqa: S102
        passed = bool(verify_fn(ns))
        signal.alarm(0)
        return passed, "" if passed else "verification failed"
    except _TimeoutError:
        return False, "timeout"
    except Exception as e:
        signal.alarm(0)
        return False, f"{type(e).__name__}: {e}"
    finally:
        signal.signal(signal.SIGALRM, old)


# ═════════════════════════════════════════════════════════════════════════════
# HumanEval benchmark
# ═════════════════════════════════════════════════════════════════════════════


def load_humaneval() -> list[dict[str, Any]]:
    """Load HumanEval (164 problems) from HuggingFace."""
    from datasets import load_dataset

    ds = load_dataset("openai_humaneval", split="test")
    problems: list[dict[str, Any]] = []
    for row in ds:
        problems.append(
            {
                "task_id": row["task_id"],
                "prompt": row["prompt"],
                "canonical_solution": row["canonical_solution"],
                "test": row["test"],
                "entry_point": row["entry_point"],
            }
        )
    print(f"  HumanEval loaded: {len(problems)} problems")
    return problems


def run_humaneval(
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    experiment: str,
    problems: list[dict[str, Any]],
    temperature: float = 0.2,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run HumanEval pass@1 benchmark."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"[{experiment}] HumanEval pass@1 ({len(problems)} problems)")
    print(sep)

    results: list[dict[str, Any]] = []
    t0 = time.time()

    for i, problem in enumerate(problems):
        completion, gen_info = generate(
            model,
            tokenizer,
            problem["prompt"],
            device,
            max_tokens=512,
            temperature=temperature,
            stop_sequences=["\n\ndef ", "\n\nclass ", "\n\n\n"],
            use_early_exit=False,
        )

        full_code = (
            problem["prompt"]
            + completion
            + "\n\n"
            + problem["test"]
            + f"\n\ncheck({problem['entry_point']})\n"
        )

        passed, error = execute_with_timeout(full_code, timeout_sec=10)

        results.append(
            {
                "task_id": problem["task_id"],
                "passed": passed,
                "error": error,
                "completion_len": len(completion),
                "tokens": gen_info["n_tokens"],
            }
        )

        if verbose or passed:
            status = "\u2713" if passed else "\u2717"
            print(f"  [{i + 1:3d}/{len(problems)}] {status} {problem['task_id']}")
        elif (i + 1) % 20 == 0:
            n_pass = sum(1 for r in results if r["passed"])
            print(f"  [{i + 1:3d}/{len(problems)}] {n_pass} passed so far...")

    elapsed = time.time() - t0
    n_passed = sum(1 for r in results if r["passed"])
    pass_at_1 = n_passed / len(problems) if problems else 0

    print(
        f"\n  Result: {n_passed}/{len(problems)} = {pass_at_1:.1%} pass@1 ({elapsed:.0f}s)"
    )

    return {
        "pass_at_1": pass_at_1,
        "passed": n_passed,
        "total": len(problems),
        "elapsed_sec": round(elapsed, 2),
        "per_problem": results,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Custom eval (16 coding problems, 5 difficulty levels)
# ═════════════════════════════════════════════════════════════════════════════

CUSTOM_CASES: list[dict[str, Any]] = [
    # ── Level 1: basics ──────────────────────────────────────────────────
    {
        "level": 1,
        "desc": "Count vowels",
        "prompt": (
            "### Problem:\n"
            "Write a Python function `contar_vocales(texto)` that returns the number "
            "of vowels (a, e, i, o, u, case-insensitive) in the string.\n"
            "### Solution:\n"
        ),
        "verify": lambda ns: (
            ns["contar_vocales"]("hola mundo") == 4
            and ns["contar_vocales"]("") == 0
            and ns["contar_vocales"]("xyz") == 0
        ),
    },
    {
        "level": 1,
        "desc": "Sum digits",
        "prompt": (
            "### Problem:\n"
            "Write a Python function `suma_digitos(n)` that returns the sum of all "
            "digits of the non-negative integer n.\n"
            "### Solution:\n"
        ),
        "verify": lambda ns: (
            ns["suma_digitos"](123) == 6
            and ns["suma_digitos"](0) == 0
            and ns["suma_digitos"](999) == 27
        ),
    },
    {
        "level": 1,
        "desc": "Palindrome",
        "prompt": (
            "### Problem:\n"
            "Write a Python function `es_palindromo(s)` that returns True if the "
            "string is a palindrome, False otherwise.\n"
            "### Solution:\n"
        ),
        "verify": lambda ns: (
            ns["es_palindromo"]("racecar") is True
            and ns["es_palindromo"]("hello") is False
            and ns["es_palindromo"]("a") is True
        ),
    },
    {
        "level": 1,
        "desc": "Max of list",
        "prompt": (
            "### Problem:\n"
            "Write a Python function `maximo(lista)` that returns the maximum element "
            "of a non-empty list without using the built-in max().\n"
            "### Solution:\n"
        ),
        "verify": lambda ns: (
            ns["maximo"]([3, 1, 4, 1, 5, 9]) == 9
            and ns["maximo"]([0]) == 0
            and ns["maximo"]([-1, -5, -2]) == -1
        ),
    },
    {
        "level": 1,
        "desc": "FizzBuzz",
        "prompt": (
            "### Problem:\n"
            "Write a Python function `fizzbuzz(n)` that returns 'FizzBuzz' if n is "
            "divisible by both 3 and 5, 'Fizz' if divisible by 3, 'Buzz' if divisible "
            "by 5, or the string representation of n otherwise.\n"
            "### Solution:\n"
        ),
        "verify": lambda ns: (
            ns["fizzbuzz"](15) == "FizzBuzz"
            and ns["fizzbuzz"](3) == "Fizz"
            and ns["fizzbuzz"](5) == "Buzz"
            and ns["fizzbuzz"](7) == "7"
        ),
    },
    # ── Level 2: lists / dicts ───────────────────────────────────────────
    {
        "level": 2,
        "desc": "Flatten one level",
        "prompt": (
            "### Problem:\n"
            "Write a Python function `aplanar(lista)` that flattens a list of lists "
            "by one level and returns the result as a single list.\n"
            "### Solution:\n"
        ),
        "verify": lambda ns: (
            ns["aplanar"]([[1, 2], [3, 4], [5]]) == [1, 2, 3, 4, 5]
            and ns["aplanar"]([]) == []
        ),
    },
    {
        "level": 2,
        "desc": "Element frequency",
        "prompt": (
            "### Problem:\n"
            "Write a Python function `frecuencia(lista)` that returns a dictionary "
            "mapping each element to its count in the list.\n"
            "### Solution:\n"
        ),
        "verify": lambda ns: (
            ns["frecuencia"]([1, 2, 2, 3, 3, 3]) == {1: 1, 2: 2, 3: 3}
            and ns["frecuencia"]([]) == {}
        ),
    },
    {
        "level": 2,
        "desc": "Squares of evens",
        "prompt": (
            "### Problem:\n"
            "Write a Python function `cuadrados_pares(n)` that returns a list of "
            "squares of all even numbers from 2 to n inclusive.\n"
            "### Solution:\n"
        ),
        "verify": lambda ns: (
            ns["cuadrados_pares"](6) == [4, 16, 36] and ns["cuadrados_pares"](1) == []
        ),
    },
    {
        "level": 2,
        "desc": "Invert dict",
        "prompt": (
            "### Problem:\n"
            "Write a Python function `invertir_dict(d)` that returns a new dictionary "
            "with keys and values swapped.\n"
            "### Solution:\n"
        ),
        "verify": lambda ns: (
            ns["invertir_dict"]({"a": 1, "b": 2}) == {1: "a", 2: "b"}
            and ns["invertir_dict"]({}) == {}
        ),
    },
    # ── Level 3: algorithms ──────────────────────────────────────────────
    {
        "level": 3,
        "desc": "Fibonacci iterative",
        "prompt": (
            "### Problem:\n"
            "Write a Python function `fibonacci(n)` that returns the n-th Fibonacci "
            "number (0-indexed: fibonacci(0)=0, fibonacci(1)=1, fibonacci(7)=13).\n"
            "### Solution:\n"
        ),
        "verify": lambda ns: (
            ns["fibonacci"](0) == 0
            and ns["fibonacci"](1) == 1
            and ns["fibonacci"](7) == 13
            and ns["fibonacci"](10) == 55
        ),
    },
    {
        "level": 3,
        "desc": "Binary search",
        "prompt": (
            "### Problem:\n"
            "Write a Python function `busqueda_binaria(lista, objetivo)` that returns "
            "the index of the target in a sorted list, or -1 if not found.\n"
            "### Solution:\n"
        ),
        "verify": lambda ns: (
            ns["busqueda_binaria"]([1, 3, 5, 7, 9], 5) == 2
            and ns["busqueda_binaria"]([1, 3, 5, 7, 9], 4) == -1
            and ns["busqueda_binaria"]([], 1) == -1
        ),
    },
    {
        "level": 3,
        "desc": "Merge sort",
        "prompt": (
            "### Problem:\n"
            "Write a Python function `merge_sort(lista)` that returns a new sorted "
            "list using the merge sort algorithm.\n"
            "### Solution:\n"
        ),
        "verify": lambda ns: (
            ns["merge_sort"]([3, 1, 4, 1, 5, 9, 2, 6]) == [1, 1, 2, 3, 4, 5, 6, 9]
            and ns["merge_sort"]([]) == []
            and ns["merge_sort"]([1]) == [1]
        ),
    },
    # ── Level 4: OOP ─────────────────────────────────────────────────────
    {
        "level": 4,
        "desc": "Stack class",
        "prompt": (
            "### Problem:\n"
            "Write a Python class `Stack` with methods `push(item)` and `pop()` "
            "implementing a LIFO stack.\n"
            "### Solution:\n"
        ),
        "verify": lambda ns: (
            (s := ns["Stack"]()) is not None
            and (s.push(1) or True)
            and (s.push(2) or True)
            and s.pop() == 2
            and s.pop() == 1
        ),
    },
    {
        "level": 4,
        "desc": "Punto with distance",
        "prompt": (
            "### Problem:\n"
            "Write a Python class `Punto` with attributes `x` and `y`, and a method "
            "`distancia(otro)` that returns the Euclidean distance to another Punto.\n"
            "### Solution:\n"
        ),
        "verify": lambda ns: ns["Punto"](0, 0).distancia(ns["Punto"](3, 4)) == 5.0,
    },
    # ── Level 5: advanced ────────────────────────────────────────────────
    {
        "level": 5,
        "desc": "Memoize decorator",
        "prompt": (
            "### Problem:\n"
            "Write a Python higher-order function `memoize(fn)` that returns a "
            "wrapped version of fn that caches results by argument.\n"
            "### Solution:\n"
        ),
        "verify": lambda ns: (
            (fn := ns["memoize"](lambda x: x * 2)) is not None and fn(5) == 10
        ),
    },
    {
        "level": 5,
        "desc": "Prime generator",
        "prompt": (
            "### Problem:\n"
            "Write a Python generator function `primos_hasta(n)` that yields all "
            "prime numbers up to and including n.\n"
            "### Solution:\n"
        ),
        "verify": lambda ns: (
            list(ns["primos_hasta"](20)) == [2, 3, 5, 7, 11, 13, 17, 19]
        ),
    },
]


def _extract_code(text: str) -> str:
    """Extract clean Python code from model output."""
    # Strip markdown fences
    if "```python" in text:
        start = text.index("```python") + len("```python")
        end = text.index("```", start) if "```" in text[start:] else len(text)
        text = text[start:end]
    elif "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start) if "```" in text[start:] else len(text)
        text = text[start:end]

    # Take only the first function/class definition
    lines = text.split("\n")
    result: list[str] = []
    found_def = False

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith(("def ", "class ")):
            if found_def:
                indent = len(line) - len(stripped)
                if indent == 0 and any(r.strip() for r in result):
                    break
            found_def = True
        if found_def:
            result.append(line)

    code = "\n".join(result).strip()

    # Normalize indentation to multiples of 4
    fixed_lines = []
    for i, line in enumerate(code.split("\n")):
        stripped = line.lstrip()
        if not stripped:
            fixed_lines.append("")
            continue
        spaces = len(line) - len(stripped)
        normalized = round(spaces / 4) * 4
        if (
            i > 0
            and normalized < 4
            and code.split("\n")[0].lstrip().startswith(("def ", "class "))
        ):
            normalized = max(4, normalized)
        fixed_lines.append(" " * normalized + stripped)

    return "\n".join(fixed_lines)


def run_custom_eval(
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    experiment: str,
    temperature: float = 0.2,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run custom 16-problem eval suite."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"[{experiment}] Custom eval ({len(CUSTOM_CASES)} problems)")
    print(sep)

    results: list[dict[str, Any]] = []
    by_level: dict[int, dict[str, int]] = defaultdict(lambda: {"passed": 0, "total": 0})

    for i, caso in enumerate(CUSTOM_CASES):
        completion, gen_info = generate(
            model,
            tokenizer,
            caso["prompt"],
            device,
            max_tokens=384,
            temperature=temperature,
            stop_sequences=["\n\n\n", "\n\ndef ", "\n\nclass "],
            use_early_exit=False,
        )

        code = _extract_code(completion)
        passed, error = execute_and_verify(code, caso["verify"])

        results.append(
            {
                "level": caso["level"],
                "desc": caso["desc"],
                "passed": passed,
                "error": error,
                "tokens": gen_info["n_tokens"],
            }
        )
        by_level[caso["level"]]["total"] += 1
        if passed:
            by_level[caso["level"]]["passed"] += 1

        status = "\u2713" if passed else "\u2717"
        extra = f" ({error[:50]})" if not passed and verbose else ""
        print(
            f"  [{i + 1:2d}/{len(CUSTOM_CASES)}] L{caso['level']} {status} {caso['desc']}{extra}"
        )

    n_passed = sum(1 for r in results if r["passed"])
    accuracy = n_passed / len(CUSTOM_CASES) if CUSTOM_CASES else 0

    level_summary: dict[str, str] = {}
    for lvl in sorted(by_level):
        d = by_level[lvl]
        level_summary[f"level_{lvl}"] = f"{d['passed']}/{d['total']}"

    print(f"\n  Result: {n_passed}/{len(CUSTOM_CASES)} = {accuracy:.1%}")
    for lvl, s in level_summary.items():
        print(f"    {lvl}: {s}")

    return {
        "accuracy": accuracy,
        "passed": n_passed,
        "total": len(CUSTOM_CASES),
        "by_level": level_summary,
        "per_problem": results,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Throughput benchmark
# ═════════════════════════════════════════════════════════════════════════════

THROUGHPUT_PROMPTS = [
    'def factorial(n):\n    """Return factorial of n."""\n',
    'def is_prime(n):\n    """Check if n is prime."""\n',
    'class LinkedList:\n    """Singly linked list."""\n',
    'def quicksort(arr):\n    """Sort using quicksort."""\n',
    'def find_duplicates(lst):\n    """Find duplicate elements."""\n',
    'def matrix_multiply(a, b):\n    """Multiply two matrices."""\n',
    'def depth_first_search(graph, start):\n    """DFS traversal."""\n',
    'class BinarySearchTree:\n    """BST implementation."""\n',
    'def longest_common_subsequence(s1, s2):\n    """Find LCS."""\n',
    'def topological_sort(graph):\n    """Topological sort of DAG."""\n',
]


def run_throughput(
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    experiment: str,
    max_tokens: int = 256,
) -> dict[str, Any]:
    """Measure generation throughput (tokens/sec)."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(
        f"[{experiment}] Throughput ({len(THROUGHPUT_PROMPTS)} prompts x {max_tokens} max tokens)"
    )
    print(sep)

    # Warmup
    generate(model, tokenizer, "def hello():\n", device, max_tokens=32, temperature=0.0)

    total_tokens = 0
    t0 = time.time()

    for prompt in THROUGHPUT_PROMPTS:
        _, info = generate(
            model,
            tokenizer,
            prompt,
            device,
            max_tokens=max_tokens,
            temperature=0.0,
            use_early_exit=True,
        )
        total_tokens += info["n_tokens"]

    elapsed = time.time() - t0
    tps = total_tokens / elapsed if elapsed > 0 else 0

    print(f"  {total_tokens} tokens in {elapsed:.1f}s = {tps:.1f} tok/s")

    return {
        "tokens_per_sec": round(tps, 2),
        "total_tokens": total_tokens,
        "elapsed_sec": round(elapsed, 2),
        "n_prompts": len(THROUGHPUT_PROMPTS),
        "max_tokens": max_tokens,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Early exit stats (PAMPAr variants only)
# ═════════════════════════════════════════════════════════════════════════════


def run_early_exit_stats(
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    experiment: str,
) -> dict[str, Any] | None:
    """Measure early exit depth distribution. PAMPAr variants only."""
    if experiment not in PAMPAR_VARIANTS:
        return None

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"[{experiment}] Early exit stats")
    print(sep)

    all_levels: list[int] = []

    for prompt in THROUGHPUT_PROMPTS[:5]:
        _, info = generate(
            model,
            tokenizer,
            prompt,
            device,
            max_tokens=128,
            temperature=0.0,
            collect_exit_levels=True,
            use_early_exit=True,
        )
        all_levels.extend(info["exit_levels"])

    if not all_levels:
        print("  No exit levels collected")
        return None

    mean_level = sum(all_levels) / len(all_levels)
    counts = Counter(all_levels)
    histogram = {str(k): v for k, v in sorted(counts.items())}

    max_level = 0
    cfg = getattr(model, "config", None)
    if cfg and hasattr(cfg, "n_levels"):
        max_level = cfg.n_levels
    else:
        max_level = max(all_levels) if all_levels else 0

    pct_early = sum(1 for lv in all_levels if lv < max_level) / len(all_levels) * 100

    print(f"  Mean exit level: {mean_level:.2f} / {max_level}")
    print(f"  Early exit %: {pct_early:.1f}%")
    print(f"  Distribution: {histogram}")

    return {
        "mean_level": round(mean_level, 3),
        "max_level": max_level,
        "n_tokens_measured": len(all_levels),
        "pct_early_exit": round(pct_early, 2),
        "histogram": histogram,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation inference benchmark")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=EXPERIMENTS,
        choices=EXPERIMENTS,
        help="Which experiments to benchmark",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(ROOT / "ablation_results"),
        help="Directory with experiment checkpoints and results",
    )
    parser.add_argument(
        "--skip-humaneval",
        action="store_true",
        help="Skip HumanEval (slow but most informative)",
    )
    parser.add_argument(
        "--skip-custom", action="store_true", help="Skip custom eval suite"
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    results_dir = Path(args.results_dir)
    print(f"Device: {device}")
    print(f"Results dir: {results_dir}")
    print(f"Experiments: {args.experiments}")

    # Load HumanEval once (shared across experiments)
    humaneval_problems: list[dict[str, Any]] | None = None
    if not args.skip_humaneval:
        print("\nLoading HumanEval dataset...")
        humaneval_problems = load_humaneval()

    all_results: dict[str, dict[str, Any]] = {}

    for experiment in args.experiments:
        print(f"\n{'#' * 70}")
        print(f"# Experiment: {experiment}")
        print(f"{'#' * 70}")

        ckpt_path = results_dir / experiment / "checkpoint.pt"
        if not ckpt_path.exists():
            print(f"  SKIP: checkpoint not found at {ckpt_path}")
            continue

        model, tokenizer = load_experiment(experiment, ckpt_path, device)
        exp_results: dict[str, Any] = {"experiment": experiment}

        # 1. HumanEval
        if humaneval_problems is not None:
            exp_results["humaneval"] = run_humaneval(
                model,
                tokenizer,
                device,
                experiment,
                humaneval_problems,
                args.temperature,
                args.verbose,
            )

        # 2. Custom eval
        if not args.skip_custom:
            exp_results["custom_eval"] = run_custom_eval(
                model, tokenizer, device, experiment, args.temperature, args.verbose
            )

        # 3. Throughput
        exp_results["throughput"] = run_throughput(model, tokenizer, device, experiment)

        # 4. Early exit stats
        exit_stats = run_early_exit_stats(model, tokenizer, device, experiment)
        if exit_stats:
            exp_results["early_exit"] = exit_stats

        # Save per-experiment results
        out_path = results_dir / experiment / "benchmark_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(exp_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n  Results saved to {out_path}")

        all_results[experiment] = exp_results

        # Free memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Summary table ──
    sep = "=" * 80
    print(f"\n\n{sep}")
    print("SUMMARY")
    print(sep)

    header = (
        f"{'Experiment':<16} {'HumanEval':>12} {'Custom':>10} "
        f"{'tok/s':>10} {'Exit lvl':>10}"
    )
    print(header)
    print("-" * len(header))

    for exp, res in all_results.items():
        he = res.get("humaneval", {})
        ce = res.get("custom_eval", {})
        tp = res.get("throughput", {})
        ee = res.get("early_exit", {})

        he_str = f"{he['pass_at_1']:.1%}" if he else "\u2014"
        ce_str = f"{ce['accuracy']:.1%}" if ce else "\u2014"
        tp_str = f"{tp['tokens_per_sec']:.1f}" if tp else "\u2014"
        ee_str = f"{ee['mean_level']:.2f}" if ee else "\u2014"

        print(f"{exp:<16} {he_str:>12} {ce_str:>10} {tp_str:>10} {ee_str:>10}")

    # Save combined summary
    summary_path = results_dir / "benchmark_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nCombined results: {summary_path}")


if __name__ == "__main__":
    main()
