#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Generador de datos textbook para continual pretrain — Fase 1.

Genera ~540K-640K tokens de texto tipo textbook distribuidos en 6 pilares
de razonamiento fundamental, multi-lenguaje (Python, JS, Rust, SQL, Bash).

Pilares:
  1. Lógica y razonamiento
  2. Estructuras de datos (cross-language)
  3. Patrones de código
  4. Comprensión de docs (cómo leer API refs, man pages)
  5. Debugging (stack traces, bisección, logging)
  6. Sintaxis multi-language (equivalencias entre lenguajes)

Usa GitHub Models API (gratis): GPT-4.1, GPT-4.1-mini, Llama-405B.

Uso:
  $env:GITHUB_TOKEN = "ghp_xxx"
  python scripts/generate_textbook_v3.py --pillar all --examples-per-pillar 200
  python scripts/generate_textbook_v3.py --pillar logic --examples-per-pillar 50
  python scripts/generate_textbook_v3.py --pillar data_structures --dry-run
  python scripts/generate_textbook_v3.py --resume
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Optional

# ============================================================================
# Modelos GitHub (mismos que generate_textbook_data.py, verificados)
# ============================================================================

GITHUB_API_URL = "https://models.inference.ai.azure.com"

MODELS = {
    "gpt41": {
        "id": "gpt-4.1",
        "temperature": 0.7,
    },
    "gpt41mini": {
        "id": "gpt-4.1-mini",
        "temperature": 0.8,
    },
    "llama": {
        "id": "Meta-Llama-3.1-405B-Instruct",
        "temperature": 0.7,
    },
}

# ============================================================================
# Pilares con temas y sub-temas
# ============================================================================

PILLARS = {
    "logic": {
        "name": "Lógica y Razonamiento",
        "topics": [
            "propositional logic: AND, OR, NOT, implication, biconditional",
            "truth tables: building truth tables for compound propositions",
            "logical equivalences: De Morgan's laws, contrapositive, distribution",
            "deductive reasoning: modus ponens, modus tollens, syllogisms",
            "inductive reasoning: generalizing from examples, finding patterns",
            "proof techniques: direct proof, proof by contradiction, proof by induction",
            "predicate logic: universal and existential quantifiers",
            "boolean algebra: simplification, normal forms, XOR properties",
            "algorithm correctness: loop invariants, pre/post conditions",
            "state machines: modeling state transitions, FSM in code",
            "recursion reasoning: base case, inductive step, stack depth",
            "complexity analysis: counting operations, Big-O derivation step by step",
            "divide and conquer reasoning: subproblem decomposition, merge step",
            "greedy correctness: exchange argument, matroid theory basics",
            "dynamic programming reasoning: optimal substructure, overlapping subproblems",
        ],
    },
    "data_structures": {
        "name": "Estructuras de Datos",
        "topics": [
            "arrays and lists: indexing, slicing, dynamic resizing — Python list vs JS Array vs Rust Vec",
            "linked lists: singly, doubly — implementation in Python and Rust",
            "stacks: LIFO, push/pop, applications (balanced parentheses, expression eval)",
            "queues: FIFO, deque, circular buffer — Python deque vs JS queue patterns",
            "hash maps: hashing, collision resolution, load factor — Python dict vs JS Map vs Rust HashMap",
            "sets: membership, union, intersection, difference — Python set vs JS Set vs Rust HashSet",
            "binary trees: traversal (inorder, preorder, postorder), BST invariant",
            "balanced trees: AVL rotations, red-black tree concept, when to use B-trees",
            "heaps: min-heap, max-heap, heapify, priority queue — Python heapq vs manual impl",
            "graphs: adjacency list vs matrix, BFS, DFS — comparative implementations",
            "tries: prefix trees, autocomplete, IP routing",
            "disjoint sets: union-find with path compression and rank",
            "bloom filters: probabilistic membership, false positive rate",
            "LRU cache: eviction policy, OrderedDict implementation vs doubly-linked list + hashmap",
            "skip lists: probabilistic alternative to balanced trees",
        ],
    },
    "patterns": {
        "name": "Patrones de Código",
        "topics": [
            "iterator pattern: lazy evaluation, generators in Python, iterators in Rust",
            "builder pattern: constructing complex objects step by step",
            "strategy pattern: swappable algorithms at runtime",
            "observer pattern: event-driven programming, pub/sub",
            "decorator pattern: wrapping functions, Python decorators vs JS middleware",
            "singleton pattern: controlled instantiation, module-level in Python",
            "factory pattern: creating objects without specifying exact class",
            "adaptor pattern: interface compatibility between different APIs",
            "error handling patterns: Result type in Rust, try/except in Python, try/catch in JS",
            "guard clauses: early return to reduce nesting",
            "null object pattern: avoiding None/null checks",
            "pipeline pattern: chaining transformations, Unix pipes, method chaining",
            "retry with backoff: exponential backoff, jitter, max retries",
            "resource management: context managers (Python with), RAII (Rust), try-finally (JS)",
            "dependency injection: passing dependencies, inversion of control",
        ],
    },
    "docs": {
        "name": "Comprensión de Documentación",
        "topics": [
            "reading Python docstrings: Args, Returns, Raises, Examples (Google style)",
            "reading Python stdlib docs: module structure, function signatures, see-also links",
            "reading MDN Web Docs: JavaScript method signatures, return values, browser compat",
            "reading Rust Book: ownership explanation, borrow checker diagrams",
            "reading man pages: synopsis, options, exit codes, examples section",
            "reading API references: HTTP methods, request/response schemas, auth headers",
            "reading SQL documentation: syntax diagrams, data type tables, constraints",
            "reading type signatures: generics, union types, optional params, return types",
            "reading error messages: traceback structure, error codes, suggested fixes",
            "reading changelog/migration guides: breaking changes, deprecation notices",
            "consulting reference during coding: searching docs for the right function",
            "example-driven learning: extracting patterns from code examples in docs",
            "cross-referencing: combining info from multiple doc sections to solve a problem",
            "reading source code as docs: navigating open source repos, reading tests as specs",
            "writing good docs: how to write clear docstrings, README, API references",
        ],
    },
    "debugging": {
        "name": "Debugging",
        "topics": [
            "reading Python tracebacks: line numbers, call chain, exception type, message",
            "reading JavaScript errors: TypeError, ReferenceError, stack trace in Node.js",
            "reading Rust compiler errors: borrow checker messages, lifetime annotations, suggestions",
            "reading SQL errors: syntax error position, constraint violation, deadlock info",
            "systematic debugging: reproduce → isolate → identify → fix → verify cycle",
            "binary search debugging: bisecting code changes to find the bug",
            "print debugging: strategic placement, formatting output, removing after fix",
            "rubber duck debugging: explaining the problem out loud, step-by-step walkthrough",
            "common bug patterns: off-by-one, null reference, infinite loop, race condition",
            "type errors across languages: Python dynamic typing pitfalls, JS coercion, Rust type mismatch",
            "logic errors: wrong condition, swapped arguments, missing edge case",
            "performance debugging: profiling, bottleneck identification, algorithmic vs constant factor",
            "memory debugging: leaks, dangling references, Rust ownership as prevention",
            "debugging with tests: writing a failing test first, regression tests",
            "logging strategy: log levels, structured logging, what to log and when",
        ],
    },
    "multilang": {
        "name": "Sintaxis Multi-Lenguaje",
        "topics": [
            "variable declaration: Python vs JS (let/const/var) vs Rust (let/let mut) vs Bash",
            "functions: def/lambda (Python) vs function/arrow (JS) vs fn (Rust) vs bash functions",
            "conditionals: if/elif/else (Python) vs if/else if/else (JS/Rust) vs if/then/fi (Bash)",
            "loops: for/while (Python) vs for/for..of/while (JS) vs for/loop/while (Rust) vs for/while (Bash)",
            "string handling: f-strings (Python) vs template literals (JS) vs format! (Rust) vs $var (Bash)",
            "collections: list/dict/set (Python) vs Array/Object/Map (JS) vs Vec/HashMap/HashSet (Rust)",
            "error handling: try/except (Python) vs try/catch (JS) vs Result/Option (Rust) vs trap (Bash)",
            "modules and imports: import (Python) vs import/require (JS) vs use/mod (Rust) vs source (Bash)",
            "classes and structs: class (Python) vs class (JS) vs struct+impl (Rust)",
            "type systems: dynamic (Python) vs dynamic+TS (JS) vs static (Rust) vs untyped (Bash)",
            "closures and lambdas: lambda (Python) vs arrow functions (JS) vs closures with move (Rust)",
            "pattern matching: match (Python 3.10+) vs switch (JS) vs match (Rust)",
            "async programming: async/await (Python) vs async/await (JS) vs async/.await (Rust)",
            "SQL basics: SELECT, INSERT, UPDATE, DELETE with WHERE, JOIN, GROUP BY, ORDER BY",
            "iterators and functional: map/filter (Python) vs map/filter (JS) vs iter().map() (Rust)",
        ],
    },
}

# Lenguajes a cubrir por pilar (no todos aplican a todos)
LANG_PER_PILLAR = {
    "logic": ["python", "pseudocode"],
    "data_structures": ["python", "javascript", "rust"],
    "patterns": ["python", "javascript", "rust"],
    "docs": ["python", "javascript", "rust", "sql", "bash"],
    "debugging": ["python", "javascript", "rust", "sql"],
    "multilang": ["python", "javascript", "rust", "sql", "bash"],
}

# ============================================================================
# System prompts por pilar
# ============================================================================

SYSTEM_PROMPTS = {
    "logic": """You are a computer science professor writing a textbook chapter on logic and reasoning
for programmers. Write clear, rigorous explanations with code examples that demonstrate the concept.
Every concept must include:
1. A clear explanation of the principle
2. A concrete code example showing it in practice
3. A step-by-step reasoning walkthrough
Use the <textbook> format. All code must be correct and executable.""",
    "data_structures": """You are a computer science professor writing a textbook on data structures.
Your explanations must be cross-language: show the same concept in multiple programming languages.
Include complexity analysis (time and space) for every operation.
Focus on WHEN to use each structure, not just HOW. Include trade-offs.
Use the <textbook> format. All code must be correct and executable.""",
    "patterns": """You are a software engineering professor writing a textbook on design patterns
and coding idioms. Show patterns across languages (Python, JavaScript, Rust).
Focus on the PROBLEM each pattern solves, then the solution.
Include real-world scenarios, not toy examples.
Use the <textbook> format. All code must be correct and executable.""",
    "docs": """You are a technical writing professor teaching developers how to read documentation.
Show real-world examples of consulting documentation to solve problems.
Demonstrate the skill of finding the right function/method by reading docs.
Include the reference snippet, then the reasoning, then the solution.
Use the <textbook> format.""",
    "debugging": """You are a debugging instructor writing a textbook on systematic debugging.
Show realistic error scenarios with actual error messages/tracebacks.
Walk through the debugging process step by step: what you see, what you think, what you try.
Include the buggy code, the error, the diagnosis AND the fix.
Use the <textbook> format. All code must be correct and executable.""",
    "multilang": """You are a polyglot programming instructor writing a textbook that teaches
concepts across multiple languages simultaneously. For each concept, show the equivalent
code in Python, JavaScript, Rust, and when applicable SQL and Bash.
Highlight key differences and common pitfalls when switching languages.
Use the <textbook> format. All code must be correct and executable.""",
}

# ============================================================================
# Prompt de generación
# ============================================================================

TEXTBOOK_FORMAT_INSTRUCTION = """
Output your response in EXACTLY this format (plain text, NO markdown code fences):

<textbook>
## Chapter: [Pillar Name]
### Section: [Topic]

[Clear explanation of the concept — 3-5 paragraphs for a textbook reader]

### Example
[Complete, working code with comments explaining each step]

### Step-by-step reasoning
[Numbered walkthrough of how to think about this problem]

### Exercise
[A problem for the reader to solve, with the solution below]

### Solution
[Complete solution with explanation]
</textbook>

RULES:
- All code must be complete and correct — no placeholders, no "..."
- Include AT LEAST one code example per language mentioned in the topic
- Explanations should be suitable for a textbook — clear, rigorous, pedagogical
- Each response should be 400-800 tokens of useful content
- Use real, practical examples — not abstract toy problems
"""


def make_textbook_prompt(pillar: str, topic: str, lang: str, seed: int) -> str:
    """Genera el prompt de usuario para un tema textbook."""
    pillar_info = PILLARS[pillar]
    lang_label = {
        "python": "Python",
        "javascript": "JavaScript/TypeScript",
        "rust": "Rust",
        "sql": "SQL",
        "bash": "Bash/Shell",
        "pseudocode": "pseudocode with Python implementation",
    }.get(lang, lang)

    variants = [
        f"""Write a textbook section about: {topic}

Primary language: {lang_label}
Pillar: {pillar_info["name"]}

{TEXTBOOK_FORMAT_INSTRUCTION}""",
        f"""Create an educational chapter section teaching: {topic}

Use {lang_label} for code examples.
This is part of the "{pillar_info["name"]}" pillar of a programming textbook.

Focus on building intuition through concrete examples and step-by-step reasoning.

{TEXTBOOK_FORMAT_INSTRUCTION}""",
        f"""Write a detailed textbook entry about: {topic}

Show code in {lang_label}. Where relevant, briefly compare with other languages.
Part of: {pillar_info["name"]}

The reader is an intermediate programmer learning to think more rigorously.

{TEXTBOOK_FORMAT_INSTRUCTION}""",
    ]

    return variants[seed % len(variants)]


# ============================================================================
# Cliente API (reutilizado de generate_textbook_data.py)
# ============================================================================


class GitHubModelsClient:
    """Cliente para GitHub Models API con retry y rate limiting."""

    def __init__(self, token: str):
        self.token = token
        self.base_url = GITHUB_API_URL
        self._request_count = 0
        self._last_request_time = 0.0
        self._min_interval = 3.0

    def generate(
        self,
        model_id: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 3000,
    ) -> Optional[str]:
        """Genera texto con un modelo de GitHub Models."""
        import urllib.request

        timeout = 120

        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        payload = json.dumps(
            {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers=headers,
            method="POST",
        )

        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                    self._last_request_time = time.time()
                    self._request_count += 1
                    return result["choices"][0]["message"]["content"]

            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "rate" in error_msg.lower():
                    wait = (attempt + 1) * 30
                    print(f"  Rate limited, esperando {wait}s...")
                    time.sleep(wait)
                elif "401" in error_msg or "403" in error_msg:
                    print(f"  Error de auth: {error_msg}")
                    return None
                else:
                    wait = (attempt + 1) * 5
                    print(f"  Error (intento {attempt + 1}/3): {error_msg}")
                    time.sleep(wait)

        return None


# ============================================================================
# Validación y limpieza
# ============================================================================


def clean_response(text: str) -> str:
    """Limpia la respuesta del modelo."""
    # Remover <think>...</think> (DeepSeek-R1)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Remover code fences markdown
    text = re.sub(r"```\w*\s*\n?", "", text)
    # Normalizar whitespace
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def validate_textbook_entry(text: str) -> dict[str, object]:
    """Valida la calidad de una entrada textbook."""
    result: dict[str, object] = {
        "valid": True,
        "has_textbook_tags": False,
        "has_code": False,
        "has_explanation": False,
        "has_example": False,
        "issues": [],
        "word_count": 0,
    }

    result["word_count"] = len(text.split())

    if "<textbook>" in text or "## Chapter" in text or "### " in text:
        result["has_textbook_tags"] = True

    code_indicators = [
        "def ",
        "function ",
        "fn ",
        "class ",
        "struct ",
        "SELECT ",
        "#!/bin/bash",
        "import ",
        "const ",
        "let ",
    ]
    if any(ind in text for ind in code_indicators):
        result["has_code"] = True
    else:
        issues = result["issues"]
        assert isinstance(issues, list)
        issues.append("no code found")

    if len(text.split()) > 100:
        result["has_explanation"] = True
    else:
        issues = result["issues"]
        assert isinstance(issues, list)
        issues.append("too short (< 100 words)")

    if "example" in text.lower() or "ejemplo" in text.lower():
        result["has_example"] = True

    wc = result["word_count"]
    assert isinstance(wc, int)
    if wc < 80:
        result["valid"] = False
        issues = result["issues"]
        assert isinstance(issues, list)
        issues.append(f"only {wc} words")

    score = sum(
        [
            bool(result["has_textbook_tags"]) * 2,
            bool(result["has_code"]) * 3,
            bool(result["has_explanation"]) * 2,
            bool(result["has_example"]) * 1,
            (wc > 200) * 1,
            (wc > 400) * 1,
        ]
    )
    result["quality_score"] = score  # max 10

    return result


# ============================================================================
# Curriculum
# ============================================================================


def build_curriculum(
    pillars: list[str],
    examples_per_pillar: int,
) -> list[dict[str, str | int]]:
    """Construye el curriculum de generación."""
    curriculum: list[dict[str, str | int]] = []
    model_keys = list(MODELS.keys())
    idx = 0

    for pillar in pillars:
        topics = PILLARS[pillar]["topics"]
        langs = LANG_PER_PILLAR[pillar]

        for i in range(examples_per_pillar):
            topic = topics[i % len(topics)]
            lang = langs[i % len(langs)]
            model_key = model_keys[idx % len(model_keys)]

            curriculum.append(
                {
                    "index": idx,
                    "pillar": pillar,
                    "topic": topic,
                    "lang": lang,
                    "model": model_key,
                }
            )
            idx += 1

    random.seed(42)
    random.shuffle(curriculum)
    return curriculum


# ============================================================================
# Pipeline
# ============================================================================


def run_generation(
    pillars: list[str],
    examples_per_pillar: int = 200,
    output_dir: str = "data/textbook_v3",
    resume: bool = False,
    dry_run: bool = False,
) -> None:
    """Ejecuta la generación de datos textbook."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token and not dry_run:
        print("Error: GITHUB_TOKEN no configurado")
        print("  $env:GITHUB_TOKEN = 'ghp_xxx'  (PowerShell)")
        sys.exit(1)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    output_file = out_path / "textbook_pretrain.jsonl"
    rejected_file = out_path / "rejected.jsonl"
    progress_file = out_path / "progress.json"

    # Resume
    completed_indices: set[int] = set()
    if resume and progress_file.exists():
        progress = json.loads(progress_file.read_text(encoding="utf-8"))
        completed_indices = set(progress.get("completed", []))
        print(f"Retomando: {len(completed_indices)} ejemplos ya generados")

    curriculum = build_curriculum(pillars, examples_per_pillar)
    total = len(curriculum)
    pending = [c for c in curriculum if c["index"] not in completed_indices]

    if dry_run:
        print(
            f"\nDRY RUN — {total} ejemplos planificados ({len(pending)} pendientes)\n"
        )
        by_pillar: dict[str, int] = {}
        by_model: dict[str, int] = {}
        by_lang: dict[str, int] = {}
        for c in curriculum:
            p = str(c["pillar"])
            m = str(c["model"])
            la = str(c["lang"])
            by_pillar[p] = by_pillar.get(p, 0) + 1
            by_model[m] = by_model.get(m, 0) + 1
            by_lang[la] = by_lang.get(la, 0) + 1

        print("Por pilar:")
        for k, v in sorted(by_pillar.items()):
            print(f"  {PILLARS[k]['name']:35s} {v:4d}")
        print("\nPor modelo:")
        for k, v in sorted(by_model.items()):
            print(f"  {k:15s} {v:4d}")
        print("\nPor lenguaje:")
        for k, v in sorted(by_lang.items()):
            print(f"  {k:15s} {v:4d}")

        print("\nPrimeros 10 ejemplos:")
        for c in curriculum[:10]:
            print(
                f"  [{c['index']:4d}] {str(c['model']):10s} | "
                f"{str(c['pillar']):15s} | {str(c['lang']):12s} | "
                f"{str(c['topic'])[:50]}"
            )
        return

    client = GitHubModelsClient(token)  # type: ignore[arg-type]

    stats = {
        "generated": len(completed_indices),
        "rejected": 0,
        "total": total,
        "by_pillar": {p: 0 for p in pillars},
        "quality_scores": [],
    }

    print(f"\nGenerando {total} ejemplos textbook ({len(pending)} pendientes)")
    print(f"Pilares: {', '.join(PILLARS[p]['name'] for p in pillars)}")
    print(f"Salida: {output_file}\n")

    start_time = time.time()

    for item in pending:
        idx = int(item["index"])
        pillar = str(item["pillar"])
        topic = str(item["topic"])
        lang = str(item["lang"])
        model_key = str(item["model"])
        model_info = MODELS[model_key]

        n_done = stats["generated"] + 1
        print(
            f"[{n_done}/{total}] {model_key:10s} | {pillar:15s} | "
            f"{lang:12s} | {topic[:45]}"
        )

        system_prompt = SYSTEM_PROMPTS[pillar]
        user_prompt = make_textbook_prompt(pillar, topic, lang, seed=idx)

        raw = client.generate(
            model_id=model_info["id"],
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=model_info["temperature"],
        )

        if not raw:
            print("  Sin respuesta, saltando")
            continue

        text = clean_response(raw)
        validation = validate_textbook_entry(text)

        qs = validation.get("quality_score", 0)
        assert isinstance(qs, int)

        if not validation["valid"] or qs < 4:
            stats["rejected"] += 1
            issues = validation.get("issues", [])
            print(f"  RECHAZADO (score={qs}): {issues}")
            with open(rejected_file, "a", encoding="utf-8") as f:
                json.dump(
                    {
                        "pillar": pillar,
                        "topic": topic,
                        "lang": lang,
                        "model": model_key,
                        "text": text,
                        "validation": {
                            k: v
                            for k, v in validation.items()
                            if k != "issues" or isinstance(v, (str, int, float, bool))
                        },
                        "issues": validation.get("issues", []),
                    },
                    f,
                    ensure_ascii=False,
                )
                f.write("\n")
            continue

        # Guardar ejemplo aprobado
        entry = {
            "text": text,
            "pillar": pillar,
            "topic": topic,
            "lang": lang,
            "model": model_key,
            "quality_score": qs,
            "word_count": validation["word_count"],
        }
        with open(output_file, "a", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

        stats["generated"] += 1
        stats["by_pillar"][pillar] = stats["by_pillar"].get(pillar, 0) + 1
        stats["quality_scores"].append(qs)
        completed_indices.add(idx)

        # Guardar progreso cada 10 ejemplos
        if stats["generated"] % 10 == 0:
            progress_file.write_text(
                json.dumps(
                    {
                        "completed": sorted(completed_indices),
                        "stats": {
                            k: v for k, v in stats.items() if k != "quality_scores"
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    # Stats finales
    elapsed = time.time() - start_time
    avg_q = (
        sum(stats["quality_scores"]) / len(stats["quality_scores"])
        if stats["quality_scores"]
        else 0
    )

    print(f"\n{'=' * 50}")
    print(f"Generacion completada en {elapsed / 60:.1f} min")
    print(f"  Generados: {stats['generated']}")
    print(f"  Rechazados: {stats['rejected']}")
    print(f"  Calidad promedio: {avg_q:.1f}/10")
    print(f"\nPor pilar:")
    for p, count in stats["by_pillar"].items():
        print(f"  {PILLARS[p]['name']:35s} {count:4d}")
    print(f"\nArchivo: {output_file}")
    print(f"{'=' * 50}")

    # Guardar progreso final
    progress_file.write_text(
        json.dumps(
            {
                "completed": sorted(completed_indices),
                "stats": {k: v for k, v in stats.items() if k != "quality_scores"},
                "elapsed_seconds": elapsed,
                "avg_quality": avg_q,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera datos textbook para continual pretrain (Fase 1)",
    )
    parser.add_argument(
        "--pillar",
        choices=list(PILLARS.keys()) + ["all"],
        default="all",
        help="Pilar a generar (default: all)",
    )
    parser.add_argument(
        "--examples-per-pillar",
        type=int,
        default=200,
        help="Ejemplos por pilar (default: 200, ~90-100K tokens/pilar)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/textbook_v3",
        help="Directorio de salida (default: data/textbook_v3)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Retomar generacion desde el ultimo checkpoint",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostrar plan sin generar",
    )
    args = parser.parse_args()

    pillars = list(PILLARS.keys()) if args.pillar == "all" else [args.pillar]

    run_generation(
        pillars=pillars,
        examples_per_pillar=args.examples_per_pillar,
        output_dir=args.output_dir,
        resume=args.resume,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
