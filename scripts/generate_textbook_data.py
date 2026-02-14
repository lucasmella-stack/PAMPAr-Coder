#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Generador de datos sintéticos "textbook quality" para PAMPAr-Coder.

Usa GitHub Models API (GPT, Gemini, Claude) como "profesores" para generar
funciones Python de alta calidad con explicaciones didácticas.

Estrategia phi-1: calidad > cantidad. Cada ejemplo incluye:
  - Explicación conceptual clara
  - Docstring con ejemplos
  - Implementación limpia
  - Tests/assertions

Uso:
  set GITHUB_TOKEN=ghp_xxx
  python scripts/generate_textbook_data.py --num-examples 1000
  python scripts/generate_textbook_data.py --num-examples 5000 --difficulty hard
  python scripts/generate_textbook_data.py --resume  # retoma desde donde quedó
"""

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
# Configuración de modelos GitHub
# ============================================================================

GITHUB_API_URL = "https://models.inference.ai.azure.com"

# Modelos disponibles en GitHub Models API (verificados 2026-02-14)
# IDs sin prefijo de proveedor — la API los requiere así
MODELS = {
    "gpt41": {
        "id": "gpt-4.1",
        "role": "algorithmic",
        "strength": "Algoritmos, edge cases, clean code, optimización",
        "temperature": 0.7,
    },
    "gpt41mini": {
        "id": "gpt-4.1-mini",
        "role": "didactic",
        "strength": "Explicaciones rápidas, variedad, buen balance calidad/velocidad",
        "temperature": 0.8,
    },
    "gpt41nano": {
        "id": "gpt-4.1-nano",
        "role": "volume",
        "strength": "Alto volumen, funciones simples, rate limit generoso",
        "temperature": 0.7,
    },
    "deepseek": {
        "id": "DeepSeek-R1",
        "role": "rigorous",
        "strength": "Razonamiento paso a paso (chain-of-thought), lógica, math",
        "temperature": 0.6,
    },
    "llama": {
        "id": "Meta-Llama-3.1-405B-Instruct",
        "role": "diverse",
        "strength": "Diversidad de estilo, open-source perspective, robustez",
        "temperature": 0.7,
    },
}

# ============================================================================
# Temas y dificultades para generación diversa
# ============================================================================

TOPICS = {
    "basics": [
        "string manipulation", "list operations", "dictionary methods",
        "set operations", "tuple unpacking", "f-strings formatting",
        "type conversion", "conditional expressions", "loop patterns",
        "basic file operations", "string slicing", "list comprehensions",
    ],
    "data_structures": [
        "stack implementation", "queue implementation", "linked list",
        "binary tree traversal", "hash map", "heap/priority queue",
        "graph representation", "trie", "deque operations",
        "doubly linked list", "circular buffer", "LRU cache",
    ],
    "algorithms": [
        "binary search", "merge sort", "quick sort", "BFS", "DFS",
        "dynamic programming - fibonacci", "dynamic programming - knapsack",
        "greedy algorithms", "two pointer technique", "sliding window",
        "backtracking", "divide and conquer", "topological sort",
    ],
    "math": [
        "prime numbers", "GCD/LCM", "matrix operations", "combinatorics",
        "modular arithmetic", "number theory", "statistics functions",
        "polynomial evaluation", "numerical methods", "geometry basics",
        "probability calculations", "base conversion",
    ],
    "functional": [
        "map/filter/reduce", "decorators", "generators", "itertools usage",
        "closures", "partial functions", "lambda expressions",
        "recursive patterns", "memoization", "higher-order functions",
        "context managers", "custom iterators",
    ],
    "text_processing": [
        "regex patterns", "CSV parsing", "JSON handling", "text tokenization",
        "string matching algorithms", "text normalization", "encoding/decoding",
        "template formatting", "markdown parsing basics", "log parsing",
        "word frequency counting", "text similarity",
    ],
    "practical": [
        "input validation", "error handling patterns", "configuration parsing",
        "CLI argument processing", "date/time manipulation", "path handling",
        "environment variables", "retry logic", "rate limiting",
        "caching strategies", "batching operations", "pagination",
    ],
}

DIFFICULTIES = {
    "easy": {
        "description": "funciones simples, 5-15 líneas, un solo concepto",
        "complexity": "O(n) o mejor, sin recursión compleja",
    },
    "medium": {
        "description": "funciones con múltiples pasos, 15-40 líneas, combina 2-3 conceptos",
        "complexity": "puede incluir recursión, estructuras auxiliares",
    },
    "hard": {
        "description": "funciones complejas, 30-80 líneas, edge cases, optimización",
        "complexity": "algoritmos avanzados, manejo de errores robusto, tests exhaustivos",
    },
}

# ============================================================================
# Prompts de generación (el corazón de la calidad)
# ============================================================================

SYSTEM_PROMPT = """You are an expert Python programming instructor writing a textbook.
Your code examples must be:
1. COMPLETE - every function must be fully implemented, never use 'pass' or '...'
2. CORRECT - the code must actually work if executed
3. CLEAN - follow PEP 8, meaningful variable names, no unnecessary complexity
4. DOCUMENTED - clear docstrings with Args, Returns, Examples (with >>> doctests)
5. TESTED - include assert statements that verify correctness

You write in a teaching style: explain the concept, show the code, verify with tests.
All responses must be pure Python code that can be executed directly.
NEVER use markdown formatting, code fences, or explanations outside of Python comments/docstrings."""

def make_generation_prompt(topic: str, difficulty: str, seed: int) -> str:
    """Crea un prompt para generar un ejemplo textbook quality."""
    diff_info = DIFFICULTIES[difficulty]
    
    # Variedad en el tipo de prompt para evitar repetición
    prompt_variants = [
        # Variant 1: Función directa
        f"""Write a Python function related to "{topic}" at {difficulty} difficulty level.
{diff_info['description']}. Complexity: {diff_info['complexity']}.

The output must follow EXACTLY this format (pure Python, no markdown):

# Topic: {topic}
# Difficulty: {difficulty}
# Concept: [brief concept explanation as comment]

def function_name(param1: type, param2: type) -> return_type:
    \"\"\"Brief description.
    
    [2-3 sentence explanation of the approach/algorithm]
    
    Args:
        param1: description
        param2: description
    
    Returns:
        description
    
    Examples:
        >>> function_name(example_input)
        expected_output
    \"\"\"
    # Step-by-step implementation with comments
    ...

# Tests
assert function_name(test_input1) == expected1
assert function_name(test_input2) == expected2
assert function_name(edge_case) == expected3
print("All tests passed!")""",

        # Variant 2: Problema con múltiples funciones relacionadas
        f"""Write 2-3 related Python functions about "{topic}" ({difficulty} level).
{diff_info['description']}.

Format (pure Python, no markdown):

# Topic: {topic}
# Concept: [what these functions teach together]

def helper_function(...):
    \"\"\"Helper with docstring and examples.\"\"\"
    ...

def main_function(...):
    \"\"\"Main function that may use the helper. Full docstring with >>> examples.\"\"\"
    ...

# Comprehensive tests
assert main_function(...) == ...
assert helper_function(...) == ...
# Edge cases
assert main_function(edge) == ...
print("All tests passed!")""",

        # Variant 3: Clase simple
        f"""Write a Python class implementing a concept related to "{topic}" ({difficulty} level).
{diff_info['description']}.

Format (pure Python, no markdown):

# Topic: {topic}
# Concept: [what this class teaches]

class ClassName:
    \"\"\"Description with usage examples in docstring.
    
    Examples:
        >>> obj = ClassName(params)
        >>> obj.method(input)
        expected_output
    \"\"\"
    
    def __init__(self, ...):
        ...
    
    def method(self, ...) -> type:
        \"\"\"Method with full docstring.\"\"\"
        ...

# Tests
obj = ClassName(test_params)
assert obj.method(input1) == expected1
assert obj.method(input2) == expected2
print("All tests passed!")""",
    ]
    
    # Usar seed para determinismo pero variedad
    return prompt_variants[seed % len(prompt_variants)]


# ============================================================================
# Cliente GitHub Models API
# ============================================================================

class GitHubModelsClient:
    """Cliente para GitHub Models API con retry y rate limiting."""
    
    def __init__(self, token: str):
        self.token = token
        self.base_url = GITHUB_API_URL
        self._request_count = 0
        self._last_request_time = 0
        self._min_interval = 3.0  # segundos entre requests (rate limit safety)
    
    def generate(self, model_id: str, prompt: str, temperature: float = 0.7,
                 max_tokens: int = 2048) -> Optional[str]:
        """Genera texto con un modelo de GitHub Models."""
        import urllib.request
        
        # Timeout más largo para modelos de razonamiento (DeepSeek-R1)
        timeout = 180 if "DeepSeek" in model_id else 120
        
        # Rate limiting
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        
        payload = json.dumps({
            "model": model_id,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode("utf-8")
        
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
                    
                    content = result["choices"][0]["message"]["content"]
                    return content
                    
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "rate" in error_msg.lower():
                    wait = (attempt + 1) * 30  # 30s, 60s, 90s
                    print(f"  ⏳ Rate limited, esperando {wait}s...")
                    time.sleep(wait)
                elif "401" in error_msg or "403" in error_msg:
                    print(f"  ❌ Error de autenticación: {error_msg}")
                    print("  Verificá tu GITHUB_TOKEN")
                    return None
                else:
                    wait = (attempt + 1) * 5
                    print(f"  ⚠️ Error (intento {attempt+1}/3): {error_msg}")
                    time.sleep(wait)
        
        return None


# ============================================================================
# Validación de calidad
# ============================================================================

def validate_python_code(code: str) -> dict:
    """Valida que el código Python generado sea de calidad."""
    result = {
        "is_valid": True,
        "has_docstring": False,
        "has_tests": False,
        "has_type_hints": False,
        "compiles": False,
        "issues": [],
    }
    
    # ¿Compila?
    try:
        compile(code, "<generated>", "exec")
        result["compiles"] = True
    except SyntaxError as e:
        result["is_valid"] = False
        result["issues"].append(f"SyntaxError: {e}")
        result["quality_score"] = 0
        return result
    
    # ¿Tiene docstrings?
    if '"""' in code or "'''" in code:
        result["has_docstring"] = True
    else:
        result["issues"].append("Sin docstring")
    
    # ¿Tiene tests/assertions?
    if "assert " in code:
        result["has_tests"] = True
        # Contar assertions
        num_asserts = code.count("assert ")
        if num_asserts < 2:
            result["issues"].append(f"Solo {num_asserts} assertion(s)")
    else:
        result["issues"].append("Sin assertions/tests")
    
    # ¿Tiene type hints?
    if "->" in code or ": str" in code or ": int" in code or ": list" in code:
        result["has_type_hints"] = True
    
    # ¿Tiene def/class?
    if "def " not in code and "class " not in code:
        result["is_valid"] = False
        result["issues"].append("Sin función ni clase definida")
    
    # ¿Es demasiado corto?
    lines = [l for l in code.strip().split("\n") if l.strip() and not l.strip().startswith("#")]
    if len(lines) < 5:
        result["is_valid"] = False
        result["issues"].append(f"Muy corto: {len(lines)} líneas")
    
    # ¿Tiene 'pass' o '...' como implementación?
    for line in code.split("\n"):
        stripped = line.strip()
        if stripped == "pass" or stripped == "...":
            result["is_valid"] = False
            result["issues"].append("Contiene 'pass' o '...' sin implementación")
            break
    
    # Limpiar markdown residual
    if "```" in code:
        result["issues"].append("Contiene markdown (se limpiará)")
    
    # Calidad general
    quality_score = sum([
        result["compiles"] * 3,
        result["has_docstring"] * 2,
        result["has_tests"] * 2,
        result["has_type_hints"] * 1,
        (len(lines) >= 10) * 1,
    ])
    result["quality_score"] = quality_score  # max 9
    
    return result


def clean_generated_code(code: str) -> str:
    """Limpia código generado removiendo markdown y artefactos."""
    # Remover <think>...</think> blocks (DeepSeek-R1 chain-of-thought)
    code = re.sub(r"<think>.*?</think>", "", code, flags=re.DOTALL)
    
    # Remover code fences
    code = re.sub(r"```python\s*\n?", "", code)
    code = re.sub(r"```\s*\n?", "", code)
    
    # Remover líneas vacías excesivas (más de 2 seguidas)
    code = re.sub(r"\n{4,}", "\n\n\n", code)
    
    # Asegurar que termina con newline
    if not code.endswith("\n"):
        code += "\n"
    
    return code.strip()


# ============================================================================
# Pipeline principal de generación
# ============================================================================

def generate_curriculum(num_examples: int, difficulty: str = "mixed") -> list:
    """Genera un curriculum balanceado de temas y dificultades."""
    curriculum = []
    all_topics = []
    
    for category, topics in TOPICS.items():
        for topic in topics:
            all_topics.append((category, topic))
    
    for i in range(num_examples):
        cat, topic = all_topics[i % len(all_topics)]
        
        if difficulty == "mixed":
            # Distribución: 30% easy, 50% medium, 20% hard
            r = random.random()
            if r < 0.3:
                diff = "easy"
            elif r < 0.8:
                diff = "medium"
            else:
                diff = "hard"
        else:
            diff = difficulty
        
        # Rotar entre modelos
        model_keys = list(MODELS.keys())
        model_key = model_keys[i % len(model_keys)]
        
        curriculum.append({
            "index": i,
            "category": cat,
            "topic": topic,
            "difficulty": diff,
            "model": model_key,
        })
    
    # Shuffle para que no sea monótono (pero con seed fijo para reproducibilidad)
    random.seed(42)
    random.shuffle(curriculum)
    
    return curriculum


def run_generation(
    num_examples: int = 1000,
    difficulty: str = "mixed",
    output_dir: str = "data/textbook",
    resume: bool = False,
    models_to_use: Optional[list] = None,
    dry_run: bool = False,
):
    """Ejecuta el pipeline completo de generación."""
    
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("❌ Error: GITHUB_TOKEN no configurado")
        print("   Ejecutá: $env:GITHUB_TOKEN = 'ghp_xxx'  (PowerShell)")
        print("   O:       set GITHUB_TOKEN=ghp_xxx       (CMD)")
        sys.exit(1)
    
    # Directorio de salida
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    output_file = out_path / "textbook_python.jsonl"
    rejected_file = out_path / "rejected.jsonl"
    progress_file = out_path / "generation_progress.json"
    
    # Resumir si se pide
    completed_indices = set()
    if resume and progress_file.exists():
        progress = json.loads(progress_file.read_text())
        completed_indices = set(progress.get("completed", []))
        print(f"📂 Retomando: {len(completed_indices)} ejemplos ya generados")
    
    # Generar curriculum
    curriculum = generate_curriculum(num_examples, difficulty)
    
    # Filtrar por modelos disponibles
    if models_to_use:
        available = {m.lower() for m in models_to_use}
        # Reasignar modelos no disponibles
        for item in curriculum:
            if item["model"] not in available:
                item["model"] = random.choice(list(available))
    
    # Stats
    stats = {
        "total_requested": num_examples,
        "generated": len(completed_indices),
        "rejected": 0,
        "by_model": {k: 0 for k in MODELS},
        "by_difficulty": {"easy": 0, "medium": 0, "hard": 0},
        "by_category": {k: 0 for k in TOPICS},
        "avg_quality": 0,
        "quality_scores": [],
    }
    
    if dry_run:
        print("\n🔍 DRY RUN — preview del curriculum:")
        for item in curriculum[:20]:
            idx_s = str(item['index']).rjust(4)
            mod_s = item['model'].ljust(10)
            dif_s = item['difficulty'].ljust(6)
            cat_s = item['category'].ljust(20)
            print(f"  [{idx_s}] {mod_s} | {dif_s} | {cat_s} | {item['topic']}")
        if len(curriculum) > 20:
            print(f"\n  ... y {len(curriculum) - 20} más")
        # Model distribution
        model_dist = ", ".join(
            k + ": " + str(sum(1 for c in curriculum if c["model"] == k))
            for k in MODELS
        )
        diff_dist = ", ".join(
            d + ": " + str(sum(1 for c in curriculum if c["difficulty"] == d))
            for d in DIFFICULTIES
        )
        print(f"\n  Distribución de modelos: {model_dist}")
        print(f"  Distribución de dificultad: {diff_dist}")
        return
    
    # Cliente API
    client = GitHubModelsClient(token)
    
    print(f"\n🚀 Generando {num_examples} ejemplos textbook quality")
    print(f"   Modelos: {', '.join(MODELS.keys())}")
    print(f"   Dificultad: {difficulty}")
    print(f"   Salida: {output_file}")
    print()
    
    start_time = time.time()
    
    for item in curriculum:
        idx = item["index"]
        if idx in completed_indices:
            continue
        
        model_key = item["model"]
        model_info = MODELS[model_key]
        topic = item["topic"]
        diff = item["difficulty"]
        
        print(f"[{stats['generated']+1}/{num_examples}] {model_key:8s} | {diff:6s} | {topic}")
        
        # Generar prompt
        prompt = make_generation_prompt(topic, diff, seed=idx)
        
        # Llamar API
        raw_code = client.generate(
            model_id=model_info["id"],
            prompt=prompt,
            temperature=model_info["temperature"],
            max_tokens=2048,
        )
        
        if not raw_code:
            print(f"  ⚠️ Sin respuesta, saltando")
            continue
        
        # Limpiar
        code = clean_generated_code(raw_code)
        
        # Validar calidad
        validation = validate_python_code(code)
        
        if validation["is_valid"] and validation["quality_score"] >= 5:
            # Aceptado — guardar
            entry = {
                "text": code,
                "source": f"textbook-{model_key}",
                "topic": topic,
                "category": item["category"],
                "difficulty": diff,
                "quality_score": validation["quality_score"],
            }
            
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
            stats["generated"] += 1
            stats["by_model"][model_key] += 1
            stats["by_difficulty"][diff] += 1
            stats["by_category"][item["category"]] += 1
            stats["quality_scores"].append(validation["quality_score"])
            
            completed_indices.add(idx)
            
            quality_str = "★" * min(validation["quality_score"], 9)
            print(f"  ✅ Calidad: {quality_str} ({validation['quality_score']}/9)")
        else:
            # Rechazado — guardar para análisis
            reject_entry = {
                "code": code,
                "model": model_key,
                "topic": topic,
                "issues": validation["issues"],
                "quality_score": validation["quality_score"],
            }
            with open(rejected_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(reject_entry, ensure_ascii=False) + "\n")
            
            stats["rejected"] += 1
            print(f"  ❌ Rechazado: {', '.join(validation['issues'])}")
        
        # Guardar progreso cada 10 ejemplos
        if stats["generated"] % 10 == 0:
            progress = {
                "completed": list(completed_indices),
                "stats": stats,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            progress_file.write_text(json.dumps(progress, indent=2))
    
    # Stats finales
    elapsed = time.time() - start_time
    avg_quality = sum(stats["quality_scores"]) / max(len(stats["quality_scores"]), 1)
    
    print(f"\n{'='*60}")
    print(f"📊 Generación completada en {elapsed/60:.1f} minutos")
    print(f"   ✅ Aceptados: {stats['generated']}")
    print(f"   ❌ Rechazados: {stats['rejected']}")
    print(f"   📈 Calidad promedio: {avg_quality:.1f}/9")
    print(f"   📁 Archivo: {output_file}")
    print(f"\n   Por modelo:")
    for k, v in stats["by_model"].items():
        print(f"      {k}: {v}")
    print(f"\n   Por dificultad:")
    for k, v in stats["by_difficulty"].items():
        print(f"      {k}: {v}")
    
    # Guardar progreso final
    stats["avg_quality"] = avg_quality
    stats["elapsed_minutes"] = elapsed / 60
    progress = {
        "completed": list(completed_indices),
        "stats": stats,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    progress_file.write_text(json.dumps(progress, indent=2))


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Genera datos textbook quality para PAMPAr-Coder"
    )
    parser.add_argument(
        "--num-examples", type=int, default=1000,
        help="Número de ejemplos a generar (default: 1000)"
    )
    parser.add_argument(
        "--difficulty", choices=["easy", "medium", "hard", "mixed"],
        default="mixed",
        help="Dificultad de los ejemplos (default: mixed)"
    )
    parser.add_argument(
        "--output-dir", default="data/textbook",
        help="Directorio de salida (default: data/textbook)"
    )
    parser.add_argument(
        "--models", nargs="+", choices=list(MODELS.keys()),
        default=None,
        help="Modelos a usar (default: todos). Opciones: " + ", ".join(MODELS.keys())
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Retomar generación previa"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Solo mostrar el plan sin generar"
    )
    
    args = parser.parse_args()
    
    run_generation(
        num_examples=args.num_examples,
        difficulty=args.difficulty,
        output_dir=args.output_dir,
        resume=args.resume,
        models_to_use=args.models,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
