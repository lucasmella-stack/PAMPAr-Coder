#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Poblar Biblioteca — Clasifica los datos curados en temas de la biblioteca.

Lee todos los JSONL que ya tenemos en data/ y los distribuye en
biblioteca/<tema>/<tema>.jsonl según el contenido de cada sample.

El MotorCuriosidad puede entonces estudiarlos inmediatamente.

Uso:
  python scripts/poblar_biblioteca.py

  # Solo algunos datasets:
  python scripts/poblar_biblioteca.py --solo distillation code

  # Límite de samples por tema:
  python scripts/poblar_biblioteca.py --max-por-tema 2000
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# REGLAS DE CLASIFICACIÓN
# Cada tema tiene keywords que detectan si un sample pertenece ahí.
# La clasificación es por scoring: el tema con más keywords gana.
# =============================================================================

REGLAS: dict[str, dict] = {
    # ── Python básico ────────────────────────────────────────────────────────
    "variables_y_tipos": {
        "nivel": 1,
        "carpeta": "python_basico",
        "keywords": ["variable", "int(", "float(", "str(", "bool", "type(",
                     "= 0", "= 1", "= \"", "= '", "casting", "assignment"],
        "anti": ["def ", "class ", "for ", "while ", "import "],
    },
    "strings_y_formato": {
        "nivel": 1,
        "carpeta": "python_basico",
        "keywords": ["f\"", "f'", ".format(", ".upper(", ".lower(", ".strip(",
                     ".split(", ".join(", ".replace(", "string", "str("],
        "anti": ["class ", "import numpy", "import pandas"],
    },
    "listas_y_tuplas": {
        "nivel": 1,
        "carpeta": "python_basico",
        "keywords": ["list", "tuple", ".append(", ".extend(", ".pop(",
                     ".insert(", ".index(", "my_list", "items = [", "= []"],
        "anti": ["dict", "{}", "class ", "import "],
    },
    "diccionarios_y_sets": {
        "nivel": 2,
        "carpeta": "python_basico",
        "keywords": ["dict", "set(", ".keys()", ".values()", ".items()",
                     ".get(", "= {}", "in d[", "d = {", "my_dict"],
        "anti": ["class ", "import numpy"],
    },
    "control_de_flujo": {
        "nivel": 2,
        "carpeta": "python_basico",
        "keywords": ["if ", "elif ", "else:", "for ", "while ", "break",
                     "continue", "range(", "in range"],
        "anti": ["def ", "class "],
    },
    "comprensiones": {
        "nivel": 2,
        "carpeta": "python_basico",
        "keywords": ["for x in", "for i in", "[x for", "[i for",
                     "list comprehension", "{k: v for", "generator"],
        "anti": ["class "],
    },
    "funciones_basicas": {
        "nivel": 3,
        "carpeta": "python_basico",
        "keywords": ["def ", "return ", "parameter", "argument", "function",
                     "def add(", "def calculate(", "def get_", "def set_"],
        "anti": ["class ", "self.", "*args", "**kwargs", "lambda"],
    },
    "funciones_avanzadas": {
        "nivel": 3,
        "carpeta": "python_basico",
        "keywords": ["*args", "**kwargs", "lambda", "closure", "inner function",
                     "higher-order", "map(", "filter(", "reduce("],
        "anti": ["class "],
    },
    "decoradores": {
        "nivel": 4,
        "carpeta": "python_basico",
        "keywords": ["@", "decorator", "functools.wraps", "@property",
                     "@staticmethod", "@classmethod", "wrapper"],
        "anti": [],
    },
    "generadores": {
        "nivel": 4,
        "carpeta": "python_basico",
        "keywords": ["yield", "yield from", "generator", "next(", "iter(",
                     "StopIteration", "__iter__", "__next__"],
        "anti": [],
    },
    "context_managers": {
        "nivel": 4,
        "carpeta": "python_basico",
        "keywords": ["with open(", "with ", "__enter__", "__exit__",
                     "contextmanager", "with lock", "context manager"],
        "anti": [],
    },
    "clases_y_oop": {
        "nivel": 4,
        "carpeta": "python_basico",
        "keywords": ["class ", "self.", "__init__", "__str__", "__repr__",
                     "object", "instance", "attribute", "method"],
        "anti": ["abstract", "super()", "multiple inheritance"],
    },
    "herencia_y_polimorfismo": {
        "nivel": 4,
        "carpeta": "python_basico",
        "keywords": ["super()", "inheritance", "polymorphism", "override",
                     "class Child(", "class Sub(", "ABC", "abstractmethod"],
        "anti": [],
    },
    "manejo_de_errores": {
        "nivel": 3,
        "carpeta": "python_basico",
        "keywords": ["try:", "except", "finally:", "raise", "Exception",
                     "ValueError", "TypeError", "IndexError", "try/except"],
        "anti": [],
    },
    "modulos_y_paquetes": {
        "nivel": 3,
        "carpeta": "python_basico",
        "keywords": ["import ", "from ", "__name__", "__main__",
                     "module", "package", "sys.path", "pip install"],
        "anti": [],
    },
    "type_hints": {
        "nivel": 3,
        "carpeta": "python_basico",
        "keywords": ["-> ", ": int", ": str", ": list", ": dict",
                     "Optional[", "Union[", "List[", "Dict[", "typing"],
        "anti": [],
    },
    "async_y_await": {
        "nivel": 5,
        "carpeta": "python_basico",
        "keywords": ["async def", "await ", "asyncio", "coroutine",
                     "async for", "async with", "aiohttp"],
        "anti": [],
    },

    # ── Algoritmos ───────────────────────────────────────────────────────────
    "busqueda_binaria": {
        "nivel": 3,
        "carpeta": "algoritmos",
        "keywords": ["binary search", "bisect", "mid = ", "left", "right",
                     "sorted array", "O(log n)", "lo = ", "hi = "],
        "anti": [],
    },
    "sorting_clasico": {
        "nivel": 3,
        "carpeta": "algoritmos",
        "keywords": ["sort", "bubble sort", "merge sort", "quick sort",
                     "insertion sort", "sorted(", "comparator", "O(n log n)"],
        "anti": [],
    },
    "recursion": {
        "nivel": 4,
        "carpeta": "algoritmos",
        "keywords": ["recursive", "recursion", "base case", "call itself",
                     "fibonacci", "factorial", "return func(", "memoiz"],
        "anti": [],
    },
    "programacion_dinamica": {
        "nivel": 5,
        "carpeta": "algoritmos",
        "keywords": ["dynamic programming", "dp[", "memoization", "tabulation",
                     "knapsack", "longest common", "subproblem", "optimal substructure"],
        "anti": [],
    },
    "grafos_bfs_dfs": {
        "nivel": 5,
        "carpeta": "algoritmos",
        "keywords": ["graph", "bfs", "dfs", "breadth", "depth first",
                     "adjacency", "visited", "queue", "stack", "node", "edge"],
        "anti": [],
    },
    "arboles_binarios": {
        "nivel": 5,
        "carpeta": "algoritmos",
        "keywords": ["binary tree", "tree node", "left.val", "right.val",
                     "inorder", "preorder", "postorder", "bst", "root"],
        "anti": [],
    },
    "heap_y_cola_prioridad": {
        "nivel": 5,
        "carpeta": "algoritmos",
        "keywords": ["heap", "heapq", "priority queue", "heappush",
                     "heappop", "min heap", "max heap"],
        "anti": [],
    },
    "two_pointers": {
        "nivel": 4,
        "carpeta": "algoritmos",
        "keywords": ["two pointer", "left = 0", "right = len",
                     "while left < right", "sliding window"],
        "anti": [],
    },
    "sliding_window": {
        "nivel": 4,
        "carpeta": "algoritmos",
        "keywords": ["sliding window", "window size", "max_sum", "min_sum",
                     "subarray", "substring of length"],
        "anti": [],
    },
    "backtracking": {
        "nivel": 5,
        "carpeta": "algoritmos",
        "keywords": ["backtrack", "backtracking", "permutation", "combination",
                     "subset", "n-queens", "sudoku", "path.append"],
        "anti": [],
    },

    # ── Patrones ─────────────────────────────────────────────────────────────
    "singleton_factory": {
        "nivel": 5,
        "carpeta": "patrones_diseno",
        "keywords": ["singleton", "factory", "_instance", "get_instance",
                     "create_", "factory method", "__new__"],
        "anti": [],
    },
    "observer_strategy": {
        "nivel": 5,
        "carpeta": "patrones_diseno",
        "keywords": ["observer", "strategy", "subscribe", "notify", "listener",
                     "event", "handler", "on_event"],
        "anti": [],
    },
    "decorator_pattern": {
        "nivel": 5,
        "carpeta": "patrones_diseno",
        "keywords": ["decorator pattern", "component", "concrete component",
                     "ConcreteDecorator", "wraps component"],
        "anti": [],
    },

    # ── stdlib ───────────────────────────────────────────────────────────────
    "collections": {
        "nivel": 3,
        "carpeta": "stdlib_python",
        "keywords": ["Counter(", "defaultdict(", "OrderedDict", "deque(",
                     "namedtuple", "from collections"],
        "anti": [],
    },
    "itertools": {
        "nivel": 4,
        "carpeta": "stdlib_python",
        "keywords": ["itertools", "chain(", "product(", "permutations(",
                     "combinations(", "groupby(", "islice("],
        "anti": [],
    },
    "functools": {
        "nivel": 4,
        "carpeta": "stdlib_python",
        "keywords": ["functools", "lru_cache", "partial(", "reduce(",
                     "@cache", "wraps("],
        "anti": [],
    },
    "dataclasses": {
        "nivel": 3,
        "carpeta": "stdlib_python",
        "keywords": ["@dataclass", "dataclass", "field(", "from dataclasses",
                     "asdict(", "astuple("],
        "anti": [],
    },
    "pathlib_y_os": {
        "nivel": 3,
        "carpeta": "stdlib_python",
        "keywords": ["pathlib", "Path(", "os.path", "os.listdir", "os.makedirs",
                     "glob(", ".exists(", ".mkdir("],
        "anti": [],
    },
    "regex": {
        "nivel": 4,
        "carpeta": "stdlib_python",
        "keywords": ["import re", "re.match", "re.search", "re.findall",
                     "re.sub", "pattern", "regex", "r\""],
        "anti": [],
    },
    "unittest_pytest": {
        "nivel": 3,
        "carpeta": "stdlib_python",
        "keywords": ["import pytest", "import unittest", "def test_", "assert ",
                     "assertEqual", "assertRaises", "fixture"],
        "anti": [],
    },

    # ── Ingeniería ───────────────────────────────────────────────────────────
    "clean_code": {
        "nivel": 4,
        "carpeta": "ingenieria_software",
        "keywords": ["clean code", "refactor", "readable", "maintainable",
                     "naming", "single responsibility", "meaningful"],
        "anti": [],
    },
    "solid_principles": {
        "nivel": 5,
        "carpeta": "ingenieria_software",
        "keywords": ["solid", "single responsibility", "open/closed",
                     "liskov", "interface segregation", "dependency inversion"],
        "anti": [],
    },
    "api_design_rest": {
        "nivel": 5,
        "carpeta": "ingenieria_software",
        "keywords": ["rest", "api", "endpoint", "http", "get request",
                     "post request", "json response", "flask", "fastapi"],
        "anti": [],
    },
}

# Tema genérico cuando nada coincide
TEMA_GENERICO = "funciones_basicas"


# =============================================================================
# CLASIFICADOR
# =============================================================================

def _score_tema(texto: str, regla: dict) -> int:
    """Cuenta cuántas keywords del tema están en el texto."""
    texto_lower = texto.lower()

    # Si tiene anti-keywords, puntúa 0
    for anti in regla.get("anti", []):
        if anti.lower() in texto_lower:
            return 0

    return sum(1 for kw in regla["keywords"] if kw.lower() in texto_lower)


def clasificar_sample(texto: str) -> str:
    """
    Clasifica un sample en el tema de la biblioteca más apropiado.

    Returns:
        Nombre del tema (key de REGLAS).
    """
    mejor_tema = TEMA_GENERICO
    mejor_score = 0

    for tema, regla in REGLAS.items():
        score = _score_tema(texto, regla)
        if score > mejor_score:
            mejor_score = score
            mejor_tema = tema

    return mejor_tema


def extraer_texto(linea: str) -> str:
    """Extrae el texto de un sample JSONL (formato PAMPAr o genérico)."""
    try:
        obj = json.loads(linea)
        # Formatos soportados
        if "text" in obj:
            return obj["text"]
        if "content" in obj:
            return obj["content"]
        if "instruction" in obj and "response" in obj:
            return f"{obj['instruction']}\n{obj['response']}"
        if "question" in obj and "answer" in obj:
            return f"{obj['question']}\n{obj['answer']}"
        # Fallback: serializar todo
        return str(obj)
    except (json.JSONDecodeError, TypeError):
        return linea.strip()


def normalizar_sample(linea: str, source: str = "") -> dict:
    """Convierte un sample al formato PAMPAr estándar."""
    try:
        obj = json.loads(linea)
        texto = extraer_texto(linea)
        return {
            "text": texto,
            "source": obj.get("source", source),
            "license": obj.get("license", "open"),
            "lang": "python",
        }
    except Exception:
        return {
            "text": linea.strip(),
            "source": source,
            "license": "open",
            "lang": "python",
        }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clasifica datos curados en la biblioteca de temas"
    )
    parser.add_argument(
        "--data", type=Path, default=Path("data"),
        help="Carpeta raíz con los JSONL de entrada",
    )
    parser.add_argument(
        "--biblioteca", type=Path, default=Path("biblioteca"),
        help="Carpeta de destino (biblioteca/)",
    )
    parser.add_argument(
        "--solo", nargs="+", default=None,
        help="Solo procesar subcarpetas específicas de data/ (ej: distillation code)",
    )
    parser.add_argument(
        "--max-por-tema", type=int, default=3000,
        help="Máximo de samples por tema (default=3000)",
    )
    parser.add_argument(
        "--min-longitud", type=int, default=80,
        help="Mínimo de caracteres por sample (descarta junk)",
    )
    args = parser.parse_args()

    # Encontrar todos los JSONL de entrada
    fuentes: list[Path] = []
    for p in sorted(args.data.rglob("*.jsonl")):
        if args.solo:
            # Solo incluir si su carpeta padre está en --solo
            if not any(s in str(p) for s in args.solo):
                continue
        if p.stat().st_size == 0:
            print(f"  [SKIP] {p.name} — vacío")
            continue
        fuentes.append(p)

    if not fuentes:
        print("No se encontraron archivos JSONL. Revisa --data y --solo.")
        return

    print(f"\nFuentes encontradas: {len(fuentes)}")
    for f in fuentes:
        mb = f.stat().st_size / 1024**2
        print(f"  {f.name:<45} {mb:>7.1f} MB")

    # Contadores por tema
    contadores: dict[str, int] = defaultdict(int)
    escritores: dict[str, object] = {}
    omitidos = 0
    procesados = 0

    # Crear carpetas de la biblioteca
    for tema, regla in REGLAS.items():
        carpeta = args.biblioteca / regla["carpeta"]
        carpeta.mkdir(parents=True, exist_ok=True)

    # Abrir writers
    handles = {}
    for tema, regla in REGLAS.items():
        ruta = args.biblioteca / regla["carpeta"] / f"{tema}.jsonl"
        handles[tema] = open(ruta, "a", encoding="utf-8")

    print(f"\nClasificando y distribuyendo samples...\n")

    try:
        for fuente in fuentes:
            source_name = fuente.stem
            print(f"  Procesando: {fuente.name}")

            with open(fuente, encoding="utf-8", errors="ignore") as f:
                for linea in f:
                    linea = linea.strip()
                    if not linea:
                        continue

                    texto = extraer_texto(linea)

                    # Filtrar samples muy cortos
                    if len(texto) < args.min_longitud:
                        omitidos += 1
                        continue

                    # Clasificar
                    tema = clasificar_sample(texto)

                    # Respetar límite por tema
                    if contadores[tema] >= args.max_por_tema:
                        omitidos += 1
                        continue

                    # Escribir en formato PAMPAr
                    sample = normalizar_sample(linea, source=source_name)
                    sample["tema"] = tema  # Metadato extra
                    handles[tema].write(json.dumps(sample, ensure_ascii=False) + "\n")
                    contadores[tema] += 1
                    procesados += 1

                    if procesados % 10000 == 0:
                        print(f"    {procesados:,} samples distribuidos...")

    finally:
        for h in handles.values():
            h.close()

    # Resumen
    print(f"\n{'='*60}")
    print(f"DISTRIBUCIÓN COMPLETADA")
    print(f"{'='*60}")
    print(f"  Samples procesados: {procesados:,}")
    print(f"  Samples omitidos:   {omitidos:,}")
    print(f"\n  Por tema (top 20):")

    temas_con_datos = [(t, c) for t, c in contadores.items() if c > 0]
    temas_con_datos.sort(key=lambda x: x[1], reverse=True)

    for tema, count in temas_con_datos[:20]:
        regla = REGLAS[tema]
        print(f"    {tema:<35} {count:>5} samples  (nivel {regla['nivel']})")

    sin_datos = [t for t in REGLAS if contadores[t] == 0]
    if sin_datos:
        print(f"\n  Temas sin datos ({len(sin_datos)}): necesitan generación futura")
        for t in sin_datos[:10]:
            print(f"    - {t}")

    # Actualizar indice.json con rutas reales
    indice_path = args.biblioteca / "indice.json"
    if indice_path.exists():
        indice = json.loads(indice_path.read_text())
        # Marcar temas que tienen datos
        for cat, temas in indice.items():
            for tema_entry in temas:
                nombre = tema_entry["nombre"]
                tema_entry["tiene_datos"] = contadores.get(nombre, 0) > 0
                tema_entry["n_samples"] = contadores.get(nombre, 0)
        indice_path.write_text(json.dumps(indice, indent=2, ensure_ascii=False))
        print(f"\n  Índice actualizado: {indice_path}")

    print(f"\nListo. La biblioteca está en: {args.biblioteca}/")
    print(f"Correr el viaje intelectual:")
    print(f"  python scripts/aprender_solo.py --checkpoint checkpoints/pampar_v2_best.pt\n")


if __name__ == "__main__":
    main()
