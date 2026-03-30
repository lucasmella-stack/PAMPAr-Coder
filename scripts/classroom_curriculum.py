"""classroom_curriculum.py — Configuración, árbol de conceptos y perfil del alumno."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class ClassroomConfig:
    """Configuración del aula."""

    # Modelo
    checkpoint_in: str = "checkpoints/v3_ghidra_v9.pt"
    checkpoint_out: str = "checkpoints/v3_classroom.pt"
    device: str = "auto"

    # Teacher
    teacher_backend: str = "github"  # "github" | "openrouter"
    teacher_model: str = "openai/gpt-4o-mini"
    api_key: str = ""

    # Entrenamiento bio-inspirado
    lr_base: float = 5e-6  # LR base (conservador)
    lr_llaves_mult: float = 0.01  # LLAVES/Tálamo: 1% del LR base
    lr_attn_mult: float = 0.1  # Atención: 10% del LR base
    lr_embed_mult: float = 0.1  # Embeddings: 10% del LR base
    lr_ffn_mult: float = 1.0  # FFN/StreamFFN: 100% del LR base

    # EWC
    ewc_lambda: float = 500.0  # Fuerza de la penalización EWC
    ewc_samples: int = 200  # Muestras para calcular Fisher

    # Replay buffer
    replay_size: int = 100  # Tamaño del buffer
    replay_ratio: float = 0.5  # 50% replay, 50% nuevo

    # Curriculum
    start_level: int = 1  # Nivel inicial (1-5)
    advance_threshold: float = 0.7  # 70% correcto para avanzar
    window_size: int = 10  # Ventana para calcular accuracy

    # Sesión
    max_lessons: int = 200  # Máximo de lecciones por sesión
    guardar_cada: int = 20  # Guardar checkpoint cada N lecciones
    seq_len: int = 256  # Longitud máx de secuencia para training

    # Bio-inspired mechanisms
    bio_enabled: bool = True  # Activar mecanismos bio-inspirados
    sleep_every: int = 15  # Consolidación de sueño cada N lecciones
    prune_every: int = 30  # Poda sináptica cada N lecciones

    # Server
    port: int = 8888

    # Recording
    record: bool = True  # Grabar sesión como video reproducible


CURRICULUM: dict[int, dict] = {
    1: {
        "nombre": "Fundamentos",
        "desc": "Variables, funciones simples, operaciones básicas",
        "ejercicios": [
            "Write a Python function `suma(a, b)` that returns the sum of two numbers.",
            "Write a Python function `es_par(n)` that returns True if n is even, False otherwise.",
            "Write a Python function `longitud(texto)` that returns the length of a string without using len().",
            "Write a Python function `invertir(texto)` that returns the reversed string.",
            "Write a Python function `contar_vocales(texto)` that counts vowels (a,e,i,o,u) case-insensitive.",
            "Write a Python function `suma_digitos(n)` that returns the sum of all digits of a non-negative integer.",
            "Write a Python function `es_palindromo(s)` that returns True if the string is a palindrome.",
            "Write a Python function `maximo(a, b)` that returns the larger of two numbers without using max().",
            "Write a Python function `absoluto(n)` that returns the absolute value without using abs().",
            "Write a Python function `celsius_a_fahrenheit(c)` that converts Celsius to Fahrenheit.",
            "Write a Python function `factorial(n)` that returns n! using a loop.",
            "Write a Python function `potencia(base, exp)` that returns base**exp using a loop.",
            "Write a Python function `duplicar_lista(lst)` that returns a new list with each element doubled.",
            "Write a Python function `minimo_lista(lst)` that returns the smallest element without using min().",
            "Write a Python function `contar_mayusculas(texto)` that counts uppercase letters in a string.",
        ],
    },
    2: {
        "nombre": "Estructuras de control",
        "desc": "Loops, condicionales, listas, diccionarios",
        "ejercicios": [
            "Write a Python function `fizzbuzz(n)` that returns 'FizzBuzz' if n divisible by 3 and 5, 'Fizz' if by 3, 'Buzz' if by 5, else str(n).",
            "Write a Python function `fibonacci(n)` that returns the n-th Fibonacci number (0-indexed).",
            "Write a Python function `frecuencia(lista)` that returns a dict mapping each element to its count.",
            "Write a Python function `aplanar(lista)` that flattens a list of lists by one level.",
            "Write a Python function `cuadrados_pares(n)` that returns squares of all even numbers from 2 to n.",
            "Write a Python function `invertir_dict(d)` that returns a new dict with keys and values swapped.",
            "Write a Python function `busqueda_lineal(lista, objetivo)` that returns the index or -1 if not found.",
            "Write a Python function `eliminar_duplicados(lista)` that returns a list without duplicates, preserving order.",
            "Write a Python function `es_primo(n)` that returns True if n is prime.",
            "Write a Python function `ordenar_burbuja(lista)` that sorts a list using bubble sort.",
            "Write a Python function `interseccion(a, b)` that returns elements common to both lists.",
            "Write a Python function `rotar_lista(lst, k)` that rotates list left by k positions.",
        ],
    },
    3: {
        "nombre": "Funciones avanzadas",
        "desc": "Recursión, generadores, comprensiones complejas",
        "ejercicios": [
            "Write a Python function `merge_sort(lista)` that returns a new sorted list using merge sort.",
            "Write a Python function `busqueda_binaria(lista, objetivo)` that returns the index or -1.",
            "Write a Python function `primos_hasta(n)` that yields all primes up to n using a generator.",
            "Write a Python function `memoize(fn)` that returns a cached version of fn.",
            "Write a Python function `aplanar_profundo(lst)` that recursively flattens nested lists.",
            "Write a Python function `permutaciones(lst)` that returns all permutations of a list.",
            "Write a Python function `cifrado_cesar(texto, k)` that shifts each letter by k positions.",
            "Write a Python function `potencia_recursiva(base, exp)` that calculates power recursively.",
            "Write a Python function `torre_hanoi(n, origen, destino, auxiliar)` that prints the moves.",
            "Write a Python function `zip_manual(a, b)` that zips two lists without using zip().",
        ],
    },
    4: {
        "nombre": "Clases y OOP",
        "desc": "Clases, herencia, métodos especiales",
        "ejercicios": [
            "Write a Python class `Stack` with methods `push(item)`, `pop()`, `is_empty()`, `peek()`.",
            "Write a Python class `Punto` with x, y attributes and a method `distancia(otro)` for Euclidean distance.",
            "Write a Python class `Cola` implementing a FIFO queue with `enqueue(item)` and `dequeue()`.",
            "Write a Python class `Fraccion` with add, sub, mul, and __str__ using GCD simplification.",
            "Write a Python class `Contador` that counts how many times it has been called (using __call__).",
            "Write a Python class `Vector` with __add__, __sub__, __mul__ (scalar) and __repr__.",
            "Write a Python class `ListaEnlazada` with `agregar(valor)`, `buscar(valor)`, `__len__`.",
            "Write a Python class `Matriz` with __add__ and __mul__ for 2D matrix operations.",
        ],
    },
    5: {
        "nombre": "Patrones avanzados",
        "desc": "Decoradores, context managers, algoritmos complejos",
        "ejercicios": [
            "Write a Python decorator `cronometrar` that prints how long a function takes to execute.",
            "Write a Python context manager class `TempFile` that creates a temp file and deletes it on exit.",
            "Write a Python function `lru_cache(maxsize)` decorator that caches the last maxsize unique calls.",
            "Write a Python function `dijkstra(grafo, inicio)` that returns shortest distances from inicio.",
            "Write a Python async function `fetch_all(urls)` that fetches URLs concurrently with asyncio.",
            "Write a Python function `quick_sort(lista)` implementing quicksort with median-of-three pivot.",
        ],
    },
}


# =============================================================================
# Árbol de conceptos — el mentor elige qué enseñar basándose en esto
# =============================================================================

# Orden de prerequisitos: cada concepto requiere dominar los anteriores en su grupo
CONCEPT_TREE: list[dict[str, str | list[str]]] = [
    # Nivel 1 — Fundamentos
    {
        "id": "arithmetic",
        "name": "Arithmetic operations",
        "desc": "suma, resta, multiplicación, división, módulo, potencia",
        "prereqs": [],
    },
    {
        "id": "variables_types",
        "name": "Variables and types",
        "desc": "int, float, str, bool, type conversion, f-strings",
        "prereqs": ["arithmetic"],
    },
    {
        "id": "conditionals",
        "name": "Conditionals",
        "desc": "if/elif/else, comparadores, operadores lógicos (and, or, not)",
        "prereqs": ["variables_types"],
    },
    {
        "id": "strings",
        "name": "String operations",
        "desc": "slicing, split, join, replace, find, lower/upper, f-strings",
        "prereqs": ["variables_types"],
    },
    {
        "id": "functions_basic",
        "name": "Basic functions",
        "desc": "def, parámetros, return, valores por defecto, docstrings",
        "prereqs": ["variables_types"],
    },
    # Nivel 2 — Control de flujo
    {
        "id": "loops_for",
        "name": "For loops",
        "desc": "for, range, enumerate, iteración sobre secuencias",
        "prereqs": ["functions_basic", "conditionals"],
    },
    {
        "id": "loops_while",
        "name": "While loops",
        "desc": "while, break, continue, centinela, acumulador",
        "prereqs": ["loops_for"],
    },
    {
        "id": "lists",
        "name": "Lists",
        "desc": "crear, indexar, append, extend, slicing, list comprehensions",
        "prereqs": ["loops_for"],
    },
    {
        "id": "tuples_sets",
        "name": "Tuples and sets",
        "desc": "tuplas inmutables, sets, operaciones de conjuntos",
        "prereqs": ["lists"],
    },
    {
        "id": "dicts",
        "name": "Dictionaries",
        "desc": "crear, acceder, items, keys, values, dict comprehensions",
        "prereqs": ["lists"],
    },
    # Nivel 3 — Funciones avanzadas
    {
        "id": "recursion",
        "name": "Recursion",
        "desc": "caso base, caso recursivo, stack de llamadas, fibonacci, factorial",
        "prereqs": ["functions_basic", "conditionals"],
    },
    {
        "id": "higher_order",
        "name": "Higher-order functions",
        "desc": "map, filter, reduce, lambda, funciones como argumento",
        "prereqs": ["functions_basic", "lists"],
    },
    {
        "id": "generators",
        "name": "Generators",
        "desc": "yield, generadores, iteradores, lazy evaluation",
        "prereqs": ["loops_for", "functions_basic"],
    },
    {
        "id": "error_handling",
        "name": "Error handling",
        "desc": "try/except/finally, raise, excepciones custom",
        "prereqs": ["functions_basic"],
    },
    # Nivel 4 — OOP
    {
        "id": "classes_basic",
        "name": "Classes",
        "desc": "class, __init__, self, atributos, métodos",
        "prereqs": ["functions_basic", "dicts"],
    },
    {
        "id": "inheritance",
        "name": "Inheritance",
        "desc": "herencia, super(), override, polimorfismo",
        "prereqs": ["classes_basic"],
    },
    {
        "id": "dunder_methods",
        "name": "Dunder methods",
        "desc": "__str__, __repr__, __len__, __add__, __eq__, __iter__",
        "prereqs": ["classes_basic"],
    },
    # Nivel 5 — Avanzado
    {
        "id": "decorators",
        "name": "Decorators",
        "desc": "decoradores, functools.wraps, patrones de decorador",
        "prereqs": ["higher_order"],
    },
    {
        "id": "context_managers",
        "name": "Context managers",
        "desc": "with, __enter__/__exit__, contextlib",
        "prereqs": ["classes_basic", "error_handling"],
    },
    {
        "id": "algorithms",
        "name": "Algorithms",
        "desc": "sorting, searching, complejidad, divide and conquer",
        "prereqs": ["recursion", "lists"],
    },
    {
        "id": "file_io",
        "name": "File I/O",
        "desc": "open, read, write, with, json, csv",
        "prereqs": ["error_handling", "strings"],
    },
]

# Lookup rápido por id
_CONCEPT_BY_ID = {c["id"]: c for c in CONCEPT_TREE}


# =============================================================================
# StudentProfile — tracking de qué sabe el alumno
# =============================================================================


class StudentProfile:
    """Perfil adaptativo del alumno: trackea dominio por concepto."""

    def __init__(self) -> None:
        # concept_id → {"correct": int, "total": int, "last_errors": [str]}
        self.concepts: dict[str, dict] = defaultdict(
            lambda: {"correct": 0, "total": 0, "last_errors": []}
        )
        self.lesson_count: int = 0
        self.total_correct: int = 0

    def record(self, concept_id: str, correct: bool, error_desc: str = "") -> None:
        """Registra un intento del alumno en un concepto."""
        c = self.concepts[concept_id]
        c["total"] += 1
        if correct:
            c["correct"] += 1
        elif error_desc:
            c["last_errors"] = (c["last_errors"] + [error_desc])[-3:]
        self.lesson_count += 1
        if correct:
            self.total_correct += 1

    def mastery(self, concept_id: str) -> float:
        """Porcentaje de dominio de un concepto (0.0 a 1.0)."""
        c = self.concepts[concept_id]
        if c["total"] == 0:
            return 0.0
        return c["correct"] / c["total"]

    def is_mastered(self, concept_id: str, threshold: float = 0.7) -> bool:
        """Un concepto se domina si tiene >= threshold accuracy y >= 3 intentos."""
        c = self.concepts[concept_id]
        return c["total"] >= 3 and self.mastery(concept_id) >= threshold

    def prereqs_met(self, concept_id: str) -> bool:
        """Verifica que los prerequisitos estén dominados (o no vistos aún)."""
        concept = _CONCEPT_BY_ID.get(concept_id)
        if not concept:
            return True
        for prereq in concept.get("prereqs", []):
            # Prerequisito cumplido si: dominado O nunca intentado (permitir explorar)
            c = self.concepts[prereq]
            if c["total"] > 0 and not self.is_mastered(prereq):
                return False
        return True

    def select_next_concept(self) -> str:
        """Elige el siguiente concepto a enseñar.

        Prioridad:
        1. Conceptos con intentos pero no dominados (reforzar)
        2. Nuevos conceptos cuyos prereqs están cumplidos
        3. Random de los primeros conceptos si todo es nuevo
        """
        # 1. Conceptos que necesitan refuerzo (intentados pero no dominados)
        needs_work = []
        for concept in CONCEPT_TREE:
            cid = concept["id"]
            c = self.concepts[cid]
            if c["total"] > 0 and not self.is_mastered(cid):
                needs_work.append((cid, self.mastery(cid)))

        if needs_work:
            # Priorizar los de menor mastery
            needs_work.sort(key=lambda x: x[1])
            return needs_work[0][0]

        # 2. Nuevos conceptos disponibles (prereqs cumplidos, no intentados)
        available_new = []
        for concept in CONCEPT_TREE:
            cid = concept["id"]
            if self.concepts[cid]["total"] == 0 and self.prereqs_met(cid):
                available_new.append(cid)

        if available_new:
            return available_new[0]  # Primero en orden del árbol

        # 3. Conceptos dominados para consolidar (repaso espaciado)
        mastered = [c["id"] for c in CONCEPT_TREE if self.is_mastered(c["id"])]
        if mastered:
            return random.choice(mastered)

        # Fallback
        return CONCEPT_TREE[0]["id"]

    def summary(self) -> str:
        """Genera un resumen textual para el mentor."""
        lines = [
            f"Lessons completed: {self.lesson_count}, "
            f"Overall accuracy: {self.total_correct}/{self.lesson_count} "
            f"({100 * self.total_correct / max(1, self.lesson_count):.0f}%)"
        ]

        mastered = []
        struggling = []
        untouched = []

        for concept in CONCEPT_TREE:
            cid = concept["id"]
            c = self.concepts[cid]
            if c["total"] == 0:
                untouched.append(concept["name"])
            elif self.is_mastered(cid):
                mastered.append(concept["name"])
            else:
                pct = 100 * self.mastery(cid)
                info = f"{concept['name']} ({pct:.0f}%)"
                if c["last_errors"]:
                    info += f" — errors: {'; '.join(c['last_errors'][-2:])}"
                struggling.append(info)

        if mastered:
            lines.append(f"Mastered: {', '.join(mastered)}")
        if struggling:
            lines.append(f"Struggling: {', '.join(struggling)}")
        if untouched:
            lines.append(f"Not yet taught: {', '.join(untouched[:5])}")

        return "\n".join(lines)


# ── Utilidades ──────────────────────────────────────────────────────────

_LEVEL_MAP: dict[str, int] = {
    "arithmetic": 1, "variables_types": 1, "conditionals": 1,
    "strings": 1, "functions_basic": 1,
    "loops_for": 2, "loops_while": 2, "lists": 2,
    "tuples_sets": 2, "dicts": 2,
    "recursion": 3, "higher_order": 3, "generators": 3,
    "error_handling": 3,
    "classes_basic": 4, "inheritance": 4, "dunder_methods": 4,
    "decorators": 5, "context_managers": 5, "algorithms": 5,
    "file_io": 5,
}


def concept_level(concept_id: str) -> int:
    """Mapea concept_id a nivel del curriculum (1-5)."""
    return _LEVEL_MAP.get(concept_id, 1)
