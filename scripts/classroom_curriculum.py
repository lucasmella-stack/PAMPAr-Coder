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

# Tipos de concepto:
#   "conceptual" — sin código, lenguaje natural (etapas 0-3)
#   "bridge"     — mezcla: concepto cotidiano + correspondencia Python (etapa 4)
#   "coding"     — Python puro (etapas 5+, curriculum original)
#
# Orden de prerequisitos: cada concepto requiere dominar los anteriores en su grupo
CONCEPT_TREE: list[dict] = [
    # =========================================================================
    # ETAPA 0 — PRESENCIA Y COMUNICACIÓN (sin código)
    # El alumno aprende que existe, que puede responder, que hay un interlocutor.
    # =========================================================================
    {
        "id": "greeting",
        "name": "Saludos y presentación",
        "type": "conceptual",
        "stage": 0,
        "desc": "Intercambio de saludos. El alumno aprende a responder y articular quién es.",
        "prereqs": [],
    },
    # =========================================================================
    # ETAPA 1 — CONCEPTOS PRIMITIVOS (sin código)
    # Lo mismo que aprender letras y números antes de aprender a leer.
    # =========================================================================
    {
        "id": "concept_number",
        "name": "¿Qué es un número?",
        "type": "conceptual",
        "stage": 1,
        "desc": "Un número representa una cantidad real: 1 manzana, 3 personas, 0 lluvia.",
        "prereqs": ["greeting"],
    },
    {
        "id": "concept_sequence",
        "name": "Secuencias y orden",
        "type": "conceptual",
        "stage": 1,
        "desc": "Una secuencia es algo que viene en orden. Contar: 1, 2, 3... Los días de la semana.",
        "prereqs": ["concept_number"],
    },
    {
        "id": "concept_word",
        "name": "Palabras y significado",
        "type": "conceptual",
        "stage": 1,
        "desc": "Una palabra es un símbolo con significado: 'casa', 'rojo', 'correr'.",
        "prereqs": ["greeting"],
    },
    {
        "id": "concept_true_false",
        "name": "Verdadero y Falso",
        "type": "conceptual",
        "stage": 1,
        "desc": "Solo hay dos opciones: algo es verdad o mentira. El cielo es azul: verdad. Las vacas vuelan: mentira.",
        "prereqs": ["greeting"],
    },
    # =========================================================================
    # ETAPA 2 — LÓGICA COTIDIANA (sin código)
    # Como aprender a combinar letras en sílabas.
    # =========================================================================
    {
        "id": "concept_compare",
        "name": "Comparar cosas",
        "type": "conceptual",
        "stage": 2,
        "desc": "Mayor, menor, igual. 5 es mayor que 3. Una jirafa es más alta que un gato.",
        "prereqs": ["concept_number", "concept_word"],
    },
    {
        "id": "concept_if_then",
        "name": "Si... entonces...",
        "type": "conceptual",
        "stage": 2,
        "desc": "Una regla: SI llueve, ENTONCES llevás paraguas. SI tienes hambre, ENTONCES comés.",
        "prereqs": ["concept_true_false"],
    },
    {
        "id": "concept_repeat",
        "name": "Repetir acciones",
        "type": "conceptual",
        "stage": 2,
        "desc": "A veces repetimos lo mismo varias veces hasta que algo cambia. Contar, respirar, caminar.",
        "prereqs": ["concept_sequence"],
    },
    # =========================================================================
    # ETAPA 3 — PROCEDIMIENTOS Y ABSTRACCIÓN (sin código)
    # Como combinar sílabas en palabras y palabras en oraciones.
    # =========================================================================
    {
        "id": "concept_steps",
        "name": "Recetas y procedimientos",
        "type": "conceptual",
        "stage": 3,
        "desc": "Una receta es una lista de pasos ordenados que llevan a un resultado. Primero X, luego Y.",
        "prereqs": ["concept_sequence", "concept_if_then"],
    },
    {
        "id": "concept_variable_box",
        "name": "Cajas con nombre (variables)",
        "type": "conceptual",
        "stage": 3,
        "desc": "Una caja con etiqueta que guarda algo. La caja 'nombre' guarda 'Ana'. La caja 'edad' guarda 25.",
        "prereqs": ["concept_number", "concept_word"],
    },
    {
        "id": "concept_function_machine",
        "name": "Máquinas que procesan (funciones)",
        "type": "conceptual",
        "stage": 3,
        "desc": "Una máquina recibe algo, lo transforma, y devuelve algo. Una licuadora: recibe fruta, devuelve jugo.",
        "prereqs": ["concept_steps"],
    },
    # =========================================================================
    # ETAPA 4 — PUENTE: CONCEPTOS → CÓDIGO (mezcla de lenguaje y Python)
    # Como pasar de leer oraciones simples a leer textos más formales.
    # =========================================================================
    {
        "id": "code_intro",
        "name": "Python: instrucciones para la computadora",
        "type": "bridge",
        "stage": 4,
        "desc": "Python es cómo le escribimos instrucciones a la computadora, igual que una receta.",
        "prereqs": ["concept_function_machine", "concept_variable_box"],
    },
    {
        "id": "code_values",
        "name": "Números y texto en Python",
        "type": "bridge",
        "stage": 4,
        "desc": "Los números son iguales: 5, 3.14. El texto va entre comillas: 'hola'.",
        "prereqs": ["code_intro"],
    },
    {
        "id": "code_true_false",
        "name": "True y False en Python",
        "type": "bridge",
        "stage": 4,
        "desc": "Verdadero se escribe True. Falso se escribe False. Es lo mismo que sí/no.",
        "prereqs": ["code_intro", "concept_true_false"],
    },
    {
        "id": "code_variables",
        "name": "Variables en Python",
        "type": "bridge",
        "stage": 4,
        "desc": "edad = 25 → la caja 'edad' ahora guarda 25. nombre = 'Ana'.",
        "prereqs": ["concept_variable_box", "code_values"],
    },
    {
        "id": "code_if",
        "name": "if/else en Python",
        "type": "bridge",
        "stage": 4,
        "desc": "El 'si... entonces...' cotidiano se escribe: if condicion: ... else: ...",
        "prereqs": ["concept_if_then", "code_true_false"],
    },
    # =========================================================================
    # ETAPA 5+ — CÓDIGO PYTHON (curriculum original, tipo "coding")
    # =========================================================================
    # Nivel 1 — Fundamentos
    {
        "id": "arithmetic",
        "name": "Arithmetic operations",
        "type": "coding",
        "stage": 5,
        "desc": "suma, resta, multiplicación, división, módulo, potencia",
        "prereqs": ["code_variables"],
    },
    {
        "id": "variables_types",
        "name": "Variables and types",
        "type": "coding",
        "stage": 5,
        "desc": "int, float, str, bool, type conversion, f-strings",
        "prereqs": ["arithmetic"],
    },
    {
        "id": "conditionals",
        "name": "Conditionals",
        "type": "coding",
        "stage": 5,
        "desc": "if/elif/else, comparadores, operadores lógicos (and, or, not)",
        "prereqs": ["variables_types"],
    },
    {
        "id": "strings",
        "name": "String operations",
        "type": "coding",
        "stage": 5,
        "desc": "slicing, split, join, replace, find, lower/upper, f-strings",
        "prereqs": ["variables_types"],
    },
    {
        "id": "functions_basic",
        "name": "Basic functions",
        "type": "coding",
        "stage": 5,
        "desc": "def, parámetros, return, valores por defecto, docstrings",
        "prereqs": ["variables_types"],
    },
    # Nivel 6 — Control de flujo
    {
        "id": "loops_for",
        "name": "For loops",
        "type": "coding",
        "stage": 6,
        "desc": "for, range, enumerate, iteración sobre secuencias",
        "prereqs": ["functions_basic", "conditionals"],
    },
    {
        "id": "loops_while",
        "name": "While loops",
        "type": "coding",
        "stage": 6,
        "desc": "while, break, continue, centinela, acumulador",
        "prereqs": ["loops_for"],
    },
    {
        "id": "lists",
        "name": "Lists",
        "type": "coding",
        "stage": 6,
        "desc": "crear, indexar, append, extend, slicing, list comprehensions",
        "prereqs": ["loops_for"],
    },
    {
        "id": "tuples_sets",
        "name": "Tuples and sets",
        "type": "coding",
        "stage": 6,
        "desc": "tuplas inmutables, sets, operaciones de conjuntos",
        "prereqs": ["lists"],
    },
    {
        "id": "dicts",
        "name": "Dictionaries",
        "type": "coding",
        "stage": 6,
        "desc": "crear, acceder, items, keys, values, dict comprehensions",
        "prereqs": ["lists"],
    },
    # Nivel 7 — Funciones avanzadas
    {
        "id": "recursion",
        "name": "Recursion",
        "type": "coding",
        "stage": 7,
        "desc": "caso base, caso recursivo, stack de llamadas, fibonacci, factorial",
        "prereqs": ["functions_basic", "conditionals"],
    },
    {
        "id": "higher_order",
        "name": "Higher-order functions",
        "type": "coding",
        "stage": 7,
        "desc": "map, filter, reduce, lambda, funciones como argumento",
        "prereqs": ["functions_basic", "lists"],
    },
    {
        "id": "generators",
        "name": "Generators",
        "type": "coding",
        "stage": 7,
        "desc": "yield, generadores, iteradores, lazy evaluation",
        "prereqs": ["loops_for", "functions_basic"],
    },
    {
        "id": "error_handling",
        "name": "Error handling",
        "type": "coding",
        "stage": 7,
        "desc": "try/except/finally, raise, excepciones custom",
        "prereqs": ["functions_basic"],
    },
    # Nivel 8 — OOP
    {
        "id": "classes_basic",
        "name": "Classes",
        "type": "coding",
        "stage": 8,
        "desc": "class, __init__, self, atributos, métodos",
        "prereqs": ["functions_basic", "dicts"],
    },
    {
        "id": "inheritance",
        "name": "Inheritance",
        "type": "coding",
        "stage": 8,
        "desc": "herencia, super(), override, polimorfismo",
        "prereqs": ["classes_basic"],
    },
    {
        "id": "dunder_methods",
        "name": "Dunder methods",
        "type": "coding",
        "stage": 8,
        "desc": "__str__, __repr__, __len__, __add__, __eq__, __iter__",
        "prereqs": ["classes_basic"],
    },
    # Nivel 9 — Avanzado
    {
        "id": "decorators",
        "name": "Decorators",
        "type": "coding",
        "stage": 9,
        "desc": "decoradores, functools.wraps, patrones de decorador",
        "prereqs": ["higher_order"],
    },
    {
        "id": "context_managers",
        "name": "Context managers",
        "type": "coding",
        "stage": 9,
        "desc": "with, __enter__/__exit__, contextlib",
        "prereqs": ["classes_basic", "error_handling"],
    },
    {
        "id": "algorithms",
        "name": "Algorithms",
        "type": "coding",
        "stage": 9,
        "desc": "sorting, searching, complejidad, divide and conquer",
        "prereqs": ["recursion", "lists"],
    },
    {
        "id": "file_io",
        "name": "File I/O",
        "type": "coding",
        "stage": 9,
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

    def select_next_concept(self, max_drills: int = 6) -> str:
        """Elige el siguiente concepto a enseñar.

        Prioridad:
        1. Nuevos conceptos disponibles (prereqs cumplidos) — siempre introduce al menos uno nuevo
           antes de volver a reforzar, una vez que el concepto anterior supera max_drills intentos.
        2. Conceptos con intentos pero no dominados Y con < max_drills intentos (reforzar)
        3. Nuevos conceptos cuyos prereqs están cumplidos (avanzar)
        4. Conceptos dominados para repaso espaciado

        max_drills: máximo de intentos seguidos en un concepto sin haberlo dominado
                    antes de forzar la introducción de uno nuevo.
        """
        # Separar conceptos que necesitan refuerzo y los que ya se han taladrado mucho
        drillable = []
        overdrilled = []
        for concept in CONCEPT_TREE:
            cid = concept["id"]
            c = self.concepts[cid]
            if c["total"] > 0 and not self.is_mastered(cid):
                if c["total"] < max_drills:
                    drillable.append((cid, self.mastery(cid)))
                else:
                    overdrilled.append((cid, self.mastery(cid)))

        # Conceptos nuevos disponibles (prereqs cumplidos, no intentados)
        available_new = [
            concept["id"]
            for concept in CONCEPT_TREE
            if self.concepts[concept["id"]]["total"] == 0
            and self.prereqs_met(concept["id"])
        ]

        # Si hay conceptos taladrados en exceso (≥ max_drills sin dominar)
        # forzar introducción de algo nuevo antes de volver a drillarlo
        if overdrilled and available_new:
            return available_new[0]

        # Reforzar conceptos con pocos intentos (todavía útil)
        if drillable:
            drillable.sort(key=lambda x: x[1])
            return drillable[0][0]

        # Avanzar a nuevo concepto aunque los anteriores no estén dominados
        if available_new:
            return available_new[0]

        # Conceptos overdrilled: volver con ellos una vez agotados los nuevos
        if overdrilled:
            overdrilled.sort(key=lambda x: x[1])
            return overdrilled[0][0]

        # Repaso espaciado de conceptos dominados
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
    # Etapa 0: presencia
    "greeting": 0,
    # Etapa 1: conceptos primitivos
    "concept_number": 1,
    "concept_sequence": 1,
    "concept_word": 1,
    "concept_true_false": 1,
    # Etapa 2: lógica cotidiana
    "concept_compare": 2,
    "concept_if_then": 2,
    "concept_repeat": 2,
    # Etapa 3: procedimientos
    "concept_steps": 3,
    "concept_variable_box": 3,
    "concept_function_machine": 3,
    # Etapa 4: puente
    "code_intro": 4,
    "code_values": 4,
    "code_true_false": 4,
    "code_variables": 4,
    "code_if": 4,
    # Etapa 5+: código Python
    "arithmetic": 5,
    "variables_types": 5,
    "conditionals": 5,
    "strings": 5,
    "functions_basic": 5,
    "loops_for": 6,
    "loops_while": 6,
    "lists": 6,
    "tuples_sets": 6,
    "dicts": 6,
    "recursion": 7,
    "higher_order": 7,
    "generators": 7,
    "error_handling": 7,
    "classes_basic": 8,
    "inheritance": 8,
    "dunder_methods": 8,
    "decorators": 9,
    "context_managers": 9,
    "algorithms": 9,
    "file_io": 9,
}


def concept_level(concept_id: str) -> int:
    """Mapea concept_id a nivel del curriculum (0-9).

    0 = presencia, 1-3 = conceptual, 4 = puente, 5-9 = código Python.
    """
    return _LEVEL_MAP.get(concept_id, 5)
