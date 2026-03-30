"""classroom_curriculum.py — Configuración y currículo del Classroom."""

from __future__ import annotations

from dataclasses import dataclass


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
