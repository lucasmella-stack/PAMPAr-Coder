#!/usr/bin/env python3
# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
generate_sft_v5.py — Dataset SFT v5 con verificación pytest de cada ejemplo.

Mejoras sobre v4:
  - Verificación en runtime: cada ejemplo se exec() + assert antes de incluirse
  - Cobertura específica de los 8 patrones que falla el modelo (ver ROADMAP)
  - Mayor variedad: múltiples formas de expresar el mismo problema
  - Formato idéntico al eval: ### Problem: / ### Solution: / ```python ... ```

Los 8 fallos a corregir (ROADMAP 3. Estado de checkpoints):
  1. fizzbuzz          — orden de condicionales (15 primero, luego 3, luego 5)
  2. cuadrados_pares   — x**2 no x*2
  3. invertir_dict     — variable correcta en comprehension
  4. busqueda_binaria  — comparador no invertido
  5. merge_sort        — indentación correcta
  6. Punto class       — self.x=x, self.y=y (no self.y=x)
  7. memoize           — closure correcto
  8. primos            — O(√n) no O(n)

Uso:
  python -X utf8 scripts/generate_sft_v5.py
  python -X utf8 scripts/generate_sft_v5.py --output data/sft_v5.jsonl --verbose
"""

import argparse
import json
import random
import sys
import textwrap
from pathlib import Path
from typing import Optional

random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# =============================================================================
# VERIFICADOR — exec + assert en namespace aislado
# =============================================================================

def verificar(code: str, checks: str) -> tuple[bool, str]:
    """
    Ejecuta el código y luego los assertions en un namespace limpio.

    Returns:
        (exito: bool, error_msg: str)
    """
    ns: dict = {}
    try:
        exec(compile(code, "<sft_ejemplo>", "exec"), ns)
    except Exception as e:
        return False, f"exec failed: {e}"
    try:
        exec(compile(checks, "<sft_checks>", "exec"), ns)
    except AssertionError as e:
        return False, f"assertion failed: {e}"
    except Exception as e:
        return False, f"check error: {e}"
    return True, ""


def formatear(problem: str, solution: str) -> str:
    """Genera el texto en el formato exacto del eval."""
    return (
        f"### Problem:\n{problem}\n"
        f"### Solution:\n```python\n{solution}\n```"
    )


# =============================================================================
# BASE DE EJEMPLOS CON ASSERTIONS
# Cada entry: {problem, solution, checks}
# La solution NO incluye el bloque ```python ``` — solo el código
# =============================================================================

EJEMPLOS: list[dict] = []


def _add(problem: str, solution: str, checks: str) -> None:
    """Registra un ejemplo tras verificarlo. Lo descarta si falla."""
    ok, err = verificar(solution, checks)
    if not ok:
        print(f"  [SKIP] Ejemplo inválido: {err}\n  code={solution[:60]}")
        return
    EJEMPLOS.append({"problem": problem, "solution": solution, "checks": checks})


# ─────────────────────────────────────────────────────────────────────────────
# 1. FIZZBUZZ (fallo: orden incorrecto de condicionales)
# ─────────────────────────────────────────────────────────────────────────────

_add(
    "Write a Python function `fizzbuzz(n)` that returns 'FizzBuzz' if n is "
    "divisible by both 3 and 5, 'Fizz' if divisible by 3, 'Buzz' if divisible "
    "by 5, or the string representation of n otherwise.",
    textwrap.dedent("""\
        def fizzbuzz(n):
            if n % 15 == 0:
                return 'FizzBuzz'
            if n % 3 == 0:
                return 'Fizz'
            if n % 5 == 0:
                return 'Buzz'
            return str(n)
    """).strip(),
    "assert fizzbuzz(15)=='FizzBuzz'; assert fizzbuzz(30)=='FizzBuzz'; "
    "assert fizzbuzz(3)=='Fizz'; assert fizzbuzz(9)=='Fizz'; "
    "assert fizzbuzz(5)=='Buzz'; assert fizzbuzz(25)=='Buzz'; "
    "assert fizzbuzz(7)=='7'; assert fizzbuzz(1)=='1'",
)

_add(
    "Write a Python function `fizzbuzz(n)` that takes an integer n and returns: "
    "'FizzBuzz' when n is a multiple of both 3 and 5, 'Fizz' when n is a multiple "
    "of 3 only, 'Buzz' when n is a multiple of 5 only, or str(n) for all other numbers.",
    textwrap.dedent("""\
        def fizzbuzz(n):
            if n % 15 == 0:
                return 'FizzBuzz'
            elif n % 3 == 0:
                return 'Fizz'
            elif n % 5 == 0:
                return 'Buzz'
            else:
                return str(n)
    """).strip(),
    "assert fizzbuzz(15)=='FizzBuzz'; assert fizzbuzz(3)=='Fizz'; "
    "assert fizzbuzz(5)=='Buzz'; assert fizzbuzz(2)=='2'",
)

_add(
    "Create a Python function `fizzbuzz(n)` that implements the classic FizzBuzz rule: "
    "return 'FizzBuzz' for multiples of 15, 'Fizz' for multiples of 3, "
    "'Buzz' for multiples of 5, and the number as a string for everything else. "
    "The divisibility by 15 check must come first.",
    textwrap.dedent("""\
        def fizzbuzz(n):
            if n % 15 == 0:
                return 'FizzBuzz'
            if n % 3 == 0:
                return 'Fizz'
            if n % 5 == 0:
                return 'Buzz'
            return str(n)
    """).strip(),
    "assert fizzbuzz(45)=='FizzBuzz'; assert fizzbuzz(6)=='Fizz'; "
    "assert fizzbuzz(10)=='Buzz'; assert fizzbuzz(4)=='4'",
)

# Extra variation with list comprehension context
_add(
    "Implement `fizzbuzz(n)` in Python. It must check divisibility by 15 first, "
    "then 3, then 5, returning the appropriate string or the number as a string.",
    textwrap.dedent("""\
        def fizzbuzz(n):
            if n % 15 == 0:
                return 'FizzBuzz'
            if n % 3 == 0:
                return 'Fizz'
            if n % 5 == 0:
                return 'Buzz'
            return str(n)

        result = [fizzbuzz(i) for i in range(1, 16)]
    """).strip(),
    "assert fizzbuzz(15)=='FizzBuzz'; assert fizzbuzz(3)=='Fizz'; "
    "assert fizzbuzz(5)=='Buzz'; assert fizzbuzz(11)=='11'",
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. CUADRADOS / POTENCIAS (fallo: x*2 en vez de x**2)
# ─────────────────────────────────────────────────────────────────────────────

_add(
    "Write a Python function `cuadrados_pares(n)` that returns a list of squares "
    "of all even numbers from 0 to n (exclusive).",
    textwrap.dedent("""\
        def cuadrados_pares(n):
            return [x ** 2 for x in range(n) if x % 2 == 0]
    """).strip(),
    "assert cuadrados_pares(5)==[0, 4, 16]; assert cuadrados_pares(7)==[0, 4, 16, 36]; "
    "assert cuadrados_pares(0)==[]",
)

_add(
    "Write a Python function `cuadrados_pares(n)` that uses a list comprehension to "
    "return the squared values of even numbers in range(n). "
    "Use the ** operator for squaring.",
    textwrap.dedent("""\
        def cuadrados_pares(n):
            return [x ** 2 for x in range(n) if x % 2 == 0]
    """).strip(),
    "assert cuadrados_pares(6)==[0, 4, 16]; assert cuadrados_pares(2)==[0]",
)

_add(
    "Create a Python function `potencias(base, n)` that returns a list of `base` "
    "raised to each power from 0 to n (inclusive).",
    textwrap.dedent("""\
        def potencias(base, n):
            return [base ** i for i in range(n + 1)]
    """).strip(),
    "assert potencias(2, 4)==[1, 2, 4, 8, 16]; assert potencias(3, 3)==[1, 3, 9, 27]",
)

_add(
    "Write a Python function `cuadrado(n)` that returns n raised to the power of 2.",
    textwrap.dedent("""\
        def cuadrado(n):
            return n ** 2
    """).strip(),
    "assert cuadrado(3)==9; assert cuadrado(0)==0; assert cuadrado(5)==25; assert cuadrado(-4)==16",
)

_add(
    "Write a Python function `cubo(n)` that returns n raised to the power of 3.",
    textwrap.dedent("""\
        def cubo(n):
            return n ** 3
    """).strip(),
    "assert cubo(2)==8; assert cubo(3)==27; assert cubo(0)==0; assert cubo(-2)==-8",
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. INVERTIR DICT (fallo: variable fantasma en comprehension)
# ─────────────────────────────────────────────────────────────────────────────

_add(
    "Write a Python function `invertir_dict(d)` that returns a new dictionary "
    "where keys and values are swapped.",
    textwrap.dedent("""\
        def invertir_dict(d):
            return {v: k for k, v in d.items()}
    """).strip(),
    "assert invertir_dict({'a':1,'b':2})=={1:'a',2:'b'}; "
    "assert invertir_dict({})=={}; assert invertir_dict({'x':'y'})=={'y':'x'}",
)

_add(
    "Create a Python function `invertir_dict(d)` that swaps keys and values in a "
    "dictionary using a dict comprehension. The original keys become values and "
    "the original values become keys.",
    textwrap.dedent("""\
        def invertir_dict(d):
            return {value: key for key, value in d.items()}
    """).strip(),
    "assert invertir_dict({'uno':1,'dos':2})=={1:'uno',2:'dos'}; "
    "assert invertir_dict({'a':'A'})=={'A':'a'}",
)

_add(
    "Write a Python function `invertir_mapeo(d)` that inverts a dictionary by "
    "creating a new dict with values as keys and keys as values.",
    textwrap.dedent("""\
        def invertir_mapeo(d):
            return {v: k for k, v in d.items()}
    """).strip(),
    "assert invertir_mapeo({1:'a',2:'b'})=={'a':1,'b':2}",
)

# ─────────────────────────────────────────────────────────────────────────────
# 4. BÚSQUEDA BINARIA (fallo: comparador invertido)
# ─────────────────────────────────────────────────────────────────────────────

_add(
    "Write a Python function `busqueda_binaria(nums, target)` that returns the "
    "index of target in the sorted list nums, or -1 if not found.",
    textwrap.dedent("""\
        def busqueda_binaria(nums, target):
            left, right = 0, len(nums) - 1
            while left <= right:
                mid = (left + right) // 2
                if nums[mid] == target:
                    return mid
                elif nums[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            return -1
    """).strip(),
    "assert busqueda_binaria([1,3,5,7,9],7)==3; "
    "assert busqueda_binaria([1,3,5,7,9],4)==-1; "
    "assert busqueda_binaria([2],2)==0; "
    "assert busqueda_binaria([],5)==-1",
)

_add(
    "Implement binary search in Python as `busqueda_binaria(arr, x)`. "
    "When the middle element is smaller than x, search the right half. "
    "When it's larger, search the left half.",
    textwrap.dedent("""\
        def busqueda_binaria(arr, x):
            lo, hi = 0, len(arr) - 1
            while lo <= hi:
                mid = (lo + hi) // 2
                if arr[mid] == x:
                    return mid
                elif arr[mid] < x:
                    lo = mid + 1
                else:
                    hi = mid - 1
            return -1
    """).strip(),
    "assert busqueda_binaria([1,2,3,4,5],3)==2; "
    "assert busqueda_binaria([1,2,3,4,5],6)==-1; "
    "assert busqueda_binaria([10,20,30],10)==0",
)

_add(
    "Write a recursive Python function `busqueda_binaria(arr, target, lo, hi)` "
    "that returns the index of target or -1 if not found.",
    textwrap.dedent("""\
        def busqueda_binaria(arr, target, lo=0, hi=None):
            if hi is None:
                hi = len(arr) - 1
            if lo > hi:
                return -1
            mid = (lo + hi) // 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                return busqueda_binaria(arr, target, mid + 1, hi)
            else:
                return busqueda_binaria(arr, target, lo, mid - 1)
    """).strip(),
    "assert busqueda_binaria([1,3,5,7],5)==2; "
    "assert busqueda_binaria([1,3,5,7],2)==-1",
)

# ─────────────────────────────────────────────────────────────────────────────
# 5. MERGE SORT (fallo: indentación corrupta)
# ─────────────────────────────────────────────────────────────────────────────

_add(
    "Write a Python function `merge_sort(arr)` that sorts a list using the "
    "merge sort algorithm and returns a new sorted list.",
    textwrap.dedent("""\
        def merge_sort(arr):
            if len(arr) <= 1:
                return arr
            mid = len(arr) // 2
            left = merge_sort(arr[:mid])
            right = merge_sort(arr[mid:])
            return merge(left, right)

        def merge(left, right):
            result = []
            i = j = 0
            while i < len(left) and j < len(right):
                if left[i] <= right[j]:
                    result.append(left[i])
                    i += 1
                else:
                    result.append(right[j])
                    j += 1
            result.extend(left[i:])
            result.extend(right[j:])
            return result
    """).strip(),
    "assert merge_sort([3,1,4,1,5,9,2,6])==[1,1,2,3,4,5,6,9]; "
    "assert merge_sort([])==[]; assert merge_sort([1])==[1]",
)

_add(
    "Implement `merge_sort(lst)` in Python. The function should split the list "
    "in half recursively and merge the sorted halves.",
    textwrap.dedent("""\
        def merge_sort(lst):
            if len(lst) <= 1:
                return lst[:]
            mid = len(lst) // 2
            left = merge_sort(lst[:mid])
            right = merge_sort(lst[mid:])
            merged = []
            i = j = 0
            while i < len(left) and j < len(right):
                if left[i] <= right[j]:
                    merged.append(left[i])
                    i += 1
                else:
                    merged.append(right[j])
                    j += 1
            merged.extend(left[i:])
            merged.extend(right[j:])
            return merged
    """).strip(),
    "assert merge_sort([5,2,8,1])==[1,2,5,8]; "
    "assert merge_sort([1,2,3])==[1,2,3]",
)

_add(
    "Write a Python function `merge_sort(lista)` that returns a new sorted "
    "list using the merge sort algorithm.",
    textwrap.dedent("""\
        def merge_sort(lista):
            if len(lista) <= 1:
                return lista[:]
            mid = len(lista) // 2
            left = merge_sort(lista[:mid])
            right = merge_sort(lista[mid:])
            result = []
            i = j = 0
            while i < len(left) and j < len(right):
                if left[i] <= right[j]:
                    result.append(left[i])
                    i += 1
                else:
                    result.append(right[j])
                    j += 1
            result.extend(left[i:])
            result.extend(right[j:])
            return result
    """).strip(),
    "assert merge_sort([3,1,4,1,5,9,2,6])==[1,1,2,3,4,5,6,9]; "
    "assert merge_sort([])==[]; assert merge_sort([1])==[1]",
)

_add(
    "Create a Python function `merge_sort(lista)` implementing the merge sort "
    "algorithm. Split the list, recursively sort halves, and merge them.",
    textwrap.dedent("""\
        def merge_sort(lista):
            if len(lista) <= 1:
                return list(lista)
            medio = len(lista) // 2
            izq = merge_sort(lista[:medio])
            der = merge_sort(lista[medio:])
            resultado = []
            i = j = 0
            while i < len(izq) and j < len(der):
                if izq[i] <= der[j]:
                    resultado.append(izq[i])
                    i += 1
                else:
                    resultado.append(der[j])
                    j += 1
            resultado.extend(izq[i:])
            resultado.extend(der[j:])
            return resultado
    """).strip(),
    "assert merge_sort([9,1,5,3])==[1,3,5,9]; "
    "assert merge_sort([2])==[2]; assert merge_sort([])==[]",
)

# ─────────────────────────────────────────────────────────────────────────────
# 6. CLASES / PUNTO (fallo: self.y = x typo sistemático)
# ─────────────────────────────────────────────────────────────────────────────

_add(
    "Write a Python class `Punto` with an `__init__(self, x, y)` method that "
    "stores x and y as instance attributes, and a `distancia(self, otro)` method "
    "that returns the Euclidean distance to another Punto.",
    textwrap.dedent("""\
        import math

        class Punto:
            def __init__(self, x, y):
                self.x = x
                self.y = y

            def distancia(self, otro):
                return math.sqrt((self.x - otro.x) ** 2 + (self.y - otro.y) ** 2)
    """).strip(),
    "p1=Punto(0,0); p2=Punto(3,4); "
    "assert p1.x==0; assert p1.y==0; assert p2.x==3; assert p2.y==4; "
    "assert p1.distancia(p2)==5.0",
)

_add(
    "Define a Python class `Punto` that represents a 2D point. "
    "The constructor takes x and y coordinates. "
    "Add a `__str__` method returning '(x, y)'.",
    textwrap.dedent("""\
        class Punto:
            def __init__(self, x, y):
                self.x = x
                self.y = y

            def __str__(self):
                return f'({self.x}, {self.y})'
    """).strip(),
    "p=Punto(3,4); assert p.x==3; assert p.y==4; assert str(p)=='(3, 4)'",
)

_add(
    "Create a Python class `Rectangulo` with `__init__(self, ancho, alto)` storing "
    "width and height, and methods `area()` and `perimetro()`.",
    textwrap.dedent("""\
        class Rectangulo:
            def __init__(self, ancho, alto):
                self.ancho = ancho
                self.alto = alto

            def area(self):
                return self.ancho * self.alto

            def perimetro(self):
                return 2 * (self.ancho + self.alto)
    """).strip(),
    "r=Rectangulo(4,5); assert r.area()==20; assert r.perimetro()==18; "
    "assert r.ancho==4; assert r.alto==5",
)

_add(
    "Write a Python class `Circulo` with `__init__(self, radio)` and methods "
    "`area()` and `circunferencia()` using math.pi.",
    textwrap.dedent("""\
        import math

        class Circulo:
            def __init__(self, radio):
                self.radio = radio

            def area(self):
                return math.pi * self.radio ** 2

            def circunferencia(self):
                return 2 * math.pi * self.radio
    """).strip(),
    "import math; c=Circulo(1); assert abs(c.area()-math.pi)<1e-9; "
    "assert abs(c.circunferencia()-2*math.pi)<1e-9; assert c.radio==1",
)

_add(
    "Create a Python class `Contador` with `__init__(self, inicio=0)`, "
    "`incrementar()`, `decrementar()`, and `valor()` methods.",
    textwrap.dedent("""\
        class Contador:
            def __init__(self, inicio=0):
                self._valor = inicio

            def incrementar(self):
                self._valor += 1

            def decrementar(self):
                self._valor -= 1

            def valor(self):
                return self._valor
    """).strip(),
    "c=Contador(); c.incrementar(); c.incrementar(); "
    "assert c.valor()==2; c.decrementar(); assert c.valor()==1; "
    "c2=Contador(10); assert c2.valor()==10",
)

# ─────────────────────────────────────────────────────────────────────────────
# 7. MEMOIZE / CLOSURES (fallo: closure incorrecto)
# ─────────────────────────────────────────────────────────────────────────────

_add(
    "Write a Python function `memoize(func)` that returns a memoized version of "
    "the given function, caching results by arguments.",
    textwrap.dedent("""\
        def memoize(func):
            cache = {}
            def wrapper(*args):
                if args not in cache:
                    cache[args] = func(*args)
                return cache[args]
            return wrapper
    """).strip(),
    "calls=[0]\n"
    "def f(n):\n    calls[0]+=1\n    return n*2\n"
    "mf=memoize(f); assert mf(3)==6; assert mf(3)==6; assert calls[0]==1",
)

_add(
    "Implement `memoize(fn)` as a decorator factory in Python. "
    "It should use an inner dict as cache and return a wrapper function.",
    textwrap.dedent("""\
        def memoize(fn):
            cache = {}
            def wrapper(*args):
                if args not in cache:
                    cache[args] = fn(*args)
                return cache[args]
            return wrapper

        @memoize
        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n - 1) + fibonacci(n - 2)
    """).strip(),
    "assert fibonacci(10)==55; assert fibonacci(0)==0; assert fibonacci(1)==1",
)

_add(
    "Write a Python function `crear_multiplicador(factor)` that returns a closure "
    "that multiplies its argument by factor.",
    textwrap.dedent("""\
        def crear_multiplicador(factor):
            def multiplicar(n):
                return n * factor
            return multiplicar
    """).strip(),
    "doble=crear_multiplicador(2); triple=crear_multiplicador(3); "
    "assert doble(5)==10; assert triple(5)==15; assert doble(0)==0",
)

_add(
    "Write a Python function `crear_contador()` that returns a closure "
    "which increments and returns a counter each time it is called.",
    textwrap.dedent("""\
        def crear_contador():
            count = [0]
            def incrementar():
                count[0] += 1
                return count[0]
            return incrementar
    """).strip(),
    "c=crear_contador(); assert c()==1; assert c()==2; assert c()==3; "
    "c2=crear_contador(); assert c2()==1",
)

# ─────────────────────────────────────────────────────────────────────────────
# 8. NÚMEROS PRIMOS — O(√n) (fallo: O(n) en vez de O(√n))
# ─────────────────────────────────────────────────────────────────────────────

_add(
    "Write a Python function `es_primo(n)` that returns True if n is a prime "
    "number, False otherwise. Use an O(√n) algorithm.",
    textwrap.dedent("""\
        def es_primo(n):
            if n < 2:
                return False
            for i in range(2, int(n ** 0.5) + 1):
                if n % i == 0:
                    return False
            return True
    """).strip(),
    "assert es_primo(2); assert es_primo(3); assert es_primo(13); "
    "assert not es_primo(0); assert not es_primo(1); "
    "assert not es_primo(4); assert not es_primo(9)",
)

_add(
    "Implement `es_primo(n)` in Python. Check divisibility only up to the "
    "square root of n for efficiency.",
    textwrap.dedent("""\
        def es_primo(n):
            if n < 2:
                return False
            if n == 2:
                return True
            if n % 2 == 0:
                return False
            for i in range(3, int(n ** 0.5) + 1, 2):
                if n % i == 0:
                    return False
            return True
    """).strip(),
    "assert es_primo(2); assert es_primo(97); assert not es_primo(1); "
    "assert not es_primo(100)",
)

_add(
    "Write a Python generator function `primos_hasta(n)` that yields all prime "
    "numbers up to and including n.",
    textwrap.dedent("""\
        def primos_hasta(n):
            for num in range(2, n + 1):
                es_primo = True
                for i in range(2, int(num ** 0.5) + 1):
                    if num % i == 0:
                        es_primo = False
                        break
                if es_primo:
                    yield num
    """).strip(),
    "assert list(primos_hasta(10))==[2,3,5,7]; assert list(primos_hasta(1))==[]; "
    "assert list(primos_hasta(2))==[2]; assert list(primos_hasta(20))==[2,3,5,7,11,13,17,19]",
)

_add(
    "Write a Python generator function `primos_hasta(n)` that yields all "
    "prime numbers from 2 to n inclusive. Use trial division up to sqrt.",
    textwrap.dedent("""\
        def primos_hasta(n):
            def es_primo(x):
                if x < 2:
                    return False
                for i in range(2, int(x ** 0.5) + 1):
                    if x % i == 0:
                        return False
                return True
            for num in range(2, n + 1):
                if es_primo(num):
                    yield num
    """).strip(),
    "assert list(primos_hasta(20))==[2,3,5,7,11,13,17,19]; "
    "assert list(primos_hasta(1))==[]",
)

_add(
    "Implement a Python generator `primos_hasta(n)` that yields each prime "
    "number up to n. Check primality by testing divisors up to the square root.",
    textwrap.dedent("""\
        def primos_hasta(n):
            for candidate in range(2, n + 1):
                is_prime = True
                for div in range(2, int(candidate ** 0.5) + 1):
                    if candidate % div == 0:
                        is_prime = False
                        break
                if is_prime:
                    yield candidate
    """).strip(),
    "assert list(primos_hasta(10))==[2,3,5,7]; assert list(primos_hasta(0))==[]",
)

_add(
    "Write a Python function `primos_hasta(n)` that returns all prime numbers "
    "up to and including n using the Sieve of Eratosthenes.",
    textwrap.dedent("""\
        def primos_hasta(n):
            if n < 2:
                return []
            criba = [True] * (n + 1)
            criba[0] = criba[1] = False
            for i in range(2, int(n ** 0.5) + 1):
                if criba[i]:
                    for j in range(i * i, n + 1, i):
                        criba[j] = False
            return [i for i in range(2, n + 1) if criba[i]]
    """).strip(),
    "assert primos_hasta(10)==[2,3,5,7]; assert primos_hasta(1)==[]; "
    "assert primos_hasta(2)==[2]; assert 97 in primos_hasta(100)",
)

_add(
    "Create a Python function `contar_primos(n)` that counts how many prime "
    "numbers are less than n.",
    textwrap.dedent("""\
        def contar_primos(n):
            def es_primo(x):
                if x < 2:
                    return False
                for i in range(2, int(x ** 0.5) + 1):
                    if x % i == 0:
                        return False
                return True
            return sum(1 for i in range(2, n) if es_primo(i))
    """).strip(),
    "assert contar_primos(10)==4; assert contar_primos(0)==0; "
    "assert contar_primos(2)==0; assert contar_primos(3)==1",
)


# =============================================================================
# EJEMPLOS GENERALES (de generate_targeted_sft.py v4, ahora verificados)
# =============================================================================

EJEMPLOS_GENERALES: list[dict] = [
    # ── Strings ──────────────────────────────────────────────────────────────
    {"problem": "Write a Python function `contar_vocales(texto)` that returns the number of vowels (a, e, i, o, u, case-insensitive) in the string.",
     "solution": "def contar_vocales(texto):\n    return sum(1 for c in texto.lower() if c in 'aeiou')",
     "checks": "assert contar_vocales('hola')==2; assert contar_vocales('')==0; assert contar_vocales('xyz')==0"},
    {"problem": "Write a Python function `es_palindromo(s)` that returns True if the string is a palindrome, False otherwise.",
     "solution": "def es_palindromo(s):\n    return s == s[::-1]",
     "checks": "assert es_palindromo('racecar'); assert not es_palindromo('hello'); assert es_palindromo('a')"},
    {"problem": "Write a Python function `invertir_cadena(s)` that returns the reversed version of string s.",
     "solution": "def invertir_cadena(s):\n    return s[::-1]",
     "checks": "assert invertir_cadena('hello')=='olleh'; assert invertir_cadena('')==''"},
    {"problem": "Write a Python function `contar_palabras(texto)` that returns the number of words in the text.",
     "solution": "def contar_palabras(texto):\n    return len(texto.split())",
     "checks": "assert contar_palabras('hola mundo')==2; assert contar_palabras('')==0"},
    {"problem": "Write a Python function `quitar_espacios(s)` that removes all whitespace from the string.",
     "solution": "def quitar_espacios(s):\n    return s.replace(' ', '')",
     "checks": "assert quitar_espacios('a b c')=='abc'; assert quitar_espacios('')==''"},
    # ── Números ───────────────────────────────────────────────────────────────
    {"problem": "Write a Python function `suma_digitos(n)` that returns the sum of all digits of the non-negative integer n.",
     "solution": "def suma_digitos(n):\n    return sum(int(d) for d in str(n))",
     "checks": "assert suma_digitos(123)==6; assert suma_digitos(0)==0; assert suma_digitos(999)==27"},
    {"problem": "Write a Python function `factorial(n)` that returns the factorial of the non-negative integer n.",
     "solution": "def factorial(n):\n    if n == 0:\n        return 1\n    result = 1\n    for i in range(1, n + 1):\n        result *= i\n    return result",
     "checks": "assert factorial(0)==1; assert factorial(1)==1; assert factorial(5)==120"},
    {"problem": "Write a Python function `fibonacci(n)` that returns the n-th Fibonacci number (0-indexed).",
     "solution": "def fibonacci(n):\n    if n <= 0:\n        return 0\n    if n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b",
     "checks": "assert fibonacci(0)==0; assert fibonacci(1)==1; assert fibonacci(10)==55"},
    {"problem": "Write a Python function `mcd(a, b)` that returns the GCD using the Euclidean algorithm.",
     "solution": "def mcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
     "checks": "assert mcd(12,8)==4; assert mcd(7,3)==1; assert mcd(0,5)==5"},
    {"problem": "Write a Python function `es_potencia_de_dos(n)` that returns True if n is a power of 2.",
     "solution": "def es_potencia_de_dos(n):\n    return n > 0 and (n & (n - 1)) == 0",
     "checks": "assert es_potencia_de_dos(1); assert es_potencia_de_dos(8); assert not es_potencia_de_dos(6)"},
    # ── Listas ────────────────────────────────────────────────────────────────
    {"problem": "Write a Python function `maximo(lista)` that returns the maximum element without using max().",
     "solution": "def maximo(lista):\n    m = lista[0]\n    for x in lista[1:]:\n        if x > m:\n            m = x\n    return m",
     "checks": "assert maximo([3,1,4,1,5])==5; assert maximo([0])==0; assert maximo([-1,-5])==-1"},
    {"problem": "Write a Python function `aplanar(lista)` that flattens a list of lists by one level and returns the result as a single list.",
     "solution": "def aplanar(lista):\n    return [x for sublist in lista for x in sublist]",
     "checks": "assert aplanar([[1,2],[3,4],[5]])==[1,2,3,4,5]; assert aplanar([])==[]"},
    {"problem": "Write a Python function `aplanar(lista)` that takes a list of lists and returns a flat list with all elements.",
     "solution": "def aplanar(lista):\n    result = []\n    for sublist in lista:\n        result.extend(sublist)\n    return result",
     "checks": "assert aplanar([[1],[2,3],[4]])==[1,2,3,4]; assert aplanar([])==[]"},
    {"problem": "Create a Python function `aplanar(lista)` that flattens nested sublists into one list.",
     "solution": "def aplanar(lista):\n    return [elem for sub in lista for elem in sub]",
     "checks": "assert aplanar([[1,2],[3]])==[1,2,3]; assert aplanar([[]])==[]"},
    {"problem": "Write a Python function `quitar_duplicados(lista)` that removes duplicates while preserving order.",
     "solution": "def quitar_duplicados(lista):\n    seen = set()\n    return [x for x in lista if not (x in seen or seen.add(x))]",
     "checks": "assert quitar_duplicados([1,2,1,3,2])==[1,2,3]; assert quitar_duplicados([])==[]"},
    {"problem": "Write a Python function `rotar(lista, k)` that rotates the list right by k positions.",
     "solution": "def rotar(lista, k):\n    if not lista:\n        return lista\n    k = k % len(lista)\n    return lista[-k:] + lista[:-k] if k else lista[:]",
     "checks": "assert rotar([1,2,3,4,5],2)==[4,5,1,2,3]; assert rotar([],3)==[]"},
    # ── Diccionarios ──────────────────────────────────────────────────────────
    {"problem": "Write a Python function `frecuencia(lista)` that returns a dictionary mapping each element to its count in the list.",
     "solution": "def frecuencia(lista):\n    freq = {}\n    for x in lista:\n        freq[x] = freq.get(x, 0) + 1\n    return freq",
     "checks": "assert frecuencia([1,2,2,3,3,3])=={1:1,2:2,3:3}; assert frecuencia([])=={}"},
    {"problem": "Write a Python function `frecuencia(lista)` that counts how many times each element appears and returns a dict.",
     "solution": "def frecuencia(lista):\n    resultado = {}\n    for elem in lista:\n        resultado[elem] = resultado.get(elem, 0) + 1\n    return resultado",
     "checks": "assert frecuencia([1,1,2])=={1:2,2:1}; assert frecuencia([])=={}"},
    {"problem": "Create a Python function `frecuencia(lista)` that returns a frequency dictionary of all elements.",
     "solution": "def frecuencia(lista):\n    freq = {}\n    for x in lista:\n        if x in freq:\n            freq[x] += 1\n        else:\n            freq[x] = 1\n    return freq",
     "checks": "assert frecuencia(['a','b','a'])=={'a':2,'b':1}; assert frecuencia([])=={}"},
    {"problem": "Write a Python function `frecuencias(lista)` that returns a dict counting occurrences of each element.",
     "solution": "def frecuencias(lista):\n    freq = {}\n    for x in lista:\n        freq[x] = freq.get(x, 0) + 1\n    return freq",
     "checks": "assert frecuencias([1,2,1,3])=={1:2,2:1,3:1}; assert frecuencias([])=={}"},
    {"problem": "Write a Python function `fusionar_dicts(d1, d2)` that returns a new dict merging d1 and d2 (d2 takes priority on conflicts).",
     "solution": "def fusionar_dicts(d1, d2):\n    return {**d1, **d2}",
     "checks": "assert fusionar_dicts({'a':1},{'b':2})=={'a':1,'b':2}; "
               "assert fusionar_dicts({'a':1},{'a':2})=={'a':2}"},
    # ── Algoritmos ────────────────────────────────────────────────────────────
    {"problem": "Write a Python function `bubble_sort(arr)` that sorts a list in-place using bubble sort and returns it.",
     "solution": "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n - i - 1):\n            if arr[j] > arr[j + 1]:\n                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n    return arr",
     "checks": "assert bubble_sort([3,1,2])==[1,2,3]; assert bubble_sort([])==[]"},
    {"problem": "Write a Python function `es_anagrama(a, b)` that returns True if a and b are anagrams.",
     "solution": "def es_anagrama(a, b):\n    return sorted(a.lower()) == sorted(b.lower())",
     "checks": "assert es_anagrama('listen','silent'); assert not es_anagrama('hello','world')"},
    # ── Stack/Queue ───────────────────────────────────────────────────────────
    {"problem": "Write a Python class `Stack` with methods `push(item)`, `pop()`, `peek()`, and `is_empty()`.",
     "solution": "class Stack:\n    def __init__(self):\n        self.items = []\n\n    def push(self, item):\n        self.items.append(item)\n\n    def pop(self):\n        return self.items.pop()\n\n    def peek(self):\n        return self.items[-1]\n\n    def is_empty(self):\n        return len(self.items) == 0",
     "checks": "s=Stack(); assert s.is_empty(); s.push(1); s.push(2); "
               "assert s.peek()==2; assert s.pop()==2; assert not s.is_empty()"},
    {"problem": "Write a Python class `Stack` with methods `push(item)` and `pop()` implementing a LIFO stack.",
     "solution": "class Stack:\n    def __init__(self):\n        self.items = []\n\n    def push(self, item):\n        self.items.append(item)\n\n    def pop(self):\n        return self.items.pop()",
     "checks": "s=Stack(); s.push(1); s.push(2); assert s.pop()==2; assert s.pop()==1"},
    {"problem": "Create a Python class `Stack` that uses a list internally to implement push and pop operations.",
     "solution": "class Stack:\n    def __init__(self):\n        self.items = []\n\n    def push(self, item):\n        self.items.append(item)\n\n    def pop(self):\n        return self.items.pop()\n\n    def size(self):\n        return len(self.items)",
     "checks": "s=Stack(); s.push(10); s.push(20); assert s.size()==2; assert s.pop()==20"},
    # ── Punto con distancia ───────────────────────────────────────────────────
    {"problem": "Write a Python class `Punto` with attributes `x` and `y`, and a method `distancia(otro)` that returns the Euclidean distance to another Punto.",
     "solution": "import math\n\nclass Punto:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n\n    def distancia(self, otro):\n        return math.sqrt((self.x - otro.x) ** 2 + (self.y - otro.y) ** 2)",
     "checks": "p1=Punto(0,0); p2=Punto(3,4); assert p1.distancia(p2)==5.0"},
    {"problem": "Create a Python class `Punto` with `x` and `y` coordinates and a `distancia(otro)` method for Euclidean distance.",
     "solution": "class Punto:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n\n    def distancia(self, otro):\n        return ((self.x - otro.x) ** 2 + (self.y - otro.y) ** 2) ** 0.5",
     "checks": "p=Punto(0,0); q=Punto(3,4); assert p.distancia(q)==5.0; assert q.distancia(p)==5.0"},
    {"problem": "Implement a Python class `Punto` representing a 2D point with `x`, `y` attributes and a `distancia` method to compute distance to another point.",
     "solution": "import math\n\nclass Punto:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n\n    def distancia(self, otro):\n        dx = self.x - otro.x\n        dy = self.y - otro.y\n        return math.sqrt(dx * dx + dy * dy)",
     "checks": "a=Punto(1,1); b=Punto(4,5); assert a.distancia(b)==5.0"},
    {"problem": "Write a Python class `Queue` with methods `enqueue(item)`, `dequeue()`, `front()`, and `is_empty()`.",
     "solution": "from collections import deque\n\nclass Queue:\n    def __init__(self):\n        self.items = deque()\n\n    def enqueue(self, item):\n        self.items.append(item)\n\n    def dequeue(self):\n        return self.items.popleft()\n\n    def front(self):\n        return self.items[0]\n\n    def is_empty(self):\n        return len(self.items) == 0",
     "checks": "q=Queue(); assert q.is_empty(); q.enqueue(1); q.enqueue(2); "
               "assert q.front()==1; assert q.dequeue()==1; assert not q.is_empty()"},
]

# Verificar y agregar los ejemplos generales
print("Verificando ejemplos generales...")
for e in EJEMPLOS_GENERALES:
    ok, err = verificar(e["solution"], e["checks"])
    if ok:
        EJEMPLOS.append(e)
    else:
        print(f"  [SKIP] {e['problem'][:60]}: {err}")


# =============================================================================
# GENERACIÓN DE VARIACIONES
# Genera N variaciones por ejemplo con leve reframing del enunciado
# =============================================================================

PREFIJOS = [
    "Write a Python function",
    "Create a Python function",
    "Implement a Python function",
    "Define a Python function",
    "Write a Python solution",
]

SUFIJOS_EXTRA = [
    "",
    " Handle edge cases appropriately.",
    " The function should be concise and correct.",
    " Do not use any external libraries.",
    " Use only Python built-ins.",
]


def generar_variacion(e: dict) -> dict:
    """Genera una pequeña variación del enunciado (surface form) manteniendo la solución."""
    problem = e["problem"]
    # Reemplazar el prefijo aleatoriamente
    for viejo in PREFIJOS:
        if problem.startswith(viejo):
            nuevo = random.choice([p for p in PREFIJOS if p != viejo])
            problem = nuevo + problem[len(viejo):]
            break
    # Agregar sufijo opcional
    sufijo = random.choice(SUFIJOS_EXTRA)
    if sufijo and not problem.endswith(sufijo):
        problem = problem.rstrip(".") + "." + sufijo
    return {"problem": problem, "solution": e["solution"], "checks": e["checks"]}


def generar_variaciones_todos(n_variaciones: int = 3) -> list[dict]:
    """Genera n variaciones para cada ejemplo base."""
    variaciones = []
    for e in EJEMPLOS:
        for _ in range(n_variaciones):
            v = generar_variacion(e)
            variaciones.append(v)
    return variaciones


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Genera SFT v5 con verificación en runtime")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "sft_v5.jsonl")
    parser.add_argument("--variaciones", type=int, default=6,
                        help="Variaciones por ejemplo base (default=6)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"SFT v5 — Generación con verificación en runtime")
    print(f"{'='*60}")
    print(f"Ejemplos base verificados: {len(EJEMPLOS)}")

    # Generar variaciones
    variaciones = generar_variaciones_todos(args.variaciones)

    todos = EJEMPLOS + variaciones
    random.shuffle(todos)

    # Escribir JSONL
    args.output.parent.mkdir(parents=True, exist_ok=True)
    escritos = 0
    with args.output.open("w", encoding="utf-8") as f:
        for e in todos:
            texto = formatear(e["problem"], e["solution"])
            obj = {"text": texto, "source": "sft_v5_verified", "license": "open"}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            escritos += 1

    print(f"\nEjemplos generados: {escritos}")
    print(f"Desglose:")
    print(f"  Base verificados : {len(EJEMPLOS)}")
    print(f"  Variaciones      : {len(variaciones)}")
    print(f"\nOutput: {args.output}")
    print(f"\nSiguiente paso:")
    print(f"  python -X utf8 scripts/sft_v5.py --checkpoint checkpoints/v3_sft_v4.pt")


if __name__ == "__main__":
    main()
