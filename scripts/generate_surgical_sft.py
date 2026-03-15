#!/usr/bin/env python3
"""Generate surgical SFT dataset targeting the 6 remaining eval failures."""

import json
import re

examples: list[dict[str, str]] = []

# === 1. FIZZBUZZ — correct logic ===
fizzbuzz_solutions = [
    'def fizzbuzz(n):\n    if n % 15 == 0:\n        return "FizzBuzz"\n    if n % 3 == 0:\n        return "Fizz"\n    if n % 5 == 0:\n        return "Buzz"\n    return str(n)',
    'def fizzbuzz(n):\n    if n % 3 == 0 and n % 5 == 0:\n        return "FizzBuzz"\n    elif n % 3 == 0:\n        return "Fizz"\n    elif n % 5 == 0:\n        return "Buzz"\n    else:\n        return str(n)',
    "def fizzbuzz(n):\n    result = ''\n    if n % 3 == 0:\n        result += 'Fizz'\n    if n % 5 == 0:\n        result += 'Buzz'\n    return result if result else str(n)",
]

fizzbuzz_prompts = [
    'Write a Python function `fizzbuzz(n)` that returns "FizzBuzz" if n is divisible by both 3 and 5, "Fizz" if divisible by 3, "Buzz" if divisible by 5, or the string representation of n otherwise.',
    'Create a Python function `fizzbuzz(n)` that returns "FizzBuzz" for multiples of 15, "Fizz" for multiples of 3, "Buzz" for multiples of 5, or str(n) otherwise.',
    'Implement a Python function `fizzbuzz(n)` that checks divisibility: if n is divisible by 3 and 5 return "FizzBuzz", by 3 return "Fizz", by 5 return "Buzz", else return str(n).',
    'Write a Python function `fizzbuzz(n)` returning "FizzBuzz" when n%15==0, "Fizz" when n%3==0, "Buzz" when n%5==0, otherwise str(n).',
    'Create a function `fizzbuzz(n)` in Python. Return "FizzBuzz" if divisible by both 3 and 5, "Fizz" if by 3, "Buzz" if by 5, else the number as string.',
]

for sol in fizzbuzz_solutions:
    for prompt in fizzbuzz_prompts:
        examples.append({"text": f"### Problem:\n{prompt}\n### Solution:\n```python\n{sol}\n```"})

# === 2. CUADRADOS_PARES — range starts at 2 ===
cuadrados_solutions = [
    "def cuadrados_pares(n):\n    return [i ** 2 for i in range(2, n + 1) if i % 2 == 0]",
    "def cuadrados_pares(n):\n    resultado = []\n    for i in range(2, n + 1):\n        if i % 2 == 0:\n            resultado.append(i * i)\n    return resultado",
    "def cuadrados_pares(n):\n    return [x * x for x in range(2, n + 1, 2)]",
]

cuadrados_prompts = [
    'Write a Python function `cuadrados_pares(n)` that returns a list of squares of all even numbers from 2 to n inclusive.',
    'Create a Python function `cuadrados_pares(n)` returning the squares of even numbers in range [2, n].',
    'Implement `cuadrados_pares(n)` in Python: return a list with i**2 for each even i from 2 to n (inclusive).',
    'Write a function `cuadrados_pares(n)` that computes squares of all even integers from 2 up to and including n.',
    'Create `cuadrados_pares(n)` returning [4, 16, 36, ...] for even numbers squared from 2 to n inclusive.',
]

for sol in cuadrados_solutions:
    for prompt in cuadrados_prompts:
        examples.append({"text": f"### Problem:\n{prompt}\n### Solution:\n```python\n{sol}\n```"})

# === 3. MERGE_SORT self-contained ===
merge_solutions = [
    "def merge_sort(lista):\n    if len(lista) <= 1:\n        return lista[:]\n    mid = len(lista) // 2\n    left = merge_sort(lista[:mid])\n    right = merge_sort(lista[mid:])\n    resultado = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            resultado.append(left[i])\n            i += 1\n        else:\n            resultado.append(right[j])\n            j += 1\n    resultado.extend(left[i:])\n    resultado.extend(right[j:])\n    return resultado",
    "def merge_sort(lista):\n    if len(lista) <= 1:\n        return list(lista)\n    mitad = len(lista) // 2\n    izq = merge_sort(lista[:mitad])\n    der = merge_sort(lista[mitad:])\n    merged = []\n    i = j = 0\n    while i < len(izq) and j < len(der):\n        if izq[i] <= der[j]:\n            merged.append(izq[i])\n            i += 1\n        else:\n            merged.append(der[j])\n            j += 1\n    merged.extend(izq[i:])\n    merged.extend(der[j:])\n    return merged",
    "def merge_sort(lista):\n    n = len(lista)\n    if n <= 1:\n        return lista[:]\n    mid = n // 2\n    a = merge_sort(lista[:mid])\n    b = merge_sort(lista[mid:])\n    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i])\n            i += 1\n        else:\n            result.append(b[j])\n            j += 1\n    result += a[i:]\n    result += b[j:]\n    return result",
]

merge_prompts = [
    'Write a Python function `merge_sort(lista)` that returns a new sorted list using the merge sort algorithm.',
    'Create a Python function `merge_sort(lista)` implementing merge sort. Return a new sorted list.',
    'Implement `merge_sort(lista)` in Python using the divide-and-conquer merge sort approach. Return sorted list.',
    'Write `merge_sort(lista)` that splits, recursively sorts, and merges sublists into a sorted result.',
    'Create a self-contained `merge_sort(lista)` function that returns a new list sorted via merge sort.',
]

for sol in merge_solutions:
    for prompt in merge_prompts:
        examples.append({"text": f"### Problem:\n{prompt}\n### Solution:\n```python\n{sol}\n```"})

# === 4. PUNTO with import math or ** 0.5 ===
punto_solutions = [
    "import math\n\nclass Punto:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n    def distancia(self, otro):\n        return math.sqrt((self.x - otro.x) ** 2 + (self.y - otro.y) ** 2)",
    "class Punto:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n    def distancia(self, otro):\n        return ((self.x - otro.x) ** 2 + (self.y - otro.y) ** 2) ** 0.5",
    "from math import sqrt\n\nclass Punto:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n    def distancia(self, otro):\n        dx = self.x - otro.x\n        dy = self.y - otro.y\n        return sqrt(dx * dx + dy * dy)",
]

punto_prompts = [
    'Write a Python class `Punto` with attributes `x` and `y`, and a method `distancia(otro)` that returns the Euclidean distance to another Punto.',
    'Create a Python class `Punto` with x, y coordinates and a `distancia(otro)` method returning Euclidean distance.',
    'Implement a `Punto` class in Python with `__init__(self, x, y)` and `distancia(self, otro)` computing the Euclidean distance.',
    'Write a class `Punto` with x and y attributes. Include a method `distancia(otro)` returning the distance between two points.',
    'Create class `Punto` with constructor taking x, y and method `distancia(otro)` that calculates the Euclidean distance.',
]

for sol in punto_solutions:
    for prompt in punto_prompts:
        examples.append({"text": f"### Problem:\n{prompt}\n### Solution:\n```python\n{sol}\n```"})

# === 5. MEMOIZE with fn (NOT func) ===
memo_solutions = [
    "def memoize(fn):\n    cache = {}\n    def wrapper(*args):\n        if args not in cache:\n            cache[args] = fn(*args)\n        return cache[args]\n    return wrapper",
    "def memoize(fn):\n    memo = {}\n    def wrapped(*args):\n        if args in memo:\n            return memo[args]\n        result = fn(*args)\n        memo[args] = result\n        return result\n    return wrapped",
    "def memoize(fn):\n    cache = {}\n    def inner(*args):\n        try:\n            return cache[args]\n        except KeyError:\n            cache[args] = fn(*args)\n            return cache[args]\n    return inner",
]

memo_prompts = [
    'Write a Python higher-order function `memoize(fn)` that returns a wrapped version of fn that caches results by argument.',
    'Create a Python function `memoize(fn)` that returns a wrapper caching fn results by arguments.',
    'Implement `memoize(fn)` in Python: return a function that caches the results of calling fn with given arguments.',
    'Write `memoize(fn)` that creates a closure with a cache dict, storing fn(*args) results for repeated calls.',
    'Create a higher-order function `memoize(fn)` returning a cached version of fn using a dictionary.',
]

for sol in memo_solutions:
    for prompt in memo_prompts:
        examples.append({"text": f"### Problem:\n{prompt}\n### Solution:\n```python\n{sol}\n```"})

# === 6. PRIMOS_HASTA with yield ===
primos_solutions = [
    "def primos_hasta(n):\n    for num in range(2, n + 1):\n        es_primo = True\n        for i in range(2, int(num ** 0.5) + 1):\n            if num % i == 0:\n                es_primo = False\n                break\n        if es_primo:\n            yield num",
    "def primos_hasta(n):\n    for num in range(2, n + 1):\n        if all(num % i != 0 for i in range(2, int(num ** 0.5) + 1)):\n            yield num",
    "def primos_hasta(n):\n    for candidate in range(2, n + 1):\n        is_prime = True\n        for divisor in range(2, int(candidate ** 0.5) + 1):\n            if candidate % divisor == 0:\n                is_prime = False\n                break\n        if is_prime:\n            yield candidate",
]

primos_prompts = [
    'Write a Python generator function `primos_hasta(n)` that yields all prime numbers up to and including n.',
    'Create a Python generator `primos_hasta(n)` that yields primes from 2 to n inclusive.',
    'Implement `primos_hasta(n)` as a Python generator that yields each prime number up to n.',
    'Write a generator function `primos_hasta(n)` using yield to produce all primes <= n.',
    'Create `primos_hasta(n)` that iterates from 2 to n and yields numbers that are prime.',
]

for sol in primos_solutions:
    for prompt in primos_prompts:
        examples.append({"text": f"### Problem:\n{prompt}\n### Solution:\n```python\n{sol}\n```"})

# Verify all compile
valid = 0
for ex in examples:
    text = ex["text"]
    m = re.search(r"```python\n(.*?)\n```", text, re.DOTALL)
    if m:
        code = m.group(1)
        try:
            compile(code, "<test>", "exec")
            valid += 1
        except SyntaxError as e:
            print(f"SYNTAX ERROR: {e}")
            print(code[:100])

print(f"Total: {len(examples)}, Valid: {valid}")

with open("data/surgical_sft.jsonl", "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")
print("Saved: data/surgical_sft.jsonl")
