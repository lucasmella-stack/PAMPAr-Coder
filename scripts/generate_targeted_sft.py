#!/usr/bin/env python3
"""
generate_targeted_sft.py — Dataset dirigido para SFT de instruction-following.

Genera ejemplos con el formato EXACTO del eval:
  ### Problem:
  Write a Python function `nombre(args)` that ...
  ### Solution:
  ```python
  def nombre(args):
      ...
  ```

Cada ejemplo:
  - El nombre de función en backticks coincide exactamente con el def
  - Implementación correcta y concisa
  - Termina con el cierre del bloque ``` sin texto adicional

Uso:
  python scripts/generate_targeted_sft.py --output data/targeted_sft.jsonl
"""

import json
import random
import argparse
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Dataset base: ejemplos curados manualmente, variados y correctos
# ─────────────────────────────────────────────────────────────────────────────

EJEMPLOS_BASE = [
    # ── Strings ──────────────────────────────────────────────────────────────
    {
        "problem": "Write a Python function `contar_vocales(texto)` that returns the number of vowels (a, e, i, o, u, case-insensitive) in the string.",
        "solution": "def contar_vocales(texto):\n    return sum(1 for c in texto.lower() if c in 'aeiou')",
    },
    {
        "problem": "Write a Python function `es_palindromo(s)` that returns True if the string is a palindrome, False otherwise.",
        "solution": "def es_palindromo(s):\n    return s == s[::-1]",
    },
    {
        "problem": "Write a Python function `invertir_cadena(s)` that returns the reversed version of string s.",
        "solution": "def invertir_cadena(s):\n    return s[::-1]",
    },
    {
        "problem": "Write a Python function `contar_palabras(texto)` that returns the number of words in the text.",
        "solution": "def contar_palabras(texto):\n    return len(texto.split())",
    },
    {
        "problem": "Write a Python function `primera_letra_mayuscula(s)` that capitalizes the first letter of each word.",
        "solution": "def primera_letra_mayuscula(s):\n    return s.title()",
    },
    {
        "problem": "Write a Python function `quitar_espacios(s)` that removes all whitespace from the string.",
        "solution": "def quitar_espacios(s):\n    return s.replace(' ', '')",
    },
    {
        "problem": "Write a Python function `repetir(s, n)` that returns the string s repeated n times.",
        "solution": "def repetir(s, n):\n    return s * n",
    },
    {
        "problem": "Write a Python function `contiene_numero(s)` that returns True if the string contains at least one digit.",
        "solution": "def contiene_numero(s):\n    return any(c.isdigit() for c in s)",
    },
    {
        "problem": "Write a Python function `contar_consonantes(texto)` that returns the count of consonants in the string.",
        "solution": "def contar_consonantes(texto):\n    vocales = set('aeiouAEIOU')\n    return sum(1 for c in texto if c.isalpha() and c not in vocales)",
    },
    {
        "problem": "Write a Python function `es_anagrama(a, b)` that returns True if strings a and b are anagrams of each other.",
        "solution": "def es_anagrama(a, b):\n    return sorted(a.lower()) == sorted(b.lower())",
    },
    {
        "problem": "Write a Python function `truncar(s, longitud)` that truncates s to the given length, adding '...' if truncated.",
        "solution": "def truncar(s, longitud):\n    if len(s) <= longitud:\n        return s\n    return s[:longitud] + '...'",
    },
    {
        "problem": "Write a Python function `contar_apariciones(texto, letra)` that returns how many times letra appears in texto.",
        "solution": "def contar_apariciones(texto, letra):\n    return texto.count(letra)",
    },
    {
        "problem": "Write a Python function `es_mayuscula(s)` that returns True if all alphabetic characters in s are uppercase.",
        "solution": "def es_mayuscula(s):\n    return s.isupper()",
    },
    {
        "problem": "Write a Python function `quitar_duplicados_str(s)` that removes duplicate characters from s preserving order.",
        "solution": "def quitar_duplicados_str(s):\n    seen = set()\n    return ''.join(c for c in s if not (c in seen or seen.add(c)))",
    },
    {
        "problem": "Write a Python function `comprimir(s)` that returns a run-length encoding of the string (e.g. 'aabb' -> 'a2b2').",
        "solution": "def comprimir(s):\n    if not s:\n        return ''\n    resultado = []\n    i = 0\n    while i < len(s):\n        c = s[i]\n        count = 1\n        while i + count < len(s) and s[i + count] == c:\n            count += 1\n        resultado.append(c + str(count))\n        i += count\n    return ''.join(resultado)",
    },
    # ── Números ───────────────────────────────────────────────────────────────
    {
        "problem": "Write a Python function `suma_digitos(n)` that returns the sum of all digits of the non-negative integer n.",
        "solution": "def suma_digitos(n):\n    return sum(int(d) for d in str(n))",
    },
    {
        "problem": "Write a Python function `es_primo(n)` that returns True if n is a prime number, False otherwise.",
        "solution": "def es_primo(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
    },
    {
        "problem": "Write a Python function `factorial(n)` that returns the factorial of the non-negative integer n.",
        "solution": "def factorial(n):\n    if n == 0:\n        return 1\n    result = 1\n    for i in range(1, n + 1):\n        result *= i\n    return result",
    },
    {
        "problem": "Write a Python function `fibonacci(n)` that returns the n-th Fibonacci number (0-indexed: fibonacci(0)=0, fibonacci(1)=1).",
        "solution": "def fibonacci(n):\n    if n <= 0:\n        return 0\n    if n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b",
    },
    {
        "problem": "Write a Python function `fizzbuzz(n)` that returns 'FizzBuzz' if n is divisible by both 3 and 5, 'Fizz' if divisible by 3, 'Buzz' if divisible by 5, or the string representation of n otherwise.",
        "solution": "def fizzbuzz(n):\n    if n % 15 == 0:\n        return 'FizzBuzz'\n    if n % 3 == 0:\n        return 'Fizz'\n    if n % 5 == 0:\n        return 'Buzz'\n    return str(n)",
    },
    {
        "problem": "Write a Python function `mcd(a, b)` that returns the greatest common divisor of a and b using the Euclidean algorithm.",
        "solution": "def mcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
    },
    {
        "problem": "Write a Python function `mcm(a, b)` that returns the least common multiple of a and b.",
        "solution": "def mcm(a, b):\n    return a * b // mcd(a, b)\n\ndef mcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
    },
    {
        "problem": "Write a Python function `es_perfecto(n)` that returns True if n is a perfect number (equal to the sum of its proper divisors).",
        "solution": "def es_perfecto(n):\n    if n < 2:\n        return False\n    return sum(i for i in range(1, n) if n % i == 0) == n",
    },
    {
        "problem": "Write a Python function `potencia(base, exp)` that returns base raised to the power exp without using the ** operator.",
        "solution": "def potencia(base, exp):\n    result = 1\n    for _ in range(exp):\n        result *= base\n    return result",
    },
    {
        "problem": "Write a Python function `es_potencia_de_dos(n)` that returns True if n is a power of 2.",
        "solution": "def es_potencia_de_dos(n):\n    return n > 0 and (n & (n - 1)) == 0",
    },
    {
        "problem": "Write a Python function `primos_hasta(n)` that returns a list of all prime numbers up to and including n.",
        "solution": "def primos_hasta(n):\n    if n < 2:\n        return []\n    criba = [True] * (n + 1)\n    criba[0] = criba[1] = False\n    for i in range(2, int(n ** 0.5) + 1):\n        if criba[i]:\n            for j in range(i * i, n + 1, i):\n                criba[j] = False\n    return [i for i in range(2, n + 1) if criba[i]]",
    },
    {
        "problem": "Write a Python function `es_cuadrado_perfecto(n)` that returns True if n is a perfect square.",
        "solution": "def es_cuadrado_perfecto(n):\n    if n < 0:\n        return False\n    r = int(n ** 0.5)\n    return r * r == n",
    },
    {
        "problem": "Write a Python function `digitos_de(n)` that returns a list of digits of the non-negative integer n in order.",
        "solution": "def digitos_de(n):\n    return [int(d) for d in str(n)]",
    },
    # ── Listas ────────────────────────────────────────────────────────────────
    {
        "problem": "Write a Python function `maximo(lista)` that returns the maximum element of a non-empty list without using the built-in max().",
        "solution": "def maximo(lista):\n    m = lista[0]\n    for x in lista[1:]:\n        if x > m:\n            m = x\n    return m",
    },
    {
        "problem": "Write a Python function `minimo(lista)` that returns the minimum element of a non-empty list without using the built-in min().",
        "solution": "def minimo(lista):\n    m = lista[0]\n    for x in lista[1:]:\n        if x < m:\n            m = x\n    return m",
    },
    {
        "problem": "Write a Python function `promedio(lista)` that returns the average of the elements in the list. Return 0 for an empty list.",
        "solution": "def promedio(lista):\n    if not lista:\n        return 0\n    return sum(lista) / len(lista)",
    },
    {
        "problem": "Write a Python function `aplanar(lista)` that flattens a list of lists by one level and returns the result as a single list.",
        "solution": "def aplanar(lista):\n    return [x for sublista in lista for x in sublista]",
    },
    {
        "problem": "Write a Python function `frecuencia(lista)` that returns a dictionary mapping each element to its count in the list.",
        "solution": "def frecuencia(lista):\n    resultado = {}\n    for x in lista:\n        resultado[x] = resultado.get(x, 0) + 1\n    return resultado",
    },
    {
        "problem": "Write a Python function `cuadrados_pares(n)` that returns a list of squares of all even numbers from 2 to n inclusive.",
        "solution": "def cuadrados_pares(n):\n    return [i * i for i in range(2, n + 1, 2)]",
    },
    {
        "problem": "Write a Python function `quitar_duplicados(lista)` that removes duplicates from a list preserving the original order.",
        "solution": "def quitar_duplicados(lista):\n    seen = set()\n    return [x for x in lista if not (x in seen or seen.add(x))]",
    },
    {
        "problem": "Write a Python function `rotar(lista, k)` that rotates the list k positions to the left.",
        "solution": "def rotar(lista, k):\n    if not lista:\n        return lista\n    k = k % len(lista)\n    return lista[k:] + lista[:k]",
    },
    {
        "problem": "Write a Python function `intercalar(a, b)` that merges two lists by alternating elements from each.",
        "solution": "def intercalar(a, b):\n    resultado = []\n    for i in range(max(len(a), len(b))):\n        if i < len(a):\n            resultado.append(a[i])\n        if i < len(b):\n            resultado.append(b[i])\n    return resultado",
    },
    {
        "problem": "Write a Python function `esta_ordenada(lista)` that returns True if the list is sorted in non-decreasing order.",
        "solution": "def esta_ordenada(lista):\n    return all(lista[i] <= lista[i + 1] for i in range(len(lista) - 1))",
    },
    {
        "problem": "Write a Python function `chunk(lista, n)` that splits the list into sublists of size n (the last chunk may be smaller).",
        "solution": "def chunk(lista, n):\n    return [lista[i:i + n] for i in range(0, len(lista), n)]",
    },
    {
        "problem": "Write a Python function `sumar_matrices(a, b)` that returns the element-wise sum of two 2D matrices of the same shape.",
        "solution": "def sumar_matrices(a, b):\n    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]",
    },
    {
        "problem": "Write a Python function `transponer(matriz)` that returns the transpose of a 2D matrix.",
        "solution": "def transponer(matriz):\n    return [list(fila) for fila in zip(*matriz)]",
    },
    {
        "problem": "Write a Python function `producto_punto(a, b)` that returns the dot product of two equal-length lists.",
        "solution": "def producto_punto(a, b):\n    return sum(x * y for x, y in zip(a, b))",
    },
    {
        "problem": "Write a Python function `indices_de(lista, valor)` that returns a list of all indices where valor appears in lista.",
        "solution": "def indices_de(lista, valor):\n    return [i for i, x in enumerate(lista) if x == valor]",
    },
    # ── Diccionarios ──────────────────────────────────────────────────────────
    {
        "problem": "Write a Python function `invertir_dict(d)` that returns a new dictionary with keys and values swapped.",
        "solution": "def invertir_dict(d):\n    return {v: k for k, v in d.items()}",
    },
    {
        "problem": "Write a Python function `fusionar_dicts(a, b)` that merges two dictionaries. Values from b overwrite values from a.",
        "solution": "def fusionar_dicts(a, b):\n    return {**a, **b}",
    },
    {
        "problem": "Write a Python function `clave_max_valor(d)` that returns the key with the highest value in the dictionary.",
        "solution": "def clave_max_valor(d):\n    return max(d, key=d.get)",
    },
    {
        "problem": "Write a Python function `filtrar_dict(d, umbral)` that returns a new dict keeping only entries where the value is greater than umbral.",
        "solution": "def filtrar_dict(d, umbral):\n    return {k: v for k, v in d.items() if v > umbral}",
    },
    {
        "problem": "Write a Python function `contar_valores(d)` that returns a new dict mapping each unique value to how many times it appears.",
        "solution": "def contar_valores(d):\n    conteo = {}\n    for v in d.values():\n        conteo[v] = conteo.get(v, 0) + 1\n    return conteo",
    },
    {
        "problem": "Write a Python function `agrupar_por(lista, clave)` that groups a list of dicts by the given key, returning a dict of lists.",
        "solution": "def agrupar_por(lista, clave):\n    grupos = {}\n    for item in lista:\n        k = item[clave]\n        grupos.setdefault(k, []).append(item)\n    return grupos",
    },
    # ── Búsqueda y ordenamiento ───────────────────────────────────────────────
    {
        "problem": "Write a Python function `busqueda_binaria(lista, objetivo)` that returns the index of the target in a sorted list, or -1 if not found.",
        "solution": "def busqueda_binaria(lista, objetivo):\n    izq, der = 0, len(lista) - 1\n    while izq <= der:\n        mid = (izq + der) // 2\n        if lista[mid] == objetivo:\n            return mid\n        elif lista[mid] < objetivo:\n            izq = mid + 1\n        else:\n            der = mid - 1\n    return -1",
    },
    {
        "problem": "Write a Python function `merge_sort(lista)` that returns a new sorted list using the merge sort algorithm.",
        "solution": "def merge_sort(lista):\n    if len(lista) <= 1:\n        return lista\n    mid = len(lista) // 2\n    izq = merge_sort(lista[:mid])\n    der = merge_sort(lista[mid:])\n    resultado = []\n    i = j = 0\n    while i < len(izq) and j < len(der):\n        if izq[i] <= der[j]:\n            resultado.append(izq[i])\n            i += 1\n        else:\n            resultado.append(der[j])\n            j += 1\n    return resultado + izq[i:] + der[j:]",
    },
    {
        "problem": "Write a Python function `burbuja(lista)` that returns a new sorted list using the bubble sort algorithm.",
        "solution": "def burbuja(lista):\n    lista = lista[:]\n    n = len(lista)\n    for i in range(n):\n        for j in range(0, n - i - 1):\n            if lista[j] > lista[j + 1]:\n                lista[j], lista[j + 1] = lista[j + 1], lista[j]\n    return lista",
    },
    {
        "problem": "Write a Python function `quick_sort(lista)` that returns a new sorted list using the quicksort algorithm.",
        "solution": "def quick_sort(lista):\n    if len(lista) <= 1:\n        return lista\n    pivote = lista[len(lista) // 2]\n    menores = [x for x in lista if x < pivote]\n    iguales = [x for x in lista if x == pivote]\n    mayores = [x for x in lista if x > pivote]\n    return quick_sort(menores) + iguales + quick_sort(mayores)",
    },
    # ── Clases básicas ────────────────────────────────────────────────────────
    {
        "problem": "Write a Python class `Stack` with methods `push(item)` and `pop()` implementing a LIFO stack. `pop()` should raise IndexError if the stack is empty.",
        "solution": "class Stack:\n    def __init__(self):\n        self._data = []\n\n    def push(self, item):\n        self._data.append(item)\n\n    def pop(self):\n        if not self._data:\n            raise IndexError('Stack is empty')\n        return self._data.pop()",
    },
    {
        "problem": "Write a Python class `Queue` with methods `enqueue(item)` and `dequeue()` implementing a FIFO queue. `dequeue()` should raise IndexError if the queue is empty.",
        "solution": "from collections import deque\n\nclass Queue:\n    def __init__(self):\n        self._data = deque()\n\n    def enqueue(self, item):\n        self._data.append(item)\n\n    def dequeue(self):\n        if not self._data:\n            raise IndexError('Queue is empty')\n        return self._data.popleft()",
    },
    {
        "problem": "Write a Python class `Punto` with attributes `x` and `y`, and a method `distancia(otro)` that returns the Euclidean distance to another Punto.",
        "solution": "import math\n\nclass Punto:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n\n    def distancia(self, otro):\n        return math.sqrt((self.x - otro.x) ** 2 + (self.y - otro.y) ** 2)",
    },
    {
        "problem": "Write a Python class `Rectangulo` with attributes `ancho` and `alto`, and methods `area()` and `perimetro()`.",
        "solution": "class Rectangulo:\n    def __init__(self, ancho, alto):\n        self.ancho = ancho\n        self.alto = alto\n\n    def area(self):\n        return self.ancho * self.alto\n\n    def perimetro(self):\n        return 2 * (self.ancho + self.alto)",
    },
    {
        "problem": "Write a Python class `Circulo` with attribute `radio` and methods `area()` and `circunferencia()`.",
        "solution": "import math\n\nclass Circulo:\n    def __init__(self, radio):\n        self.radio = radio\n\n    def area(self):\n        return math.pi * self.radio ** 2\n\n    def circunferencia(self):\n        return 2 * math.pi * self.radio",
    },
    {
        "problem": "Write a Python class `Contador` with a method `incrementar()` that increases the count by 1, `decrementar()` that decreases it, and `valor()` that returns the current count.",
        "solution": "class Contador:\n    def __init__(self):\n        self._count = 0\n\n    def incrementar(self):\n        self._count += 1\n\n    def decrementar(self):\n        self._count -= 1\n\n    def valor(self):\n        return self._count",
    },
    {
        "problem": "Write a Python class `NodoLista` and a class `ListaEnlazada` with methods `agregar(valor)` (append to end) and `a_lista()` (return as Python list).",
        "solution": "class NodoLista:\n    def __init__(self, valor):\n        self.valor = valor\n        self.siguiente = None\n\nclass ListaEnlazada:\n    def __init__(self):\n        self.cabeza = None\n\n    def agregar(self, valor):\n        nuevo = NodoLista(valor)\n        if self.cabeza is None:\n            self.cabeza = nuevo\n            return\n        actual = self.cabeza\n        while actual.siguiente:\n            actual = actual.siguiente\n        actual.siguiente = nuevo\n\n    def a_lista(self):\n        resultado = []\n        actual = self.cabeza\n        while actual:\n            resultado.append(actual.valor)\n            actual = actual.siguiente\n        return resultado",
    },
    # ── Funciones de orden superior ───────────────────────────────────────────
    {
        "problem": "Write a Python higher-order function `memoize(fn)` that returns a wrapped version of fn that caches results by argument.",
        "solution": "def memoize(fn):\n    cache = {}\n    def wrapper(*args):\n        if args not in cache:\n            cache[args] = fn(*args)\n        return cache[args]\n    return wrapper",
    },
    {
        "problem": "Write a Python higher-order function `aplicar_n_veces(fn, n)` that returns a function that applies fn n times to its argument.",
        "solution": "def aplicar_n_veces(fn, n):\n    def wrapper(x):\n        for _ in range(n):\n            x = fn(x)\n        return x\n    return wrapper",
    },
    {
        "problem": "Write a Python higher-order function `componer(f, g)` that returns the function composition f(g(x)).",
        "solution": "def componer(f, g):\n    return lambda x: f(g(x))",
    },
    {
        "problem": "Write a Python higher-order function `parcial(fn, *args_fijos)` that returns a new function with the first arguments pre-filled.",
        "solution": "def parcial(fn, *args_fijos):\n    def wrapper(*args):\n        return fn(*args_fijos, *args)\n    return wrapper",
    },
    # ── Generadores ───────────────────────────────────────────────────────────
    {
        "problem": "Write a Python generator function `primos_gen(limite)` that yields all prime numbers up to limite.",
        "solution": "def primos_gen(limite):\n    def es_primo(n):\n        if n < 2:\n            return False\n        for i in range(2, int(n ** 0.5) + 1):\n            if n % i == 0:\n                return False\n        return True\n    for n in range(2, limite + 1):\n        if es_primo(n):\n            yield n",
    },
    {
        "problem": "Write a Python generator function `fibonacci_gen(n)` that yields the first n Fibonacci numbers.",
        "solution": "def fibonacci_gen(n):\n    a, b = 0, 1\n    for _ in range(n):\n        yield a\n        a, b = b, a + b",
    },
    {
        "problem": "Write a Python generator function `rango_cuadrados(inicio, fin)` that yields the squares of all integers from inicio to fin inclusive.",
        "solution": "def rango_cuadrados(inicio, fin):\n    for i in range(inicio, fin + 1):\n        yield i * i",
    },
    # ── Recursión ─────────────────────────────────────────────────────────────
    {
        "problem": "Write a Python function `suma_recursiva(lista)` that computes the sum of a list recursively without using sum().",
        "solution": "def suma_recursiva(lista):\n    if not lista:\n        return 0\n    return lista[0] + suma_recursiva(lista[1:])",
    },
    {
        "problem": "Write a Python function `potencia_recursiva(base, exp)` that computes base raised to exp using recursion.",
        "solution": "def potencia_recursiva(base, exp):\n    if exp == 0:\n        return 1\n    return base * potencia_recursiva(base, exp - 1)",
    },
    {
        "problem": "Write a Python function `aplanar_recursivo(lista)` that recursively flattens a nested list to any depth.",
        "solution": "def aplanar_recursivo(lista):\n    resultado = []\n    for item in lista:\n        if isinstance(item, list):\n            resultado.extend(aplanar_recursivo(item))\n        else:\n            resultado.append(item)\n    return resultado",
    },
    {
        "problem": "Write a Python function `invertir_recursivo(lista)` that reverses a list using recursion.",
        "solution": "def invertir_recursivo(lista):\n    if len(lista) <= 1:\n        return lista\n    return invertir_recursivo(lista[1:]) + [lista[0]]",
    },
    # ── Manejo de errores ─────────────────────────────────────────────────────
    {
        "problem": "Write a Python function `dividir_seguro(a, b)` that returns a / b, or None if b is zero.",
        "solution": "def dividir_seguro(a, b):\n    if b == 0:\n        return None\n    return a / b",
    },
    {
        "problem": "Write a Python function `parsear_entero(s)` that tries to parse string s as an integer, returning None on failure.",
        "solution": "def parsear_entero(s):\n    try:\n        return int(s)\n    except (ValueError, TypeError):\n        return None",
    },
    # ── Operaciones de conjuntos ──────────────────────────────────────────────
    {
        "problem": "Write a Python function `union_listas(a, b)` that returns a sorted list of all unique elements from both lists.",
        "solution": "def union_listas(a, b):\n    return sorted(set(a) | set(b))",
    },
    {
        "problem": "Write a Python function `interseccion_listas(a, b)` that returns a sorted list of elements that appear in both lists.",
        "solution": "def interseccion_listas(a, b):\n    return sorted(set(a) & set(b))",
    },
    {
        "problem": "Write a Python function `diferencia_listas(a, b)` that returns a sorted list of elements in a but not in b.",
        "solution": "def diferencia_listas(a, b):\n    return sorted(set(a) - set(b))",
    },
    # ── Matemáticas ───────────────────────────────────────────────────────────  
    {
        "problem": "Write a Python function `es_armstrong(n)` that returns True if n is an Armstrong number (sum of its digits each raised to the power of the number of digits equals n).",
        "solution": "def es_armstrong(n):\n    digitos = str(n)\n    p = len(digitos)\n    return sum(int(d) ** p for d in digitos) == n",
    },
    {
        "problem": "Write a Python function `raiz_cuadrada(n)` that computes the integer square root of n without using math.sqrt.",
        "solution": "def raiz_cuadrada(n):\n    if n < 0:\n        raise ValueError('No se puede calcular la raiz de un numero negativo')\n    if n == 0:\n        return 0\n    x = n\n    y = (x + 1) // 2\n    while y < x:\n        x = y\n        y = (x + n // x) // 2\n    return x",
    },
    {
        "problem": "Write a Python function `celsius_a_fahrenheit(c)` that converts Celsius to Fahrenheit.",
        "solution": "def celsius_a_fahrenheit(c):\n    return c * 9 / 5 + 32",
    },
    {
        "problem": "Write a Python function `fahrenheit_a_celsius(f)` that converts Fahrenheit to Celsius.",
        "solution": "def fahrenheit_a_celsius(f):\n    return (f - 32) * 5 / 9",
    },
    # ── Validación ────────────────────────────────────────────────────────────
    {
        "problem": "Write a Python function `es_email_valido(email)` that returns True if the string has a basic valid email format (contains exactly one @ and at least one dot after the @).",
        "solution": "def es_email_valido(email):\n    partes = email.split('@')\n    if len(partes) != 2:\n        return False\n    dominio = partes[1]\n    return '.' in dominio and not dominio.startswith('.') and not dominio.endswith('.')",
    },
    {
        "problem": "Write a Python function `es_solo_letras(s)` that returns True if the string contains only alphabetic characters.",
        "solution": "def es_solo_letras(s):\n    return s.isalpha()",
    },
    {
        "problem": "Write a Python function `es_numero(s)` that returns True if the string represents a valid number (integer or float).",
        "solution": "def es_numero(s):\n    try:\n        float(s)\n        return True\n    except ValueError:\n        return False",
    },
    # ── Lógica combinatoria ───────────────────────────────────────────────────
    {
        "problem": "Write a Python function `combinaciones(lista, k)` that returns all k-length combinations of the list as a list of lists.",
        "solution": "def combinaciones(lista, k):\n    if k == 0:\n        return [[]]\n    if not lista:\n        return []\n    primero = lista[0]\n    resto = lista[1:]\n    con = [[primero] + c for c in combinaciones(resto, k - 1)]\n    sin = combinaciones(resto, k)\n    return con + sin",
    },
    {
        "problem": "Write a Python function `permutaciones(lista)` that returns all permutations of the list as a list of lists.",
        "solution": "def permutaciones(lista):\n    if len(lista) <= 1:\n        return [lista[:]]\n    resultado = []\n    for i, elem in enumerate(lista):\n        resto = lista[:i] + lista[i+1:]\n        for perm in permutaciones(resto):\n            resultado.append([elem] + perm)\n    return resultado",
    },
    # ── Cadenas avanzadas ─────────────────────────────────────────────────────
    {
        "problem": "Write a Python function `palabras_mas_largas(texto, n)` that returns a list of the n longest words in the text (sorted by length descending).",
        "solution": "def palabras_mas_largas(texto, n):\n    palabras = texto.split()\n    return sorted(set(palabras), key=len, reverse=True)[:n]",
    },
    {
        "problem": "Write a Python function `rotar_cadena(s, k)` that rotates the string k positions to the left.",
        "solution": "def rotar_cadena(s, k):\n    if not s:\n        return s\n    k = k % len(s)\n    return s[k:] + s[:k]",
    },
    {
        "problem": "Write a Python function `formato_moneda(n, simbolo)` that formats a number as currency with 2 decimal places and the given symbol prefix.",
        "solution": "def formato_moneda(n, simbolo):\n    return f'{simbolo}{n:.2f}'",
    },
    # ── Más estructuras de datos ───────────────────────────────────────────────
    {
        "problem": "Write a Python class `MinHeap` with methods `insertar(val)` and `extraer_min()` that implements a min-heap using a list.",
        "solution": "import heapq\n\nclass MinHeap:\n    def __init__(self):\n        self._datos = []\n\n    def insertar(self, val):\n        heapq.heappush(self._datos, val)\n\n    def extraer_min(self):\n        if not self._datos:\n            raise IndexError('Heap vacio')\n        return heapq.heappop(self._datos)",
    },
    {
        "problem": "Write a Python class `Grafo` with method `agregar_arista(u, v)` to add an undirected edge and `vecinos(u)` to return neighbors of node u.",
        "solution": "class Grafo:\n    def __init__(self):\n        self._adj = {}\n\n    def agregar_arista(self, u, v):\n        self._adj.setdefault(u, []).append(v)\n        self._adj.setdefault(v, []).append(u)\n\n    def vecinos(self, u):\n        return self._adj.get(u, [])",
    },
    # ── Algoritmos de grafos ───────────────────────────────────────────────────
    {
        "problem": "Write a Python function `bfs(grafo, inicio)` that performs a breadth-first search on an adjacency list graph and returns the list of visited nodes in order.",
        "solution": "from collections import deque\n\ndef bfs(grafo, inicio):\n    visitados = []\n    cola = deque([inicio])\n    seen = {inicio}\n    while cola:\n        nodo = cola.popleft()\n        visitados.append(nodo)\n        for vecino in grafo.get(nodo, []):\n            if vecino not in seen:\n                seen.add(vecino)\n                cola.append(vecino)\n    return visitados",
    },
    {
        "problem": "Write a Python function `dfs(grafo, inicio)` that performs a depth-first search on an adjacency list graph and returns the list of visited nodes in order.",
        "solution": "def dfs(grafo, inicio):\n    visitados = []\n    pila = [inicio]\n    seen = set()\n    while pila:\n        nodo = pila.pop()\n        if nodo in seen:\n            continue\n        seen.add(nodo)\n        visitados.append(nodo)\n        for vecino in reversed(grafo.get(nodo, [])):\n            if vecino not in seen:\n                pila.append(vecino)\n    return visitados",
    },
]


def _generar_variantes(ejemplo: dict) -> list[dict]:
    """Genera variantes sintácticas del mismo ejemplo para aumentar el dataset."""
    problem = ejemplo["problem"]
    solution = ejemplo["solution"]
    variantes = [{"problem": problem, "solution": solution}]

    # Pequeñas variaciones del enunciado
    variaciones_intro = [
        "Write a Python function",
        "Implement a Python function",
        "Create a Python function",
        "Define a Python function",
        "Write a Python",
    ]
    for intro in variaciones_intro:
        if problem.startswith("Write a Python function") and intro != "Write a Python function":
            variante_problem = problem.replace("Write a Python function", intro, 1)
            variantes.append({"problem": variante_problem, "solution": solution})
        elif problem.startswith("Write a Python class") and "class" in intro.lower():
            break

    # Solo las 3 primeras variantes para no explotar el dataset
    return variantes[:3]


def _ejemplo_a_texto(problem: str, solution: str) -> str:
    """Convierte un par problem/solution al formato de entrenamiento."""
    return f"### Problem:\n{problem}\n### Solution:\n```python\n{solution}\n```"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent.parent / "data" / "targeted_sft.jsonl")
    parser.add_argument("--repeticiones", type=int, default=5,
                        help="Cuántas veces repetir el dataset base")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    todos: list[str] = []
    for ej in EJEMPLOS_BASE:
        for variante in _generar_variantes(ej):
            todos.append(_ejemplo_a_texto(variante["problem"], variante["solution"]))

    # Repetir para dar más épocas de exposición
    dataset = (todos * args.repeticiones)
    random.shuffle(dataset)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for texto in dataset:
            f.write(json.dumps({"text": texto}, ensure_ascii=False) + "\n")

    print(f"Dataset generado: {len(dataset)} ejemplos → {args.output}")
    print(f"Ejemplos base: {len(EJEMPLOS_BASE)}")
    print(f"Con variantes: {len(todos)}")
    print(f"Total con x{args.repeticiones} repeticiones: {len(dataset)}")


if __name__ == "__main__":
    main()
