#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Generador de datos sintéticos para temas con pocos ejemplos en la biblioteca.

Genera ejemplos de código Python usando templates con variaciones,
SIN requerir ninguna API externa. Todo local.

Uso:
  python scripts/generar_temas_vacios.py
  python scripts/generar_temas_vacios.py --temas collections dataclasses
  python scripts/generar_temas_vacios.py --por-tema 500 --modo append
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

random.seed(42)

# =============================================================================
# HELPERS
# =============================================================================

def jsonl_entry(text: str, tema: str) -> str:
    """Devuelve una línea JSONL con el formato estándar PAMPAr."""
    return json.dumps(
        {"text": text, "source": "generated_templates", "license": "open",
         "lang": "python", "tema": tema},
        ensure_ascii=False,
    )


def shuffle_and_sample(items: list, n: int) -> list:
    """Devuelve n items del listado (con repetición si n > len)."""
    if n <= len(items):
        return random.sample(items, n)
    # Repetir con shuffles distintos para alcanzar n
    result = []
    while len(result) < n:
        shuffled = items[:]
        random.shuffle(shuffled)
        result.extend(shuffled)
    return result[:n]


# =============================================================================
# GENERADORES POR TEMA
# =============================================================================

def generar_collections(n: int) -> list[str]:
    """Genera ejemplos del módulo collections."""
    entries = []

    # --- Counter ---
    counter_examples = [
        """from collections import Counter

words = ['apple', 'banana', 'apple', 'orange', 'banana', 'apple']
count = Counter(words)
print(count)  # Counter({'apple': 3, 'banana': 2, 'orange': 1})
print(count.most_common(2))  # [('apple', 3), ('banana', 2)]""",

        """from collections import Counter

text = "hello world hello python"
letter_count = Counter(text.split())
print(letter_count['hello'])  # 2
print(letter_count.most_common(1))  # [('hello', 2)]""",

        """from collections import Counter

def most_frequent(items):
    \"\"\"Retorna el elemento más frecuente en una lista.\"\"\"
    c = Counter(items)
    return c.most_common(1)[0][0]

result = most_frequent([1, 2, 2, 3, 3, 3])
print(result)  # 3""",

        """from collections import Counter

# Aritmética de Counters
a = Counter({'x': 3, 'y': 2})
b = Counter({'x': 1, 'y': 5})
print(a + b)  # Counter({'y': 7, 'x': 4})
print(a - b)  # Counter({'x': 2})
print(a & b)  # Counter({'x': 1, 'y': 2})  # intersección (mínimo)
print(a | b)  # Counter({'y': 5, 'x': 3})  # unión (máximo)""",

        """from collections import Counter

# Contar caracteres en un string
word = "mississippi"
freq = Counter(word)
print(freq.most_common(3))
# [('s', 4), ('i', 4), ('p', 2)]""",

        """from collections import Counter

def contar_duplicados(lista):
    \"\"\"Retorna solo los elementos que aparecen más de una vez.\"\"\"
    c = Counter(lista)
    return {elem: count for elem, count in c.items() if count > 1}

print(contar_duplicados([1, 2, 2, 3, 3, 3, 4]))
# {2: 2, 3: 3}""",
    ]

    # --- defaultdict ---
    defaultdict_examples = [
        """from collections import defaultdict

# Agrupar palabras por longitud
words = ['hi', 'hello', 'hey', 'world', 'hi']
by_length = defaultdict(list)
for word in words:
    by_length[len(word)].append(word)

print(dict(by_length))
# {2: ['hi', 'hi'], 5: ['hello', 'world'], 3: ['hey']}""",

        """from collections import defaultdict

# defaultdict con int — contar sin KeyError
counter = defaultdict(int)
for char in "banana":
    counter[char] += 1

print(dict(counter))
# {'b': 1, 'a': 3, 'n': 2}""",

        """from collections import defaultdict

# Grafo como lista de adyacencia
def construir_grafo(edges):
    grafo = defaultdict(list)
    for u, v in edges:
        grafo[u].append(v)
        grafo[v].append(u)
    return grafo

g = construir_grafo([(1, 2), (1, 3), (2, 4)])
print(g[1])  # [2, 3]""",

        """from collections import defaultdict

# defaultdict con set — evitar duplicados
grupos = defaultdict(set)
datos = [('A', 1), ('B', 2), ('A', 3), ('B', 2)]
for key, val in datos:
    grupos[key].add(val)

print(dict(grupos))  # {'A': {1, 3}, 'B': {2}}""",

        """from collections import defaultdict

# Nested defaultdict
tabla = defaultdict(lambda: defaultdict(int))
puntos = [('Alice', 'math', 95), ('Bob', 'math', 80), ('Alice', 'english', 90)]
for nombre, materia, nota in puntos:
    tabla[nombre][materia] = nota

print(tabla['Alice']['math'])   # 95
print(tabla['Bob']['english'])  # 0 (default)""",
    ]

    # --- deque ---
    deque_examples = [
        """from collections import deque

# deque como cola doble eficiente (O(1) en ambos extremos)
d = deque([1, 2, 3])
d.appendleft(0)
d.append(4)
print(d)        # deque([0, 1, 2, 3, 4])
print(d.popleft())  # 0
print(d.pop())      # 4""",

        """from collections import deque

# Ventana deslizante con maxlen
def moving_average(nums, window):
    \"\"\"Calcula el promedio móvil con deque de ancho fijo.\"\"\"
    dq = deque(maxlen=window)
    result = []
    for n in nums:
        dq.append(n)
        if len(dq) == window:
            result.append(sum(dq) / window)
    return result

print(moving_average([1, 2, 3, 4, 5], 3))
# [2.0, 3.0, 4.0]""",

        """from collections import deque

# BFS usando deque
def bfs(grafo, inicio):
    visitados = set()
    cola = deque([inicio])
    orden = []
    while cola:
        nodo = cola.popleft()
        if nodo not in visitados:
            visitados.add(nodo)
            orden.append(nodo)
            cola.extend(grafo.get(nodo, []))
    return orden

grafo = {1: [2, 3], 2: [4], 3: [4], 4: []}
print(bfs(grafo, 1))  # [1, 2, 3, 4]""",

        """from collections import deque

# deque con rotación
d = deque([1, 2, 3, 4, 5])
d.rotate(2)   # Rotar 2 posiciones a la derecha
print(d)      # deque([4, 5, 1, 2, 3])
d.rotate(-2)  # Rotar 2 a la izquierda
print(d)      # deque([1, 2, 3, 4, 5])""",
    ]

    # --- namedtuple ---
    namedtuple_examples = [
        """from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])
p = Point(3, 4)
print(p.x, p.y)   # 3 4
print(p)           # Point(x=3, y=4)

# Inmutable como tupla normal
distance = (p.x**2 + p.y**2)**0.5
print(f"Distancia: {distance:.2f}")  # 5.00""",

        """from collections import namedtuple

# namedtuple como DTO ligero
Employee = namedtuple('Employee', ['name', 'salary', 'department'])
e = Employee(name='Alice', salary=75000, department='Engineering')
print(e.name)        # Alice
print(e._asdict())   # OrderedDict([...])
e2 = e._replace(salary=80000)
print(e2.salary)     # 80000""",

        """from collections import namedtuple

# Usar namedtuple para parsear CSV
Row = namedtuple('Row', ['name', 'age', 'city'])
lines = [
    "Alice,30,Madrid",
    "Bob,25,Barcelona",
]
for line in lines:
    row = Row(*line.split(','))
    print(f"{row.name} vive en {row.city}")""",
    ]

    # --- OrderedDict ---
    ordereddict_examples = [
        """from collections import OrderedDict

# OrderedDict recuerda el orden de inserción (útil en Python < 3.7)
od = OrderedDict()
od['first'] = 1
od['second'] = 2
od['third'] = 3

for key, val in od.items():
    print(key, val)

# Mover al final o al inicio
od.move_to_end('first')
print(list(od.keys()))  # ['second', 'third', 'first']""",

        """from collections import OrderedDict

# Implementar LRU Cache simple con OrderedDict
class LRUCache:
    \"\"\"Implementación simple de LRU Cache.\"\"\"

    def __init__(self, capacidad):
        self.cache = OrderedDict()
        self.capacidad = capacidad

    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)  # Marcar como reciente
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacidad:
            self.cache.popitem(last=False)  # Eliminar el más antiguo

cache = LRUCache(2)
cache.put(1, 'a')
cache.put(2, 'b')
print(cache.get(1))   # 'a'
cache.put(3, 'c')     # Evicta el 2
print(cache.get(2))   # -1""",
    ]

    # Preguntas y respuestas sobre collections
    qa_examples = [
        """# ¿Cómo encontrar el elemento más común en una lista con collections?
# Usa Counter.most_common()

from collections import Counter

def elemento_mas_comun(lista):
    return Counter(lista).most_common(1)[0][0]

print(elemento_mas_comun(['a', 'b', 'a', 'c', 'a']))  # 'a'""",

        """# ¿Diferencia entre dict y defaultdict?
# defaultdict devuelve un valor por defecto sin KeyError

from collections import defaultdict

# dict normal
d = {}
# d['key'] += 1  # KeyError si 'key' no existe

# defaultdict
dd = defaultdict(int)
dd['key'] += 1   # OK: empieza en 0 automáticamente
print(dd['key'])  # 1""",

        """# ¿Cuándo usar deque vs list para una cola?
# deque es O(1) para append/popleft; list.pop(0) es O(n)

from collections import deque
import time

# List como cola (ineficiente)
# queue = []
# queue.append(x)  # O(1)
# queue.pop(0)     # O(n) — desplaza todos los elementos

# deque como cola (eficiente)
cola = deque()
cola.append('tarea1')
cola.append('tarea2')
primera = cola.popleft()  # O(1)
print(primera)  # tarea1""",
    ]

    all_examples = (
        counter_examples + defaultdict_examples + deque_examples +
        namedtuple_examples + ordereddict_examples + qa_examples
    )

    # Generar n entries con variaciones
    for _ in range(n):
        ex = random.choice(all_examples)
        entries.append(jsonl_entry(ex, "collections"))

    return entries


def generar_dataclasses(n: int) -> list[str]:
    """Genera ejemplos del módulo dataclasses."""
    entries = []

    examples = [
        """from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

p = Point(3.0, 4.0)
print(p)          # Point(x=3.0, y=4.0)
print(p.x, p.y)   # 3.0 4.0""",

        """from dataclasses import dataclass, field

@dataclass
class Student:
    name: str
    age: int
    grades: list[float] = field(default_factory=list)

    def average(self) -> float:
        if not self.grades:
            return 0.0
        return sum(self.grades) / len(self.grades)

s = Student("Alice", 20, [9.0, 8.5, 9.5])
print(s.average())  # 9.0""",

        """from dataclasses import dataclass

@dataclass(frozen=True)
class Vector:
    \"\"\"Vector inmutable — puede usarse como clave de dict o en sets.\"\"\"
    x: float
    y: float

    def magnitude(self) -> float:
        return (self.x**2 + self.y**2)**0.5

    def __add__(self, other: 'Vector') -> 'Vector':
        return Vector(self.x + other.x, self.y + other.y)

v1 = Vector(3.0, 4.0)
v2 = Vector(1.0, 2.0)
print(v1 + v2)          # Vector(x=4.0, y=6.0)
print(v1.magnitude())   # 5.0""",

        """from dataclasses import dataclass, field, asdict

@dataclass
class Config:
    host: str = "localhost"
    port: int = 8080
    debug: bool = False
    tags: list[str] = field(default_factory=list)

cfg = Config(port=9000, tags=["api", "v1"])
print(cfg)
# Config(host='localhost', port=9000, debug=False, tags=['api', 'v1'])

# Convertir a dict para serializar
d = asdict(cfg)
print(d['port'])  # 9000""",

        """from dataclasses import dataclass, field
import json

@dataclass
class Product:
    id: int
    name: str
    price: float
    tags: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> 'Product':
        return cls(**data)

p = Product(1, "Laptop", 1499.99, ["tech", "computers"])
print(p.to_json())""",

        """from dataclasses import dataclass, field
from typing import ClassVar

@dataclass
class Counter:
    \"\"\"Contador con máximo configurable.\"\"\"
    MAX_VALUE: ClassVar[int] = 100
    value: int = 0

    def increment(self, by: int = 1) -> None:
        self.value = min(self.value + by, self.MAX_VALUE)

    def reset(self) -> None:
        self.value = 0

c = Counter()
c.increment(50)
c.increment(70)  # No supera MAX_VALUE
print(c.value)   # 100""",

        """from dataclasses import dataclass, field

@dataclass(order=True)
class Task:
    \"\"\"Tarea con prioridad — comparable por prioridad.\"\"\"
    priority: int
    name: str = field(compare=False)  # No entra en comparación
    done: bool = field(default=False, compare=False)

tasks = [Task(2, "Email"), Task(1, "Bug fix"), Task(3, "Docs")]
tasks.sort()
for t in tasks:
    print(f"[{t.priority}] {t.name}")
# [1] Bug fix
# [2] Email
# [3] Docs""",

        """from dataclasses import dataclass, asdict, astuple

@dataclass
class RGB:
    red: int
    green: int
    blue: int

    def to_hex(self) -> str:
        return f"#{self.red:02X}{self.green:02X}{self.blue:02X}"

color = RGB(255, 128, 0)
print(color.to_hex())           # #FF8000
print(asdict(color))            # {'red': 255, 'green': 128, 'blue': 0}
print(astuple(color))           # (255, 128, 0)
r, g, b = astuple(color)
print(r, g, b)                  # 255 128 0""",

        """from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Node:
    \"\"\"Nodo de lista enlazada implementado con dataclass.\"\"\"
    value: int
    next: Optional['Node'] = field(default=None, repr=False)

def build_linked_list(values: list[int]) -> Optional[Node]:
    if not values:
        return None
    head = Node(values[0])
    curr = head
    for v in values[1:]:
        curr.next = Node(v)
        curr = curr.next
    return head

head = build_linked_list([1, 2, 3])
print(head.value)       # 1
print(head.next.value)  # 2""",

        """from dataclasses import dataclass

# ¿Por qué usar @dataclass en lugar de __init__ manual?
# Genera automáticamente: __init__, __repr__, __eq__

# Sin dataclass (verboso)
class PointManual:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

# Con dataclass (limpio)
@dataclass
class Point:
    x: float
    y: float

# Equivalentes pero dataclass es mucho más conciso
p1 = Point(1.0, 2.0)
p2 = Point(1.0, 2.0)
print(p1 == p2)   # True (automático)
print(repr(p1))   # Point(x=1.0, y=2.0) (automático)""",

        """from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class LogEntry:
    message: str
    level: str = "INFO"
    timestamp: datetime = field(default_factory=datetime.now)
    tags: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        ts = self.timestamp.strftime("%H:%M:%S")
        return f"[{ts}] [{self.level}] {self.message}"

entry = LogEntry("Server started", "INFO", tags=["startup"])
print(entry)  # [HH:MM:SS] [INFO] Server started""",

        """from dataclasses import dataclass

@dataclass
class BankAccount:
    owner: str
    balance: float = 0.0

    def deposit(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("El monto debe ser positivo")
        self.balance += amount

    def withdraw(self, amount: float) -> None:
        if amount > self.balance:
            raise ValueError("Saldo insuficiente")
        self.balance -= amount

    def __str__(self) -> str:
        return f"Cuenta de {self.owner}: ${self.balance:.2f}"

acc = BankAccount("Alice")
acc.deposit(1000)
acc.withdraw(250)
print(acc)  # Cuenta de Alice: $750.00""",
    ]

    for _ in range(n):
        ex = random.choice(examples)
        entries.append(jsonl_entry(ex, "dataclasses"))

    return entries


def generar_functools(n: int) -> list[str]:
    """Genera ejemplos del módulo functools."""
    entries = []

    examples = [
        """from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci(n: int) -> int:
    \"\"\"Fibonacci con memoización automática.\"\"\"
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(10))   # 55
print(fibonacci(30))   # 832040
print(fibonacci.cache_info())
# CacheInfo(hits=28, misses=31, maxsize=None, currsize=31)""",

        """from functools import cache

# @cache: shorthand de @lru_cache(maxsize=None) desde Python 3.9
@cache
def factorial(n: int) -> int:
    if n <= 1:
        return 1
    return n * factorial(n - 1)

print(factorial(10))  # 3628800
print(factorial.cache_info())""",

        """from functools import partial

def potencia(base: float, exponente: float) -> float:
    return base ** exponente

cuadrado = partial(potencia, exponente=2)
cubo = partial(potencia, exponente=3)

print(cuadrado(5))   # 25.0
print(cubo(3))       # 27.0

# Útil para callbacks y event handlers
numeros = [1, 2, 3, 4, 5]
cuadrados = list(map(cuadrado, numeros))
print(cuadrados)  # [1.0, 4.0, 9.0, 16.0, 25.0]""",

        """from functools import partial

def log(level: str, message: str, prefix: str = "App") -> str:
    return f"[{prefix}][{level}] {message}"

# Crear funciones especializadas con partial
info = partial(log, "INFO")
error = partial(log, "ERROR")
debug = partial(log, "DEBUG", prefix="Debug")

print(info("Server started"))          # [App][INFO] Server started
print(error("Connection failed"))      # [App][ERROR] Connection failed
print(debug("Processing request"))     # [Debug][DEBUG] Processing request""",

        """from functools import reduce
from operator import add, mul

numeros = [1, 2, 3, 4, 5]

# Suma con reduce
total = reduce(add, numeros)
print(total)  # 15

# Producto con reduce
producto = reduce(mul, numeros)
print(producto)  # 120

# Implementación manual de flatten
listas = [[1, 2], [3, 4], [5, 6]]
plana = reduce(lambda a, b: a + b, listas)
print(plana)  # [1, 2, 3, 4, 5, 6]""",

        """from functools import wraps
from typing import Callable
import time

def timer(func: Callable) -> Callable:
    \"\"\"Decorador que mide el tiempo de ejecución.\"\"\"
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} tardó {elapsed:.4f}s")
        return result
    return wrapper

@timer
def process(n: int) -> list[int]:
    return [i**2 for i in range(n)]

result = process(10000)
# process tardó 0.001s""",

        """from functools import wraps

def retry(times: int = 3):
    \"\"\"Decorador que reintenta una función si lanza excepción.\"\"\"
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(times):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    print(f"Intento {attempt + 1} fallido: {e}")
            raise last_error
        return wrapper
    return decorator

@retry(times=3)
def fetch_data(url: str) -> dict:
    # Simular error de red
    raise ConnectionError("Timeout")""",

        """from functools import wraps

# ¿Por qué usar @wraps en decoradores?
# Sin @wraps, el decorador destruye los metadatos de la función original

def sin_wraps(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def con_wraps(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@sin_wraps
def mi_funcion():
    \"\"\"Esta es mi función.\"\"\"
    pass

print(mi_funcion.__name__)  # 'wrapper' (incorrecto!)
print(mi_funcion.__doc__)   # None (perdido!)

@con_wraps
def mi_funcion2():
    \"\"\"Esta es mi función.\"\"\"
    pass

print(mi_funcion2.__name__)  # 'mi_funcion2' (correcto)
print(mi_funcion2.__doc__)   # 'Esta es mi función.' (preservado)""",

        """from functools import lru_cache

# lru_cache para memoizar llamadas costosas
@lru_cache(maxsize=128)
def calcular_ruta(origen: str, destino: str) -> int:
    \"\"\"Simula cálculo costoso de ruta.\"\"\"
    # En producción aquí iría un algoritmo real
    return hash(origen + destino) % 1000 + 1

# Primera llamada: calcula
r1 = calcular_ruta("Madrid", "Barcelona")
# Segunda llamada con mismos args: usa caché (no recalcula)
r2 = calcular_ruta("Madrid", "Barcelona")
print(r1 == r2)  # True

# Limpiar caché si los datos subyacentes cambian
calcular_ruta.cache_clear()""",

        """from functools import total_ordering

@total_ordering
class Temperatura:
    \"\"\"Con @total_ordering solo hay que definir __eq__ y __lt__.
    El resto (__le__, __gt__, __ge__) se generan automáticamente.\"\"\"

    def __init__(self, grados: float):
        self.grados = grados

    def __eq__(self, other) -> bool:
        return self.grados == other.grados

    def __lt__(self, other) -> bool:
        return self.grados < other.grados

    def __repr__(self) -> str:
        return f"Temperatura({self.grados}°C)"

t1 = Temperatura(20.0)
t2 = Temperatura(30.0)
print(t1 < t2)   # True
print(t1 > t2)   # False  (generado por @total_ordering)
print(t2 >= t1)  # True   (generado)""",
    ]

    for _ in range(n):
        ex = random.choice(examples)
        entries.append(jsonl_entry(ex, "functools"))

    return entries


def generar_singleton_factory(n: int) -> list[str]:
    """Genera ejemplos de patrones Singleton y Factory."""
    entries = []

    examples = [
        # --- Singleton ---
        """# Patrón Singleton — garantiza una sola instancia de la clase

class DatabaseConnection:
    \"\"\"Singleton para conexión a base de datos.\"\"\"
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._connected = False
        return cls._instance

    def connect(self, url: str) -> None:
        self._connected = True
        self._url = url

    def is_connected(self) -> bool:
        return self._connected

db1 = DatabaseConnection()
db2 = DatabaseConnection()
db1.connect("postgresql://localhost/mydb")
print(db1 is db2)              # True — misma instancia
print(db2.is_connected())      # True""",

        """# Singleton con metaclase — más Pythónico

class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Config(metaclass=SingletonMeta):
    def __init__(self):
        self.settings = {}

    def set(self, key: str, value) -> None:
        self.settings[key] = value

    def get(self, key: str, default=None):
        return self.settings.get(key, default)

c1 = Config()
c2 = Config()
c1.set("debug", True)
print(c2.get("debug"))  # True — misma instancia""",

        """# Singleton thread-safe con lock

import threading

class Logger:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._logs = []
        return cls._instance

    def log(self, message: str) -> None:
        self._logs.append(message)

    def get_logs(self) -> list[str]:
        return self._logs.copy()

log1 = Logger()
log2 = Logger()
log1.log("Inicio del servidor")
print(log2.get_logs())  # ['Inicio del servidor']""",

        # --- Factory ---
        """# Patrón Factory — crea objetos sin exponer la lógica de creación

from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def speak(self) -> str:
        ...

class Dog(Animal):
    def speak(self) -> str:
        return "Woof!"

class Cat(Animal):
    def speak(self) -> str:
        return "Meow!"

class Bird(Animal):
    def speak(self) -> str:
        return "Tweet!"

def animal_factory(tipo: str) -> Animal:
    \"\"\"Factory function — elige la clase correcta según el tipo.\"\"\"
    registro = {"dog": Dog, "cat": Cat, "bird": Bird}
    if tipo not in registro:
        raise ValueError(f"Tipo desconocido: {tipo}")
    return registro[tipo]()

for tipo in ["dog", "cat", "bird"]:
    animal = animal_factory(tipo)
    print(f"{tipo}: {animal.speak()}")""",

        """# Factory Method como método de clase

from typing import Literal

class Connection:
    def __init__(self, host: str, port: int, protocol: str):
        self.host = host
        self.port = port
        self.protocol = protocol

    @classmethod
    def http(cls, host: str) -> 'Connection':
        return cls(host, 80, "HTTP")

    @classmethod
    def https(cls, host: str) -> 'Connection':
        return cls(host, 443, "HTTPS")

    @classmethod
    def ftp(cls, host: str) -> 'Connection':
        return cls(host, 21, "FTP")

    def __repr__(self) -> str:
        return f"Connection({self.protocol}://{self.host}:{self.port})"

c1 = Connection.https("api.example.com")
c2 = Connection.http("old.example.com")
print(c1)  # Connection(HTTPS://api.example.com:443)""",

        """# Abstract Factory — familias de objetos relacionados

from abc import ABC, abstractmethod

class Button(ABC):
    @abstractmethod
    def render(self) -> str: ...

class TextInput(ABC):
    @abstractmethod
    def render(self) -> str: ...

class DarkButton(Button):
    def render(self) -> str:
        return "<button class='dark'>Click</button>"

class DarkInput(TextInput):
    def render(self) -> str:
        return "<input class='dark' />"

class LightButton(Button):
    def render(self) -> str:
        return "<button class='light'>Click</button>"

class LightInput(TextInput):
    def render(self) -> str:
        return "<input class='light' />"

class UIFactory(ABC):
    @abstractmethod
    def create_button(self) -> Button: ...
    @abstractmethod
    def create_input(self) -> TextInput: ...

class DarkThemeFactory(UIFactory):
    def create_button(self) -> Button:
        return DarkButton()
    def create_input(self) -> TextInput:
        return DarkInput()

class LightThemeFactory(UIFactory):
    def create_button(self) -> Button:
        return LightButton()
    def create_input(self) -> TextInput:
        return LightInput()

# Uso
factory: UIFactory = DarkThemeFactory()
print(factory.create_button().render())
print(factory.create_input().render())""",

        """# Singleton como módulo — la forma más Pythónica

# En Python, los módulos ya son singletons por naturaleza.
# La mejor implementación de Singleton es a menudo simplemente un módulo.

# config.py (módulo singleton)
class _Config:
    def __init__(self):
        self.debug = False
        self.db_url = "sqlite:///default.db"
        self.api_key = ""

    def load_from_env(self) -> None:
        import os
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.db_url = os.getenv("DATABASE_URL", self.db_url)

# Instancia única a nivel de módulo
config = _Config()

# Uso desde otros módulos:
# from config import config
# config.load_from_env()
# print(config.debug)""",

        """# Registro de productos con Factory

class Shape:
    pass

class Circle(Shape):
    def __init__(self, radius: float):
        self.radius = radius
    def area(self) -> float:
        import math
        return math.pi * self.radius ** 2

class Rectangle(Shape):
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
    def area(self) -> float:
        return self.width * self.height

class ShapeFactory:
    _registry: dict = {}

    @classmethod
    def register(cls, name: str, shape_class):
        cls._registry[name] = shape_class

    @classmethod
    def create(cls, name: str, **kwargs) -> Shape:
        if name not in cls._registry:
            raise ValueError(f"Shape '{name}' no registrada")
        return cls._registry[name](**kwargs)

ShapeFactory.register("circle", Circle)
ShapeFactory.register("rectangle", Rectangle)

c = ShapeFactory.create("circle", radius=5.0)
r = ShapeFactory.create("rectangle", width=4.0, height=3.0)
print(f"Círculo: {c.area():.2f}")      # 78.54
print(f"Rectángulo: {r.area():.2f}")   # 12.00""",
    ]

    for _ in range(n):
        ex = random.choice(examples)
        entries.append(jsonl_entry(ex, "singleton_factory"))

    return entries


def generar_itertools(n: int) -> list[str]:
    """Genera ejemplos adicionales para itertools."""
    entries = []

    examples = [
        """import itertools

# chain — iterar sobre múltiples iterables como si fueran uno
a = [1, 2, 3]
b = [4, 5, 6]
c = [7, 8, 9]
resultado = list(itertools.chain(a, b, c))
print(resultado)  # [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Útil para aplanar una lista de listas
listas = [[1, 2], [3, 4], [5]]
plana = list(itertools.chain.from_iterable(listas))
print(plana)  # [1, 2, 3, 4, 5]""",

        """import itertools

# combinations — todas las combinaciones sin repetición
items = ['A', 'B', 'C', 'D']
for combo in itertools.combinations(items, 2):
    print(combo)
# ('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'C'), ('B', 'D'), ('C', 'D')

total = len(list(itertools.combinations(items, 2)))
print(f"Total: {total}")  # 6""",

        """import itertools

# permutations — todos los ordenamientos posibles
items = [1, 2, 3]
for perm in itertools.permutations(items):
    print(perm)
# (1,2,3), (1,3,2), (2,1,3), (2,3,1), (3,1,2), (3,2,1)

# Solo permutaciones de longitud r
for perm in itertools.permutations([1, 2, 3], 2):
    print(perm)""",

        """import itertools

# product — producto cartesiano (equivalente a bucles anidados)
colores = ['rojo', 'azul']
tamaños = ['S', 'M', 'L']

variantes = list(itertools.product(colores, tamaños))
print(variantes)
# [('rojo','S'),('rojo','M'),('rojo','L'),('azul','S'),('azul','M'),('azul','L')]

# product consigo mismo — equivalente a repetición
dados = list(itertools.product(range(1, 7), repeat=2))
print(len(dados))  # 36 combinaciones""",

        """import itertools

# groupby — agrupa elementos consecutivos
from operator import itemgetter

datos = [
    ('Alice', 'Engineering'),
    ('Bob', 'Engineering'),
    ('Carol', 'Marketing'),
    ('Dave', 'Marketing'),
    ('Eve', 'HR'),
]
# IMPORTANTE: datos deben estar ordenados por la clave para groupby
datos.sort(key=itemgetter(1))

for departamento, empleados in itertools.groupby(datos, key=itemgetter(1)):
    print(f"{departamento}: {[e[0] for e in empleados]}")""",

        """import itertools

# islice — slicing de iteradores (sin cargar todo en memoria)
def numeros_infinitos():
    n = 0
    while True:
        yield n
        n += 1

# Tomar los primeros 10 sin materializar el infinito
primeros_10 = list(itertools.islice(numeros_infinitos(), 10))
print(primeros_10)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# islice con start, stop, step
pares = list(itertools.islice(numeros_infinitos(), 0, 20, 2))
print(pares)  # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]""",

        """import itertools

# takewhile y dropwhile
datos = [2, 4, 6, 7, 8, 10, 11, 12]

# takewhile: toma mientras se cumpla la condición
pares_iniciales = list(itertools.takewhile(lambda x: x % 2 == 0, datos))
print(pares_iniciales)  # [2, 4, 6]

# dropwhile: salta mientras se cumpla, luego toma todo
desde_impar = list(itertools.dropwhile(lambda x: x % 2 == 0, datos))
print(desde_impar)  # [7, 8, 10, 11, 12]""",

        """import itertools

# accumulate — sumas acumuladas (también otras operaciones)
import operator

nums = [1, 2, 3, 4, 5]
sumas = list(itertools.accumulate(nums))
print(sumas)  # [1, 3, 6, 10, 15]

# Producto acumulado
productos = list(itertools.accumulate(nums, operator.mul))
print(productos)  # [1, 2, 6, 24, 120]

# Máximo acumulado (running max)
datos = [3, 1, 4, 1, 5, 9, 2, 6]
running_max = list(itertools.accumulate(datos, max))
print(running_max)  # [3, 3, 4, 4, 5, 9, 9, 9]""",

        """import itertools

# combinations_with_replacement — combinaciones con repetición
# (ej: cuántas formas de elegir 2 monedas de {1, 5, 10} con repetición)
monedas = [1, 5, 10]
for combo in itertools.combinations_with_replacement(monedas, 2):
    print(combo)
# (1,1), (1,5), (1,10), (5,5), (5,10), (10,10)""",
    ]

    for _ in range(n):
        ex = random.choice(examples)
        entries.append(jsonl_entry(ex, "itertools"))

    return entries


def generar_backtracking(n: int) -> list[str]:
    """Genera ejemplos adicionales para backtracking."""
    entries = []

    examples = [
        """# Backtracking: N-Queens problem

def n_queens(n: int) -> list[list[int]]:
    \"\"\"Retorna todas las soluciones al problema de N reinas.\"\"\"
    solutions = []

    def es_valida(tablero: list[int], fila: int, col: int) -> bool:
        for r in range(fila):
            c = tablero[r]
            if c == col or abs(c - col) == abs(r - fila):
                return False
        return True

    def backtrack(fila: int, tablero: list[int]):
        if fila == n:
            solutions.append(tablero[:])
            return
        for col in range(n):
            if es_valida(tablero, fila, col):
                tablero.append(col)
                backtrack(fila + 1, tablero)
                tablero.pop()

    backtrack(0, [])
    return solutions

print(len(n_queens(8)))  # 92 soluciones""",

        """# Backtracking: generar todas las permutaciones

def permutaciones(nums: list[int]) -> list[list[int]]:
    resultado = []

    def backtrack(actual: list[int], restantes: list[int]):
        if not restantes:
            resultado.append(actual[:])
            return
        for i, num in enumerate(restantes):
            actual.append(num)
            backtrack(actual, restantes[:i] + restantes[i+1:])
            actual.pop()

    backtrack([], nums)
    return resultado

print(permutaciones([1, 2, 3]))
# [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]""",

        """# Backtracking: subset sum

def subset_sum(nums: list[int], target: int) -> list[list[int]]:
    \"\"\"Retorna todos los subsets que suman exactamente target.\"\"\"
    resultado = []

    def backtrack(inicio: int, actual: list[int], suma: int):
        if suma == target:
            resultado.append(actual[:])
            return
        if suma > target:
            return
        for i in range(inicio, len(nums)):
            actual.append(nums[i])
            backtrack(i + 1, actual, suma + nums[i])
            actual.pop()

    nums.sort()
    backtrack(0, [], 0)
    return resultado

print(subset_sum([2, 3, 6, 7], 7))
# [[7], [2, 2, 3]] — con duplicados, variante diferente""",

        """# Backtracking: resolver Sudoku

def es_valido(tablero, fila, col, num):
    if num in tablero[fila]:
        return False
    if num in [tablero[r][col] for r in range(9)]:
        return False
    box_row, box_col = (fila // 3) * 3, (col // 3) * 3
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            if tablero[r][c] == num:
                return False
    return True

def resolver_sudoku(tablero: list[list[int]]) -> bool:
    for fila in range(9):
        for col in range(9):
            if tablero[fila][col] == 0:
                for num in range(1, 10):
                    if es_valido(tablero, fila, col, num):
                        tablero[fila][col] = num
                        if resolver_sudoku(tablero):
                            return True
                        tablero[fila][col] = 0  # Backtrack
                return False
    return True  # Tablero completo""",

        """# Backtracking: combinaciones de suma

def combination_sum(candidates: list[int], target: int) -> list[list[int]]:
    \"\"\"Encuentra todas las combinaciones que sumen exactamente target.
    Los candidatos pueden reutilizarse múltiples veces.\"\"\"
    resultado = []
    candidates.sort()

    def backtrack(inicio: int, actual: list[int], restante: int):
        if restante == 0:
            resultado.append(actual[:])
            return
        for i in range(inicio, len(candidates)):
            if candidates[i] > restante:
                break
            actual.append(candidates[i])
            backtrack(i, actual, restante - candidates[i])  # i, no i+1
            actual.pop()

    backtrack(0, [], target)
    return resultado

print(combination_sum([2, 3, 6, 7], 7))
# [[2, 2, 3], [7]]""",

        """# Backtracking: generar paréntesis válidos

def generar_parentesis(n: int) -> list[str]:
    \"\"\"Genera todas las combinaciones válidas de n pares de paréntesis.\"\"\"
    resultado = []

    def backtrack(actual: str, abiertos: int, cerrados: int):
        if len(actual) == 2 * n:
            resultado.append(actual)
            return
        if abiertos < n:
            backtrack(actual + "(", abiertos + 1, cerrados)
        if cerrados < abiertos:
            backtrack(actual + ")", abiertos, cerrados + 1)

    backtrack("", 0, 0)
    return resultado

print(generar_parentesis(3))
# ['((()))', '(()())', '(())()', '()(())', '()()()']""",
    ]

    for _ in range(n):
        ex = random.choice(examples)
        entries.append(jsonl_entry(ex, "backtracking"))

    return entries


def generar_clean_code(n: int) -> list[str]:
    """Genera ejemplos de clean code y principios de código limpio."""
    entries = []

    examples = [
        """# Clean Code: nombres descriptivos

# MAL — nombres sin significado
def calc(a, b, c):
    return a * b + c

# BIEN — nombres que explican la intención
def calcular_precio_total(precio_unitario: float, cantidad: int, impuesto: float) -> float:
    \"\"\"Calcula el precio total incluyendo impuesto.\"\"\"
    return precio_unitario * cantidad * (1 + impuesto)

# Los nombres correctos hacen innecesarios los comentarios explicativos
precio = calcular_precio_total(10.0, 5, 0.21)  # $60.50""",

        """# Clean Code: funciones pequeñas con una sola responsabilidad

# MAL — función que hace demasiado
def procesar_usuario(user_data: dict) -> None:
    # Valida, guarda en BD, manda email, todo junto
    if not user_data.get('email'):
        raise ValueError("Email requerido")
    # ... 50 líneas más

# BIEN — funciones pequeñas y enfocadas
def validar_usuario(user_data: dict) -> None:
    \"\"\"Solo valida. No hace nada más.\"\"\"
    if not user_data.get('email'):
        raise ValueError("Email requerido")
    if not user_data.get('nombre'):
        raise ValueError("Nombre requerido")

def guardar_usuario(user_data: dict) -> int:
    \"\"\"Solo persiste. Retorna el ID creado.\"\"\"
    # ... lógica de BD
    return 1

def enviar_bienvenida(email: str) -> None:
    \"\"\"Solo envía el email.\"\"\"
    # ... lógica de email""",

        """# Clean Code: evitar números mágicos

# MAL
if velocidad > 120:
    multa = velocidad * 0.15 + 50

# BIEN — usar constantes con nombres claros
VELOCIDAD_MAXIMA_KMH = 120
FACTOR_MULTA = 0.15
MULTA_BASE = 50

def calcular_multa(velocidad: float) -> float:
    \"\"\"Calcula multa si supera la velocidad máxima.\"\"\"
    if velocidad <= VELOCIDAD_MAXIMA_KMH:
        return 0.0
    return velocidad * FACTOR_MULTA + MULTA_BASE

print(calcular_multa(150))  # 72.5""",

        """# Clean Code: funciones que retornan o modifican, no ambas

# MAL — modifica Y retorna (confuso)
def agregar_y_contar(lista: list, item) -> int:
    lista.append(item)
    return len(lista)  # Hace dos cosas

# BIEN — separar responsabilidades
def agregar_elemento(lista: list, item) -> None:
    \"\"\"Solo agrega, sin retornar.\"\"\"
    lista.append(item)

def contar_elementos(lista: list) -> int:
    \"\"\"Solo cuenta, sin modificar.\"\"\"
    return len(lista)

mi_lista = [1, 2, 3]
agregar_elemento(mi_lista, 4)
total = contar_elementos(mi_lista)
print(total)  # 4""",

        """# Clean Code: usar Early Return para reducir anidamiento

# MAL — código con anidamiento profundo
def procesar_pedido(pedido: dict) -> str:
    if pedido:
        if pedido.get('activo'):
            if pedido.get('cantidad', 0) > 0:
                if pedido.get('precio', 0) > 0:
                    return "Pedido válido"
                else:
                    return "Precio inválido"
            else:
                return "Cantidad inválida"
        else:
            return "Pedido inactivo"
    else:
        return "Pedido vacío"

# BIEN — Early Return (guard clauses)
def procesar_pedido_limpio(pedido: dict) -> str:
    if not pedido:
        return "Pedido vacío"
    if not pedido.get('activo'):
        return "Pedido inactivo"
    if pedido.get('cantidad', 0) <= 0:
        return "Cantidad inválida"
    if pedido.get('precio', 0) <= 0:
        return "Precio inválido"
    return "Pedido válido\"""",

        """# Clean Code: evitar comentarios obvios, documentar el PORQUÉ

# MAL — comenta lo que el código ya dice
i = 0  # Inicializar i a 0
i += 1  # Incrementar i

# MAL — comentario redundante
def sumar(a: int, b: int) -> int:
    # Esta función suma a y b
    return a + b

# BIEN — documentar PORQUÉ, no QUÉ
CACHE_TTL_SECONDS = 300  # 5 min: suficiente para no saturar el rate limit de la API

def get_user(user_id: int) -> dict:
    # Usamos caché aquí porque la API externa tiene rate limit de 60 req/min.
    # TODO: migrar a Redis cuando tengamos > 1000 usuarios concurrentes.
    return fetch_from_cache_or_api(user_id)""",

        """# Clean Code: no usar flags booleans como argumentos

# MAL — ¿qué significa True aquí?
def crear_usuario(nombre: str, admin: bool) -> dict:
    if admin:
        return {'nombre': nombre, 'rol': 'admin', 'permisos': ['todo']}
    return {'nombre': nombre, 'rol': 'user', 'permisos': ['leer']}

usuario = crear_usuario("Alice", True)  # ¿Qué significa True?

# BIEN — funciones separadas con nombres explícitos
def crear_usuario_admin(nombre: str) -> dict:
    return {'nombre': nombre, 'rol': 'admin', 'permisos': ['todo']}

def crear_usuario_normal(nombre: str) -> dict:
    return {'nombre': nombre, 'rol': 'user', 'permisos': ['leer']}

alice = crear_usuario_admin("Alice")  # Claro y legible""",
    ]

    for _ in range(n):
        ex = random.choice(examples)
        entries.append(jsonl_entry(ex, "clean_code"))

    return entries


# =============================================================================
# MAPA DE GENERADORES Y ARCHIVOS DESTINO
# =============================================================================

GENERADORES = {
    "collections":       (generar_collections,     "stdlib_python/collections.jsonl"),
    "dataclasses":       (generar_dataclasses,     "stdlib_python/dataclasses.jsonl"),
    "functools":         (generar_functools,        "stdlib_python/functools.jsonl"),
    "singleton_factory": (generar_singleton_factory, "patrones_diseno/singleton_factory.jsonl"),
    "itertools":         (generar_itertools,        "stdlib_python/itertools.jsonl"),
    "backtracking":      (generar_backtracking,     "algoritmos/backtracking.jsonl"),
    "clean_code":        (generar_clean_code,       "ingenieria_software/clean_code.jsonl"),
}


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera datos sintéticos para temas con pocos ejemplos"
    )
    parser.add_argument(
        "--temas", nargs="+", default=list(GENERADORES.keys()),
        help=f"Temas a generar (default: todos). Opciones: {list(GENERADORES.keys())}",
    )
    parser.add_argument(
        "--por-tema", type=int, default=300,
        help="Número de ejemplos por tema (default: 300)",
    )
    parser.add_argument(
        "--biblioteca", type=Path, default=Path("biblioteca"),
        help="Directorio de la biblioteca (default: biblioteca/)",
    )
    parser.add_argument(
        "--modo", choices=["append", "overwrite"], default="append",
        help="'append' agrega al archivo existente, 'overwrite' lo reemplaza",
    )
    args = parser.parse_args()

    temas_desconocidos = [t for t in args.temas if t not in GENERADORES]
    if temas_desconocidos:
        print(f"ERROR: Temas desconocidos: {temas_desconocidos}")
        print(f"Temas disponibles: {list(GENERADORES.keys())}")
        sys.exit(1)

    print(f"\nGenerando datos para {len(args.temas)} temas ({args.por_tema} ejemplos c/u)")
    print(f"Modo: {args.modo}\n")

    total_escritos = 0

    for tema in args.temas:
        generador_fn, ruta_relativa = GENERADORES[tema]
        ruta = args.biblioteca / ruta_relativa
        ruta.parent.mkdir(parents=True, exist_ok=True)

        # Contar líneas existentes
        lineas_previas = 0
        if ruta.exists():
            lineas_previas = sum(1 for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip())

        # Generar
        entries = generador_fn(args.por_tema)

        # Escribir
        modo_abrir = "a" if args.modo == "append" else "w"
        with open(ruta, modo_abrir, encoding="utf-8") as f:
            for entry in entries:
                f.write(entry + "\n")

        lineas_nuevas = sum(1 for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip())
        escritos = len(entries)
        total_escritos += escritos

        print(f"  ✓ {tema:<25} {lineas_previas:>4} → {lineas_nuevas:>4} líneas  (+{escritos})")

    print(f"\nTotal generados: {total_escritos:,} ejemplos")
    print(f"Biblioteca en: {args.biblioteca}/")


if __name__ == "__main__":
    main()
