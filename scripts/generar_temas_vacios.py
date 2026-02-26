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


def generar_sliding_window(n: int) -> list[str]:
    """Genera ejemplos del patrón Sliding Window."""
    examples = [
        """# Sliding Window: máximo en ventana de tamaño k

from collections import deque

def max_sliding_window(nums: list[int], k: int) -> list[int]:
    \"\"\"Retorna el máximo de cada ventana de tamaño k — O(n).\"\"\"
    dq: deque[int] = deque()  # Índices, de mayor a menor valor
    result = []
    for i, val in enumerate(nums):
        # Eliminar índices fuera de la ventana
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        # Eliminar índices con valor menor al actual (ya no útiles)
        while dq and nums[dq[-1]] < val:
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            result.append(nums[dq[0]])
    return result

print(max_sliding_window([1, 3, -1, -3, 5, 3, 6, 7], 3))
# [3, 3, 5, 5, 6, 7]""",

        """# Sliding Window: subcadena más larga sin repetir

def longest_unique_substring(s: str) -> int:
    \"\"\"Longitud de la subcadena más larga sin caracteres repetidos.\"\"\"
    seen: dict[str, int] = {}  # char → último índice
    left = 0
    max_len = 0
    for right, char in enumerate(s):
        if char in seen and seen[char] >= left:
            left = seen[char] + 1
        seen[char] = right
        max_len = max(max_len, right - left + 1)
    return max_len

print(longest_unique_substring("abcabcbb"))  # 3 ("abc")
print(longest_unique_substring("bbbbb"))     # 1 ("b")
print(longest_unique_substring("pwwkew"))    # 3 ("wke")""",

        """# Sliding Window: suma máxima de subarray de tamaño k

def max_subarray_sum(nums: list[int], k: int) -> int:
    \"\"\"Suma máxima de un subarray contíguo de exactamente k elementos.\"\"\"
    if len(nums) < k:
        raise ValueError("Array shorter than window size")
    window_sum = sum(nums[:k])
    max_sum = window_sum
    for i in range(k, len(nums)):
        window_sum += nums[i] - nums[i - k]  # Deslizar ventana
        max_sum = max(max_sum, window_sum)
    return max_sum

print(max_subarray_sum([2, 1, 5, 1, 3, 2], 3))  # 9 (5+1+3)""",

        """# Sliding Window: mínimo de ventana que contiene todos los caracteres

def min_window_substring(s: str, t: str) -> str:
    \"\"\"Subcadena mínima de s que contiene todos los chars de t.\"\"\"
    from collections import Counter
    need = Counter(t)
    missing = len(t)
    left = start = 0
    best = float('inf'), 0, 0

    for right, char in enumerate(s, 1):
        if need[char] > 0:
            missing -= 1
        need[char] -= 1

        if missing == 0:
            while need[s[left]] < 0:
                need[s[left]] += 1
                left += 1
            if right - left < best[0]:
                best = right - left, left, right
            need[s[left]] += 1
            missing += 1
            left += 1

    return s[best[1]:best[2]] if best[0] != float('inf') else ""

print(min_window_substring("ADOBECODEBANC", "ABC"))  # "BANC\"""",

        """# Sliding Window: promedio de todas las ventanas de tamaño k

def ventana_promedios(nums: list[float], k: int) -> list[float]:
    \"\"\"Promedio de cada sub-ventana de tamaño k — O(n).\"\"\"
    suma = sum(nums[:k])
    promedios = [suma / k]
    for i in range(k, len(nums)):
        suma += nums[i] - nums[i - k]
        promedios.append(suma / k)
    return promedios

datos = [1.0, 3.0, 5.0, 7.0, 9.0]
print(ventana_promedios(datos, 3))  # [3.0, 5.0, 7.0]""",

        """# Sliding Window: contar subarrays con suma <= k

def count_subarrays_sum_le_k(nums: list[int], k: int) -> int:
    \"\"\"Cuenta cuántos subarrays tienen suma <= k (solo enteros no negativos).\"\"\"
    left = 0
    current_sum = 0
    count = 0
    for right in range(len(nums)):
        current_sum += nums[right]
        while current_sum > k and left <= right:
            current_sum -= nums[left]
            left += 1
        count += right - left + 1  # Todos los subarrays que terminan en right
    return count

print(count_subarrays_sum_le_k([1, 2, 3, 4], 5))  # 6""",

        """# Sliding Window: longitud mínima con suma >= target

def min_subarray_len(target: int, nums: list[int]) -> int:
    \"\"\"Longitud mínima de subarray con suma >= target.\"\"\"
    left = 0
    current_sum = 0
    min_len = float('inf')
    for right in range(len(nums)):
        current_sum += nums[right]
        while current_sum >= target:
            min_len = min(min_len, right - left + 1)
            current_sum -= nums[left]
            left += 1
    return min_len if min_len != float('inf') else 0

print(min_subarray_len(7, [2, 3, 1, 2, 4, 3]))  # 2 (subarray [4,3])""",
    ]
    return [jsonl_entry(random.choice(examples), "sliding_window") for _ in range(n)]


def generar_decorator_pattern(n: int) -> list[str]:
    """Genera ejemplos del patrón Decorator (estructural)."""
    examples = [
        """# Patrón Decorator: añadir comportamiento sin modificar la clase

from abc import ABC, abstractmethod

class Component(ABC):
    @abstractmethod
    def operation(self) -> str:
        ...

class ConcreteComponent(Component):
    def operation(self) -> str:
        return "ConcreteComponent"

class Decorator(Component):
    def __init__(self, component: Component):
        self._component = component

    def operation(self) -> str:
        return self._component.operation()

class LoggingDecorator(Decorator):
    def operation(self) -> str:
        result = super().operation()
        print(f"LOG: operación ejecutada → {result}")
        return result

class UpperCaseDecorator(Decorator):
    def operation(self) -> str:
        return super().operation().upper()

comp = ConcreteComponent()
logged = LoggingDecorator(UpperCaseDecorator(comp))
print(logged.operation())  # LOG: ... → CONCRETECOMPONENT""",

        """# Decorator pattern con funciones Python (@wraps)

from functools import wraps

def bold(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return "<b>" + func(*args, **kwargs) + "</b>"
    return wrapper

def italic(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return "<i>" + func(*args, **kwargs) + "</i>"
    return wrapper

@bold
@italic
def greet(name: str) -> str:
    return f"Hello, {name}!"

print(greet("Alice"))    # <b><i>Hello, Alice!</i></b>
print(greet.__name__)    # greet (preservado por @wraps)""",

        """# Decorator pattern: Coffee con condimentos

class Coffee:
    def cost(self) -> float:
        return 1.0
    def description(self) -> str:
        return "Simple coffee"

class CoffeeDecorator:
    def __init__(self, coffee: Coffee):
        self._coffee = coffee
    def cost(self) -> float:
        return self._coffee.cost()
    def description(self) -> str:
        return self._coffee.description()

class Milk(CoffeeDecorator):
    def cost(self) -> float:
        return self._coffee.cost() + 0.25
    def description(self) -> str:
        return self._coffee.description() + ", milk"

class Sugar(CoffeeDecorator):
    def cost(self) -> float:
        return self._coffee.cost() + 0.10
    def description(self) -> str:
        return self._coffee.description() + ", sugar"

class Vanilla(CoffeeDecorator):
    def cost(self) -> float:
        return self._coffee.cost() + 0.50
    def description(self) -> str:
        return self._coffee.description() + ", vanilla"

coffee = Vanilla(Milk(Sugar(Coffee())))
print(coffee.description())  # Simple coffee, sugar, milk, vanilla
print(f"${coffee.cost():.2f}")  # $1.85""",

        """# Decorator pattern: validación de inputs

def validate_positive(func):
    \"\"\"Decorador que valida que todos los args numéricos sean positivos.\"\"\"
    @wraps(func)
    def wrapper(*args, **kwargs):
        for arg in args:
            if isinstance(arg, (int, float)) and arg <= 0:
                raise ValueError(f"Argumento debe ser positivo, recibido: {arg}")
        return func(*args, **kwargs)
    return wrapper

from functools import wraps

@validate_positive
def area_circulo(radio: float) -> float:
    import math
    return math.pi * radio ** 2

print(f"{area_circulo(5):.2f}")  # 78.54
# area_circulo(-1)  # ValueError""",

        """# Decorator pattern: cache con TTL

import time
from functools import wraps

def cache_con_ttl(segundos: int):
    \"\"\"Decorador factory: cachea el resultado durante N segundos.\"\"\"
    def decorator(func):
        _cache: dict = {}
        @wraps(func)
        def wrapper(*args):
            now = time.time()
            if args in _cache:
                result, ts = _cache[args]
                if now - ts < segundos:
                    return result
            result = func(*args)
            _cache[args] = (result, now)
            return result
        return wrapper
    return decorator

@cache_con_ttl(60)
def obtener_precio(ticker: str) -> float:
    # Simula llamada a API costosa
    return hash(ticker) % 1000 / 10.0

print(obtener_precio("AAPL"))  # llamada real
print(obtener_precio("AAPL"))  # desde caché""",

        """# Decorator pattern: medir rendimiento

from functools import wraps
import time
import statistics

def benchmark(repeticiones: int = 10):
    \"\"\"Ejecuta la función N veces y reporta estadísticas de tiempo.\"\"\"
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tiempos = []
            resultado = None
            for _ in range(repeticiones):
                t0 = time.perf_counter()
                resultado = func(*args, **kwargs)
                tiempos.append(time.perf_counter() - t0)
            media = statistics.mean(tiempos) * 1000
            stdev = statistics.stdev(tiempos) * 1000 if len(tiempos) > 1 else 0
            print(f"{func.__name__}: {media:.3f}ms ± {stdev:.3f}ms ({repeticiones} runs)")
            return resultado
        return wrapper
    return decorator

@benchmark(repeticiones=5)
def sort_million():
    import random
    data = random.sample(range(1_000_000), 1_000_000)
    return sorted(data)""",
    ]
    return [jsonl_entry(random.choice(examples), "decorator_pattern") for _ in range(n)]


def generar_unittest_pytest(n: int) -> list[str]:
    """Genera ejemplos de testing con pytest y unittest."""
    examples = [
        """import pytest

# Fixtures: setup reutilizable entre tests

@pytest.fixture
def usuario_base():
    return {"nombre": "Alice", "email": "alice@test.com", "edad": 30}

@pytest.fixture
def lista_vacia():
    return []

def test_usuario_tiene_nombre(usuario_base):
    assert usuario_base["nombre"] == "Alice"

def test_usuario_tiene_email_valido(usuario_base):
    assert "@" in usuario_base["email"]

def test_agregar_a_lista(lista_vacia):
    lista_vacia.append(1)
    assert len(lista_vacia) == 1""",

        """import pytest

# pytest.mark.parametrize — tests con múltiples inputs

@pytest.mark.parametrize("entrada,esperado", [
    (2, 4),
    (3, 9),
    (0, 0),
    (-2, 4),
    (5, 25),
])
def test_cuadrado(entrada, esperado):
    assert entrada ** 2 == esperado

@pytest.mark.parametrize("texto,resultado", [
    ("hello", "HELLO"),
    ("WORLD", "WORLD"),
    ("", ""),
    ("Python 3", "PYTHON 3"),
])
def test_upper(texto, resultado):
    assert texto.upper() == resultado""",

        """import pytest

# Testear excepciones

def dividir(a: float, b: float) -> float:
    if b == 0:
        raise ZeroDivisionError("No se puede dividir por cero")
    return a / b

def test_division_normal():
    assert dividir(10, 2) == 5.0

def test_division_por_cero():
    with pytest.raises(ZeroDivisionError, match="No se puede dividir"):
        dividir(10, 0)

def test_division_negativa():
    assert dividir(-6, 2) == -3.0

def test_division_flotante():
    assert pytest.approx(dividir(1, 3)) == 0.333, 0.001""",

        """import pytest
from unittest.mock import Mock, patch, MagicMock

# Mocking: aislar dependencias externas

class EmailService:
    def send(self, to: str, subject: str, body: str) -> bool:
        # En producción enviaría un email real
        raise NotImplementedError

class UserRegistration:
    def __init__(self, email_service: EmailService):
        self.email_service = email_service

    def register(self, email: str) -> dict:
        user = {"email": email, "active": True}
        self.email_service.send(email, "Bienvenido", "Tu cuenta fue creada")
        return user

def test_registro_exitoso():
    mock_email = Mock(spec=EmailService)
    mock_email.send.return_value = True

    reg = UserRegistration(mock_email)
    user = reg.register("alice@test.com")

    assert user["email"] == "alice@test.com"
    assert user["active"] is True
    mock_email.send.assert_called_once_with(
        "alice@test.com", "Bienvenido", "Tu cuenta fue creada"
    )""",

        """import pytest

# Fixtures con scope y teardown

@pytest.fixture(scope="module")
def conexion_db():
    \"\"\"Fixture de módulo: se crea una vez para todos los tests.\"\"\"
    print("\\nAbriendo conexión DB (una vez)")
    db = {"connected": True, "data": {}}
    yield db
    print("\\nCerrando conexión DB")
    db["connected"] = False

def test_insertar(conexion_db):
    conexion_db["data"]["user1"] = "Alice"
    assert "user1" in conexion_db["data"]

def test_leer(conexion_db):
    conexion_db["data"]["user2"] = "Bob"
    assert conexion_db["data"].get("user2") == "Bob"

def test_db_sigue_conectada(conexion_db):
    assert conexion_db["connected"] is True""",

        """import unittest

# unittest clásico — compatible con pytest runner

class TestCalculadora(unittest.TestCase):

    def setUp(self):
        \"\"\"Ejecutado antes de cada test.\"\"\"
        self.calc_history = []

    def test_suma(self):
        result = 2 + 3
        self.assertEqual(result, 5)

    def test_resta(self):
        self.assertEqual(10 - 4, 6)

    def test_division_entera(self):
        self.assertEqual(7 // 2, 3)

    def test_raises_on_zero_division(self):
        with self.assertRaises(ZeroDivisionError):
            _ = 1 / 0

    def test_casi_igual(self):
        self.assertAlmostEqual(0.1 + 0.2, 0.3, places=10)

if __name__ == "__main__":
    unittest.main()""",

        """import pytest

# conftest.py — fixtures compartidas entre múltiples archivos de test
# (este código irá en tests/conftest.py)

# conftest.py
@pytest.fixture
def datos_usuario():
    return {
        "id": 1,
        "nombre": "Test User",
        "email": "test@example.com",
        "roles": ["viewer"],
    }

@pytest.fixture
def admin_usuario(datos_usuario):
    \"\"\"Fixture que extiende otra fixture.\"\"\"
    datos_usuario["roles"] = ["viewer", "editor", "admin"]
    return datos_usuario

# test_permisos.py
def test_viewer_no_puede_editar(datos_usuario):
    assert "editor" not in datos_usuario["roles"]

def test_admin_puede_todo(admin_usuario):
    assert "admin" in admin_usuario["roles"]
    assert "editor" in admin_usuario["roles"]""",

        """import pytest

# Marcadores para organizar y filtrar tests

@pytest.mark.slow
def test_proceso_largo():
    import time
    time.sleep(0.01)  # Simulación
    assert True

@pytest.mark.smoke
def test_servidor_responde():
    assert True  # Aquí iría un health check real

@pytest.mark.skip(reason="Feature en desarrollo")
def test_nueva_funcionalidad():
    assert False

@pytest.mark.xfail(reason="Bug conocido #123")
def test_comportamiento_con_bug():
    raise ValueError("Bug reproducible")

# Ejecutar solo smoke tests:
# pytest -m smoke
# Excluir lentos:
# pytest -m "not slow\"""",
    ]
    return [jsonl_entry(random.choice(examples), "unittest_pytest") for _ in range(n)]


def generar_heap_y_cola(n: int) -> list[str]:
    """Genera ejemplos de heap y cola de prioridad."""
    examples = [
        """import heapq

# heapq: min-heap en Python
# Por defecto es MIN-heap (el más pequeño al frente)

nums = [5, 1, 8, 2, 9, 3]
heapq.heapify(nums)           # Convierte lista a heap in-place O(n)
print(nums[0])                 # 1 (mínimo)
print(heapq.heappop(nums))    # 1 (pop del mínimo)
print(heapq.heappop(nums))    # 2

heapq.heappush(nums, 0)       # Insertar 0
print(heapq.heappop(nums))    # 0""",

        """import heapq

# MAX-heap: negar los valores
nums = [5, 1, 8, 2, 9, 3]
max_heap = [-x for x in nums]
heapq.heapify(max_heap)

# Extraer el mayor
mayor = -heapq.heappop(max_heap)
print(mayor)  # 9

# Insertar nuevo máximo
heapq.heappush(max_heap, -10)
print(-heapq.heappop(max_heap))  # 10""",

        """import heapq

# K elementos más grandes con heap de tamaño k

def k_largest(nums: list[int], k: int) -> list[int]:
    \"\"\"Retorna los k elementos más grandes — O(n log k).\"\"\"
    return heapq.nlargest(k, nums)

def k_smallest(nums: list[int], k: int) -> list[int]:
    \"\"\"Retorna los k elementos más pequeños — O(n log k).\"\"\"
    return heapq.nsmallest(k, nums)

data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
print(k_largest(data, 3))   # [9, 6, 5]
print(k_smallest(data, 3))  # [1, 1, 2]""",

        """import heapq
from dataclasses import dataclass, field
from typing import Any

@dataclass(order=True)
class PriorityItem:
    priority: int
    item: Any = field(compare=False)

class ColaDePrioridad:
    \"\"\"Cola de prioridad usando heapq.\"\"\"
    def __init__(self):
        self._heap: list[PriorityItem] = []

    def push(self, item: Any, priority: int) -> None:
        heapq.heappush(self._heap, PriorityItem(priority, item))

    def pop(self) -> Any:
        return heapq.heappop(self._heap).item

    def peek(self) -> Any:
        return self._heap[0].item

    def __len__(self) -> int:
        return len(self._heap)

cola = ColaDePrioridad()
cola.push("tarea baja", priority=10)
cola.push("tarea urgente", priority=1)
cola.push("tarea media", priority=5)

print(cola.pop())  # tarea urgente (prioritad 1)
print(cola.pop())  # tarea media""",

        """import heapq

# Merge de K listas ordenadas con heap

def merge_k_sorted(listas: list[list[int]]) -> list[int]:
    \"\"\"Combina k listas ordenadas en una sola — O(N log k).\"\"\"
    heap: list[tuple[int, int, int]] = []
    # (valor, índice_lista, índice_elemento)
    for i, lista in enumerate(listas):
        if lista:
            heapq.heappush(heap, (lista[0], i, 0))

    resultado = []
    while heap:
        val, lista_idx, elem_idx = heapq.heappop(heap)
        resultado.append(val)
        if elem_idx + 1 < len(listas[lista_idx]):
            siguiente = listas[lista_idx][elem_idx + 1]
            heapq.heappush(heap, (siguiente, lista_idx, elem_idx + 1))

    return resultado

listas = [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
print(merge_k_sorted(listas))  # [1, 2, 3, 4, 5, 6, 7, 8, 9]""",

        """import heapq

# Dijkstra con heap — camino más corto

def dijkstra(grafo: dict[int, list[tuple[int, int]]], inicio: int) -> dict[int, int]:
    \"\"\"Distancias mínimas desde inicio. grafo[u] = [(peso, v), ...].\"\"\"
    distancias = {inicio: 0}
    heap = [(0, inicio)]   # (distancia_acumulada, nodo)

    while heap:
        dist, nodo = heapq.heappop(heap)
        if dist > distancias.get(nodo, float('inf')):
            continue
        for peso, vecino in grafo.get(nodo, []):
            nueva_dist = dist + peso
            if nueva_dist < distancias.get(vecino, float('inf')):
                distancias[vecino] = nueva_dist
                heapq.heappush(heap, (nueva_dist, vecino))

    return distancias

grafo = {
    0: [(1, 1), (4, 2)],
    1: [(2, 2), (6, 3)],
    2: [(3, 3)],
    3: [],
}
print(dijkstra(grafo, 0))  # {0:0, 1:1, 2:3, 3:6}""",
    ]
    return [jsonl_entry(random.choice(examples), "heap_y_cola_prioridad") for _ in range(n)]


def generar_programacion_dinamica(n: int) -> list[str]:
    """Genera ejemplos de programación dinámica."""
    examples = [
        """# DP: Fibonacci con memoización (top-down)

from functools import cache

@cache
def fib(n: int) -> int:
    \"\"\"Fibonacci con memoización automática — O(n).\"\"\"
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

print([fib(i) for i in range(10)])  # [0,1,1,2,3,5,8,13,21,34]

# Bottom-up (tabulation) — más eficiente en memoria
def fib_tabulation(n: int) -> int:
    if n < 2:
        return n
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b

print(fib_tabulation(10))  # 55""",

        """# DP: Problema de la Mochila 0/1

def knapsack(pesos: list[int], valores: list[int], capacidad: int) -> int:
    \"\"\"Valor máximo que cabe en la mochila de capacidad dada.\"\"\"
    n = len(pesos)
    # dp[i][w] = max valor con los primeros i items y capacidad w
    dp = [[0] * (capacidad + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacidad + 1):
            # No tomar el item i-1
            dp[i][w] = dp[i-1][w]
            # Tomar el item i-1 si cabe
            if pesos[i-1] <= w:
                dp[i][w] = max(dp[i][w], dp[i-1][w - pesos[i-1]] + valores[i-1])

    return dp[n][capacidad]

pesos =  [2, 3, 4, 5]
valores = [3, 4, 5, 6]
print(knapsack(pesos, valores, 8))  # 10""",

        """# DP: Longest Common Subsequence (LCS)

def lcs(s1: str, s2: str) -> int:
    \"\"\"Longitud de la subsecuencia común más larga.\"\"\"
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]

print(lcs("ABCBDAB", "BDCAB"))  # 4 (BCAB o BDAB)
print(lcs("AGGTAB", "GXTXAYB"))  # 4 (GTAB)""",

        """# DP: Coin Change — mínimo número de monedas

def coin_change(monedas: list[int], monto: int) -> int:
    \"\"\"Mínimo de monedas para alcanzar el monto. -1 si imposible.\"\"\"
    dp = [float('inf')] * (monto + 1)
    dp[0] = 0

    for cantidad in range(1, monto + 1):
        for moneda in monedas:
            if moneda <= cantidad:
                dp[cantidad] = min(dp[cantidad], dp[cantidad - moneda] + 1)

    return dp[monto] if dp[monto] != float('inf') else -1

print(coin_change([1, 5, 6, 9], 11))  # 2 (5+6)
print(coin_change([2], 3))             # -1""",

        """# DP: Longest Increasing Subsequence (LIS)

def lis(nums: list[int]) -> int:
    \"\"\"Longitud de la subsecuencia creciente más larga — O(n²).\"\"\"
    if not nums:
        return 0
    dp = [1] * len(nums)
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)

print(lis([10, 9, 2, 5, 3, 7, 101, 18]))  # 4 ([2,3,7,18] o [2,5,7,18])

# Versión O(n log n) con bisect
import bisect

def lis_fast(nums: list[int]) -> int:
    tails: list[int] = []
    for x in nums:
        pos = bisect.bisect_left(tails, x)
        if pos == len(tails):
            tails.append(x)
        else:
            tails[pos] = x
    return len(tails)""",

        """# DP: Edit Distance (Levenshtein)

def edit_distance(s1: str, s2: str) -> int:
    \"\"\"Mínimo de operaciones (insertar, eliminar, reemplazar) para convertir s1 en s2.\"\"\"
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i  # Eliminar todos los chars de s1
    for j in range(n + 1):
        dp[0][j] = j  # Insertar todos los chars de s2

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]  # Gratis
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # Eliminar
                    dp[i][j-1],    # Insertar
                    dp[i-1][j-1],  # Reemplazar
                )

    return dp[m][n]

print(edit_distance("kitten", "sitting"))  # 3
print(edit_distance("horse", "ros"))       # 3""",

        """# DP: Maximum Subarray (Kadane's Algorithm)

def max_subarray(nums: list[int]) -> int:
    \"\"\"Suma máxima de subarray contiguo — O(n).\"\"\"
    max_sum = current = nums[0]
    for x in nums[1:]:
        current = max(x, current + x)
        max_sum = max(max_sum, current)
    return max_sum

print(max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]))  # 6 ([4,-1,2,1])
print(max_subarray([-1, -2, -3]))  # -1 (elemento menos negativo)

# Con índices
def max_subarray_con_indices(nums: list[int]) -> tuple[int, int, int]:
    max_sum = current = nums[0]
    start = end = temp_start = 0
    for i, x in enumerate(nums[1:], 1):
        if x > current + x:
            current = x
            temp_start = i
        else:
            current += x
        if current > max_sum:
            max_sum = current
            start, end = temp_start, i
    return max_sum, start, end""",
    ]
    return [jsonl_entry(random.choice(examples), "programacion_dinamica") for _ in range(n)]


def generar_solid_principles(n: int) -> list[str]:
    """Genera ejemplos de principios SOLID."""
    examples = [
        """# S — Single Responsibility Principle
# Una clase debe tener una sola razón para cambiar.

# MAL: la clase hace demasiado
class UserManagerMal:
    def create_user(self, data: dict) -> dict:
        # Valida, guarda y manda email en el mismo lugar
        pass

# BIEN: responsabilidades separadas
class UserValidator:
    \"\"\"Solo valida datos de usuario.\"\"\"
    def validate(self, data: dict) -> None:
        if not data.get("email"):
            raise ValueError("Email requerido")
        if not data.get("nombre"):
            raise ValueError("Nombre requerido")

class UserRepository:
    \"\"\"Solo persiste usuarios.\"\"\"
    def save(self, user: dict) -> int:
        # Lógica de BD aquí
        return 1

class UserNotifier:
    \"\"\"Solo envía notificaciones.\"\"\"
    def send_welcome(self, email: str) -> None:
        print(f"Email de bienvenida enviado a {email}")

# Orquestador
class UserService:
    def __init__(self, validator, repo, notifier):
        self.validator = validator
        self.repo = repo
        self.notifier = notifier

    def create_user(self, data: dict) -> int:
        self.validator.validate(data)
        user_id = self.repo.save(data)
        self.notifier.send_welcome(data["email"])
        return user_id""",

        """# O — Open/Closed Principle
# Abierto para extensión, cerrado para modificación.

from abc import ABC, abstractmethod

class Descuento(ABC):
    @abstractmethod
    def aplicar(self, precio: float) -> float:
        ...

class SinDescuento(Descuento):
    def aplicar(self, precio: float) -> float:
        return precio

class DescuentoPorcentaje(Descuento):
    def __init__(self, porcentaje: float):
        self.porcentaje = porcentaje

    def aplicar(self, precio: float) -> float:
        return precio * (1 - self.porcentaje / 100)

class DescuentoFijo(Descuento):
    def __init__(self, monto: float):
        self.monto = monto

    def aplicar(self, precio: float) -> float:
        return max(0.0, precio - self.monto)

# Para añadir un nuevo tipo de descuento, NO modificamos código existente:
class DescuentoBOGO(Descuento):  # Buy One Get One
    def aplicar(self, precio: float) -> float:
        return precio / 2

def calcular_total(precio: float, descuento: Descuento) -> float:
    return descuento.aplicar(precio)

print(calcular_total(100.0, DescuentoPorcentaje(20)))  # 80.0
print(calcular_total(100.0, DescuentoFijo(15)))        # 85.0""",

        """# L — Liskov Substitution Principle
# Los subtipos deben ser sustituibles por sus tipos base.

class Rectangulo:
    def __init__(self, ancho: float, alto: float):
        self._ancho = ancho
        self._alto = alto

    @property
    def ancho(self) -> float:
        return self._ancho

    @ancho.setter
    def ancho(self, valor: float) -> None:
        self._ancho = valor

    @property
    def alto(self) -> float:
        return self._alto

    @alto.setter
    def alto(self, valor: float) -> None:
        self._alto = valor

    def area(self) -> float:
        return self._ancho * self._alto

# MAL: Cuadrado viola LSP
# class Cuadrado(Rectangulo):
#     @Rectangulo.ancho.setter
#     def ancho(self, valor):
#         self._ancho = self._alto = valor  # Rompe el contrato del setter!

# BIEN: Cuadrado NO hereda de Rectangulo
class Cuadrado:
    def __init__(self, lado: float):
        self._lado = lado

    def area(self) -> float:
        return self._lado ** 2

def test_area(forma: Rectangulo) -> None:
    forma.ancho = 5
    forma.alto = 4
    assert forma.area() == 20  # Cuadrado rompería esto

r = Rectangulo(2, 3)
test_area(r)  # OK""",

        """# I — Interface Segregation Principle
# Mejor varias interfaces específicas que una grande.

from abc import ABC, abstractmethod

# MAL: interfaz "gorda"
class WorkerMal(ABC):
    @abstractmethod
    def trabajar(self): ...
    @abstractmethod
    def comer(self): ...
    @abstractmethod
    def dormir(self): ...
    # Un Robot no puede comer/dormir pero tendría que implementarlo!

# BIEN: interfaces separadas
class Trabajador(ABC):
    @abstractmethod
    def trabajar(self): ...

class SereVivo(ABC):
    @abstractmethod
    def comer(self): ...
    @abstractmethod
    def dormir(self): ...

class HumanoTrabajador(Trabajador, SereVivo):
    def trabajar(self): print("Humano trabajando")
    def comer(self): print("Humano comiendo")
    def dormir(self): print("Humano durmiendo")

class Robot(Trabajador):
    \"\"\"Solo implementa lo que necesita.\"\"\"
    def trabajar(self): print("Robot trabajando 24/7")""",

        """# D — Dependency Inversion Principle
# Depender de abstracciones, no de implementaciones concretas.

from abc import ABC, abstractmethod

class BaseDeDatos(ABC):
    @abstractmethod
    def guardar(self, datos: dict) -> int: ...
    @abstractmethod
    def buscar(self, id: int) -> dict: ...

class PostgreSQLDB(BaseDeDatos):
    def guardar(self, datos: dict) -> int:
        print(f"Guardando en PostgreSQL: {datos}")
        return 1

    def buscar(self, id: int) -> dict:
        return {"id": id, "source": "PostgreSQL"}

class InMemoryDB(BaseDeDatos):
    def __init__(self):
        self._store: dict[int, dict] = {}
        self._next_id = 1

    def guardar(self, datos: dict) -> int:
        self._store[self._next_id] = datos
        result = self._next_id
        self._next_id += 1
        return result

    def buscar(self, id: int) -> dict:
        return self._store.get(id, {})

# UserService depende de la ABSTRACCIÓN, no de PostgreSQL
class UserService:
    def __init__(self, db: BaseDeDatos):  # Inversión de dependencia!
        self.db = db

    def crear_usuario(self, nombre: str) -> int:
        return self.db.guardar({"nombre": nombre})

# Fácil de testear con InMemoryDB
service = UserService(InMemoryDB())
id_ = service.crear_usuario("Alice")
print(id_)  # 1""",
    ]
    return [jsonl_entry(random.choice(examples), "solid_principles") for _ in range(n)]


def generar_herencia_y_polimorfismo(n: int) -> list[str]:
    """Genera ejemplos de herencia y polimorfismo."""
    examples = [
        """# Herencia básica en Python

class Animal:
    def __init__(self, nombre: str, edad: int):
        self.nombre = nombre
        self.edad = edad

    def hablar(self) -> str:
        raise NotImplementedError("Subclase debe implementar hablar()")

    def __str__(self) -> str:
        return f"{self.nombre} ({self.edad} años)"

class Perro(Animal):
    def hablar(self) -> str:
        return "¡Guau!"

    def fetch(self) -> str:
        return f"{self.nombre} trae la pelota"

class Gato(Animal):
    def hablar(self) -> str:
        return "¡Miau!"

    def ronronear(self) -> str:
        return "Purrr..."

# Polimorfismo: mismo método, comportamiento distinto
animales: list[Animal] = [Perro("Rex", 3), Gato("Whiskers", 5)]
for animal in animales:
    print(f"{animal}: {animal.hablar()}")""",

        """# super() — llamar al constructor del padre

class Vehiculo:
    def __init__(self, marca: str, velocidad_max: float):
        self.marca = marca
        self.velocidad_max = velocidad_max

    def descripcion(self) -> str:
        return f"{self.marca} (max {self.velocidad_max} km/h)"

class Coche(Vehiculo):
    def __init__(self, marca: str, velocidad_max: float, puertas: int):
        super().__init__(marca, velocidad_max)  # Llamar al padre
        self.puertas = puertas

    def descripcion(self) -> str:
        base = super().descripcion()  # Reutilizar lógica del padre
        return f"{base}, {self.puertas} puertas"

class ElectricCar(Coche):
    def __init__(self, marca: str, velocidad_max: float, puertas: int, autonomia_km: int):
        super().__init__(marca, velocidad_max, puertas)
        self.autonomia_km = autonomia_km

    def descripcion(self) -> str:
        return f"{super().descripcion()}, {self.autonomia_km}km autonomía"

tesla = ElectricCar("Tesla", 250.0, 4, 500)
print(tesla.descripcion())""",

        """from abc import ABC, abstractmethod

# Clases abstractas — definen contratos sin implementación completa

class Figura(ABC):
    \"\"\"Clase base abstracta para figuras geométricas.\"\"\"

    @abstractmethod
    def area(self) -> float:
        \"\"\"Retorna el área de la figura.\"\"\"
        ...

    @abstractmethod
    def perimetro(self) -> float:
        \"\"\"Retorna el perímetro de la figura.\"\"\"
        ...

    def describe(self) -> str:
        return (f"{type(self).__name__}: "
                f"área={self.area():.2f}, perím={self.perimetro():.2f}")

class Circulo(Figura):
    def __init__(self, radio: float):
        self.radio = radio

    def area(self) -> float:
        import math
        return math.pi * self.radio ** 2

    def perimetro(self) -> float:
        import math
        return 2 * math.pi * self.radio

class Triangulo(Figura):
    def __init__(self, a: float, b: float, c: float):
        self.a, self.b, self.c = a, b, c

    def area(self) -> float:
        s = (self.a + self.b + self.c) / 2
        return (s * (s-self.a) * (s-self.b) * (s-self.c)) ** 0.5

    def perimetro(self) -> float:
        return self.a + self.b + self.c

figuras: list[Figura] = [Circulo(5), Triangulo(3, 4, 5)]
for f in figuras:
    print(f.describe())""",

        """# Herencia múltiple y MRO (Method Resolution Order)

class Loggable:
    def log(self, msg: str) -> None:
        print(f"[LOG] {msg}")

class Serializable:
    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('_')}

class Validatable:
    def validate(self) -> bool:
        return True

class Model(Loggable, Serializable, Validatable):
    def save(self) -> None:
        if self.validate():
            self.log(f"Guardando {self.to_dict()}")

class User(Model):
    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email

    def validate(self) -> bool:
        return "@" in self.email

u = User("Alice", "alice@test.com")
u.save()             # [LOG] Guardando {'name': 'Alice', 'email': 'alice@test.com'}
print(u.to_dict())   # {'name': 'Alice', 'email': 'alice@test.com'}
print(User.__mro__)  # Orden de resolución de métodos""",

        """# __init_subclass__ — hook al crear subclases

class Plugin:
    \"\"\"Base que registra automáticamente cada subclase.\"\"\"
    _registry: dict[str, type] = {}

    def __init_subclass__(cls, nombre: str = "", **kwargs):
        super().__init_subclass__(**kwargs)
        key = nombre or cls.__name__.lower()
        Plugin._registry[key] = cls

    @classmethod
    def obtener(cls, nombre: str) -> 'Plugin':
        if nombre not in cls._registry:
            raise KeyError(f"Plugin '{nombre}' no registrado")
        return cls._registry[nombre]()

class PluginJSON(Plugin, nombre="json"):
    def procesar(self, data: str) -> dict:
        import json
        return json.loads(data)

class PluginCSV(Plugin, nombre="csv"):
    def procesar(self, data: str) -> list:
        return [row.split(',') for row in data.strip().splitlines()]

plugin = Plugin.obtener("json")
print(plugin.procesar('{"key": "value"}'))
print(Plugin._registry.keys())""",
    ]
    return [jsonl_entry(random.choice(examples), "herencia_y_polimorfismo") for _ in range(n)]


def generar_async_y_await(n: int) -> list[str]:
    """Genera ejemplos de async/await."""
    examples = [
        """import asyncio

# async/await básico

async def saludar(nombre: str, delay: float) -> str:
    await asyncio.sleep(delay)   # No bloquea el event loop
    return f"Hola, {nombre}!"

async def main():
    # Ejecutar secuencialmente
    r1 = await saludar("Alice", 0.1)
    r2 = await saludar("Bob", 0.1)
    print(r1)
    print(r2)

asyncio.run(main())""",

        """import asyncio

# asyncio.gather — ejecutar corutinas en paralelo

async def tarea(nombre: str, segundos: float) -> str:
    print(f"{nombre}: iniciando")
    await asyncio.sleep(segundos)
    print(f"{nombre}: terminada")
    return f"Resultado de {nombre}"

async def main():
    # gather ejecuta todas en paralelo
    resultados = await asyncio.gather(
        tarea("A", 0.3),
        tarea("B", 0.1),
        tarea("C", 0.2),
    )
    # Orden garantizado aunque terminen en orden distinto
    for r in resultados:
        print(r)

asyncio.run(main())
# B termina primero, luego C, luego A
# Pero resultados = [A, B, C] (orden original)""",

        """import asyncio

# async context manager y async generator

class AsyncDatabase:
    async def __aenter__(self):
        print("Conectando a DB...")
        await asyncio.sleep(0.01)
        return self

    async def __aexit__(self, *args):
        print("Cerrando conexión DB")

    async def query(self, sql: str) -> list[dict]:
        await asyncio.sleep(0.01)  # Simula I/O
        return [{"resultado": sql}]

async def numeros_pares(limite: int):
    \"\"\"Async generator.\"\"\"
    for i in range(0, limite, 2):
        await asyncio.sleep(0)  # Cede control
        yield i

async def main():
    async with AsyncDatabase() as db:
        rows = await db.query("SELECT * FROM users")
        print(rows)

    async for num in numeros_pares(10):
        print(num, end=" ")

asyncio.run(main())""",

        """import asyncio
from typing import Any

# asyncio.create_task — ejecutar sin await inmediato

async def descargar(url: str) -> str:
    print(f"Descargando {url}...")
    await asyncio.sleep(0.1)  # Simula descarga
    return f"Contenido de {url}"

async def main():
    # Crear tasks sin esperar inmediatamente
    task1 = asyncio.create_task(descargar("https://api.example.com/users"))
    task2 = asyncio.create_task(descargar("https://api.example.com/posts"))
    task3 = asyncio.create_task(descargar("https://api.example.com/comments"))

    # Aquí podemos hacer otras cosas mientras las tasks corren...
    print("Tasks creadas, haciendo otras cosas...")

    # Esperar los resultados
    r1 = await task1
    r2 = await task2
    r3 = await task3
    print(f"Descargados: {len([r1, r2, r3])} recursos")

asyncio.run(main())""",

        """import asyncio

# Manejar timeouts y cancelación

async def operacion_lenta(n: int) -> int:
    await asyncio.sleep(n)
    return n * 2

async def main():
    # timeout: lanza asyncio.TimeoutError si supera el tiempo
    try:
        resultado = await asyncio.wait_for(operacion_lenta(10), timeout=0.5)
    except asyncio.TimeoutError:
        print("Operación cancelada por timeout")

    # asyncio.wait con control fino
    tasks = [asyncio.create_task(operacion_lenta(i)) for i in [1, 2, 3]]
    done, pending = await asyncio.wait(tasks, timeout=1.5)

    for task in done:
        print(f"Completada: {task.result()}")
    for task in pending:
        task.cancel()
        print("Cancelada una task pendiente")

asyncio.run(main())""",

        """import asyncio

# Semáforo — limitar concurrencia

async def llamar_api(session_id: int, semaforo: asyncio.Semaphore) -> str:
    async with semaforo:
        print(f"Session {session_id}: llamando API")
        await asyncio.sleep(0.1)  # Simula latencia
        return f"Respuesta {session_id}"

async def main():
    # Máximo 3 llamadas simultáneas aunque lanzamos 10
    sem = asyncio.Semaphore(3)
    tareas = [llamar_api(i, sem) for i in range(10)]
    resultados = await asyncio.gather(*tareas)
    print(f"Total: {len(resultados)} respuestas")

asyncio.run(main())""",
    ]
    return [jsonl_entry(random.choice(examples), "async_y_await") for _ in range(n)]


def generar_two_pointers(n: int) -> list[str]:
    """Genera ejemplos del patrón Two Pointers."""
    examples = [
        """# Two Pointers: par con suma objetivo en array ordenado

def two_sum_sorted(nums: list[int], target: int) -> tuple[int, int] | None:
    \"\"\"Retorna los índices del par que suma target (array ordenado).
    O(n) en tiempo, O(1) en espacio.\"\"\"
    left, right = 0, len(nums) - 1
    while left < right:
        total = nums[left] + nums[right]
        if total == target:
            return left, right
        elif total < target:
            left += 1
        else:
            right -= 1
    return None

print(two_sum_sorted([1, 2, 3, 4, 6], 6))   # (1, 3) → 2+4
print(two_sum_sorted([2, 7, 11, 15], 9))     # (0, 1) → 2+7""",

        """# Two Pointers: invertir array in-place

def invertir(nums: list[int]) -> None:
    \"\"\"Invierte el array sin espacio extra — O(n).\"\"\"
    left, right = 0, len(nums) - 1
    while left < right:
        nums[left], nums[right] = nums[right], nums[left]
        left += 1
        right -= 1

nums = [1, 2, 3, 4, 5]
invertir(nums)
print(nums)  # [5, 4, 3, 2, 1]

# También funciona para strings vía list
def invertir_string(s: str) -> str:
    chars = list(s)
    invertir(chars)
    return "".join(chars)

print(invertir_string("hello"))  # "olleh\"""",

        """# Two Pointers: detectar ciclo en lista enlazada (Floyd's)

class ListNode:
    def __init__(self, val: int, next=None):
        self.val = val
        self.next = next

def has_cycle(head: ListNode | None) -> bool:
    \"\"\"Detecta ciclo con two pointers (lento y rápido).\"\"\"
    slow = fast = head
    while fast and fast.next:
        slow = slow.next          # Avanza 1
        fast = fast.next.next     # Avanza 2
        if slow is fast:
            return True           # Hay ciclo
    return False

# Lista sin ciclo: 1 → 2 → 3 → None
n1, n2, n3 = ListNode(1), ListNode(2), ListNode(3)
n1.next, n2.next = n2, n3
print(has_cycle(n1))  # False

# Lista con ciclo: 1 → 2 → 3 → 2 (ciclo)
n3.next = n2
print(has_cycle(n1))  # True""",

        """# Two Pointers: eliminar duplicados de array ordenado in-place

def remove_duplicates(nums: list[int]) -> int:
    \"\"\"Elimina duplicados del array ordenado in-place.
    Retorna la longitud del array sin duplicados. O(n).\"\"\"
    if not nums:
        return 0
    slow = 0
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]
    return slow + 1

nums = [1, 1, 2, 3, 3, 4, 5, 5]
k = remove_duplicates(nums)
print(k)          # 5
print(nums[:k])   # [1, 2, 3, 4, 5]""",

        """# Two Pointers: contenedor con más agua (Container With Most Water)

def max_water(heights: list[int]) -> int:
    \"\"\"Área máxima de agua entre dos líneas verticales.\"\"\"
    left, right = 0, len(heights) - 1
    max_area = 0
    while left < right:
        ancho = right - left
        altura = min(heights[left], heights[right])
        max_area = max(max_area, ancho * altura)
        # Mover el puntero de la línea más corta
        if heights[left] < heights[right]:
            left += 1
        else:
            right -= 1
    return max_area

print(max_water([1, 8, 6, 2, 5, 4, 8, 3, 7]))  # 49""",

        """# Two Pointers: 3Sum — todos los tripletes que suman 0

def three_sum(nums: list[int]) -> list[list[int]]:
    \"\"\"Todos los tripletes únicos que suman 0 — O(n²).\"\"\"
    nums.sort()
    resultado = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i-1]:
            continue  # Saltar duplicados del primer elemento
        left, right = i + 1, len(nums) - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total == 0:
                resultado.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left+1]:
                    left += 1
                while left < right and nums[right] == nums[right-1]:
                    right -= 1
                left += 1
                right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1
    return resultado

print(three_sum([-1, 0, 1, 2, -1, -4]))
# [[-1, -1, 2], [-1, 0, 1]]""",
    ]
    return [jsonl_entry(random.choice(examples), "two_pointers") for _ in range(n)]


# =============================================================================
# MAPA DE GENERADORES Y ARCHIVOS DESTINO
# =============================================================================

GENERADORES = {
    "collections":            (generar_collections,          "stdlib_python/collections.jsonl"),
    "dataclasses":            (generar_dataclasses,          "stdlib_python/dataclasses.jsonl"),
    "functools":              (generar_functools,             "stdlib_python/functools.jsonl"),
    "singleton_factory":      (generar_singleton_factory,    "patrones_diseno/singleton_factory.jsonl"),
    "itertools":              (generar_itertools,             "stdlib_python/itertools.jsonl"),
    "backtracking":           (generar_backtracking,         "algoritmos/backtracking.jsonl"),
    "clean_code":             (generar_clean_code,           "ingenieria_software/clean_code.jsonl"),
    "sliding_window":         (generar_sliding_window,       "algoritmos/sliding_window.jsonl"),
    "decorator_pattern":      (generar_decorator_pattern,    "patrones_diseno/decorator_pattern.jsonl"),
    "unittest_pytest":        (generar_unittest_pytest,      "stdlib_python/unittest_pytest.jsonl"),
    "heap_y_cola_prioridad":  (generar_heap_y_cola,          "algoritmos/heap_y_cola_prioridad.jsonl"),
    "programacion_dinamica":  (generar_programacion_dinamica, "algoritmos/programacion_dinamica.jsonl"),
    "solid_principles":       (generar_solid_principles,     "ingenieria_software/solid_principles.jsonl"),
    "herencia_y_polimorfismo":(generar_herencia_y_polimorfismo, "python_basico/herencia_y_polimorfismo.jsonl"),
    "async_y_await":          (generar_async_y_await,        "python_basico/async_y_await.jsonl"),
    "two_pointers":           (generar_two_pointers,         "algoritmos/two_pointers.jsonl"),
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
