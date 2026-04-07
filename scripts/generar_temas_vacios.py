#!/usr/bin/env python3
# SPDX-License-Identifier: BUSL-1.1
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
# =============================================================================
# GENERADORES PARA TEMAS FALTANTES (añadidos en Feb 2026)
# =============================================================================

def generar_json_y_csv(n: int) -> list[str]:
    """Genera ejemplos de json y csv con la stdlib."""
    examples = [
        """import json

data = {"nombre": "Alice", "edad": 30, "activo": True}
serializado = json.dumps(data, ensure_ascii=False, indent=2)
print(serializado)
recuperado = json.loads(serializado)
print(recuperado["nombre"])  # Alice""",

        """import json
from pathlib import Path

def cargar_config(ruta: str) -> dict:
    \"\"\"Carga un archivo JSON de configuración.\"\"\"
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)

def guardar_config(ruta: str, config: dict) -> None:
    \"\"\"Guarda configuración como JSON indentado.\"\"\"
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)""",

        """import json

# Manejo de tipos no serializables
from datetime import datetime, date

class EncoderPersonalizado(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)

evento = {"fecha": datetime(2025, 1, 15), "nombre": "lanzamiento"}
resultado = json.dumps(evento, cls=EncoderPersonalizado)
print(resultado)""",

        """import csv
import io

# Escribir CSV en memoria
output = io.StringIO()
writer = csv.DictWriter(output, fieldnames=["nombre", "edad", "ciudad"])
writer.writeheader()
writer.writerows([
    {"nombre": "Alice", "edad": 30, "ciudad": "Madrid"},
    {"nombre": "Bob",   "edad": 25, "ciudad": "Buenos Aires"},
])
print(output.getvalue())""",

        """import csv
from pathlib import Path

def leer_csv(ruta: str) -> list[dict]:
    \"\"\"Lee un CSV y retorna lista de dicts.\"\"\"
    with open(ruta, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def escribir_csv(ruta: str, filas: list[dict], campos: list[str]) -> None:
    \"\"\"Escribe una lista de dicts como CSV.\"\"\"
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(filas)""",

        """import json

# json.JSONDecodeError
def parsear_seguro(texto: str) -> dict | None:
    \"\"\"Parsea JSON con manejo de error.\"\"\"
    try:
        return json.loads(texto)
    except json.JSONDecodeError as e:
        print(f"JSON inválido: {e}")
        return None

print(parsear_seguro('{"ok": 1}'))   # {'ok': 1}
print(parsear_seguro("no es json"))  # None""",

        """import csv
import sys

# Leer CSV desde stdin (útil en pipelines)
reader = csv.DictReader(sys.stdin)
for fila in reader:
    print(fila["nombre"], fila["valor"])""",

        """import json

# Filtrar y transformar JSON lines (JSONL)
def procesar_jsonl(ruta_entrada: str, ruta_salida: str, filtro) -> int:
    escritos = 0
    with open(ruta_entrada, encoding="utf-8") as fin, \
         open(ruta_salida, "w", encoding="utf-8") as fout:
        for linea in fin:
            obj = json.loads(linea)
            if filtro(obj):
                fout.write(json.dumps(obj, ensure_ascii=False) + "\\n")
                escritos += 1
    return escritos""",

        """import csv
from collections import defaultdict

def agrupar_csv_por_columna(ruta: str, columna: str) -> dict[str, list[dict]]:
    \"\"\"Agrupa filas de un CSV por el valor de una columna.\"\"\"
    grupos: dict[str, list[dict]] = defaultdict(list)
    with open(ruta, newline="", encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            grupos[fila[columna]].append(fila)
    return dict(grupos)""",

        """import json

# Pretty-print anidado con sorted keys
config = {"z": 3, "a": [1, 2], "m": {"x": 0}}
bonito = json.dumps(config, indent=4, sort_keys=True)
print(bonito)
# {
#     "a": [1, 2],
#     "m": {"x": 0},
#     "z": 3
# }""",
    ]
    return [jsonl_entry(e, "json_y_csv") for e in shuffle_and_sample(examples, n)]


def generar_logging(n: int) -> list[str]:
    """Genera ejemplos del módulo logging."""
    examples = [
        """import logging

# Configuración básica
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s — %(levelname)s — %(message)s",
)

logging.debug("Detalle de depuración")
logging.info("Aplicación iniciada")
logging.warning("Recurso bajo")
logging.error("Error al conectarse a DB")
logging.critical("Sistema crítico caído")""",

        """import logging
from pathlib import Path

def crear_logger(nombre: str, archivo: str) -> logging.Logger:
    \"\"\"Crea un logger con handler de archivo y consola.\"\"\"
    logger = logging.getLogger(nombre)
    logger.setLevel(logging.DEBUG)

    # Handler archivo
    fh = logging.FileHandler(archivo, encoding="utf-8")
    fh.setLevel(logging.DEBUG)

    # Handler consola
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)

    fmt = logging.Formatter("%(name)s — %(levelname)s — %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger""",

        """import logging

# Logger por módulo (buena práctica)
logger = logging.getLogger(__name__)

def procesar(item: dict) -> None:
    logger.debug("Procesando item id=%s", item.get("id"))
    try:
        resultado = item["valor"] * 2
        logger.info("Item %s procesado: resultado=%d", item["id"], resultado)
    except KeyError:
        logger.exception("Item sin clave 'valor': %s", item)""",

        """import logging

# Filtro personalizado
class FiltroAplicacion(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith("DEBUG_VERBOSE:")

logger = logging.getLogger("app")
logger.addFilter(FiltroAplicacion())""",

        """import logging
import json

class JSONFormatter(logging.Formatter):
    \"\"\"Formatea logs como JSON (útil para sistemas de monitoreo).\"\"\"
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }, ensure_ascii=False)""",

        """import logging
from logging.handlers import RotatingFileHandler

# Log con rotación: máximo 5 MB, guarda 3 backups
handler = RotatingFileHandler(
    "app.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
handler.setFormatter(logging.Formatter("%(asctime)s — %(message)s"))

logger = logging.getLogger("rotante")
logger.addHandler(handler)
logger.setLevel(logging.INFO)""",

        """import logging

# Niveles personalizados
TRACE = 5
logging.addLevelName(TRACE, "TRACE")

def trace(self, message, *args, **kws):
    if self.isEnabledFor(TRACE):
        self._log(TRACE, message, args, **kws)

logging.Logger.trace = trace

logger = logging.getLogger("custom")
logger.setLevel(TRACE)
logger.trace("Mensaje muy detallado")""",

        """import logging
import contextlib

@contextlib.contextmanager
def log_duracion(operacion: str):
    \"\"\"Context manager que loguea el tiempo de una operación.\"\"\"
    import time
    logger = logging.getLogger("timer")
    inicio = time.perf_counter()
    logger.info("Iniciando: %s", operacion)
    try:
        yield
    finally:
        dur = time.perf_counter() - inicio
        logger.info("Finalizado: %s (%.3fs)", operacion, dur)

with log_duracion("cálculo pesado"):
    sum(range(10_000_000))""",

        """import logging

# Configurar con dictConfig
import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "formatters": {"default": {"format": "%(levelname)s: %(message)s"}},
    "handlers": {"console": {"class": "logging.StreamHandler", "formatter": "default"}},
    "root": {"level": "INFO", "handlers": ["console"]},
}
logging.config.dictConfig(LOGGING_CONFIG)
logging.info("Configurado con dictConfig")""",
    ]
    return [jsonl_entry(e, "logging") for e in shuffle_and_sample(examples, n)]


def generar_threading_multiprocess(n: int) -> list[str]:
    """Genera ejemplos de threading y multiprocessing."""
    examples = [
        """import threading
import time

def tarea(nombre: str, segundos: float) -> None:
    print(f"{nombre} iniciando")
    time.sleep(segundos)
    print(f"{nombre} terminado")

hilos = [threading.Thread(target=tarea, args=(f"Hilo-{i}", i * 0.5)) for i in range(4)]
for h in hilos:
    h.start()
for h in hilos:
    h.join()
print("Todos los hilos terminaron")""",

        """import threading

# Lock para evitar race conditions
contador = 0
lock = threading.Lock()

def incrementar(n: int) -> None:
    global contador
    for _ in range(n):
        with lock:
            contador += 1

hilos = [threading.Thread(target=incrementar, args=(1000,)) for _ in range(5)]
for h in hilos:
    h.start()
for h in hilos:
    h.join()
print(f"Contador final: {contador}")  # 5000""",

        """from multiprocessing import Pool
import math

def calcular_factorial(n: int) -> int:
    return math.factorial(n)

numeros = [10, 20, 30, 40, 50]
with Pool(processes=4) as pool:
    resultados = pool.map(calcular_factorial, numeros)
print(resultados)""",

        """import threading
from queue import Queue

def productor(q: Queue, items: list) -> None:
    for item in items:
        q.put(item)
    q.put(None)  # Señal de fin

def consumidor(q: Queue) -> list:
    resultados = []
    while True:
        item = q.get()
        if item is None:
            break
        resultados.append(item * 2)
    return resultados

q: Queue = Queue()
t1 = threading.Thread(target=productor, args=(q, list(range(10))))
t1.start()
t1.join()
print(consumidor(q))""",

        """from multiprocessing import Process, Queue

def trabajador(nombre: str, cola: Queue) -> None:
    cola.put(f"{nombre}: resultado={42}")

cola: Queue = Queue()
procs = [Process(target=trabajador, args=(f"Worker-{i}", cola)) for i in range(3)]
for p in procs:
    p.start()
for p in procs:
    p.join()

while not cola.empty():
    print(cola.get())""",

        """import threading

class HiloWorker(threading.Thread):
    \"\"\"Thread personalizado con retorno de resultado.\"\"\"

    def __init__(self, fn, *args):
        super().__init__()
        self.fn = fn
        self.args = args
        self.resultado = None
        self.error = None

    def run(self) -> None:
        try:
            self.resultado = self.fn(*self.args)
        except Exception as e:
            self.error = e""",

        """from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request

def descargar(url: str) -> str:
    with urllib.request.urlopen(url, timeout=5) as resp:
        return resp.read(200).decode()

urls = ["http://httpbin.org/get", "http://httpbin.org/ip"]
with ThreadPoolExecutor(max_workers=4) as ex:
    futuros = {ex.submit(descargar, u): u for u in urls}
    for fut in as_completed(futuros):
        try:
            print(futuros[fut], "OK")
        except Exception as e:
            print(futuros[fut], "ERROR:", e)""",

        """from concurrent.futures import ProcessPoolExecutor

def es_primo(n: int) -> bool:
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

numeros = list(range(10_000, 10_100))
with ProcessPoolExecutor() as ex:
    primos = list(filter(None, ex.map(es_primo, numeros)))
print(f"Primos encontrados: {len(primos)}")""",

        """import threading

# Evento para coordinar hilos
evento = threading.Event()

def esperador(nombre: str) -> None:
    print(f"{nombre} esperando señal...")
    evento.wait()
    print(f"{nombre} continúa")

hilos = [threading.Thread(target=esperador, args=(f"W{i}",)) for i in range(3)]
for h in hilos:
    h.start()

import time; time.sleep(0.5)
print("Enviando señal")
evento.set()
for h in hilos:
    h.join()""",

        """import multiprocessing
import os

def info_proceso() -> None:
    print(f"PID: {os.getpid()}, PPID: {os.getppid()}")

if __name__ == "__main__":
    procs = [multiprocessing.Process(target=info_proceso) for _ in range(3)]
    for p in procs: p.start()
    for p in procs: p.join()""",
    ]
    return [jsonl_entry(e, "threading_multiprocess") for e in shuffle_and_sample(examples, n)]


def generar_git_y_ci_cd(n: int) -> list[str]:
    """Genera ejemplos de git y CI/CD con Python (subprocess, GitHub Actions, etc.)."""
    examples = [
        """import subprocess

def git_status() -> str:
    \"\"\"Retorna el estado del repositorio git.\"\"\"
    result = subprocess.run(
        ["git", "status", "--short"],
        capture_output=True, text=True, check=True
    )
    return result.stdout

def git_log(n: int = 10) -> list[str]:
    \"\"\"Retorna los últimos n commits.\"\"\"
    result = subprocess.run(
        ["git", "log", f"--max-count={n}", "--oneline"],
        capture_output=True, text=True, check=True
    )
    return result.stdout.strip().splitlines()""",

        """import subprocess
from pathlib import Path

def git_commit(mensaje: str, archivos: list[str] | None = None) -> None:
    \"\"\"Stage y commit de archivos con un mensaje convencional.\"\"\"
    if archivos:
        subprocess.run(["git", "add"] + archivos, check=True)
    else:
        subprocess.run(["git", "add", "-A"], check=True)
    subprocess.run(["git", "commit", "-m", mensaje], check=True)

def git_push(remote: str = "origin", rama: str = "main") -> None:
    subprocess.run(["git", "push", remote, rama], check=True)""",

        """# GitHub Actions workflow (YAML generado desde Python)
import yaml
from pathlib import Path

workflow = {
    "name": "CI",
    "on": ["push", "pull_request"],
    "jobs": {
        "test": {
            "runs-on": "ubuntu-latest",
            "steps": [
                {"uses": "actions/checkout@v4"},
                {"uses": "actions/setup-python@v5", "with": {"python-version": "3.13"}},
                {"run": "pip install -r requirements.txt"},
                {"run": "pytest --tb=short"},
            ]
        }
    }
}
Path(".github/workflows/ci.yml").parent.mkdir(parents=True, exist_ok=True)
Path(".github/workflows/ci.yml").write_text(yaml.dump(workflow), encoding="utf-8")""",

        """import subprocess

def rama_actual() -> str:
    return subprocess.run(
        ["git", "branch", "--show-current"],
        capture_output=True, text=True
    ).stdout.strip()

def crear_rama(nombre: str) -> None:
    subprocess.run(["git", "checkout", "-b", nombre], check=True)

def hay_cambios_sin_commitear() -> bool:
    r = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    return bool(r.stdout.strip())""",

        """import subprocess
import re

def ultimo_tag() -> str | None:
    \"\"\"Retorna el último tag semver o None.\"\"\"
    try:
        out = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            capture_output=True, text=True, check=True
        ).stdout.strip()
        return out
    except subprocess.CalledProcessError:
        return None

def siguiente_version_patch(tag: str) -> str:
    \"\"\"Incrementa el patch de un tag semver (v1.2.3 → v1.2.4).\"\"\"
    m = re.match(r"v?(\\d+)\\.(\\d+)\\.(\\d+)", tag)
    if not m:
        raise ValueError(f"Tag no semver: {tag}")
    major, minor, patch = m.groups()
    return f"v{major}.{minor}.{int(patch)+1}\"""",

        """import subprocess
from datetime import datetime

def changelog_desde(tag: str) -> str:
    \"\"\"Genera un changelog desde un tag hasta HEAD.\"\"\"
    result = subprocess.run(
        ["git", "log", f"{tag}..HEAD", "--pretty=format:- %s (%an)"],
        capture_output=True, text=True, check=True
    )
    fecha = datetime.now().strftime("%Y-%m-%d")
    return f"## [{fecha}]\\n\\n{result.stdout}\\n""",

        """import subprocess

def archivos_modificados_entre(ref_a: str, ref_b: str = "HEAD") -> list[str]:
    \"\"\"Lista archivos Python modificados entre dos refs.\"\"\"
    r = subprocess.run(
        ["git", "diff", "--name-only", ref_a, ref_b],
        capture_output=True, text=True, check=True
    )
    return [f for f in r.stdout.splitlines() if f.endswith(".py")]

# Uso: solo correr lint sobre archivos cambiados
cambiados = archivos_modificados_entre("origin/main")
if cambiados:
    subprocess.run(["ruff", "check"] + cambiados)""",

        """# pre-commit hook en Python
# Guardar como .git/hooks/pre-commit y dar permisos de ejecución

import subprocess
import sys

def ejecutar(cmd: list[str]) -> int:
    return subprocess.run(cmd).returncode

errores = 0
errores += ejecutar(["ruff", "check", "."])
errores += ejecutar(["mypy", "--ignore-missing-imports", "."])
errores += ejecutar(["pytest", "-q", "--tb=short"])

if errores:
    print("Pre-commit falló — commit cancelado")
    sys.exit(1)""",

        """import subprocess

class GitRepo:
    \"\"\"Wrapper mínimo de operaciones git.\"\"\"

    def __init__(self, directorio: str = "."):
        self.dir = directorio

    def _run(self, *args: str) -> str:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True,
            check=True, cwd=self.dir
        ).stdout.strip()

    def status(self) -> str:
        return self._run("status", "--short")

    def log(self, n: int = 5) -> list[str]:
        return self._run("log", f"-{n}", "--oneline").splitlines()

    def push(self) -> None:
        self._run("push")""",

        """# Detector de secretos en staged files
import subprocess
import re
import sys

PATRONES = [
    r"(?i)api[_-]?key\\s*=\\s*['\"][^'\"]{10,}",
    r"(?i)password\\s*=\\s*['\"][^'\"]{4,}",
    r"(?i)secret\\s*=\\s*['\"][^'\"]{10,}",
]

def detectar_secretos() -> list[str]:
    diff = subprocess.run(
        ["git", "diff", "--cached", "--unified=0"],
        capture_output=True, text=True
    ).stdout
    encontrados = []
    for linea in diff.splitlines():
        if linea.startswith("+"):
            for pat in PATRONES:
                if re.search(pat, linea):
                    encontrados.append(linea)
    return encontrados

secretos = detectar_secretos()
if secretos:
    print("SECRETOS detectados en staged files:")
    for s in secretos:
        print(" ", s)
    sys.exit(1)""",
    ]
    return [jsonl_entry(e, "git_y_ci_cd") for e in shuffle_and_sample(examples, n)]


def generar_iterator_pattern(n: int) -> list[str]:
    """Genera ejemplos del patrón Iterator en Python."""
    examples = [
        """class RangeIterator:
    \"\"\"Implementación manual del patrón Iterator.\"\"\"

    def __init__(self, inicio: int, fin: int, paso: int = 1):
        self._actual = inicio
        self._fin = fin
        self._paso = paso

    def __iter__(self):
        return self

    def __next__(self) -> int:
        if self._actual >= self._fin:
            raise StopIteration
        valor = self._actual
        self._actual += self._paso
        return valor

for n in RangeIterator(0, 10, 2):
    print(n)  # 0 2 4 6 8""",

        """from typing import Iterator, TypeVar

T = TypeVar("T")

class ListaDobleEnlazada:
    \"\"\"Lista con iterador explicit.\"\"\"

    class Nodo:
        def __init__(self, valor):
            self.valor = valor
            self.siguiente = None

    def __init__(self):
        self._cabeza = None

    def agregar(self, valor) -> None:
        nodo = self.Nodo(valor)
        nodo.siguiente = self._cabeza
        self._cabeza = nodo

    def __iter__(self) -> Iterator:
        actual = self._cabeza
        while actual:
            yield actual.valor
            actual = actual.siguiente""",

        """class ArbolBinario:
    def __init__(self, valor, izq=None, der=None):
        self.valor = valor
        self.izq = izq
        self.der = der

    def __iter__(self):
        \"\"\"Recorrido inorden usando yield.\"\"\"
        if self.izq:
            yield from self.izq
        yield self.valor
        if self.der:
            yield from self.der

arbol = ArbolBinario(4, ArbolBinario(2, ArbolBinario(1), ArbolBinario(3)), ArbolBinario(6))
print(list(arbol))  # [1, 2, 3, 4, 6]""",

        """from abc import ABC, abstractmethod
from typing import Iterator, Any

class Coleccion(ABC):
    @abstractmethod
    def crear_iterador(self) -> Iterator[Any]: ...

class ColeccionFiltrada(Coleccion):
    def __init__(self, items: list, predicado):
        self._items = items
        self._pred = predicado

    def crear_iterador(self) -> Iterator[Any]:
        return (x for x in self._items if self._pred(x))

pares = ColeccionFiltrada(range(10), lambda x: x % 2 == 0)
print(list(pares.crear_iterador()))  # [0, 2, 4, 6, 8]""",

        """class GeneradorInfinito:
    \"\"\"Iterador infinito de Fibonacci.\"\"\"

    def __iter__(self):
        return self

    def __init__(self):
        self._a, self._b = 0, 1

    def __next__(self) -> int:
        valor = self._a
        self._a, self._b = self._b, self._a + self._b
        return valor

import itertools
primeros_10 = list(itertools.islice(GeneradorInfinito(), 10))
print(primeros_10)  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]""",

        """from typing import Iterator

def csv_iterator(ruta: str) -> Iterator[dict]:
    \"\"\"Itera sobre filas de un CSV sin cargar todo en RAM.\"\"\"
    import csv
    with open(ruta, newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)

# Uso: procesar CSV de 10 GB línea a línea
for fila in csv_iterator("datos.csv"):
    procesar(fila)""",

        """class IteradorConEstado:
    \"\"\"Iterador que admite reset.\"\"\"

    def __init__(self, datos: list):
        self._datos = datos
        self._indice = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._indice >= len(self._datos):
            raise StopIteration
        v = self._datos[self._indice]
        self._indice += 1
        return v

    def reset(self) -> None:
        self._indice = 0""",

        """# Iterador de paginación de API
from typing import Iterator
import urllib.request, json

def paginar(url_base: str, pagina: int = 1) -> Iterator[list]:
    \"\"\"Itera sobre páginas de una API REST.\"\"\"
    while True:
        url = f"{url_base}?page={pagina}&per_page=20"
        with urllib.request.urlopen(url) as resp:
            data = json.loads(resp.read())
        if not data:
            break
        yield data
        pagina += 1""",
    ]
    return [jsonl_entry(e, "iterator_pattern") for e in shuffle_and_sample(examples, n)]


def generar_command_pattern(n: int) -> list[str]:
    """Genera ejemplos del patrón Command en Python."""
    examples = [
        """from abc import ABC, abstractmethod

class Comando(ABC):
    @abstractmethod
    def ejecutar(self) -> None: ...

    @abstractmethod
    def deshacer(self) -> None: ...

class Receptor:
    def __init__(self):
        self.texto = ""

    def agregar(self, texto: str) -> None:
        self.texto += texto
        print(f"Texto: {self.texto!r}")

    def eliminar(self, n: int) -> None:
        self.texto = self.texto[:-n]
        print(f"Texto: {self.texto!r}")

class ComandoAgregar(Comando):
    def __init__(self, receptor: Receptor, texto: str):
        self._r = receptor
        self._txt = texto

    def ejecutar(self) -> None:
        self._r.agregar(self._txt)

    def deshacer(self) -> None:
        self._r.eliminar(len(self._txt))""",

        """from collections import deque
from typing import Protocol

class Comando(Protocol):
    def ejecutar(self) -> None: ...
    def deshacer(self) -> None: ...

class HistorialComandos:
    \"\"\"Invocador con soporte de undo/redo.\"\"\"

    def __init__(self):
        self._historial: deque[Comando] = deque()
        self._rehace: deque[Comando] = deque()

    def ejecutar(self, cmd: Comando) -> None:
        cmd.ejecutar()
        self._historial.append(cmd)
        self._rehace.clear()

    def deshacer(self) -> None:
        if self._historial:
            cmd = self._historial.pop()
            cmd.deshacer()
            self._rehace.append(cmd)

    def rehacer(self) -> None:
        if self._rehace:
            cmd = self._rehace.pop()
            cmd.ejecutar()
            self._historial.append(cmd)""",

        """from dataclasses import dataclass, field
from typing import Callable

@dataclass
class ComandoFuncion:
    \"\"\"Comando implementado con funciones (sin clases extra).\"\"\"
    ejecutar: Callable[[], None]
    deshacer: Callable[[], None]
    descripcion: str = ""

# Uso
valores: list[int] = []
cmd = ComandoFuncion(
    ejecutar=lambda: valores.append(42),
    deshacer=lambda: valores.pop(),
    descripcion="Agregar 42",
)
cmd.ejecutar()
print(valores)  # [42]
cmd.deshacer()
print(valores)  # []""",

        """# Command + Queue: procesamiento asíncrono
from queue import Queue
from threading import Thread
from abc import ABC, abstractmethod

class Tarea(ABC):
    @abstractmethod
    def ejecutar(self) -> None: ...

class ColaTareas:
    def __init__(self):
        self._q: Queue[Tarea | None] = Queue()

    def encolar(self, tarea: Tarea) -> None:
        self._q.put(tarea)

    def procesar(self) -> None:
        while True:
            tarea = self._q.get()
            if tarea is None:
                break
            tarea.ejecutar()
            self._q.task_done()

    def iniciar_worker(self) -> Thread:
        t = Thread(target=self.procesar, daemon=True)
        t.start()
        return t""",

        """# Command para operaciones de base de datos con transacción
from typing import Any

class ComandoDB:
    def __init__(self, conexion):
        self._conn = conexion
        self._historial: list[tuple[str, tuple]] = []

    def ejecutar(self, sql: str, params: tuple = ()) -> Any:
        self._historial.append((sql, params))
        return self._conn.execute(sql, params)

    def rollback(self) -> None:
        # Deshacer en orden inverso
        for sql, params in reversed(self._historial):
            if sql.startswith("INSERT"):
                self._conn.execute("DELETE WHERE id = ?", (params[0],))
        self._historial.clear()""",

        """from typing import Callable
from functools import partial

# Command como callable (estilo funcional)
def mover_archivo(origen: str, destino: str) -> None:
    import shutil; shutil.move(origen, destino)

def crear_comando(fn: Callable, *args, **kwargs) -> Callable[[], None]:
    return partial(fn, *args, **kwargs)

# Cola de comandos diferidos
pendientes: list[Callable] = [
    crear_comando(mover_archivo, "a.txt", "tmp/a.txt"),
    crear_comando(print, "Archivos movidos"),
]

for cmd in pendientes:
    cmd()""",

        """# Macro: secuencia de comandos
from abc import ABC, abstractmethod

class Comando(ABC):
    @abstractmethod
    def ejecutar(self) -> None: ...

class Macro(Comando):
    \"\"\"Compone múltiples comandos en uno.\"\"\"

    def __init__(self, *comandos: Comando):
        self._cmds = list(comandos)

    def ejecutar(self) -> None:
        for cmd in self._cmds:
            cmd.ejecutar()

    def agregar(self, cmd: Comando) -> "Macro":
        self._cmds.append(cmd)
        return self""",
    ]
    return [jsonl_entry(e, "command_pattern") for e in shuffle_and_sample(examples, n)]


def generar_refactoring(n: int) -> list[str]:
    """Genera ejemplos de refactoring con antes/despues en Python."""
    examples = [
        """# ANTES: función larga con múltiples responsabilidades
def procesar_pedido(pedido):
    # Validar
    if not pedido.get("items"):
        raise ValueError("Sin items")
    if pedido.get("total", 0) <= 0:
        raise ValueError("Total inválido")
    # Calcular descuento
    descuento = 0
    if pedido["total"] > 100:
        descuento = pedido["total"] * 0.1
    # Guardar en DB
    import sqlite3
    conn = sqlite3.connect("db.sqlite3")
    conn.execute("INSERT INTO pedidos VALUES (?, ?)", (pedido["id"], pedido["total"] - descuento))
    conn.commit()
    return pedido["total"] - descuento

# DESPUÉS: funciones con responsabilidad única
def validar_pedido(pedido: dict) -> None:
    if not pedido.get("items"):
        raise ValueError("Sin items")
    if pedido.get("total", 0) <= 0:
        raise ValueError("Total inválido")

def calcular_descuento(total: float) -> float:
    return total * 0.1 if total > 100 else 0.0

def guardar_pedido(conn, pedido_id: int, total_final: float) -> None:
    conn.execute("INSERT INTO pedidos VALUES (?, ?)", (pedido_id, total_final))
    conn.commit()""",

        """# ANTES: números mágicos
def calcular_precio(cantidad, precio):
    if cantidad > 10:
        return cantidad * precio * 0.85
    return cantidad * precio * 1.21

# DESPUÉS: constantes con nombre descriptivo
DESCUENTO_VOLUMEN = 0.85    # 15% de descuento para > 10 unidades
IVA = 1.21                  # IVA del 21%
UMBRAL_VOLUMEN = 10

def calcular_precio(cantidad: int, precio: float) -> float:
    if cantidad > UMBRAL_VOLUMEN:
        return cantidad * precio * DESCUENTO_VOLUMEN
    return cantidad * precio * IVA""",

        """# ANTES: condicionales anidadas (arrow anti-pattern)
def procesar(usuario, pedido, pago):
    if usuario:
        if usuario.activo:
            if pedido:
                if pedido.items:
                    if pago:
                        return pago.cobrar(pedido.total)
    return None

# DESPUÉS: guard clauses (early return)
def procesar(usuario, pedido, pago):
    if not usuario or not usuario.activo:
        return None
    if not pedido or not pedido.items:
        return None
    if not pago:
        return None
    return pago.cobrar(pedido.total)""",

        """# ANTES: duplicación de código (DRY violation)
def area_circulo(r):
    return 3.14159 * r * r

def perimetro_circulo(r):
    return 2 * 3.14159 * r

def volumen_esfera(r):
    return (4 / 3) * 3.14159 * r ** 3

# DESPUÉS: constante y funciones cohesivas
import math

def area_circulo(radio: float) -> float:
    return math.pi * radio ** 2

def perimetro_circulo(radio: float) -> float:
    return 2 * math.pi * radio

def volumen_esfera(radio: float) -> float:
    return (4 / 3) * math.pi * radio ** 3""",

        """# ANTES: clase con demasiados campos (Feature Envy / Large Class)
class Pedido:
    def __init__(self, id, cliente_nombre, cliente_email, cliente_dir, items, total):
        self.id = id
        self.cliente_nombre = cliente_nombre
        self.cliente_email = cliente_email
        self.cliente_dir = cliente_dir
        self.items = items
        self.total = total

# DESPUÉS: extraer clase Cliente
from dataclasses import dataclass

@dataclass
class Cliente:
    nombre: str
    email: str
    direccion: str

@dataclass
class Pedido:
    id: int
    cliente: Cliente
    items: list
    total: float""",

        """# ANTES: string concatenación ilegible
def construir_query(tabla, columnas, condicion):
    q = "SELECT "
    for i, c in enumerate(columnas):
        q += c
        if i < len(columnas) - 1:
            q += ", "
    q += " FROM " + tabla
    if condicion:
        q += " WHERE " + condicion
    return q

# DESPUÉS: f-string y join
def construir_query(tabla: str, columnas: list[str], condicion: str = "") -> str:
    cols = ", ".join(columnas)
    base = f"SELECT {cols} FROM {tabla}"
    return f"{base} WHERE {condicion}" if condicion else base""",

        """# ANTES: parámetro booleano que bifurca comportamiento
def enviar_email(destinatario, asunto, cuerpo, es_html):
    if es_html:
        # lógica HTML
        headers = {"Content-Type": "text/html"}
    else:
        headers = {"Content-Type": "text/plain"}
    # ...enviar...

# DESPUÉS: dos funciones con nombres claros
def enviar_email_texto(destinatario: str, asunto: str, cuerpo: str) -> None:
    _enviar(destinatario, asunto, cuerpo, content_type="text/plain")

def enviar_email_html(destinatario: str, asunto: str, html: str) -> None:
    _enviar(destinatario, asunto, html, content_type="text/html")

def _enviar(dest, asunto, body, content_type): ...""",

        """# ANTES: comentarios que explican código oscuro
def calc(d, r):
    # d es descuento en %, r es precio base
    # Convertir porcentaje a decimal y restar
    return r - (r * (d / 100))

# DESPUÉS: código auto-documentado, sin comentarios redundantes
def aplicar_descuento(precio_base: float, descuento_pct: float) -> float:
    \"\"\"Retorna el precio con el descuento aplicado.\"\"\"
    return precio_base * (1 - descuento_pct / 100)""",

        """# ANTES: abuso de excepciones para control de flujo
def buscar_usuario(id_: int) -> dict:
    try:
        return DB.get(id_)
    except KeyError:
        return {}
    except Exception:
        return {}

# DESPUÉS: control explícito con Optional
from typing import Optional

def buscar_usuario(id_: int) -> Optional[dict]:
    \"\"\"Retorna el usuario o None si no existe.\"\"\"
    return DB.get(id_)  # dict.get retorna None si no existe""",
    ]
    return [jsonl_entry(e, "refactoring") for e in shuffle_and_sample(examples, n)]


def generar_testing_avanzado(n: int) -> list[str]:
    """Genera ejemplos de testing avanzado con pytest."""
    examples = [
        """import pytest
from unittest.mock import MagicMock, patch

def enviar_notificacion(servicio, usuario_id: int) -> bool:
    usuario = servicio.obtener(usuario_id)
    if not usuario:
        return False
    servicio.notificar(usuario["email"], "Bienvenido")
    return True

def test_enviar_notificacion_usuario_existente():
    servicio = MagicMock()
    servicio.obtener.return_value = {"email": "a@test.com"}
    assert enviar_notificacion(servicio, 1) is True
    servicio.notificar.assert_called_once_with("a@test.com", "Bienvenido")

def test_enviar_notificacion_usuario_inexistente():
    servicio = MagicMock()
    servicio.obtener.return_value = None
    assert enviar_notificacion(servicio, 99) is False
    servicio.notificar.assert_not_called()""",

        """import pytest

@pytest.mark.parametrize("entrada,esperado", [
    ("hola", "HOLA"),
    ("mundo", "MUNDO"),
    ("", ""),
    ("  espacios  ", "  ESPACIOS  "),
])
def test_upper(entrada: str, esperado: str) -> None:
    assert entrada.upper() == esperado

@pytest.mark.parametrize("a,b,resultado", [
    (1, 2, 3),
    (0, 0, 0),
    (-1, 1, 0),
    (100, -50, 50),
])
def test_suma(a: int, b: int, resultado: int) -> None:
    assert a + b == resultado""",

        """import pytest
from unittest.mock import patch, AsyncMock
import asyncio

async def fetch_data(url: str) -> dict:
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.json()

@pytest.mark.asyncio
async def test_fetch_data():
    mock_resp = AsyncMock()
    mock_resp.json.return_value = {"status": "ok"}
    mock_session = AsyncMock()
    mock_session.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_resp

    with patch("aiohttp.ClientSession", return_value=mock_session):
        result = await fetch_data("http://example.com")
    assert result == {"status": "ok"}""",

        """import pytest
from unittest.mock import patch
import time

# Fixture con scope compartido
@pytest.fixture(scope="module")
def config_test():
    return {"db": "sqlite:///:memory:", "debug": True}

# Fixture con teardown
@pytest.fixture
def archivo_temporal(tmp_path):
    archivo = tmp_path / "test.txt"
    archivo.write_text("contenido inicial")
    yield archivo
    # teardown automático al salir de tmp_path

def test_modificar_archivo(archivo_temporal):
    archivo_temporal.write_text("nuevo contenido")
    assert archivo_temporal.read_text() == "nuevo contenido\"""",

        """import pytest

class TestCalculadora:
    \"\"\"Suite completa con setup/teardown de clase.\"\"\"

    @pytest.fixture(autouse=True)
    def setup(self):
        self.calc = Calculadora()
        yield
        # teardown si hace falta

    def test_suma(self):
        assert self.calc.sumar(2, 3) == 5

    def test_division(self):
        assert self.calc.dividir(10, 2) == 5.0

    def test_division_por_cero(self):
        with pytest.raises(ZeroDivisionError):
            self.calc.dividir(5, 0)

    @pytest.mark.slow
    def test_operacion_pesada(self):
        assert self.calc.factorial(20) > 0""",

        """import pytest
from unittest.mock import patch, call

def procesar_batch(items: list[int], procesador) -> list[int]:
    return [procesador(i) for i in items]

def test_procesador_llamado_con_cada_item():
    mock_proc = MagicMock(side_effect=lambda x: x * 2)
    resultado = procesar_batch([1, 2, 3], mock_proc)
    assert resultado == [2, 4, 6]
    assert mock_proc.call_count == 3
    mock_proc.assert_has_calls([call(1), call(2), call(3)])

def test_procesador_lanza_excepcion():
    mock_proc = MagicMock(side_effect=ValueError("error"))
    with pytest.raises(ValueError, match="error"):
        procesar_batch([1], mock_proc)""",

        """import pytest
from unittest.mock import patch
import json

# Test con captura de stdout
def imprimir_reporte(datos: list[dict]) -> None:
    for item in datos:
        print(f"{item['nombre']}: {item['valor']}")

def test_reporte_imprime_correctamente(capsys):
    datos = [{"nombre": "A", "valor": 1}, {"nombre": "B", "valor": 2}]
    imprimir_reporte(datos)
    output = capsys.readouterr().out
    assert "A: 1" in output
    assert "B: 2" in output""",

        """import pytest

# Markers personalizados
def pytest_configure(config):
    config.addinivalue_line("markers", "integracion: tests de integración lentos")
    config.addinivalue_line("markers", "smoke: tests de humo rápidos")

@pytest.mark.smoke
def test_main_importa():
    import main  # Solo verifica que no hay errores de importación

@pytest.mark.integracion
def test_flujo_completo(db_session):
    usuario = crear_usuario(db_session, "Alice")
    assert usuario.id is not None
    assert db_session.query(Usuario).count() == 1""",

        """import pytest
from hypothesis import given, strategies as st

# Property-based testing con Hypothesis
@given(st.lists(st.integers()))
def test_sort_idempotente(lista):
    \"\"\"Ordenar dos veces da el mismo resultado que una vez.\"\"\"
    assert sorted(sorted(lista)) == sorted(lista)

@given(st.text())
def test_upper_lower_roundtrip(texto):
    \"\"\"upper().lower() preserva len pero case puede diferir.\"\"\"
    assert len(texto.upper()) == len(texto)

@given(st.integers(min_value=0, max_value=1000))
def test_factorial_positivo(n):
    import math
    assert math.factorial(n) >= 1""",
    ]
    return [jsonl_entry(e, "testing_avanzado") for e in shuffle_and_sample(examples, n)]


def generar_arboles_binarios_extra(n: int) -> list[str]:
    """Genera ejemplos adicionales de árboles binarios."""
    examples = [
        """class Nodo:
    def __init__(self, valor: int):
        self.valor = valor
        self.izq: "Nodo | None" = None
        self.der: "Nodo | None" = None

class BST:
    \"\"\"Árbol Binario de Búsqueda.\"\"\"

    def __init__(self):
        self.raiz: Nodo | None = None

    def insertar(self, valor: int) -> None:
        if not self.raiz:
            self.raiz = Nodo(valor)
        else:
            self._insertar(self.raiz, valor)

    def _insertar(self, nodo: Nodo, valor: int) -> None:
        if valor < nodo.valor:
            if nodo.izq is None:
                nodo.izq = Nodo(valor)
            else:
                self._insertar(nodo.izq, valor)
        else:
            if nodo.der is None:
                nodo.der = Nodo(valor)
            else:
                self._insertar(nodo.der, valor)""",

        """from collections import deque

class Nodo:
    def __init__(self, val: int, izq=None, der=None):
        self.val, self.izq, self.der = val, izq, der

def bfs(raiz: Nodo | None) -> list[list[int]]:
    \"\"\"Recorrido por niveles (BFS).\"\"\"
    if not raiz:
        return []
    resultado = []
    cola = deque([raiz])
    while cola:
        nivel = []
        for _ in range(len(cola)):
            nodo = cola.popleft()
            nivel.append(nodo.val)
            if nodo.izq: cola.append(nodo.izq)
            if nodo.der: cola.append(nodo.der)
        resultado.append(nivel)
    return resultado""",

        """class Nodo:
    def __init__(self, val, izq=None, der=None):
        self.val, self.izq, self.der = val, izq, der

def altura(nodo: "Nodo | None") -> int:
    if nodo is None:
        return 0
    return 1 + max(altura(nodo.izq), altura(nodo.der))

def esta_balanceado(nodo: "Nodo | None") -> bool:
    if nodo is None:
        return True
    dif = abs(altura(nodo.izq) - altura(nodo.der))
    return dif <= 1 and esta_balanceado(nodo.izq) and esta_balanceado(nodo.der)

raiz = Nodo(1, Nodo(2, Nodo(4)), Nodo(3))
print(esta_balanceado(raiz))  # True""",

        """class Nodo:
    def __init__(self, val, izq=None, der=None):
        self.val, self.izq, self.der = val, izq, der

def invertir(nodo: "Nodo | None") -> "Nodo | None":
    \"\"\"Invierte un árbol binario (mirror).\"\"\"
    if nodo is None:
        return None
    nodo.izq, nodo.der = invertir(nodo.der), invertir(nodo.izq)
    return nodo

def inorden(nodo, acc=None) -> list[int]:
    if acc is None: acc = []
    if nodo:
        inorden(nodo.izq, acc)
        acc.append(nodo.val)
        inorden(nodo.der, acc)
    return acc""",

        """class Nodo:
    def __init__(self, val, izq=None, der=None):
        self.val, self.izq, self.der = val, izq, der

def lca(raiz: "Nodo | None", p: int, q: int) -> "Nodo | None":
    \"\"\"Least Common Ancestor en BST.\"\"\"
    if raiz is None:
        return None
    if p < raiz.val and q < raiz.val:
        return lca(raiz.izq, p, q)
    if p > raiz.val and q > raiz.val:
        return lca(raiz.der, p, q)
    return raiz

raiz = Nodo(6, Nodo(2, Nodo(0), Nodo(4)), Nodo(8, Nodo(7), Nodo(9)))
print(lca(raiz, 2, 8).val)  # 6
print(lca(raiz, 0, 4).val)  # 2""",

        """class Nodo:
    def __init__(self, val, izq=None, der=None):
        self.val, self.izq, self.der = val, izq, der

def serializar(raiz: "Nodo | None") -> str:
    \"\"\"Serializa árbol a string (preorden con None=N).\"\"\"
    if raiz is None:
        return "N"
    return f"{raiz.val},{serializar(raiz.izq)},{serializar(raiz.der)}"

def deserializar(data: str) -> "Nodo | None":
    tokens = iter(data.split(","))
    def helper():
        val = next(tokens)
        if val == "N":
            return None
        nodo = Nodo(int(val))
        nodo.izq = helper()
        nodo.der = helper()
        return nodo
    return helper()""",

        """class Nodo:
    def __init__(self, val, izq=None, der=None):
        self.val, self.izq, self.der = val, izq, der

def max_path_sum(nodo: "Nodo | None") -> int:
    \"\"\"Suma máxima de camino en árbol binario.\"\"\"
    max_total = float("-inf")

    def helper(n: "Nodo | None") -> int:
        nonlocal max_total
        if n is None:
            return 0
        izq = max(helper(n.izq), 0)
        der = max(helper(n.der), 0)
        max_total = max(max_total, n.val + izq + der)
        return n.val + max(izq, der)

    helper(nodo)
    return int(max_total)

raiz = Nodo(-10, Nodo(9), Nodo(20, Nodo(15), Nodo(7)))
print(max_path_sum(raiz))  # 42""",
    ]
    return [jsonl_entry(e, "arboles_binarios") for e in shuffle_and_sample(examples, n)]


# MAPA DE GENERADORES Y ARCHIVOS DESTINO
# =============================================================================

# =============================================================================
# GENERADORES — TEMAS PRINCIPALES (agregados Feb 2026)
# =============================================================================

def generar_variables_y_tipos(n: int) -> list[str]:
    ejemplos = [
        'x = 42\ny = 3.14\nnombre = "Python"\nactivo = True\nvacio = None\nprint(type(x), type(y), type(nombre))',
        'edad = 25\naltura = 1.75\nprint(f"Edad: {edad}, Altura: {altura:.2f}m")',
        '# Conversión de tipos\nentero = int("42")\nflotante = float("3.14")\ntexto = str(100)\nbool_val = bool(0)\nprint(entero, flotante, texto, bool_val)',
        'a = 10\nb = 3\nprint(a + b)   # 13\nprint(a - b)   # 7\nprint(a * b)   # 30\nprint(a / b)   # 3.333\nprint(a // b)  # 3 (división entera)\nprint(a % b)   # 1 (módulo)\nprint(a ** b)  # 1000 (potencia)',
        'x = 5\nprint(isinstance(x, int))    # True\nprint(isinstance(x, float))  # False\nprint(type(x) is int)        # True',
        'valores = [1, "hola", 3.14, True, None]\nfor v in valores:\n    print(f"{v!r:<10} -> {type(v).__name__}")',
        '# Variables múltiples\na, b, c = 1, 2, 3\nprint(a, b, c)  # 1 2 3\n\n# Swap\na, b = b, a\nprint(a, b)  # 2 1\n\n# Desempaquetado extendido\nprimero, *resto = [1, 2, 3, 4, 5]\nprint(primero, resto)  # 1 [2, 3, 4, 5]',
        'numero = 255\nprint(bin(numero))   # 0b11111111\nprint(oct(numero))   # 0o377\nprint(hex(numero))   # 0xff\nprint(int("ff", 16)) # 255',
        'def es_numero(s: str) -> bool:\n    """Verifica si un string puede convertirse a número."""\n    try:\n        float(s)\n        return True\n    except ValueError:\n        return False\n\nprint(es_numero("3.14"))  # True\nprint(es_numero("abc"))   # False',
        'x: int = 10\ny: float = 3.14\nz: str = "hola"\nw: bool = True\nprint(x, y, z, w)',
        'MAXIMO_INT = 2**63 - 1\nMINIMO_FLOAT = 1e-308\nINFINITO = float("inf")\nNAN = float("nan")\nprint(MAXIMO_INT)\nprint(MINIMO_FLOAT)\nimport math\nprint(math.isinf(INFINITO))  # True\nprint(math.isnan(NAN))       # True',
        '# Operadores de comparación e identidad\na = [1, 2, 3]\nb = [1, 2, 3]\nc = a\nprint(a == b)   # True (igual valor)\nprint(a is b)   # False (distinto objeto)\nprint(a is c)   # True (mismo objeto)',
        'nombre = "Lucas"\nprint(nombre.upper())        # LUCAS\nprint(len(nombre))           # 5\nprint(nombre[0])             # L\nprint(nombre[-1])            # s\nprint(nombre[:3])            # Luc',
    ]
    return [jsonl_entry(e, "variables_y_tipos") for e in shuffle_and_sample(ejemplos, n)]


def generar_strings_y_formato(n: int) -> list[str]:
    ejemplos = [
        'nombre = "Lucas"\nedad = 25\nprint(f"Hola, {nombre}! Tienes {edad} años.")',
        'pi = 3.14159\nprint(f"Pi = {pi:.2f}")        # Pi = 3.14\nprint(f"Pi = {pi:10.4f}")   # Pi =     3.1416\nprint(f"Hex: {255:#x}")       # Hex: 0xff',
        'texto = "  hola mundo  "\nprint(texto.strip())         # "hola mundo"\nprint(texto.lstrip())        # "hola mundo  "\nprint(texto.rstrip())        # "  hola mundo"',
        'cadena = "python es genial"\nprint(cadena.split())           # [\'python\', \'es\', \'genial\']\nprint(cadena.split("es"))       # [\'python \', \' genial\']\nprint(", ".join(["a", "b", "c"]))  # "a, b, c"',
        'texto = "Hello, World!"\nprint(texto.upper())      # HELLO, WORLD!\nprint(texto.lower())      # hello, world!\nprint(texto.title())      # Hello, World!\nprint(texto.capitalize())  # Hello, world!',
        'cadena = "banana"\nprint(cadena.count("a"))    # 3\nprint(cadena.find("an"))    # 1\nprint(cadena.index("n"))    # 2\nprint(cadena.replace("a", "o"))  # "bonono"',
        'email = "usuario@ejemplo.com"\nprint(email.startswith("usuario"))  # True\nprint(email.endswith(".com"))        # True\nprint("@" in email)                  # True',
        '# Multiline strings\npoema = """\nRosas rojas,\nvioletas azules,\nPython es genial,\n¡y tú también!\n"""\nprint(poema.strip())',
        'frase = "el cielo es azul"\npalabras = frase.split()\nresultado = " ".join(p.capitalize() for p in palabras)\nprint(resultado)  # El Cielo Es Azul',
        'plantilla = "Nombre: {nombre}, Edad: {edad}"\nprint(plantilla.format(nombre="Ana", edad=30))',
        'texto = "abc123def456"\nimport re\nnumeros = re.findall(r"\\d+", texto)\nprint(numeros)  # [\'123\', \'456\']',
        'def truncar(texto: str, maximo: int = 50) -> str:\n    """Trunca un texto largo añadiendo \'...\'"""\n    if len(texto) <= maximo:\n        return texto\n    return texto[:maximo - 3] + "..."\n\nprint(truncar("Hola mundo", 8))  # "Hola ..."',
        'lineas = "línea 1\\nlínea 2\\nlínea 3"\nfor i, linea in enumerate(lineas.splitlines(), 1):\n    print(f"{i}: {linea}")',
    ]
    return [jsonl_entry(e, "strings_y_formato") for e in shuffle_and_sample(ejemplos, n)]


def generar_control_de_flujo(n: int) -> list[str]:
    ejemplos = [
        'x = 10\nif x > 0:\n    print("positivo")\nelif x < 0:\n    print("negativo")\nelse:\n    print("cero")',
        'for i in range(5):\n    print(i, end=" ")  # 0 1 2 3 4',
        'for i in range(10):\n    if i == 7:\n        break\n    if i % 2 == 0:\n        continue\n    print(i, end=" ")  # 1 3 5',
        'n = 1\nwhile n <= 5:\n    print(n, end=" ")\n    n += 1  # 1 2 3 4 5',
        'frutas = ["manzana", "banana", "cereza"]\nfor i, fruta in enumerate(frutas):\n    print(f"{i}: {fruta}")',
        'a = [1, 2, 3]\nb = ["x", "y", "z"]\nfor num, letra in zip(a, b):\n    print(num, letra)',
        '# Expresión ternaria\nedad = 18\nestado = "mayor" if edad >= 18 else "menor"\nprint(estado)  # mayor',
        'match comando:\n    case "start":\n        print("Iniciando...")\n    case "stop":\n        print("Deteniendo...")\n    case _:\n        print("Comando desconocido")',
        'nombres = ["Ana", "Bob", "Carlos"]\nfor nombre in reversed(nombres):\n    print(nombre)',
        '# Bucle con else\nfor i in range(5):\n    if i == 10:  # nunca ocurre\n        break\nelse:\n    print("Bucle completado sin break")',
        'numero = 42\ndescripcion = (\n    "grande" if numero > 100\n    else "mediano" if numero > 10\n    else "pequeño"\n)\nprint(descripcion)  # mediano',
        'numeros = range(1, 11)\npares = [n for n in numeros if n % 2 == 0]\nprint(pares)  # [2, 4, 6, 8, 10]',
        'def contar_hasta(n: int) -> None:\n    for i in range(1, n + 1):\n        print(i)',
    ]
    return [jsonl_entry(e, "control_de_flujo") for e in shuffle_and_sample(ejemplos, n)]


def generar_listas_y_tuplas(n: int) -> list[str]:
    ejemplos = [
        'nums = [3, 1, 4, 1, 5, 9, 2, 6]\nprint(sorted(nums))           # [1, 1, 2, 3, 4, 5, 6, 9]\nprint(sorted(nums, reverse=True))  # [9, 6, 5, 4, 3, 2, 1, 1]',
        'lista = [1, 2, 3]\nlista.append(4)\nlista.extend([5, 6])\nlista.insert(0, 0)\nprint(lista)  # [0, 1, 2, 3, 4, 5, 6]',
        '# Slicing\nnums = list(range(10))\nprint(nums[2:5])    # [2, 3, 4]\nprint(nums[::2])    # [0, 2, 4, 6, 8]\nprint(nums[::-1])   # [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]',
        'colores = ["rojo", "verde", "azul"]\nprint("verde" in colores)   # True\nprint(colores.index("azul"))  # 2\nprint(colores.count("rojo")) # 1\ncolores.remove("verde")\nprint(colores)',
        '# Tuplas — inmutables\npunto = (3, 4)\nx, y = punto\ndistancia = (x**2 + y**2) ** 0.5\nprint(distancia)  # 5.0',
        'nombres_coords = [("Ana", 1, 2), ("Bob", 3, 4), ("Carlos", 0, 1)]\nnombres_coords.sort(key=lambda t: t[1])\nprint(nombres_coords)',
        'a = [1, 2, 3]\nb = a[:]  # copia superficial\nb.append(4)\nprint(a)  # [1, 2, 3]\nprint(b)  # [1, 2, 3, 4]',
        'filas = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]\nfila = filas[1]\nprint(fila[2])  # 6\nprint(filas[0][0])  # 1',
        'nums = [1, 2, 3, 4, 5]\nprint(sum(nums))    # 15\nprint(min(nums))    # 1\nprint(max(nums))    # 5\nprint(len(nums))    # 5',
        'palabras = ["banana", "manzana", "kiwi"]\npalabras.sort(key=len)\nprint(palabras)  # [\'kiwi\', \'banana\', \'manzana\']',
        'from typing import List\n\ndef aplanar(lista: List[List]) -> List:\n    """Aplana una lista de listas."""\n    return [elem for sublista in lista for elem in sublista]\n\nprint(aplanar([[1, 2], [3, 4], [5]]))  # [1, 2, 3, 4, 5]',
        '# Comprensión de lista con condición\ncuadrados_pares = [x**2 for x in range(10) if x % 2 == 0]\nprint(cuadrados_pares)  # [0, 4, 16, 36, 64]',
        'primero, *resto, ultimo = [1, 2, 3, 4, 5]\nprint(primero)  # 1\nprint(resto)    # [2, 3, 4]\nprint(ultimo)   # 5',
    ]
    return [jsonl_entry(e, "listas_y_tuplas") for e in shuffle_and_sample(ejemplos, n)]


def generar_diccionarios_y_sets(n: int) -> list[str]:
    ejemplos = [
        'persona = {"nombre": "Ana", "edad": 30, "ciudad": "Madrid"}\nprint(persona["nombre"])         # Ana\nprint(persona.get("pais", "N/A")) # N/A',
        'inventario = {"manzanas": 5, "bananas": 3}\ninventario["naranjas"] = 10\ninventario["manzanas"] += 2\nprint(inventario)',
        '# Iterar sobre dict\npuntos = {"Ana": 95, "Bob": 87, "Carlos": 92}\nfor nombre, pts in puntos.items():\n    print(f"{nombre}: {pts}")\n\nganador = max(puntos, key=puntos.get)\nprint(f"Ganador: {ganador}")',
        'nums = {1, 2, 3, 4, 5}\npares = {2, 4, 6, 8}\nprint(nums & pares)   # {2, 4}   intersección\nprint(nums | pares)   # {1, 2, 3, 4, 5, 6, 8}  unión\nprint(nums - pares)   # {1, 3, 5}  diferencia',
        '# dict comprehension\ncuadrados = {x: x**2 for x in range(6)}\nprint(cuadrados)  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16, 5: 25}',
        'from collections import defaultdict\n\ngrupos = defaultdict(list)\ndatos = [("A", 1), ("B", 2), ("A", 3), ("B", 4)]\nfor clave, valor in datos:\n    grupos[clave].append(valor)\nprint(dict(grupos))  # {\'A\': [1, 3], \'B\': [2, 4]}',
        'd1 = {"a": 1, "b": 2}\nd2 = {"b": 3, "c": 4}\nmerged = {**d1, **d2}\nprint(merged)  # {\'a\': 1, \'b\': 3, \'c\': 4}',
        'palabras = ["hola", "mundo", "hola", "python", "mundo", "hola"]\nfrecuencia = {}\nfor p in palabras:\n    frecuencia[p] = frecuencia.get(p, 0) + 1\nprint(frecuencia)',
        'from typing import Dict, Set\n\ndef agrupar_por_longitud(palabras: list[str]) -> Dict[int, Set[str]]:\n    resultado: Dict[int, Set[str]] = {}\n    for p in palabras:\n        resultado.setdefault(len(p), set()).add(p)\n    return resultado\n\nprint(agrupar_por_longitud(["hi", "hola", "mundo", "ok"]))',
        'cache: dict = {}\n\ndef fibonacci(n: int) -> int:\n    if n in cache:\n        return cache[n]\n    if n <= 1:\n        return n\n    cache[n] = fibonacci(n-1) + fibonacci(n-2)\n    return cache[n]\n\nprint(fibonacci(10))  # 55',
        '# Eliminar duplicados preservando orden\nvistos: set = set()\nunicos = []\nfor x in [1, 3, 2, 1, 4, 3, 5]:\n    if x not in vistos:\n        vistos.add(x)\n        unicos.append(x)\nprint(unicos)  # [1, 3, 2, 4, 5]',
        'config = {\n    "host": "localhost",\n    "port": 8080,\n    "debug": True,\n    "db": {"name": "app", "pool": 5}\n}\nprint(config["db"]["name"])  # app\nconfig.update({"port": 9000, "debug": False})\nprint(config["port"])  # 9000',
    ]
    return [jsonl_entry(e, "diccionarios_y_sets") for e in shuffle_and_sample(ejemplos, n)]


def generar_funciones_basicas(n: int) -> list[str]:
    ejemplos = [
        'def saludar(nombre: str) -> str:\n    return f"Hola, {nombre}!"\n\nprint(saludar("Lucas"))  # Hola, Lucas!',
        'def sumar(a: float, b: float) -> float:\n    """Retorna la suma de a y b."""\n    return a + b\n\nprint(sumar(3, 4.5))  # 7.5',
        'def potencia(base: float, exponente: int = 2) -> float:\n    """Eleva base al exponente (default cuadrado)."""\n    return base ** exponente\n\nprint(potencia(3))     # 9\nprint(potencia(2, 10)) # 1024',
        'def dividir(a: float, b: float) -> tuple[float, int]:\n    """Retorna cociente y resto."""\n    return a // b, a % b\n\ncoc, res = dividir(17, 5)\nprint(f"Cociente: {coc}, Resto: {res}")',
        'def es_primo(n: int) -> bool:\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n\nprimos = [n for n in range(2, 30) if es_primo(n)]\nprint(primos)',
        'def maximo(a, b, c):\n    """Retorna el mayor de tres valores."""\n    return max(a, b, c)\n\nprint(maximo(3, 7, 5))   # 7',
        'def factorial(n: int) -> int:\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n\nprint(factorial(6))  # 720',
        'def celsius_a_fahrenheit(celsius: float) -> float:\n    return celsius * 9/5 + 32\n\ndef fahrenheit_a_celsius(f: float) -> float:\n    return (f - 32) * 5/9\n\nprint(celsius_a_fahrenheit(100))  # 212.0\nprint(fahrenheit_a_celsius(32))   # 0.0',
        'def contar_vocales(texto: str) -> int:\n    return sum(1 for c in texto.lower() if c in "aeiou")\n\nprint(contar_vocales("Hola Mundo"))  # 4',
        'def es_palindromo(s: str) -> bool:\n    limpio = "".join(c.lower() for c in s if c.isalnum())\n    return limpio == limpio[::-1]\n\nprint(es_palindromo("A man a plan a canal Panama"))  # True',
        'def promedio(*numeros: float) -> float:\n    if not numeros:\n        return 0.0\n    return sum(numeros) / len(numeros)\n\nprint(promedio(1, 2, 3, 4, 5))  # 3.0',
        'def invertir_string(s: str) -> str:\n    return s[::-1]\n\nprint(invertir_string("python"))  # nohtyp',
    ]
    return [jsonl_entry(e, "funciones_basicas") for e in shuffle_and_sample(ejemplos, n)]


def generar_funciones_avanzadas(n: int) -> list[str]:
    ejemplos = [
        'def args_ejemplo(*args, **kwargs):\n    print("args:", args)\n    print("kwargs:", kwargs)\n\nargs_ejemplo(1, 2, 3, nombre="Ana", edad=30)',
        'cuadrado = lambda x: x ** 2\ndoble = lambda x: x * 2\nnums = [1, 2, 3, 4, 5]\nprint(list(map(cuadrado, nums)))    # [1, 4, 9, 16, 25]\nprint(list(filter(lambda x: x > 2, nums)))  # [3, 4, 5]',
        'def hacer_operacion(operacion):\n    """Función de orden superior."""\n    def aplicar(a, b):\n        return operacion(a, b)\n    return aplicar\n\nsumar = hacer_operacion(lambda a, b: a + b)\nprint(sumar(3, 4))  # 7',
        'def contador():\n    """Closure: función que captura su entorno."""\n    n = 0\n    def incrementar():\n        nonlocal n\n        n += 1\n        return n\n    return incrementar\n\nc = contador()\nprint(c())  # 1\nprint(c())  # 2\nprint(c())  # 3',
        'from functools import reduce\n\nnums = [1, 2, 3, 4, 5]\nproducto = reduce(lambda a, b: a * b, nums)\nprint(producto)  # 120',
        'from typing import Callable, TypeVar\nT = TypeVar("T")\n\ndef aplicar_dos_veces(f: Callable[[T], T], x: T) -> T:\n    return f(f(x))\n\nprint(aplicar_dos_veces(lambda x: x + 3, 7))  # 13',
        'def registrar(func):\n    """Registra cuántas veces se llama una función."""\n    registrar.llamadas = 0\n    def wrapper(*args, **kwargs):\n        registrar.llamadas += 1\n        return func(*args, **kwargs)\n    return wrapper\n\n@registrar\ndef saludar(nombre):\n    return f"Hola {nombre}"\n\nsaludar("Ana")\nsaludar("Bob")\nprint(registrar.llamadas)  # 2',
        'from functools import partial\n\ndef multiplicar(a, b):\n    return a * b\n\ndoble = partial(multiplicar, 2)\ntriple = partial(multiplicar, 3)\nprint(doble(5))   # 10\nprint(triple(5))  # 15',
        'def memoize(func):\n    cache = {}\n    def wrapper(*args):\n        if args not in cache:\n            cache[args] = func(*args)\n        return cache[args]\n    return wrapper\n\n@memoize\ndef fib(n):\n    if n < 2: return n\n    return fib(n-1) + fib(n-2)\n\nprint(fib(35))  # 9227465',
        'from typing import Generator\n\ndef numeros_infinitos(inicio: int = 0) -> Generator[int, None, None]:\n    n = inicio\n    while True:\n        yield n\n        n += 1\n\ngen = numeros_infinitos(10)\nprint([next(gen) for _ in range(5)])  # [10, 11, 12, 13, 14]',
        'palabras = ["banana", "kiwi", "manzana", "pera"]\northograficas = sorted(palabras)\npor_len = sorted(palabras, key=len)\npor_len_desc = sorted(palabras, key=len, reverse=True)\nprint(orthograficas)\nprint(por_len)',
    ]
    return [jsonl_entry(e, "funciones_avanzadas") for e in shuffle_and_sample(ejemplos, n)]


def generar_clases_y_oop(n: int) -> list[str]:
    ejemplos = [
        'class Perro:\n    def __init__(self, nombre: str, raza: str):\n        self.nombre = nombre\n        self.raza = raza\n\n    def ladrar(self) -> str:\n        return f"{self.nombre} dice: ¡Guau!"\n\n    def __repr__(self) -> str:\n        return f"Perro({self.nombre!r}, {self.raza!r})"\n\nfido = Perro("Fido", "Labrador")\nprint(fido.ladrar())\nprint(fido)',
        'class Circulo:\n    PI = 3.14159\n\n    def __init__(self, radio: float):\n        self.radio = radio\n\n    @property\n    def area(self) -> float:\n        return self.PI * self.radio ** 2\n\n    @property\n    def perimetro(self) -> float:\n        return 2 * self.PI * self.radio\n\nc = Circulo(5)\nprint(f"Área: {c.area:.2f}")\nprint(f"Perímetro: {c.perimetro:.2f}")',
        'class Contador:\n    _instancias = 0\n\n    def __init__(self):\n        Contador._instancias += 1\n        self.id = Contador._instancias\n\n    @classmethod\n    def total(cls) -> int:\n        return cls._instancias\n\n    @staticmethod\n    def descripcion() -> str:\n        return "Clase Contador"\n\na = Contador()\nb = Contador()\nprint(Contador.total())       # 2\nprint(Contador.descripcion()) # Clase Contador',
        'class Pila:\n    def __init__(self):\n        self._items: list = []\n\n    def push(self, item) -> None:\n        self._items.append(item)\n\n    def pop(self):\n        if not self._items:\n            raise IndexError("Pila vacía")\n        return self._items.pop()\n\n    def peek(self):\n        if not self._items:\n            raise IndexError("Pila vacía")\n        return self._items[-1]\n\n    def __len__(self) -> int:\n        return len(self._items)\n\n    def __bool__(self) -> bool:\n        return bool(self._items)\n\np = Pila()\np.push(1)\np.push(2)\nprint(p.pop())   # 2\nprint(len(p))    # 1',
        'class Temperatura:\n    def __init__(self, celsius: float):\n        self._celsius = celsius\n\n    @property\n    def celsius(self) -> float:\n        return self._celsius\n\n    @celsius.setter\n    def celsius(self, valor: float) -> None:\n        if valor < -273.15:\n            raise ValueError("Por debajo del cero absoluto")\n        self._celsius = valor\n\n    @property\n    def fahrenheit(self) -> float:\n        return self._celsius * 9/5 + 32\n\nt = Temperatura(100)\nprint(t.fahrenheit)  # 212.0\nt.celsius = 0\nprint(t.fahrenheit)  # 32.0',
        'class Punto:\n    def __init__(self, x: float, y: float):\n        self.x = x\n        self.y = y\n\n    def __add__(self, otro: "Punto") -> "Punto":\n        return Punto(self.x + otro.x, self.y + otro.y)\n\n    def __eq__(self, otro: object) -> bool:\n        if not isinstance(otro, Punto):\n            return NotImplemented\n        return self.x == otro.x and self.y == otro.y\n\n    def __repr__(self) -> str:\n        return f"Punto({self.x}, {self.y})"\n\np1 = Punto(1, 2)\np2 = Punto(3, 4)\nprint(p1 + p2)  # Punto(4, 6)',
        'class Vehiculo:\n    def __init__(self, marca: str, velocidad_max: float):\n        self.marca = marca\n        self.velocidad_max = velocidad_max\n        self._velocidad = 0.0\n\n    def acelerar(self, delta: float) -> None:\n        self._velocidad = min(self._velocidad + delta, self.velocidad_max)\n\n    def frenar(self, delta: float) -> None:\n        self._velocidad = max(self._velocidad - delta, 0)\n\n    @property\n    def velocidad(self) -> float:\n        return self._velocidad\n\n    def __str__(self) -> str:\n        return f"{self.marca} a {self._velocidad:.1f} km/h"\n\nauto = Vehiculo("Toyota", 200)\nauto.acelerar(80)\nauto.acelerar(80)\nprint(auto)        # Toyota a 160.0 km/h',
        'from dataclasses import dataclass, field\nfrom typing import List\n\n@dataclass\nclass Producto:\n    nombre: str\n    precio: float\n    tags: List[str] = field(default_factory=list)\n\n    @property\n    def precio_con_iva(self) -> float:\n        return self.precio * 1.21\n\np = Producto("Laptop", 999.99, ["electrónica", "computación"])\nprint(p.precio_con_iva)\nprint(p)',
    ]
    return [jsonl_entry(e, "clases_y_oop") for e in shuffle_and_sample(ejemplos, n)]


def generar_manejo_de_errores(n: int) -> list[str]:
    ejemplos = [
        'try:\n    resultado = 10 / 0\nexcept ZeroDivisionError as e:\n    print(f"Error: {e}")\nfinally:\n    print("Bloque finally siempre se ejecuta")',
        'class ErrorDominio(Exception):\n    """Error personalizado para reglas de negocio."""\n    def __init__(self, mensaje: str, codigo: int = 0):\n        super().__init__(mensaje)\n        self.codigo = codigo\n\ntry:\n    raise ErrorDominio("Saldo insuficiente", codigo=402)\nexcept ErrorDominio as e:\n    print(f"[{e.codigo}] {e}")',
        'def leer_entero(s: str) -> int:\n    try:\n        return int(s)\n    except ValueError:\n        raise ValueError(f"No se puede convertir \'{s}\' a entero") from None\n\ntry:\n    print(leer_entero("abc"))\nexcept ValueError as e:\n    print(e)',
        'import json\nfrom pathlib import Path\n\ndef leer_config(ruta: str) -> dict:\n    try:\n        return json.loads(Path(ruta).read_text())\n    except FileNotFoundError:\n        return {}\n    except json.JSONDecodeError as e:\n        raise ValueError(f"Config inválida: {e}") from e',
        'def dividir(a: float, b: float) -> float:\n    if b == 0:\n        raise ZeroDivisionError("El divisor no puede ser cero")\n    return a / b\n\ntry:\n    print(dividir(10, 2))  # 5.0\n    print(dividir(10, 0))  # ZeroDivisionError\nexcept ZeroDivisionError as e:\n    print(f"Error: {e}")',
        'class ValidacionError(ValueError):\n    pass\n\ndef validar_edad(edad: int) -> None:\n    if not isinstance(edad, int):\n        raise TypeError(f"Se esperaba int, se recibió {type(edad).__name__}")\n    if edad < 0 or edad > 150:\n        raise ValidacionError(f"Edad inválida: {edad}")\n\ntry:\n    validar_edad(-5)\nexcept ValidacionError as e:\n    print(e)',
        '# Múltiples excepciones\ntry:\n    datos = [1, 2, 3]\n    print(datos[10])\nexcept (IndexError, KeyError) as e:\n    print(f"Error de acceso: {e}")\nexcept Exception as e:\n    print(f"Error inesperado: {e}")',
        'from contextlib import suppress\n\n# Ignorar excepciones específicas de forma limpia\nwith suppress(FileNotFoundError):\n    import os\n    os.remove("archivo_que_no_existe.txt")\nprint("Continuando sin error")',
        'def procesar_lista(items: list) -> list:\n    """Procesa items, registrando errores sin abortar."""\n    resultados = []\n    errores = []\n    for item in items:\n        try:\n            resultados.append(int(item) * 2)\n        except (ValueError, TypeError) as e:\n            errores.append((item, str(e)))\n    if errores:\n        print(f"Advertencias: {errores}")\n    return resultados\n\nprint(procesar_lista(["1", "2", "abc", "4"]))  # [2, 4, 8]',
    ]
    return [jsonl_entry(e, "manejo_de_errores") for e in shuffle_and_sample(ejemplos, n)]


def generar_comprensiones(n: int) -> list[str]:
    ejemplos = [
        '# List comprehension básica\ncuadrados = [x**2 for x in range(10)]\nprint(cuadrados)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]',
        '# Con condición\nimpares = [x for x in range(20) if x % 2 != 0]\nprint(impares)  # [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]',
        '# Dict comprehension\ninverso = {v: k for k, v in {"a": 1, "b": 2, "c": 3}.items()}\nprint(inverso)  # {1: \'a\', 2: \'b\', 3: \'c\'}',
        '# Set comprehension\nletras = {c.lower() for c in "Hola Mundo" if c.isalpha()}\nprint(sorted(letras))  # [\'a\', \'d\', \'h\', \'l\', \'m\', \'n\', \'o\', \'u\']',
        '# Generator expression (lazy)\ntotal = sum(x**2 for x in range(1000))  # no crea lista en memoria\nprint(total)',
        '# Comprensión anidada\nmatriz = [[i * j for j in range(1, 4)] for i in range(1, 4)]\nfor fila in matriz:\n    print(fila)\n# [1, 2, 3]\n# [2, 4, 6]\n# [3, 6, 9]',
        '# Aplanar matriz con comprensión\nmatriz = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]\nplana = [x for fila in matriz for x in fila]\nprint(plana)  # [1, 2, 3, 4, 5, 6, 7, 8, 9]',
        '# Comprensión con ternario\netiquetas = ["par" if x % 2 == 0 else "impar" for x in range(6)]\nprint(etiquetas)  # [\'par\', \'impar\', \'par\', \'impar\', \'par\', \'impar\']',
        'datos = [{"nombre": "Ana", "edad": 25}, {"nombre": "Bob", "edad": 17}, {"nombre": "Carlos", "edad": 30}]\nmayor_de_edad = [p["nombre"] for p in datos if p["edad"] >= 18]\nprint(mayor_de_edad)  # [\'Ana\', \'Carlos\']',
        'palabras = ["hola", "mundo", "python", "es", "genial"]\nfiltradas = [p.upper() for p in palabras if len(p) > 4]\nprint(filtradas)  # [\'MUNDO\', \'PYTHON\', \'GENIAL\']',
        '# Transponer matrix con zip + comprensión\nmatriz = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]\ntranspuesta = [list(fila) for fila in zip(*matriz)]\nprint(transpuesta)',
        'numeros = range(1, 101)\npares_cuadrados = {n: n**2 for n in numeros if n % 2 == 0}\nmaximo = max(pares_cuadrados.values())\nprint(f"Máximo cuadrado par hasta 100: {maximo}")  # 10000',
    ]
    return [jsonl_entry(e, "comprensiones") for e in shuffle_and_sample(ejemplos, n)]


def generar_generadores_tema(n: int) -> list[str]:
    ejemplos = [
        'def contar_hasta(limite: int):\n    """Generator que cuenta de 1 a limite."""\n    n = 1\n    while n <= limite:\n        yield n\n        n += 1\n\nfor num in contar_hasta(5):\n    print(num, end=" ")  # 1 2 3 4 5',
        'def fibonacci():\n    """Generator infinito de Fibonacci."""\n    a, b = 0, 1\n    while True:\n        yield a\n        a, b = b, a + b\n\nfib = fibonacci()\nprint([next(fib) for _ in range(8)])  # [0, 1, 1, 2, 3, 5, 8, 13]',
        'from typing import Generator, Iterator\n\ndef leer_chunks(texto: str, n: int) -> Generator[str, None, None]:\n    """Divide un texto en chunks de n caracteres."""\n    for i in range(0, len(texto), n):\n        yield texto[i:i+n]\n\nfor chunk in leer_chunks("abcdefghij", 3):\n    print(chunk)  # abc, def, ghi, j',
        '# Encadenamiento de generators\ndef cuadrados(n):\n    yield from (x**2 for x in range(n))\n\ndef filtrar_pares(gen):\n    for x in gen:\n        if x % 2 == 0:\n            yield x\n\nresultado = list(filtrar_pares(cuadrados(10)))\nprint(resultado)  # [0, 4, 16, 36, 64]',
        'import itertools\n\n# islice para tomar N elementos de un generator infinito\ndef naturales():\n    n = 0\n    while True:\n        yield n\n        n += 1\n\nprimeros_10 = list(itertools.islice(naturales(), 10))\nprint(primeros_10)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]',
        '# Generator con send() — coroutine básica\ndef acumulador():\n    total = 0\n    while True:\n        valor = yield total\n        if valor is None:\n            break\n        total += valor\n\ngen = acumulador()\nnext(gen)       # inicializar\ngen.send(10)    # 10\ngen.send(20)    # 30\nprint(gen.send(5))  # 35',
        'def paginar(items: list, pagina: int) -> Iterator:\n    """Generator que paginaea una lista."""\n    for i in range(0, len(items), pagina):\n        yield items[i:i+pagina]\n\nfor pagina in paginar(list(range(10)), 3):\n    print(pagina)',
        'def tee_generator(gen, n=2):\n    import itertools\n    return itertools.tee(gen, n)\n\ng1, g2 = tee_generator(x**2 for x in range(5))\nprint(list(g1))  # [0, 1, 4, 9, 16]\nprint(list(g2))  # [0, 1, 4, 9, 16]',
        '# yield from para delegar en sub-generator\ndef cadena(*iterables):\n    for it in iterables:\n        yield from it\n\nprimeros = cadena([1,2,3], "abc", range(5))\nresult = list(primeros)\nprint(result)',
    ]
    return [jsonl_entry(e, "generadores") for e in shuffle_and_sample(ejemplos, n)]


def generar_decoradores_tema(n: int) -> list[str]:
    ejemplos = [
        'from functools import wraps\nimport time\n\ndef medir_tiempo(func):\n    @wraps(func)\n    def wrapper(*args, **kwargs):\n        inicio = time.perf_counter()\n        resultado = func(*args, **kwargs)\n        fin = time.perf_counter()\n        print(f"{func.__name__} tomó {fin-inicio:.4f}s")\n        return resultado\n    return wrapper\n\n@medir_tiempo\ndef operacion_lenta():\n    return sum(range(1_000_000))\n\noperacion_lenta()',
        'from functools import wraps\n\ndef solo_enteros(func):\n    """Valida que todos los argumentos sean enteros."""\n    @wraps(func)\n    def wrapper(*args, **kwargs):\n        for a in args:\n            if not isinstance(a, int):\n                raise TypeError(f"Se esperaba int, se recibió {type(a).__name__}")\n        return func(*args, **kwargs)\n    return wrapper\n\n@solo_enteros\ndef sumar(a, b):\n    return a + b\n\nprint(sumar(3, 4))    # 7\nprint(sumar(3, 4.5))  # TypeError',
        'def reintentar(intentos: int = 3):\n    """Decorador factory: reintenta la función si falla."""\n    def decorador(func):\n        from functools import wraps\n        @wraps(func)\n        def wrapper(*args, **kwargs):\n            for i in range(intentos):\n                try:\n                    return func(*args, **kwargs)\n                except Exception as e:\n                    if i == intentos - 1:\n                        raise\n                    print(f"Intento {i+1} fallido: {e}")\n        return wrapper\n    return decorador\n\n@reintentar(intentos=3)\ndef operacion_inestable():\n    import random\n    if random.random() < 0.7:\n        raise ValueError("Fallo aleatorio")\n    return "OK"',
        'class Singleton:\n    """Decorador de clase: garantiza una sola instancia."""\n    def __init__(self, clase):\n        self._clase = clase\n        self._instancia = None\n\n    def __call__(self, *args, **kwargs):\n        if self._instancia is None:\n            self._instancia = self._clase(*args, **kwargs)\n        return self._instancia\n\n@Singleton\nclass Config:\n    def __init__(self):\n        self.datos = {}\n\na = Config()\nb = Config()\nprint(a is b)  # True',
        'from functools import wraps, lru_cache\n\n@lru_cache(maxsize=128)\ndef fibonacci(n: int) -> int:\n    """Fibonacci con memoización automática."""\n    if n < 2:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\nprint(fibonacci(40))  # 102334155\nprint(fibonacci.cache_info())',
        'def registrar(nivel="INFO"):\n    from functools import wraps\n    def decorador(func):\n        @wraps(func)\n        def wrapper(*args, **kwargs):\n            print(f"[{nivel}] Llamando {func.__name__}")\n            resultado = func(*args, **kwargs)\n            print(f"[{nivel}] {func.__name__} retornó {resultado}")\n            return resultado\n        return wrapper\n    return decorador\n\n@registrar("DEBUG")\ndef multiplicar(a, b):\n    return a * b\n\nmultiplicar(3, 4)',
        'from functools import wraps\n\ndef deprecado(mensaje: str):\n    """Marca una función como obsoleta."""\n    import warnings\n    def decorador(func):\n        @wraps(func)\n        def wrapper(*args, **kwargs):\n            warnings.warn(\n                f"{func.__name__} está obsoleto: {mensaje}",\n                DeprecationWarning, stacklevel=2\n            )\n            return func(*args, **kwargs)\n        return wrapper\n    return decorador\n\n@deprecado("usa nueva_funcion() en su lugar")\ndef funcion_vieja():\n    pass',
    ]
    return [jsonl_entry(e, "decoradores") for e in shuffle_and_sample(ejemplos, n)]


def generar_context_managers(n: int) -> list[str]:
    ejemplos = [
        '# Context manager con clase\nclass TemporizadorCM:\n    import time\n    def __enter__(self):\n        import time\n        self._inicio = time.perf_counter()\n        return self\n\n    def __exit__(self, exc_type, exc_val, exc_tb):\n        import time\n        self.elapsed = time.perf_counter() - self._inicio\n        print(f"Tiempo: {self.elapsed:.4f}s")\n        return False  # no suprimir excepciones\n\nwith TemporizadorCM() as t:\n    resultado = sum(range(100_000))\nprint(f"Resultado: {resultado}")',
        'from contextlib import contextmanager\n\n@contextmanager\ndef abrir_temporal(ruta: str):\n    """Crea un archivo temporal y lo elimina al finalizar."""\n    from pathlib import Path\n    path = Path(ruta)\n    try:\n        path.write_text("")\n        yield path\n    finally:\n        if path.exists():\n            path.unlink()\n\nwith abrir_temporal("temp.txt") as f:\n    f.write_text("datos temporales")\n    print(f.read_text())',
        'from contextlib import contextmanager\n\n@contextmanager\ndef transaccion(db):\n    """Simula una transacción de base de datos."""\n    try:\n        yield db\n        db.commit()\n        print("Commit exitoso")\n    except Exception as e:\n        db.rollback()\n        print(f"Rollback por: {e}")\n        raise',
        '# Múltiples context managers en línea\nimport tempfile\nimport os\n\nwith tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f1, \\\n     tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f2:\n    f1.write("data")\n    f2.write("col1,col2")\nprint("Ambos archivos creados y cerrados")',
        'from contextlib import suppress\n\nwith suppress(FileNotFoundError, PermissionError):\n    import os\n    os.remove("archivo_inexistente.log")\nprint("Sin error")',
        'class ConexionBD:\n    def __init__(self, url: str):\n        self.url = url\n        self.conexion = None\n\n    def __enter__(self):\n        print(f"Conectando a {self.url}")\n        self.conexion = {"url": self.url, "activa": True}\n        return self.conexion\n\n    def __exit__(self, *args):\n        if self.conexion:\n            self.conexion["activa"] = False\n            print("Conexión cerrada")\n        return False\n\nwith ConexionBD("sqlite:///app.db") as conn:\n    print(f"Conexión activa: {conn[\'activa\']}")',
        'from contextlib import contextmanager\n\n@contextmanager\ndef directorio_temporal():\n    import tempfile, os, shutil\n    tmp = tempfile.mkdtemp()\n    try:\n        yield tmp\n    finally:\n        shutil.rmtree(tmp, ignore_errors=True)\n\nwith directorio_temporal() as d:\n    ruta = os.path.join(d, "archivo.txt")\n    with open(ruta, "w") as f:\n        f.write("temporal")\n    print(os.listdir(d))',
    ]
    return [jsonl_entry(e, "context_managers") for e in shuffle_and_sample(ejemplos, n)]


def generar_recursion_tema(n: int) -> list[str]:
    ejemplos = [
        'def factorial(n: int) -> int:\n    """Factorial recursivo."""\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n\nprint(factorial(10))  # 3628800',
        'def fibonacci(n: int) -> int:\n    """Fibonacci recursivo con memoización."""\n    from functools import lru_cache\n    @lru_cache(maxsize=None)\n    def _fib(k):\n        if k < 2: return k\n        return _fib(k-1) + _fib(k-2)\n    return _fib(n)\n\nprint(fibonacci(30))  # 832040',
        'def suma_lista(lst: list) -> int:\n    """Suma recursiva de una lista."""\n    if not lst:\n        return 0\n    return lst[0] + suma_lista(lst[1:])\n\nprint(suma_lista([1, 2, 3, 4, 5]))  # 15',
        'def busqueda_binaria_rec(arr: list, objetivo, izq: int = 0, der: int = None) -> int:\n    if der is None:\n        der = len(arr) - 1\n    if izq > der:\n        return -1\n    mid = (izq + der) // 2\n    if arr[mid] == objetivo:\n        return mid\n    if arr[mid] < objetivo:\n        return busqueda_binaria_rec(arr, objetivo, mid + 1, der)\n    return busqueda_binaria_rec(arr, objetivo, izq, mid - 1)\n\nprint(busqueda_binaria_rec([1,3,5,7,9,11], 7))  # 3',
        'def potencia(base: float, exp: int) -> float:\n    """Potencia recursiva eficiente (O(log n))."""\n    if exp == 0:\n        return 1\n    if exp % 2 == 0:\n        mitad = potencia(base, exp // 2)\n        return mitad * mitad\n    return base * potencia(base, exp - 1)\n\nprint(potencia(2, 10))  # 1024',
        'def aplanar(lst: list) -> list:\n    """Aplana recursivamente una lista anidada."""\n    resultado = []\n    for item in lst:\n        if isinstance(item, list):\n            resultado.extend(aplanar(item))\n        else:\n            resultado.append(item)\n    return resultado\n\nprint(aplanar([1, [2, [3, 4], 5], [6, 7]]))  # [1, 2, 3, 4, 5, 6, 7]',
        'def hanoi(n: int, origen: str, destino: str, auxiliar: str) -> None:\n    """Torres de Hanói."""\n    if n == 1:\n        print(f"Mover disco 1 de {origen} a {destino}")\n        return\n    hanoi(n-1, origen, auxiliar, destino)\n    print(f"Mover disco {n} de {origen} a {destino}")\n    hanoi(n-1, auxiliar, destino, origen)\n\nhanoi(3, "A", "C", "B")',
        'def mcd(a: int, b: int) -> int:\n    """Máximo común divisor por algoritmo de Euclides."""\n    if b == 0:\n        return a\n    return mcd(b, a % b)\n\nprint(mcd(48, 18))  # 6',
        'def permutar(elements: list) -> list[list]:\n    """Genera todas las permutaciones de una lista."""\n    if len(elements) <= 1:\n        return [elements[:]]\n    resultado = []\n    for i, elem in enumerate(elements):\n        resto = elements[:i] + elements[i+1:]\n        for perm in permutar(resto):\n            resultado.append([elem] + perm)\n    return resultado\n\nperms = permutar([1, 2, 3])\nprint(len(perms))  # 6',
    ]
    return [jsonl_entry(e, "recursion") for e in shuffle_and_sample(ejemplos, n)]


def generar_busqueda_binaria(n: int) -> list[str]:
    ejemplos = [
        'def busqueda_binaria(arr: list, objetivo) -> int:\n    """Retorna el índice del objetivo, o -1 si no existe."""\n    izq, der = 0, len(arr) - 1\n    while izq <= der:\n        mid = (izq + der) // 2\n        if arr[mid] == objetivo:\n            return mid\n        if arr[mid] < objetivo:\n            izq = mid + 1\n        else:\n            der = mid - 1\n    return -1\n\narr = [1, 3, 5, 7, 9, 11, 13]\nprint(busqueda_binaria(arr, 7))   # 3\nprint(busqueda_binaria(arr, 4))   # -1',
        'import bisect\n\narr = [1, 3, 5, 7, 9]\nbisect.insort(arr, 6)\nprint(arr)  # [1, 3, 5, 6, 7, 9]\nprint(bisect.bisect_left(arr, 7))   # 4\nprint(bisect.bisect_right(arr, 7))  # 5',
        'def primer_mayor_o_igual(arr: list, objetivo) -> int:\n    """Lower bound: primer índice donde arr[i] >= objetivo."""\n    izq, der = 0, len(arr)\n    while izq < der:\n        mid = (izq + der) // 2\n        if arr[mid] < objetivo:\n            izq = mid + 1\n        else:\n            der = mid\n    return izq\n\narr = [1, 2, 4, 4, 5, 8, 10]\nprint(primer_mayor_o_igual(arr, 4))  # 2 (primer 4)\nprint(primer_mayor_o_igual(arr, 6))  # 5 (primer >= 6 es 8)',
        'def rotated_search(arr: list, objetivo: int) -> int:\n    """Búsqueda binaria en array rotado."""\n    izq, der = 0, len(arr) - 1\n    while izq <= der:\n        mid = (izq + der) // 2\n        if arr[mid] == objetivo:\n            return mid\n        if arr[izq] <= arr[mid]:  # izquierda ordenada\n            if arr[izq] <= objetivo < arr[mid]:\n                der = mid - 1\n            else:\n                izq = mid + 1\n        else:  # derecha ordenada\n            if arr[mid] < objetivo <= arr[der]:\n                izq = mid + 1\n            else:\n                der = mid - 1\n    return -1\n\nprint(rotated_search([4,5,6,7,0,1,2], 0))  # 4',
        'def raiz_cuadrada_entera(n: int) -> int:\n    """Raíz cuadrada entera usando búsqueda binaria."""\n    if n < 0:\n        raise ValueError("n debe ser no negativo")\n    izq, der = 0, n\n    while izq <= der:\n        mid = (izq + der) // 2\n        if mid * mid == n:\n            return mid\n        if mid * mid < n:\n            izq = mid + 1\n            ultimo = mid\n        else:\n            der = mid - 1\n    return ultimo\n\nprint(raiz_cuadrada_entera(8))   # 2\nprint(raiz_cuadrada_entera(25))  # 5',
        'def encontrar_pico(arr: list) -> int:\n    """Encontrar cualquier elemento pico (mayor a sus vecinos)."""\n    izq, der = 0, len(arr) - 1\n    while izq < der:\n        mid = (izq + der) // 2\n        if arr[mid] > arr[mid + 1]:\n            der = mid\n        else:\n            izq = mid + 1\n    return izq\n\narr = [1, 3, 5, 4, 2]\nidx = encontrar_pico(arr)\nprint(f"Pico en índice {idx}: {arr[idx]}")  # índice 2: 5',
        'def contar_ocurrencias(arr: list, objetivo) -> int:\n    """Cuenta cuántas veces aparece objetivo en arr ordenado."""\n    import bisect\n    izq = bisect.bisect_left(arr, objetivo)\n    der = bisect.bisect_right(arr, objetivo)\n    return der - izq\n\narr = [1, 2, 2, 2, 3, 4, 4]\nprint(contar_ocurrencias(arr, 2))  # 3\nprint(contar_ocurrencias(arr, 4))  # 2',
    ]
    return [jsonl_entry(e, "busqueda_binaria") for e in shuffle_and_sample(ejemplos, n)]


def generar_sorting_clasico(n: int) -> list[str]:
    ejemplos = [
        'def bubble_sort(arr: list) -> list:\n    """Ordenamiento de burbuja O(n²)."""\n    arr = arr[:]\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n - i - 1):\n            if arr[j] > arr[j + 1]:\n                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n    return arr\n\nprint(bubble_sort([64, 34, 25, 12, 22, 11, 90]))',
        'def merge_sort(arr: list) -> list:\n    """Merge sort O(n log n)."""\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    izq = merge_sort(arr[:mid])\n    der = merge_sort(arr[mid:])\n    return merge(izq, der)\n\ndef merge(izq: list, der: list) -> list:\n    resultado = []\n    i = j = 0\n    while i < len(izq) and j < len(der):\n        if izq[i] <= der[j]:\n            resultado.append(izq[i]); i += 1\n        else:\n            resultado.append(der[j]); j += 1\n    resultado.extend(izq[i:])\n    resultado.extend(der[j:])\n    return resultado\n\nprint(merge_sort([38, 27, 43, 3, 9, 82, 10]))',
        'def quick_sort(arr: list) -> list:\n    """Quick sort O(n log n) promedio."""\n    if len(arr) <= 1:\n        return arr\n    pivote = arr[len(arr) // 2]\n    menores = [x for x in arr if x < pivote]\n    iguales = [x for x in arr if x == pivote]\n    mayores = [x for x in arr if x > pivote]\n    return quick_sort(menores) + iguales + quick_sort(mayores)\n\nprint(quick_sort([3, 6, 8, 10, 1, 2, 1]))',
        'def insertion_sort(arr: list) -> list:\n    """Ordenamiento por inserción O(n²), bueno para casi-ordenados."""\n    arr = arr[:]\n    for i in range(1, len(arr)):\n        clave = arr[i]\n        j = i - 1\n        while j >= 0 and arr[j] > clave:\n            arr[j + 1] = arr[j]\n            j -= 1\n        arr[j + 1] = clave\n    return arr\n\nprint(insertion_sort([12, 11, 13, 5, 6]))',
        'def selection_sort(arr: list) -> list:\n    """Ordenamiento por selección O(n²)."""\n    arr = arr[:]\n    n = len(arr)\n    for i in range(n):\n        min_idx = i\n        for j in range(i + 1, n):\n            if arr[j] < arr[min_idx]:\n                min_idx = j\n        arr[i], arr[min_idx] = arr[min_idx], arr[i]\n    return arr\n\nprint(selection_sort([64, 25, 12, 22, 11]))',
        '# Python built-in sort (Timsort) — el más eficiente en práctica\nimport random\nnums = random.sample(range(1000), 10)\nprint("Original:", nums)\nnums.sort()\nprint("Ordenado:", nums)\n\n# Sort estable con key\npersonas = [("Ana", 30), ("Bob", 25), ("Carlos", 30)]\npersonas.sort(key=lambda p: (p[1], p[0]))\nprint(personas)',
        'def counting_sort(arr: list, max_val: int) -> list:\n    """Counting sort O(n + k), para enteros pequeños."""\n    count = [0] * (max_val + 1)\n    for x in arr:\n        count[x] += 1\n    resultado = []\n    for val, freq in enumerate(count):\n        resultado.extend([val] * freq)\n    return resultado\n\nprint(counting_sort([4, 2, 2, 8, 3, 3, 1], 8))',
    ]
    return [jsonl_entry(e, "sorting_clasico") for e in shuffle_and_sample(ejemplos, n)]


def generar_grafos_bfs_dfs(n: int) -> list[str]:
    ejemplos = [
        'from collections import deque\n\ndef bfs(grafo: dict, inicio: str) -> list:\n    """BFS — Búsqueda en Anchura."""\n    visitados = set()\n    cola = deque([inicio])\n    orden = []\n    while cola:\n        nodo = cola.popleft()\n        if nodo not in visitados:\n            visitados.add(nodo)\n            orden.append(nodo)\n            cola.extend(grafo.get(nodo, []))\n    return orden\n\ngrafo = {"A": ["B", "C"], "B": ["D", "E"], "C": ["F"], "D": [], "E": [], "F": []}\nprint(bfs(grafo, "A"))  # [\'A\', \'B\', \'C\', \'D\', \'E\', \'F\']',
        'def dfs(grafo: dict, inicio: str, visitados: set = None) -> list:\n    """DFS — Búsqueda en Profundidad recursiva."""\n    if visitados is None:\n        visitados = set()\n    visitados.add(inicio)\n    orden = [inicio]\n    for vecino in grafo.get(inicio, []):\n        if vecino not in visitados:\n            orden.extend(dfs(grafo, vecino, visitados))\n    return orden\n\ngrafo = {"A": ["B", "C"], "B": ["D"], "C": ["E"], "D": [], "E": []}\nprint(dfs(grafo, "A"))  # [\'A\', \'B\', \'D\', \'C\', \'E\']',
        'def dfs_iterativo(grafo: dict, inicio: str) -> list:\n    """DFS iterativo con pila explícita."""\n    visitados = set()\n    pila = [inicio]\n    orden = []\n    while pila:\n        nodo = pila.pop()\n        if nodo not in visitados:\n            visitados.add(nodo)\n            orden.append(nodo)\n            pila.extend(reversed(grafo.get(nodo, [])))\n    return orden\n\ngrafo = {"A": ["B", "C"], "B": ["D", "E"], "C": [], "D": [], "E": []}\nprint(dfs_iterativo(grafo, "A"))',
        'from collections import deque\n\ndef camino_mas_corto(grafo: dict, inicio: str, fin: str) -> list | None:\n    """BFS para encontrar el camino más corto."""\n    cola = deque([(inicio, [inicio])])\n    visitados = {inicio}\n    while cola:\n        nodo, camino = cola.popleft()\n        if nodo == fin:\n            return camino\n        for vecino in grafo.get(nodo, []):\n            if vecino not in visitados:\n                visitados.add(vecino)\n                cola.append((vecino, camino + [vecino]))\n    return None\n\ngrafo = {"A": ["B", "C"], "B": ["D"], "C": ["D", "E"], "D": ["E"], "E": []}\nprint(camino_mas_corto(grafo, "A", "E"))',
        'def tiene_ciclo(grafo: dict) -> bool:\n    """Detecta ciclos en un grafo dirigido con DFS."""\n    blanco, gris, negro = 0, 1, 2\n    color = {nodo: blanco for nodo in grafo}\n\n    def visitar(v) -> bool:\n        color[v] = gris\n        for w in grafo.get(v, []):\n            if color[w] == gris:\n                return True\n            if color[w] == blanco and visitar(w):\n                return True\n        color[v] = negro\n        return False\n\n    return any(visitar(v) for v in grafo if color[v] == blanco)\n\nprint(tiene_ciclo({"A": ["B"], "B": ["C"], "C": ["A"]}))  # True\nprint(tiene_ciclo({"A": ["B"], "B": ["C"], "C": []}))      # False',
        'def componentes_conectados(grafo: dict) -> list[set]:\n    """Encuentra los componentes conectados de un grafo no dirigido."""\n    visitados = set()\n    componentes = []\n\n    def dfs(nodo, componente):\n        visitados.add(nodo)\n        componente.add(nodo)\n        for vecino in grafo.get(nodo, []):\n            if vecino not in visitados:\n                dfs(vecino, componente)\n\n    for nodo in grafo:\n        if nodo not in visitados:\n            c = set()\n            dfs(nodo, c)\n            componentes.append(c)\n    return componentes\n\ngrafo = {"A": ["B"], "B": ["A"], "C": ["D"], "D": ["C"], "E": []}\nprint(componentes_conectados(grafo))',
        'from collections import defaultdict\n\nclass Grafo:\n    def __init__(self):\n        self.adj: dict = defaultdict(list)\n\n    def agregar_arista(self, u, v, dirigido=False):\n        self.adj[u].append(v)\n        if not dirigido:\n            self.adj[v].append(u)\n\n    def vecinos(self, nodo):\n        return self.adj[nodo]\n\ng = Grafo()\ng.agregar_arista("A", "B")\ng.agregar_arista("B", "C")\ng.agregar_arista("A", "C")\nprint(dict(g.adj))',
    ]
    return [jsonl_entry(e, "grafos_bfs_dfs") for e in shuffle_and_sample(ejemplos, n)]


def generar_modulos_y_paquetes(n: int) -> list[str]:
    ejemplos = [
        'import os\nimport sys\nfrom pathlib import Path\n\n# Información del sistema\nprint(sys.version)\nprint(os.getcwd())\nprint(Path.home())',
        'from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    from collections.abc import Sequence\n\ndef procesar(items: "Sequence[int]") -> list:\n    return sorted(set(items))',
        'import importlib\n\ndef importar_dinamico(modulo: str, atributo: str):\n    """Importa un atributo de un módulo dinámicamente."""\n    mod = importlib.import_module(modulo)\n    return getattr(mod, atributo)\n\nPi = importar_dinamico("math", "pi")\nprint(Pi)  # 3.141592653589793',
        '# __init__.py de un paquete\n__version__ = "1.0.0"\n__author__ = "Lucas"\n__all__ = ["Cliente", "Config", "parsear"]\n\nfrom .cliente import Cliente\nfrom .config import Config\nfrom .utils import parsear',
        'from dataclasses import dataclass\nfrom pathlib import Path\nimport json\n\n@dataclass\nclass Config:\n    host: str = "localhost"\n    port: int = 8080\n    debug: bool = False\n\n    @classmethod\n    def desde_json(cls, ruta: Path) -> "Config":\n        datos = json.loads(ruta.read_text())\n        return cls(**datos)\n\n    def a_dict(self) -> dict:\n        from dataclasses import asdict\n        return asdict(self)',
        'import functools\nimport operator\nfrom collections import Counter, defaultdict\nfrom itertools import chain, groupby\n\n# Usar stdlib correctamente\npalabras = "el gato come el ratón y el perro duerme".split()\nfrecuencias = Counter(palabras)\nprint(frecuencias.most_common(3))',
        '# Relative imports dentro de un paquete\n# En archivo: paquete/utils/formato.py\nfrom ..config import Config\nfrom .validacion import validar\n\ndef formatear(datos: dict) -> str:\n    cfg = Config()\n    validar(datos)\n    return str(datos)',
        '# Uso de __name__ == "__main__"\ndef procesar_datos(datos: list) -> list:\n    return [x * 2 for x in datos]\n\ndef main():\n    datos = [1, 2, 3, 4, 5]\n    resultado = procesar_datos(datos)\n    print(resultado)\n\nif __name__ == "__main__":\n    main()',
        'import sys\nfrom pathlib import Path\n\n# Agregar directorio al path para imports locales\nroot = Path(__file__).parent.parent\nif str(root) not in sys.path:\n    sys.path.insert(0, str(root))',
    ]
    return [jsonl_entry(e, "modulos_y_paquetes") for e in shuffle_and_sample(ejemplos, n)]


def generar_type_hints_tema(n: int) -> list[str]:
    ejemplos = [
        'from typing import Optional, Union\n\ndef buscar(items: list[str], objetivo: str) -> Optional[int]:\n    """Retorna el índice o None si no existe."""\n    try:\n        return items.index(objetivo)\n    except ValueError:\n        return None\n\nresult = buscar(["a", "b", "c"], "b")\nprint(result)  # 1',
        'from typing import TypeVar, Generic\n\nT = TypeVar("T")\n\nclass Caja(Generic[T]):\n    def __init__(self, contenido: T) -> None:\n        self.contenido = contenido\n\n    def obtener(self) -> T:\n        return self.contenido\n\ncaja_int = Caja(42)\ncaja_str = Caja("hola")\nprint(caja_int.obtener())  # 42',
        'from typing import Callable, TypeVar\n\nT = TypeVar("T")\nU = TypeVar("U")\n\ndef mapear(f: Callable[[T], U], items: list[T]) -> list[U]:\n    return [f(x) for x in items]\n\nresult = mapear(str.upper, ["a", "b", "c"])\nprint(result)  # [\'A\', \'B\', \'C\']',
        'from dataclasses import dataclass\nfrom typing import ClassVar\n\n@dataclass\nclass Empleado:\n    nombre: str\n    salario: float\n    departamento: str\n    _contador: ClassVar[int] = 0\n\n    def __post_init__(self) -> None:\n        Empleado._contador += 1\n\n    @classmethod\n    def total(cls) -> int:\n        return cls._contador\n\nana = Empleado("Ana", 50000, "IT")\nbob = Empleado("Bob", 60000, "HR")\nprint(Empleado.total())  # 2',
        'from typing import Protocol\n\nclass Comparable(Protocol):\n    def __lt__(self, other: "Comparable") -> bool: ...\n    def __le__(self, other: "Comparable") -> bool: ...\n\ndef minimo(a: Comparable, b: Comparable) -> Comparable:\n    return a if a <= b else b\n\nprint(minimo(3, 5))      # 3\nprint(minimo("b", "a"))  # a',
        'from typing import TypedDict\n\nclass UserDict(TypedDict):\n    nombre: str\n    edad: int\n    email: str\n\ndef procesar_usuario(user: UserDict) -> str:\n    return f"{user[\'nombre\']} ({user[\'edad\']}): {user[\'email\']}"\n\nusuario: UserDict = {"nombre": "Ana", "edad": 30, "email": "ana@ejemplo.com"}\nprint(procesar_usuario(usuario))',
        'from typing import overload\n\n@overload\ndef procesar(x: int) -> str: ...\n@overload\ndef procesar(x: str) -> int: ...\n\ndef procesar(x):\n    if isinstance(x, int):\n        return str(x)\n    return int(x)\n\nprint(procesar(42))     # "42"\nprint(procesar("42"))   # 42',
        'from typing import NamedTuple\n\nclass Punto(NamedTuple):\n    x: float\n    y: float\n    z: float = 0.0\n\n    def distancia_origen(self) -> float:\n        return (self.x**2 + self.y**2 + self.z**2) ** 0.5\n\np = Punto(3, 4)\nprint(p.distancia_origen())  # 5.0\nprint(p.x, p.y, p.z)',
    ]
    return [jsonl_entry(e, "type_hints") for e in shuffle_and_sample(ejemplos, n)]


def generar_regex_tema(n: int) -> list[str]:
    ejemplos = [
        'import re\n\n# Buscar patrón\ntexto = "Mi teléfono es 555-1234 y el de Ana es 555-5678"\npatron = r"\\d{3}-\\d{4}"\ntelefono = re.search(patron, texto)\nif telefono:\n    print(f"Encontrado: {telefono.group()}")  # 555-1234\n\ntodos = re.findall(patron, texto)\nprint(f"Todos: {todos}")  # [\'555-1234\', \'555-5678\']',
        'import re\n\n# Grupos de captura\nfecha = "2026-02-27"\npatron = r"(\\d{4})-(\\d{2})-(\\d{2})"\nm = re.match(patron, fecha)\nif m:\n    año, mes, dia = m.groups()\n    print(f"Año: {año}, Mes: {mes}, Día: {dia}")',
        'import re\n\n# Validar email\ndef validar_email(email: str) -> bool:\n    patron = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"\n    return bool(re.match(patron, email))\n\nprint(validar_email("usuario@ejemplo.com"))  # True\nprint(validar_email("invalido@"))            # False\nprint(validar_email("sin_arroba"))           # False',
        'import re\n\n# Sustitución\ntexto = "El gato y el perro corren. El gato come."\nnuevo = re.sub(r"\\bEl\\b", "Un", texto)\nprint(nuevo)  # Un gato y el perro corren. Un gato come.\n\n# Con función\ndef capitalizar(m):\n    return m.group(0).upper()\n\nresultado = re.sub(r"[aeiou]", capitalizar, "hola mundo")\nprint(resultado)  # hOlA mUndO',
        'import re\n\n# Grupos nombrados\nregistro = "2026-02-27 ERROR [app.db] Connection timeout"\npatron = r"(?P<fecha>\\d{4}-\\d{2}-\\d{2}) (?P<nivel>\\w+) \\[(?P<modulo>[\\w.]+)\\] (?P<mensaje>.+)"\nm = re.match(patron, registro)\nif m:\n    print(m.group("fecha"))   # 2026-02-27\n    print(m.group("nivel"))   # ERROR\n    print(m.group("mensaje")) # Connection timeout',
        'import re\n\n# Splitting\ntexto = "uno,dos;tres|cuatro"\npiezas = re.split(r"[,;|]", texto)\nprint(piezas)  # [\'uno\', \'dos\', \'tres\', \'cuatro\']',
        'import re\n\n# Flags\ntexto = "Python ES Genial"\npatrón_ci = re.compile(r"es", re.IGNORECASE)\nprint(patrón_ci.search(texto).group())  # ES\n\n# Multiline\ntexto_ml = "primero\\nsegundo\\ntercero"\nprint(re.findall(r"^\\w+", texto_ml, re.MULTILINE))',
        'import re\n\ndef extraer_urls(texto: str) -> list[str]:\n    patron = r"https?://[^\\s<>\\"]+[^\\s<>\\\".,;]"\n    return re.findall(patron, texto)\n\ntexto = "Visita https://python.org o http://ejemplo.com/path?q=1 para más info"\nprint(extraer_urls(texto))',
    ]
    return [jsonl_entry(e, "regex") for e in shuffle_and_sample(ejemplos, n)]


def generar_api_design_rest_tema(n: int) -> list[str]:
    ejemplos = [
        'from fastapi import FastAPI, HTTPException\nfrom pydantic import BaseModel\n\napp = FastAPI()\n\nclass Item(BaseModel):\n    nombre: str\n    precio: float\n    disponible: bool = True\n\nitems_db: dict[int, Item] = {}\n\n@app.post("/items", status_code=201)\ndef crear_item(item: Item) -> dict:\n    item_id = len(items_db) + 1\n    items_db[item_id] = item\n    return {"id": item_id, **item.model_dump()}',
        'from fastapi import FastAPI, HTTPException\nfrom pydantic import BaseModel\n\napp = FastAPI()\n\n@app.get("/items/{item_id}")\ndef obtener_item(item_id: int) -> dict:\n    if item_id not in items_db:\n        raise HTTPException(status_code=404, detail="Item no encontrado")\n    return {"id": item_id, **items_db[item_id].model_dump()}',
        'from pydantic import BaseModel, validator, Field\nfrom typing import Optional\n\nclass Usuario(BaseModel):\n    nombre: str = Field(..., min_length=2, max_length=50)\n    email: str\n    edad: int = Field(..., ge=0, le=150)\n    rol: Optional[str] = "user"\n\n    @validator("email")\n    def email_valido(cls, v):\n        if "@" not in v:\n            raise ValueError("Email inválido")\n        return v.lower()\n\nu = Usuario(nombre="Ana", email="ANA@test.COM", edad=25)\nprint(u)',
        'from fastapi import APIRouter, Depends\nfrom typing import Annotated\n\nrouter = APIRouter(prefix="/usuarios", tags=["usuarios"])\n\ndef obtener_db():\n    db = {"conexion": "activa"}\n    try:\n        yield db\n    finally:\n        pass  # cerrar conexión\n\nDB = Annotated[dict, Depends(obtener_db)]\n\n@router.get("/")\ndef listar(db: DB) -> list:\n    return []\n\n@router.get("/{user_id}")\ndef obtener(user_id: int, db: DB) -> dict:\n    return {"id": user_id}',
        'from fastapi import FastAPI\nfrom fastapi.middleware.cors import CORSMiddleware\n\napp = FastAPI(title="Mi API", version="1.0.0")\n\napp.add_middleware(\n    CORSMiddleware,\n    allow_origins=["https://mifrontend.com"],\n    allow_methods=["GET", "POST", "PUT", "DELETE"],\n    allow_headers=["*"],\n)',
        'from fastapi import FastAPI, Query\nfrom typing import Optional\n\napp = FastAPI()\n\n@app.get("/buscar")\ndef buscar(\n    q: str = Query(..., min_length=1, description="Término de búsqueda"),\n    pagina: int = Query(1, ge=1),\n    limite: int = Query(10, ge=1, le=100),\n    categoria: Optional[str] = None,\n) -> dict:\n    return {\n        "query": q,\n        "pagina": pagina,\n        "limite": limite,\n        "categoria": categoria,\n    }',
        'from fastapi import FastAPI, status\nfrom fastapi.responses import JSONResponse\n\napp = FastAPI()\n\n@app.exception_handler(ValueError)\nasync def manejar_valor_error(request, exc):\n    return JSONResponse(\n        status_code=status.HTTP_400_BAD_REQUEST,\n        content={"error": str(exc), "tipo": "ValueError"},\n    )',
    ]
    return [jsonl_entry(e, "api_design_rest") for e in shuffle_and_sample(ejemplos, n)]


def generar_observer_strategy_tema(n: int) -> list[str]:
    ejemplos = [
        '# Patrón Observer\nfrom abc import ABC, abstractmethod\n\nclass Observador(ABC):\n    @abstractmethod\n    def actualizar(self, evento: str, datos: dict) -> None: ...\n\nclass Sujeto:\n    def __init__(self):\n        self._observadores: list[Observador] = []\n\n    def suscribir(self, obs: Observador) -> None:\n        self._observadores.append(obs)\n\n    def notificar(self, evento: str, datos: dict = {}) -> None:\n        for obs in self._observadores:\n            obs.actualizar(evento, datos)\n\nclass Logger(Observador):\n    def actualizar(self, evento: str, datos: dict) -> None:\n        print(f"[LOG] {evento}: {datos}")\n\nbus = Sujeto()\nbus.suscribir(Logger())\nbus.notificar("usuario_creado", {"nombre": "Ana"})',
        '# Patrón Strategy\nfrom abc import ABC, abstractmethod\n\nclass EstrategiaOrden(ABC):\n    @abstractmethod\n    def ordenar(self, datos: list) -> list: ...\n\nclass OrdenBurbujas(EstrategiaOrden):\n    def ordenar(self, datos: list) -> list:\n        d = datos[:]\n        n = len(d)\n        for i in range(n):\n            for j in range(0, n-i-1):\n                if d[j] > d[j+1]:\n                    d[j], d[j+1] = d[j+1], d[j]\n        return d\n\nclass OrdenPython(EstrategiaOrden):\n    def ordenar(self, datos: list) -> list:\n        return sorted(datos)\n\nclass Ordenador:\n    def __init__(self, estrategia: EstrategiaOrden):\n        self.estrategia = estrategia\n\n    def ejecutar(self, datos: list) -> list:\n        return self.estrategia.ordenar(datos)\n\nord = Ordenador(OrdenPython())\nprint(ord.ejecutar([3, 1, 4, 1, 5, 9]))',
        '# Observer con events dict (pythónico)\nfrom collections import defaultdict\nfrom typing import Callable\n\nclass EventBus:\n    def __init__(self):\n        self._handlers: dict = defaultdict(list)\n\n    def on(self, evento: str, handler: Callable) -> None:\n        self._handlers[evento].append(handler)\n\n    def emit(self, evento: str, **datos) -> None:\n        for handler in self._handlers[evento]:\n            handler(**datos)\n\nbus = EventBus()\n\n@bus.on("login")\ndef log_login(usuario, **_):\n    print(f"Usuario {usuario} inició sesión")\n\nbus.emit("login", usuario="ana", ip="192.168.1.1")',
        '# Strategy con funciones (pythónico)\nfrom typing import Callable\n\ndef validar_email(s: str) -> bool:\n    return "@" in s and "." in s.split("@")[-1]\n\ndef validar_telefono(s: str) -> bool:\n    return s.replace("-", "").replace("+", "").isdigit()\n\ndef validar(valor: str, estrategia: Callable[[str], bool]) -> bool:\n    return estrategia(valor)\n\nprint(validar("user@mail.com", validar_email))  # True\nprint(validar("+54-911-1234", validar_telefono))  # True',
        '# Observer con weakrefs (sin memory leak)\nimport weakref\nfrom typing import Callable\n\nclass Señal:\n    def __init__(self):\n        self._slots: list = []\n\n    def conectar(self, slot: Callable) -> None:\n        self._slots.append(weakref.WeakMethod(slot) if hasattr(slot, "__self__") else weakref.ref(slot))\n\n    def emitir(self, *args, **kwargs) -> None:\n        muertos = []\n        for ref in self._slots:\n            slot = ref()\n            if slot is None:\n                muertos.append(ref)\n            else:\n                slot(*args, **kwargs)\n        for ref in muertos:\n            self._slots.remove(ref)',
        '# Patrón Strategy para serialización\nfrom abc import ABC, abstractmethod\nimport json\n\nclass Serializador(ABC):\n    @abstractmethod\n    def serializar(self, datos: dict) -> str: ...\n    @abstractmethod\n    def deserializar(self, texto: str) -> dict: ...\n\nclass JSONSerializador(Serializador):\n    def serializar(self, datos: dict) -> str:\n        return json.dumps(datos, ensure_ascii=False)\n    def deserializar(self, texto: str) -> dict:\n        return json.loads(texto)\n\ns = JSONSerializador()\ncodificado = s.serializar({"nombre": "Ana", "edad": 30})\nprint(codificado)\nprint(s.deserializar(codificado))',
    ]
    return [jsonl_entry(e, "observer_strategy") for e in shuffle_and_sample(ejemplos, n)]


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
    # --- Temas faltantes (Feb 2026) ---
    "json_y_csv":             (generar_json_y_csv,           "stdlib_python/json_y_csv.jsonl"),
    "logging":                (generar_logging,              "stdlib_python/logging.jsonl"),
    "threading_multiprocess": (generar_threading_multiprocess, "python_basico/threading_multiprocess.jsonl"),
    "git_y_ci_cd":            (generar_git_y_ci_cd,          "ingenieria_software/git_y_ci_cd.jsonl"),
    "iterator_pattern":       (generar_iterator_pattern,     "patrones_diseno/iterator_pattern.jsonl"),
    "command_pattern":        (generar_command_pattern,      "patrones_diseno/command_pattern.jsonl"),
    "refactoring":            (generar_refactoring,          "ingenieria_software/refactoring.jsonl"),
    "testing_avanzado":       (generar_testing_avanzado,     "ingenieria_software/testing_avanzado.jsonl"),
    "arboles_binarios":       (generar_arboles_binarios_extra, "algoritmos/arboles_binarios.jsonl"),
    # --- 22 nuevos generadores (Feb 2026) ---
    "variables_y_tipos":      (generar_variables_y_tipos,    "python_basico/variables_y_tipos.jsonl"),
    "strings_y_formato":      (generar_strings_y_formato,    "python_basico/strings_y_formato.jsonl"),
    "control_de_flujo":       (generar_control_de_flujo,     "python_basico/control_de_flujo.jsonl"),
    "listas_y_tuplas":        (generar_listas_y_tuplas,      "python_basico/listas_y_tuplas.jsonl"),
    "diccionarios_y_sets":    (generar_diccionarios_y_sets,  "python_basico/diccionarios_y_sets.jsonl"),
    "funciones_basicas":      (generar_funciones_basicas,    "python_basico/funciones_basicas.jsonl"),
    "funciones_avanzadas":    (generar_funciones_avanzadas,  "python_basico/funciones_avanzadas.jsonl"),
    "clases_y_oop":           (generar_clases_y_oop,         "python_basico/clases_y_oop.jsonl"),
    "manejo_de_errores":      (generar_manejo_de_errores,    "python_basico/manejo_de_errores.jsonl"),
    "comprensiones":          (generar_comprensiones,        "python_basico/comprensiones.jsonl"),
    "generadores":            (generar_generadores_tema,     "python_basico/generadores.jsonl"),
    "decoradores":            (generar_decoradores_tema,     "python_basico/decoradores.jsonl"),
    "context_managers":       (generar_context_managers,     "python_basico/context_managers.jsonl"),
    "type_hints":             (generar_type_hints_tema,      "python_basico/type_hints.jsonl"),
    "modulos_y_paquetes":     (generar_modulos_y_paquetes,   "python_basico/modulos_y_paquetes.jsonl"),
    "recursion":              (generar_recursion_tema,       "algoritmos/recursion.jsonl"),
    "busqueda_binaria":       (generar_busqueda_binaria,     "algoritmos/busqueda_binaria.jsonl"),
    "sorting_clasico":        (generar_sorting_clasico,      "algoritmos/sorting_clasico.jsonl"),
    "grafos_bfs_dfs":         (generar_grafos_bfs_dfs,       "algoritmos/grafos_bfs_dfs.jsonl"),
    "regex":                  (generar_regex_tema,           "stdlib_python/regex.jsonl"),
    "observer_strategy":      (generar_observer_strategy_tema, "patrones_diseno/observer_strategy.jsonl"),
    "api_design_rest":        (generar_api_design_rest_tema, "ingenieria_software/api_design_rest.jsonl"),
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
