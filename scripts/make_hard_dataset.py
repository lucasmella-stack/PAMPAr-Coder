#!/usr/bin/env python3
"""Generate sft_hard_v2.jsonl — SHORT targeted data for hard routing patterns.

Short examples (3-10 lines) that fit within 256-token chunks, matching exactly
the kind of patterns the brain scanner evaluates.
"""
import json
from pathlib import Path

OUTPUT = "sft_hard_v2.jsonl"

def ex(problem: str, solution: str) -> dict:
    return {
        "text": f"### Problem:\n{problem}\n### Solution:\n```python\n{solution}\n```",
        "source": "sft_hard_v2",
        "license": "open",
    }

examples: list[dict] = []

# ===========================================================================
# DECORADORES (@) — ESTR territory tokens
# ===========================================================================
examples += [
    ex(
        "Create a class with @staticmethod and @classmethod decorators.",
        "class MathUtils:\n"
        "    @staticmethod\n"
        "    def add(a, b):\n"
        "        return a + b\n\n"
        "    @classmethod\n"
        "    def create(cls):\n"
        "        return cls()",
    ),
    ex(
        "Use @property to make a read-only attribute.",
        "class Circle:\n"
        "    def __init__(self, radius):\n"
        "        self._radius = radius\n\n"
        "    @property\n"
        "    def radius(self):\n"
        "        return self._radius\n\n"
        "    @property\n"
        "    def area(self):\n"
        "        return 3.14159 * self._radius ** 2",
    ),
    ex(
        "Create a logging decorator that wraps any function.",
        "import functools\n\n"
        "def log_call(func):\n"
        "    @functools.wraps(func)\n"
        "    def wrapper(*args, **kwargs):\n"
        "        print(f'Calling {func.__name__}')\n"
        "        return func(*args, **kwargs)\n"
        "    return wrapper\n\n"
        "@log_call\n"
        "def greet(name):\n"
        "    return f'Hello, {name}'",
    ),
    ex(
        "Use @staticmethod for utility methods.",
        "class Validator:\n"
        "    @staticmethod\n"
        "    def is_positive(n):\n"
        "        return n > 0\n\n"
        "    @staticmethod\n"
        "    def is_even(n):\n"
        "        return n % 2 == 0\n\n"
        "    @staticmethod\n"
        "    def is_in_range(n, lo, hi):\n"
        "        return lo <= n <= hi",
    ),
    ex(
        "Create a class with @property setter and deleter.",
        "class Temperature:\n"
        "    def __init__(self):\n"
        "        self._celsius = 0\n\n"
        "    @property\n"
        "    def celsius(self):\n"
        "        return self._celsius\n\n"
        "    @celsius.setter\n"
        "    def celsius(self, value):\n"
        "        if value < -273.15:\n"
        "            raise ValueError('Temperature below absolute zero')\n"
        "        self._celsius = value\n\n"
        "    @celsius.deleter\n"
        "    def celsius(self):\n"
        "        del self._celsius",
    ),
    ex(
        "Create a timer decorator that measures execution time.",
        "import time\nimport functools\n\n"
        "def timer(func):\n"
        "    @functools.wraps(func)\n"
        "    def wrapper(*args, **kwargs):\n"
        "        start = time.time()\n"
        "        result = func(*args, **kwargs)\n"
        "        elapsed = time.time() - start\n"
        "        print(f'{func.__name__} took {elapsed:.4f}s')\n"
        "        return result\n"
        "    return wrapper\n\n"
        "@timer\n"
        "def process_data(items):\n"
        "    return sorted(items)",
    ),
    ex(
        "Use @classmethod as a factory method.",
        "class Point:\n"
        "    def __init__(self, x, y):\n"
        "        self.x = x\n"
        "        self.y = y\n\n"
        "    @classmethod\n"
        "    def from_tuple(cls, t):\n"
        "        return cls(t[0], t[1])\n\n"
        "    @classmethod\n"
        "    def origin(cls):\n"
        "        return cls(0, 0)",
    ),
    ex(
        "Create a decorator with arguments.",
        "import functools\n\n"
        "def repeat(n):\n"
        "    def decorator(func):\n"
        "        @functools.wraps(func)\n"
        "        def wrapper(*args, **kwargs):\n"
        "            for _ in range(n):\n"
        "                result = func(*args, **kwargs)\n"
        "            return result\n"
        "        return wrapper\n"
        "    return decorator\n\n"
        "@repeat(3)\n"
        "def say_hello():\n"
        "    print('Hello')",
    ),
    ex(
        "Use @abstractmethod to define an interface.",
        "from abc import ABC, abstractmethod\n\n"
        "class Shape(ABC):\n"
        "    @abstractmethod\n"
        "    def area(self):\n"
        "        pass\n\n"
        "    @abstractmethod\n"
        "    def perimeter(self):\n"
        "        pass\n\n"
        "    @staticmethod\n"
        "    def validate(value):\n"
        "        return value > 0",
    ),
    ex(
        "Create a memoize decorator using a dictionary.",
        "def memoize(func):\n"
        "    cache = {}\n"
        "    @functools.wraps(func)\n"
        "    def wrapper(*args):\n"
        "        if args not in cache:\n"
        "            cache[args] = func(*args)\n"
        "        return cache[args]\n"
        "    return wrapper\n\n"
        "@memoize\n"
        "def fibonacci(n):\n"
        "    if n <= 1:\n"
        "        return n\n"
        "    return fibonacci(n - 1) + fibonacci(n - 2)",
    ),
    ex(
        "Create a class using @dataclass decorator.",
        "from dataclasses import dataclass, field\n\n"
        "@dataclass\n"
        "class Employee:\n"
        "    name: str\n"
        "    salary: float\n"
        "    skills: list = field(default_factory=list)\n\n"
        "    @property\n"
        "    def annual_salary(self):\n"
        "        return self.salary * 12\n\n"
        "@dataclass\n"
        "class Department:\n"
        "    name: str\n"
        "    employees: list = field(default_factory=list)\n\n"
        "    @staticmethod\n"
        "    def create_empty(name):\n"
        "        return Department(name=name)",
    ),
    ex(
        "Use @overload for type-safe overloaded functions.",
        "from typing import overload\n\n"
        "@overload\n"
        "def process(x: int) -> int: ...\n\n"
        "@overload\n"
        "def process(x: str) -> str: ...\n\n"
        "def process(x):\n"
        "    if isinstance(x, int):\n"
        "        return x * 2\n"
        "    return x.upper()",
    ),
]

# ===========================================================================
# LAMBDA — SINT territory tokens (lambda keyword, :, *)
# ===========================================================================
examples += [
    ex(
        "Sort a list of tuples by the second element using lambda.",
        "pairs = [(1, 3), (4, 1), (2, 5)]\n"
        "sorted_pairs = sorted(pairs, key=lambda x: x[1])\n"
        "print(sorted_pairs)",
    ),
    ex(
        "Use lambda with map and filter.",
        "numbers = [1, 2, 3, 4, 5, 6]\n"
        "evens = list(filter(lambda x: x % 2 == 0, numbers))\n"
        "squares = list(map(lambda x: x ** 2, numbers))\n"
        "doubled_evens = list(map(lambda x: x * 2, filter(lambda x: x % 2 == 0, numbers)))",
    ),
    ex(
        "Create lambda function for sorting dictionaries.",
        "students = [{'name': 'Alice', 'grade': 90}, {'name': 'Bob', 'grade': 85}]\n"
        "sorted_by_grade = sorted(students, key=lambda s: s['grade'], reverse=True)\n"
        "sorted_by_name = sorted(students, key=lambda s: s['name'])",
    ),
    ex(
        "Use lambda with reduce.",
        "from functools import reduce\n"
        "numbers = [1, 2, 3, 4, 5]\n"
        "product = reduce(lambda x, y: x * y, numbers)\n"
        "total = reduce(lambda a, b: a + b, numbers, 0)",
    ),
    ex(
        "Assign lambda to variables and call them.",
        "fn = lambda x: x * 2\n"
        "double = lambda n: n * 2\n"
        "add = lambda a, b: a + b\n"
        "clamp = lambda x, lo, hi: max(lo, min(hi, x))\n"
        "result = double(add(3, 4))",
    ),
    ex(
        "Use lambda with conditional expression.",
        "classify = lambda x: 'positive' if x > 0 else 'negative' if x < 0 else 'zero'\n"
        "is_even = lambda n: n % 2 == 0\n"
        "absolute = lambda x: x if x >= 0 else -x",
    ),
    ex(
        "Sort strings by multiple keys using lambda.",
        "words = ['banana', 'apple', 'cherry', 'date']\n"
        "by_length = sorted(words, key=lambda w: len(w))\n"
        "by_length_then_alpha = sorted(words, key=lambda w: (len(w), w))\n"
        "by_last_char = sorted(words, key=lambda w: w[-1])",
    ),
    ex(
        "Use lambda with max and min.",
        "points = [(1, 2), (3, 1), (2, 4)]\n"
        "closest = min(points, key=lambda p: p[0] ** 2 + p[1] ** 2)\n"
        "farthest = max(points, key=lambda p: p[0] ** 2 + p[1] ** 2)",
    ),
    ex(
        "Use lambda in a pipeline.",
        "transform = lambda x: x * 2\n"
        "shift = lambda x: x + 1\n"
        "negate = lambda x: -x\n"
        "pipeline = lambda x: negate(shift(transform(x)))\n"
        "result = pipeline(5)",
    ),
    ex(
        "Use lambda with dict comprehension and zip.",
        "keys = ['a', 'b', 'c']\n"
        "values = [1, 2, 3]\n"
        "merge = lambda k, v: dict(zip(k, v))\n"
        "doubled = dict(map(lambda kv: (kv[0], kv[1] * 2), zip(keys, values)))",
    ),
    ex(
        "Sort a matrix by row sum using lambda.",
        "matrix = [[3, 1, 4], [1, 5, 9], [2, 6, 5]]\n"
        "sorted_rows = sorted(matrix, key=lambda row: sum(row))\n"
        "max_row = max(matrix, key=lambda row: sum(row))\n"
        "min_element = min(matrix, key=lambda row: min(row))",
    ),
    ex(
        "Use lambda for groupby operations.",
        "from itertools import groupby\n"
        "words = ['cat', 'car', 'dog', 'door', 'ant', 'ape']\n"
        "by_first = {k: list(v) for k, v in groupby(sorted(words, key=lambda w: w[0]), lambda w: w[0])}\n"
        "by_len = sorted(words, key=lambda w: len(w))",
    ),
]

# ===========================================================================
# MAGIC METHODS (__init__, __str__, __eq__ etc.)
# ===========================================================================
examples += [
    ex(
        "Implement __init__ and __str__ for a Point class.",
        "class Point:\n"
        "    def __init__(self, x, y):\n"
        "        self.x = x\n"
        "        self.y = y\n\n"
        "    def __str__(self):\n"
        "        return f'Point({self.x}, {self.y})'\n\n"
        "    def __repr__(self):\n"
        "        return f'Point(x={self.x!r}, y={self.y!r})'",
    ),
    ex(
        "Implement comparison magic methods for a Student class.",
        "class Student:\n"
        "    def __init__(self, name, grade):\n"
        "        self.name = name\n"
        "        self.grade = grade\n\n"
        "    def __eq__(self, other):\n"
        "        return self.grade == other.grade\n\n"
        "    def __lt__(self, other):\n"
        "        return self.grade < other.grade\n\n"
        "    def __le__(self, other):\n"
        "        return self.grade <= other.grade\n\n"
        "    def __gt__(self, other):\n"
        "        return self.grade > other.grade",
    ),
    ex(
        "Implement __len__ and __getitem__ for a Stack.",
        "class Stack:\n"
        "    def __init__(self):\n"
        "        self._items = []\n\n"
        "    def __len__(self):\n"
        "        return len(self._items)\n\n"
        "    def __getitem__(self, index):\n"
        "        return self._items[index]\n\n"
        "    def __bool__(self):\n"
        "        return len(self._items) > 0\n\n"
        "    def push(self, item):\n"
        "        self._items.append(item)\n\n"
        "    def pop(self):\n"
        "        return self._items.pop()",
    ),
    ex(
        "Implement __add__ and __mul__ for a Vector class.",
        "class Vector:\n"
        "    def __init__(self, x, y):\n"
        "        self.x = x\n"
        "        self.y = y\n\n"
        "    def __add__(self, other):\n"
        "        return Vector(self.x + other.x, self.y + other.y)\n\n"
        "    def __sub__(self, other):\n"
        "        return Vector(self.x - other.x, self.y - other.y)\n\n"
        "    def __mul__(self, scalar):\n"
        "        return Vector(self.x * scalar, self.y * scalar)\n\n"
        "    def __repr__(self):\n"
        "        return f'Vector({self.x}, {self.y})'",
    ),
    ex(
        "Implement __contains__ and __iter__ for a NumberRange.",
        "class NumberRange:\n"
        "    def __init__(self, start, end):\n"
        "        self.start = start\n"
        "        self.end = end\n\n"
        "    def __contains__(self, value):\n"
        "        return self.start <= value <= self.end\n\n"
        "    def __iter__(self):\n"
        "        return iter(range(self.start, self.end + 1))\n\n"
        "    def __len__(self):\n"
        "        return max(0, self.end - self.start + 1)",
    ),
    ex(
        "Create a context manager using __enter__ and __exit__.",
        "import time\n\n"
        "class Timer:\n"
        "    def __init__(self, name):\n"
        "        self.name = name\n"
        "        self.elapsed = 0\n\n"
        "    def __enter__(self):\n"
        "        self._start = time.time()\n"
        "        return self\n\n"
        "    def __exit__(self, exc_type, exc_val, exc_tb):\n"
        "        self.elapsed = time.time() - self._start\n"
        "        print(f'{self.name}: {self.elapsed:.4f}s')\n"
        "        return False",
    ),
    ex(
        "Implement __hash__ alongside __eq__ for a hashable Point.",
        "class Point:\n"
        "    def __init__(self, x, y):\n"
        "        self.x = x\n"
        "        self.y = y\n\n"
        "    def __eq__(self, other):\n"
        "        return self.x == other.x and self.y == other.y\n\n"
        "    def __hash__(self):\n"
        "        return hash((self.x, self.y))\n\n"
        "    def __repr__(self):\n"
        "        return f'Point({self.x}, {self.y})'",
    ),
    ex(
        "Implement __call__ to make objects callable.",
        "class Multiplier:\n"
        "    def __init__(self, factor):\n"
        "        self.factor = factor\n\n"
        "    def __call__(self, x):\n"
        "        return x * self.factor\n\n"
        "double = Multiplier(2)\n"
        "triple = Multiplier(3)\n"
        "result = double(5) + triple(3)\n"
        "composed = lambda x: double(triple(x))",
    ),
    ex(
        "Implement __setitem__ and __delitem__ for a custom dict.",
        "class LimitedDict:\n"
        "    def __init__(self, max_size):\n"
        "        self._data = {}\n"
        "        self.max_size = max_size\n\n"
        "    def __setitem__(self, key, value):\n"
        "        if len(self._data) >= self.max_size and key not in self._data:\n"
        "            raise OverflowError('Dict is full')\n"
        "        self._data[key] = value\n\n"
        "    def __getitem__(self, key):\n"
        "        return self._data[key]\n\n"
        "    def __delitem__(self, key):\n"
        "        del self._data[key]\n\n"
        "    def __len__(self):\n"
        "        return len(self._data)",
    ),
    ex(
        "Use __init_subclass__ to register subclasses.",
        "class Plugin:\n"
        "    _registry = {}\n\n"
        "    def __init_subclass__(cls, name=None, **kwargs):\n"
        "        super().__init_subclass__(**kwargs)\n"
        "        if name is not None:\n"
        "            Plugin._registry[name] = cls\n\n"
        "class PluginA(Plugin, name='a'):\n"
        "    def run(self):\n"
        "        return 'A'\n\n"
        "class PluginB(Plugin, name='b'):\n"
        "    def run(self):\n"
        "        return 'B'",
    ),
    ex(
        "Implement __format__ for custom string formatting.",
        "class Money:\n"
        "    def __init__(self, amount, currency='USD'):\n"
        "        self.amount = amount\n"
        "        self.currency = currency\n\n"
        "    def __format__(self, spec):\n"
        "        if spec == 'short':\n"
        "            return f'{self.amount:.0f} {self.currency}'\n"
        "        return f'{self.amount:.2f} {self.currency}'\n\n"
        "    def __repr__(self):\n"
        "        return f'Money({self.amount!r}, {self.currency!r})'",
    ),
    ex(
        "Implement __neg__ and __abs__ for a Vector.",
        "class Vector:\n"
        "    def __init__(self, x, y, z=0):\n"
        "        self.x = x\n"
        "        self.y = y\n"
        "        self.z = z\n\n"
        "    def __neg__(self):\n"
        "        return Vector(-self.x, -self.y, -self.z)\n\n"
        "    def __abs__(self):\n"
        "        return (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5\n\n"
        "    def __eq__(self, other):\n"
        "        return self.x == other.x and self.y == other.y and self.z == other.z",
    ),
]

# ===========================================================================
# COMPARISON + LOGICAL OPERATORS (==, !=, <, >, >=, <=, and, or, not, in, is)
# ===========================================================================
examples += [
    ex(
        "Write a function using comparison and logical operators.",
        "def classify_number(x):\n"
        "    if x > 0 and x % 2 == 0:\n"
        "        return 'positive even'\n"
        "    elif x > 0 and x % 2 != 0:\n"
        "        return 'positive odd'\n"
        "    elif x < 0:\n"
        "        return 'negative'\n"
        "    else:\n"
        "        return 'zero'",
    ),
    ex(
        "Use chained comparisons.",
        "def is_valid_age(age):\n"
        "    return 0 <= age <= 150\n\n"
        "def is_in_range(x, lo, hi):\n"
        "    return lo <= x <= hi\n\n"
        "def between(a, lo, hi):\n"
        "    return lo < a < hi\n\n"
        "def all_equal(a, b, c):\n"
        "    return a == b == c",
    ),
    ex(
        "Use all() and any() with comparisons.",
        "def all_positive(numbers):\n"
        "    return all(x > 0 for x in numbers)\n\n"
        "def any_negative(numbers):\n"
        "    return any(x < 0 for x in numbers)\n\n"
        "def none_zero(numbers):\n"
        "    return not any(x == 0 for x in numbers)\n\n"
        "def all_distinct(items):\n"
        "    return len(items) == len(set(items))",
    ),
    ex(
        "Use is, in, and not operators.",
        "def check_value(obj, collection):\n"
        "    if obj is None:\n"
        "        return 'none'\n"
        "    if obj not in collection:\n"
        "        return 'absent'\n"
        "    if not isinstance(obj, str):\n"
        "        return 'not string'\n"
        "    return 'found'\n\n"
        "def is_missing(x):\n"
        "    return x is None or x == '' or x == []",
    ),
    ex(
        "Write a comparator function.",
        "def compare(a, b):\n"
        "    if a == b:\n"
        "        return 0\n"
        "    elif a < b:\n"
        "        return -1\n"
        "    else:\n"
        "        return 1\n\n"
        "def compare_strings(s1, s2):\n"
        "    if s1 == s2:\n"
        "        return 0\n"
        "    return -1 if s1 < s2 else 1",
    ),
    ex(
        "Combine logical operators for validation.",
        "def is_valid_email(email):\n"
        "    has_at = '@' in email\n"
        "    has_dot = '.' in email\n"
        "    not_empty = len(email) > 0\n"
        "    return has_at and has_dot and not_empty\n\n"
        "def is_valid_password(pwd):\n"
        "    return (len(pwd) >= 8\n"
        "            and any(c.isupper() for c in pwd)\n"
        "            and any(c.isdigit() for c in pwd))",
    ),
    ex(
        "Implement equality and inequality logic.",
        "def find_different(a, b, c):\n"
        "    if a == b and b == c:\n"
        "        return None\n"
        "    if a != b and a != c:\n"
        "        return a\n"
        "    if b != a and b != c:\n"
        "        return b\n"
        "    return c\n\n"
        "def count_distinct(items):\n"
        "    return len(set(items))",
    ),
    ex(
        "Write boolean short-circuit expressions.",
        "def safe_divide(a, b):\n"
        "    return b != 0 and a / b\n\n"
        "def first_truthy(*values):\n"
        "    return next((v for v in values if v), None)\n\n"
        "def coalesce(value, default):\n"
        "    return value if value is not None else default\n\n"
        "def guard(condition, message):\n"
        "    return condition or (_ for _ in ()).throw(ValueError(message))",
    ),
    ex(
        "Use comparison operators in list comprehensions.",
        "numbers = list(range(-10, 11))\n"
        "positives = [x for x in numbers if x > 0]\n"
        "non_negatives = [x for x in numbers if x >= 0]\n"
        "odds = [x for x in numbers if x != 0 and x % 2 != 0]\n"
        "pairs = [(a, b) for a in numbers for b in numbers if a != b and a < b]",
    ),
    ex(
        "Implement a binary search using comparisons.",
        "def binary_search(arr, target):\n"
        "    lo, hi = 0, len(arr) - 1\n"
        "    while lo <= hi:\n"
        "        mid = (lo + hi) // 2\n"
        "        if arr[mid] == target:\n"
        "            return mid\n"
        "        elif arr[mid] < target:\n"
        "            lo = mid + 1\n"
        "        else:\n"
        "            hi = mid - 1\n"
        "    return -1",
    ),
    ex(
        "Use membership testing and identity checks.",
        "VALID_STATUSES = {'active', 'inactive', 'pending'}\n\n"
        "def is_valid_status(status):\n"
        "    return status in VALID_STATUSES\n\n"
        "def check_singleton(obj, singleton):\n"
        "    return obj is singleton\n\n"
        "def filter_valid(items):\n"
        "    return [x for x in items if x is not None and x != '' and x != 0]",
    ),
    ex(
        "Implement mixed comparison logic for a grading system.",
        "def letter_grade(score):\n"
        "    if score >= 90:\n"
        "        return 'A'\n"
        "    elif score >= 80:\n"
        "        return 'B'\n"
        "    elif score >= 70:\n"
        "        return 'C'\n"
        "    elif score >= 60:\n"
        "        return 'D'\n"
        "    else:\n"
        "        return 'F'\n\n"
        "def is_passing(score, threshold=60):\n"
        "    return score >= threshold and score <= 100",
    ),
]

# ===========================================================================
# MIXED: arithmetic + logic + lambda + decorators combined
# ===========================================================================
examples += [
    ex(
        "Combine arithmetic and comparison in a statistics module.",
        "def stats(numbers):\n"
        "    n = len(numbers)\n"
        "    mean = sum(numbers) / n\n"
        "    variance = sum((x - mean) ** 2 for x in numbers) / n\n"
        "    std = variance ** 0.5\n"
        "    return {'mean': mean, 'std': std, 'min': min(numbers), 'max': max(numbers)}",
    ),
    ex(
        "Implement clamp and normalize using arithmetic.",
        "def clamp(value, lo, hi):\n"
        "    return max(lo, min(hi, value))\n\n"
        "def normalize(x, x_min, x_max):\n"
        "    if x_max == x_min:\n"
        "        return 0.0\n"
        "    return (x - x_min) / (x_max - x_min)\n\n"
        "def lerp(a, b, t):\n"
        "    return a + (b - a) * t",
    ),
    ex(
        "Use augmented assignment operators to compute running totals.",
        "def running_stats(numbers):\n"
        "    total = 0\n"
        "    count = 0\n"
        "    maximum = float('-inf')\n"
        "    for n in numbers:\n"
        "        total += n\n"
        "        count += 1\n"
        "        if n > maximum:\n"
        "            maximum = n\n"
        "    mean = total / count if count != 0 else 0\n"
        "    return mean, total, maximum",
    ),
    ex(
        "Check triangle validity with arithmetic and comparison.",
        "def is_valid_triangle(a, b, c):\n"
        "    return a + b > c and b + c > a and a + c > b\n\n"
        "def is_pythagorean(a, b, c):\n"
        "    sides = sorted([a, b, c])\n"
        "    return sides[0] ** 2 + sides[1] ** 2 == sides[2] ** 2",
    ),
    ex(
        "Compute quadratic roots with arithmetic and comparisons.",
        "import math\n\n"
        "def quadratic_roots(a, b, c):\n"
        "    discriminant = b * b - 4 * a * c\n"
        "    if discriminant < 0:\n"
        "        return None\n"
        "    elif discriminant == 0:\n"
        "        return -b / (2 * a)\n"
        "    else:\n"
        "        d = math.sqrt(discriminant)\n"
        "        return (-b + d) / (2 * a), (-b - d) / (2 * a)",
    ),
    ex(
        "Count tokens by category with lambda and comparison.",
        "tokens = ['def', 'x', '==', '@', 'lambda', 'return', '+', '__init__']\n"
        "keywords = ['def', 'return', 'lambda', 'class', 'if', 'for', 'while']\n"
        "operators = ['+', '-', '*', '/', '==', '!=', '<', '>', '>=', '<=']\n\n"
        "is_keyword = lambda t: t in keywords\n"
        "is_operator = lambda t: t in operators\n\n"
        "keyword_count = sum(1 for t in tokens if is_keyword(t))\n"
        "operator_count = sum(1 for t in tokens if is_operator(t))",
    ),
    ex(
        "Build a pipeline of lambda transformations.",
        "pipeline = [\n"
        "    lambda x: x * 2,\n"
        "    lambda x: x + 1,\n"
        "    lambda x: x ** 2,\n"
        "    lambda x: x - 3,\n"
        "]\n\n"
        "def apply_pipeline(x, transforms):\n"
        "    result = x\n"
        "    for transform in transforms:\n"
        "        result = transform(result)\n"
        "    return result\n\n"
        "output = apply_pipeline(5, pipeline)",
    ),
    ex(
        "Use decorator and lambda together.",
        "import functools\n\n"
        "def validate(func):\n"
        "    @functools.wraps(func)\n"
        "    def wrapper(*args):\n"
        "        if any(a is None for a in args):\n"
        "            raise ValueError('None argument')\n"
        "        return func(*args)\n"
        "    return wrapper\n\n"
        "@validate\n"
        "def compute(a, b, c):\n"
        "    result = a * b + c\n"
        "    return result if result != 0 else None\n\n"
        "transform = lambda x: x ** 2 + 2 * x + 1",
    ),
    ex(
        "Implement a class with decorator, comparison, and lambda.",
        "class SortedList:\n"
        "    def __init__(self, key=None):\n"
        "        self._items = []\n"
        "        self._key = key if key is not None else lambda x: x\n\n"
        "    def add(self, item):\n"
        "        self._items.append(item)\n"
        "        self._items.sort(key=self._key)\n\n"
        "    def __len__(self):\n"
        "        return len(self._items)\n\n"
        "    def __getitem__(self, index):\n"
        "        return self._items[index]\n\n"
        "    @property\n"
        "    def min_item(self):\n"
        "        return self._items[0] if self._items else None\n\n"
        "    @property\n"
        "    def max_item(self):\n"
        "        return self._items[-1] if self._items else None",
    ),
    ex(
        "Create a DSL using magic methods and lambda.",
        "class Query:\n"
        "    def __init__(self, data):\n"
        "        self._data = list(data)\n\n"
        "    def where(self, predicate):\n"
        "        return Query(filter(predicate, self._data))\n\n"
        "    def select(self, transform):\n"
        "        return Query(map(transform, self._data))\n\n"
        "    def order_by(self, key=None):\n"
        "        return Query(sorted(self._data, key=key))\n\n"
        "    def to_list(self):\n"
        "        return list(self._data)\n\n"
        "result = (Query(range(20))\n"
        "          .where(lambda x: x % 2 == 0)\n"
        "          .select(lambda x: x ** 2)\n"
        "          .order_by(lambda x: -x)\n"
        "          .to_list())",
    ),
]

out_path = Path(__file__).parent.parent / "data" / "sft_hard_v1.jsonl"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    for item in examples:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Written {len(examples)} examples to {out_path}")
