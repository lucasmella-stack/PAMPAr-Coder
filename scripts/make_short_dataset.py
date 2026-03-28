#!/usr/bin/env python3
"""Generate sft_short_v1.jsonl — SHORT dense examples matching brain scanner patterns.

Each solution is ≤10 lines so it fits entirely in a single 256-token chunk.
Focused on the specific token patterns that the brain scanner finds hard:
  - @decorator (ESTR territory tokens)
  - lambda expressions (SINT tokens)
  - __magic__ methods (ESTR tokens)
  - comparison / logical operators (LOGI tokens)
  - mixed arithmetic + logic (SINT/LOGI combined)
"""
import json
from pathlib import Path

OUT = Path(__file__).parent.parent / "data" / "sft_short_v1.jsonl"

def ex(problem: str, solution: str) -> dict:
    return {
        "text": f"### Problem:\n{problem}\n### Solution:\n```python\n{solution}\n```",
        "source": "sft_short_v1",
        "license": "open",
    }


examples: list[dict] = []

# =============================================================================
# LAMBDA — many short variants
# =============================================================================
lambda_examples = [
    # Direct assignment variants
    ("Assign a lambda that doubles its input.", "fn = lambda x: x * 2"),
    ("Assign a lambda that squares its input.", "sq = lambda n: n ** 2"),
    ("Assign a lambda that adds two numbers.", "add = lambda a, b: a + b"),
    ("Assign a lambda that subtracts two numbers.", "sub = lambda a, b: a - b"),
    ("Assign a lambda that multiplies two numbers.", "mul = lambda a, b: a * b"),
    ("Assign a lambda that returns the absolute value.", "absolute = lambda x: x if x >= 0 else -x"),
    ("Assign a lambda that clamps a value.", "clamp = lambda x, lo, hi: max(lo, min(hi, x))"),
    ("Assign a lambda that checks if a number is even.", "is_even = lambda n: n % 2 == 0"),
    ("Assign a lambda that checks if a number is positive.", "is_pos = lambda x: x > 0"),
    ("Assign a lambda that returns the max of two values.", "maximum = lambda a, b: a if a > b else b"),
    ("Assign a lambda that negates a number.", "negate = lambda x: -x"),
    ("Assign a lambda that increments by one.", "inc = lambda x: x + 1"),
    ("Assign a lambda that decrements by one.", "dec = lambda x: x - 1"),
    ("Assign a lambda that halves its input.", "half = lambda x: x / 2"),
    ("Assign a lambda that triples its input.", "triple = lambda n: n * 3"),
    # sort key variants
    ("Sort a list of tuples by second element.", "pairs = [(1, 3), (2, 1)]\nresult = sorted(pairs, key=lambda x: x[1])"),
    ("Sort words by length.", "words = ['cat', 'elephant', 'ox']\nresult = sorted(words, key=lambda w: len(w))"),
    ("Sort numbers by absolute value.", "nums = [-3, 1, -2, 4]\nresult = sorted(nums, key=lambda x: abs(x))"),
    ("Sort dicts by a key.", "items = [{'v': 3}, {'v': 1}]\nresult = sorted(items, key=lambda d: d['v'])"),
    ("Find the tuple closest to origin.", "pts = [(1, 2), (3, 0)]\nclosest = min(pts, key=lambda p: p[0]**2 + p[1]**2)"),
    # filter/map variants
    ("Filter even numbers with lambda.", "nums = [1, 2, 3, 4, 5]\nevens = list(filter(lambda x: x % 2 == 0, nums))"),
    ("Map square with lambda.", "nums = [1, 2, 3]\nsquares = list(map(lambda x: x ** 2, nums))"),
    ("Double all items with lambda.", "nums = [1, 2, 3]\ndoubled = list(map(lambda x: x * 2, nums))"),
    ("Filter positives with lambda.", "nums = [-1, 2, -3, 4]\npos = list(filter(lambda x: x > 0, nums))"),
    # conditional lambda
    ("Lambda that returns sign of number.", "sign = lambda x: 1 if x > 0 else -1 if x < 0 else 0"),
    ("Lambda that classifies even or odd.", "parity = lambda n: 'even' if n % 2 == 0 else 'odd'"),
    ("Lambda that returns max of two.", "mx = lambda a, b: a if a >= b else b"),
    # reduce
    (
        "Use reduce with lambda to compute product.",
        "from functools import reduce\nnums = [1, 2, 3, 4]\nprod = reduce(lambda a, b: a * b, nums)",
    ),
    (
        "Use reduce to sum a list.",
        "from functools import reduce\nnums = [1, 2, 3]\ntotal = reduce(lambda a, b: a + b, nums, 0)",
    ),
    # combined
    (
        "Compose two lambdas.",
        "double = lambda x: x * 2\nshift = lambda x: x + 1\nresult = shift(double(5))",
    ),
    (
        "Use lambda as a key in max.",
        "words = ['hi', 'hello', 'hey']\nlongest = max(words, key=lambda w: len(w))",
    ),
    (
        "Inline lambda in sorted call.",
        "data = [3, 1, 4, 1, 5]\nsorted_data = sorted(data, key=lambda x: -x)",
    ),
]
for prob, sol in lambda_examples:
    examples.append(ex(prob, sol))

# =============================================================================
# DECORATORS — short snippets
# =============================================================================
decorator_examples = [
    (
        "Use @staticmethod in a class.",
        "class C:\n    @staticmethod\n    def add(a, b):\n        return a + b",
    ),
    (
        "Use @classmethod in a class.",
        "class C:\n    @classmethod\n    def create(cls):\n        return cls()",
    ),
    (
        "Use @property to wrap an attribute.",
        "class C:\n    def __init__(self, v):\n        self._v = v\n    @property\n    def v(self):\n        return self._v",
    ),
    (
        "Apply @staticmethod for validation.",
        "class V:\n    @staticmethod\n    def positive(n):\n        return n > 0\n    @staticmethod\n    def even(n):\n        return n % 2 == 0",
    ),
    (
        "Define a simple decorator.",
        "def dec(func):\n    def wrapper(*args):\n        return func(*args)\n    return wrapper\n\n@dec\ndef greet(name):\n    return f'hi {name}'",
    ),
    (
        "Use @staticmethod with @classmethod together.",
        "class M:\n    @staticmethod\n    def zero():\n        return 0\n    @classmethod\n    def make(cls):\n        return cls()",
    ),
    (
        "Use @property in a Temperature class.",
        "class T:\n    def __init__(self):\n        self._c = 0\n    @property\n    def celsius(self):\n        return self._c",
    ),
    (
        "Define @abstractmethod in an abstract class.",
        "from abc import ABC, abstractmethod\nclass Shape(ABC):\n    @abstractmethod\n    def area(self):\n        pass",
    ),
    (
        "Use @staticmethod for a utility function.",
        "class Util:\n    @staticmethod\n    def clamp(x, lo, hi):\n        return max(lo, min(hi, x))",
    ),
    (
        "Create a cached staticmethod.",
        "class C:\n    @staticmethod\n    def fib(n):\n        if n <= 1:\n            return n\n        return C.fib(n-1) + C.fib(n-2)",
    ),
    (
        "Use @property with setter.",
        "class N:\n    def __init__(self):\n        self._v = 0\n    @property\n    def v(self):\n        return self._v\n    @v.setter\n    def v(self, val):\n        self._v = val",
    ),
    (
        "Stack two @staticmethod decorators.",
        "class Tools:\n    @staticmethod\n    def sq(n):\n        return n * n\n    @staticmethod\n    def cube(n):\n        return n * n * n",
    ),
    ("Apply @classmethod as factory.", "class P:\n    @classmethod\n    def origin(cls):\n        return cls(0, 0)"),
    (
        "Use @property to compute area.",
        "class R:\n    def __init__(self, w, h):\n        self.w = w\n        self.h = h\n    @property\n    def area(self):\n        return self.w * self.h",
    ),
]
for prob, sol in decorator_examples:
    examples.append(ex(prob, sol))

# =============================================================================
# MAGIC METHODS — short snippets
# =============================================================================
magic_examples = [
    ("Implement __init__ for a Point.", "class Point:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y"),
    ("Implement __str__ for a Point.", "class Point:\n    def __str__(self):\n        return f'({self.x}, {self.y})'"),
    ("Implement __repr__ for a Point.", "class Point:\n    def __repr__(self):\n        return f'Point({self.x!r}, {self.y!r})'"),
    ("Implement __eq__ for equality check.", "class P:\n    def __init__(self, x):\n        self.x = x\n    def __eq__(self, other):\n        return self.x == other.x"),
    ("Implement __lt__ for comparison.", "class P:\n    def __lt__(self, other):\n        return self.x < other.x"),
    ("Implement __len__.", "class C:\n    def __init__(self):\n        self._d = []\n    def __len__(self):\n        return len(self._d)"),
    ("Implement __add__ for a Vector.", "class V:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n    def __add__(self, o):\n        return V(self.x + o.x, self.y + o.y)"),
    ("Implement __mul__ scalar multiplication.", "class V:\n    def __mul__(self, s):\n        return V(self.x * s, self.y * s)"),
    ("Implement __call__.", "class F:\n    def __init__(self, n):\n        self.n = n\n    def __call__(self, x):\n        return x * self.n"),
    ("Implement __bool__.", "class C:\n    def __init__(self):\n        self._items = []\n    def __bool__(self):\n        return len(self._items) > 0"),
    ("Implement __contains__.", "class R:\n    def __init__(self, lo, hi):\n        self.lo = lo\n        self.hi = hi\n    def __contains__(self, v):\n        return self.lo <= v <= self.hi"),
    ("Implement __getitem__.", "class W:\n    def __init__(self, data):\n        self._d = data\n    def __getitem__(self, i):\n        return self._d[i]"),
    ("Implement __hash__ with __eq__.", "class P:\n    def __eq__(self, o):\n        return self.x == o.x\n    def __hash__(self):\n        return hash(self.x)"),
    ("Implement __neg__ and __abs__.", "class N:\n    def __init__(self, v):\n        self.v = v\n    def __neg__(self):\n        return N(-self.v)\n    def __abs__(self):\n        return N(abs(self.v))"),
    ("Implement __iter__ for a range.", "class R:\n    def __init__(self, n):\n        self.n = n\n    def __iter__(self):\n        return iter(range(self.n))"),
    ("Implement __enter__ and __exit__.", "class C:\n    def __enter__(self):\n        return self\n    def __exit__(self, *a):\n        pass"),
    ("Implement __setitem__ and __delitem__.", "class D:\n    def __init__(self):\n        self._d = {}\n    def __setitem__(self, k, v):\n        self._d[k] = v\n    def __delitem__(self, k):\n        del self._d[k]"),
]
for prob, sol in magic_examples:
    examples.append(ex(prob, sol))

# =============================================================================
# COMPARISON + LOGICAL OPERATORS (==, !=, <, >, <=, >=, and, or, not, in, is)
# =============================================================================
comparison_examples = [
    ("Check if two values are equal.", "result = x == y"),
    ("Check inequality.", "different = x != y"),
    ("Check if a < b.", "less = a < b"),
    ("Check if a > b.", "greater = a > b"),
    ("Check a <= b.", "le = a <= b"),
    ("Check a >= b.", "ge = a >= b"),
    ("Combine conditions with and.", "valid = x > 0 and x < 100"),
    ("Combine conditions with or.", "either = a == 0 or b == 0"),
    ("Negate a condition with not.", "absent = not x in collection"),
    ("Use in operator.", "found = item in items"),
    ("Use is operator.", "is_none = obj is None"),
    ("Use is not operator.", "not_none = obj is not None"),
    ("Use chained comparison.", "in_range = lo <= x <= hi"),
    ("Use x == y or x != z.", "result = x == y or x != z"),
    ("Compare with zero.", "positive = n > 0\nnegative = n < 0\nzero = n == 0"),
    ("Use multiple comparisons in if.", "if a == b and b != c:\n    result = True\nelse:\n    result = False"),
    ("Use any with comparison.", "has_positive = any(x > 0 for x in nums)"),
    ("Use all with comparison.", "all_pos = all(x > 0 for x in nums)"),
    ("Combine != and and.", "valid = x != 0 and y != 0"),
    ("Combine == and or.", "either_zero = x == 0 or y == 0"),
    ("Membership and identity check.", "ok = val in allowed and val is not None"),
    ("Short-circuit with or.", "result = a or b or c"),
    ("Short-circuit with and.", "safe = cond1 and cond2 and cond3"),
    ("Ternary based on comparison.", "label = 'yes' if x > 0 else 'no'"),
    ("Filter with comparison.", "big = [x for x in nums if x > 10]"),
    ("Filter with != 0.", "nonzero = [x for x in nums if x != 0]"),
    ("Sort and compare.", "pair = (a, b)\nsmaller = min(a, b)\nlarger = max(a, b)"),
    (
        "Use all comparisons in one function.",
        "def classify(x):\n    if x > 0:\n        return 'positive'\n    elif x < 0:\n        return 'negative'\n    else:\n        return 'zero'",
    ),
    (
        "Binary search with comparisons.",
        "def bsearch(arr, t):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == t:\n            return mid\n        elif arr[mid] < t:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1",
    ),
    (
        "Validate with multiple conditions.",
        "def valid(x, lo, hi):\n    return x >= lo and x <= hi and x != 0",
    ),
]
for prob, sol in comparison_examples:
    examples.append(ex(prob, sol))

# =============================================================================
# ARITHMETIC + MIXED
# =============================================================================
arith_examples = [
    ("Compute sum and product.", "total = a + b + c\nproduct = a * b * c"),
    ("Compute difference and quotient.", "diff = a - b\nquot = a / b"),
    ("Use augmented assignment.", "total += price\ncount -= 1\nresult *= factor"),
    ("Compute power and root.", "sq = x ** 2\nroot = x ** 0.5"),
    ("Mixed arithmetic expression.", "result = a + b * c - d / e"),
    ("Arithmetic with parentheses.", "val = (a + b) * (c - d)"),
    ("Floor division and modulo.", "q, r = a // b, a % b"),
    ("Compute mean of a list.", "mean = sum(nums) / len(nums)"),
    ("Compute variance.", "mean = sum(nums) / len(nums)\nvar = sum((x - mean) ** 2 for x in nums) / len(nums)"),
    ("Normalize a value.", "norm = (x - x_min) / (x_max - x_min)"),
    ("Clamp a value.", "result = max(lo, min(hi, x))"),
    ("Mix arithmetic and comparison.", "valid = a + b > c and a * b != 0"),
    ("Accumulate in a loop.", "total = 0\nfor n in nums:\n    total += n\n    total *= 2"),
    ("Compute distance.", "dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5"),
    ("Use divmod.", "q, r = divmod(a, b)"),
]
for prob, sol in arith_examples:
    examples.append(ex(prob, sol))

# Write output
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    for item in examples:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Written {len(examples)} examples to {OUT}")
