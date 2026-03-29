# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Definición de 52 Zonas de Brodmann para código.

Inspirado en la neurociencia: cada zona procesa un tipo específico
de información, permitiendo especialización y eficiencia.

Territorios (4):
- SINTAXIS: Estructura del lenguaje (keywords, delimitadores)
- SEMANTICA: Significado (identificadores, literales)
- LOGICO: Razonamiento (operadores, control de flujo)
- ESTRUCTURAL: Patrones (bloques, formato)
"""

from enum import IntEnum, auto
from typing import Dict, Set, Tuple


class Territorio(IntEnum):
    """Los 4 macro-territorios (lóbulos cerebrales)."""

    SINTAXIS = 0
    SEMANTICA = 1
    LOGICO = 2
    ESTRUCTURAL = 3


class Zona(IntEnum):
    """
    52 zonas especializadas para procesamiento de código.

    Nomenclatura: B{num}_{funcion}
    - B01-B15: SINTAXIS
    - B16-B30: SEMANTICA
    - B31-B42: LOGICO
    - B43-B52: ESTRUCTURAL
    """

    # =========================================================================
    # SINTAXIS (15 zonas) - Estructura del lenguaje
    # =========================================================================
    B01_KW_DEF = auto()  # def, function, fn
    B02_KW_CLASS = auto()  # class, struct, interface
    B03_KW_IMPORT = auto()  # import, from, require
    B04_KW_RETURN = auto()  # return, yield
    B05_KW_CONTROL = auto()  # if, else, elif, switch
    B06_KW_LOOP = auto()  # for, while, loop
    B07_KW_EXCEPT = auto()  # try, except, catch, finally
    B08_KW_ASYNC = auto()  # async, await
    B09_KW_MOD = auto()  # public, private, static
    B10_KW_VAR = auto()  # let, const, var
    B11_DELIM_PAREN = auto()  # ( )
    B12_DELIM_BRACK = auto()  # [ ]
    B13_DELIM_BRACE = auto()  # { }
    B14_PUNCT = auto()  # , ; :
    B15_COMMENT = auto()  # # // /* */

    # =========================================================================
    # SEMANTICA (15 zonas) - Significado
    # =========================================================================
    B16_ID_VAR = auto()  # variables locales
    B17_ID_FUNC = auto()  # nombres de funciones
    B18_ID_CLASS = auto()  # nombres de clases
    B19_ID_PARAM = auto()  # parámetros
    B20_ID_ATTR = auto()  # atributos .attr
    B21_LIT_INT = auto()  # enteros
    B22_LIT_FLOAT = auto()  # decimales
    B23_LIT_STR = auto()  # strings
    B24_LIT_BOOL = auto()  # True, False
    B25_LIT_NONE = auto()  # None, null, nil
    B26_TYPE_PRIM = auto()  # int, str, float
    B27_TYPE_COLL = auto()  # list, dict, set
    B28_TYPE_GEN = auto()  # Optional, List[T]
    B29_BUILTIN = auto()  # print, len, range
    B30_MAGIC = auto()  # __init__, __str__

    # =========================================================================
    # LOGICO (12 zonas) - Razonamiento
    # =========================================================================
    B31_OP_ARITH = auto()  # + - * / % **
    B32_OP_COMP = auto()  # == != < > <= >=
    B33_OP_LOGIC = auto()  # and or not
    B34_OP_BIT = auto()  # & | ^ ~ << >>
    B35_OP_ASSIGN = auto()  # = += -= *=
    B36_OP_MEMBER = auto()  # . ->
    B37_OP_TERNARY = auto()  # ? :
    B38_FLOW_BRANCH = auto()  # decisiones if/else
    B39_FLOW_LOOP = auto()  # iteraciones
    B40_FLOW_JUMP = auto()  # break, continue
    B41_FLOW_CALL = auto()  # llamadas a función
    B42_FLOW_EXCEPT = auto()  # manejo de excepciones

    # =========================================================================
    # ESTRUCTURAL (10 zonas) - Patrones
    # =========================================================================
    B43_BLOCK_FUNC = auto()  # cuerpo de función
    B44_BLOCK_CLASS = auto()  # cuerpo de clase
    B45_BLOCK_LOOP = auto()  # cuerpo de loop
    B46_BLOCK_COND = auto()  # cuerpo de condicional
    B47_INDENT = auto()  # indentación
    B48_NEWLINE = auto()  # saltos de línea
    B49_SPACE = auto()  # espacios
    B50_PATTERN_LIST = auto()  # comprehensions
    B51_PATTERN_DICT = auto()  # dict literals
    B52_PATTERN_CALL = auto()  # f(x, y, z)


# =============================================================================
# MAPEO ZONA -> TERRITORIO
# =============================================================================


def _zona_a_territorio(zona: Zona) -> Territorio:
    """Determina el territorio de una zona."""
    z = zona.value
    if z <= 15:
        return Territorio.SINTAXIS
    elif z <= 30:
        return Territorio.SEMANTICA
    elif z <= 42:
        return Territorio.LOGICO
    else:
        return Territorio.ESTRUCTURAL


# Cache del mapeo
ZONA_TERRITORIO: Dict[Zona, Territorio] = {z: _zona_a_territorio(z) for z in Zona}

# Override: B35_OP_ASSIGN (=, +=, -=, etc.) → SINTAXIS
# Justificación lingüística: la asignación es un constructo sintáctico
# de nivel sentencia, NO una operación lógica/computacional como + o and.
# El modelo ya routea `=` a SINTAXIS de forma natural.
ZONA_TERRITORIO[Zona.B35_OP_ASSIGN] = Territorio.SINTAXIS

# Override: B07_KW_EXCEPT (try, except, finally, raise) → SEMANTICA
# Justificación: los keywords de excepción definen semántica de errores —
# QUÉ errores pueden ocurrir y CÓMO manejarlos. A diferencia de if/for
# (control flow puro), el manejo de excepciones es un concern semántico.
# El modelo los routea a SEMANTICA de forma consistente.
ZONA_TERRITORIO[Zona.B07_KW_EXCEPT] = Territorio.SEMANTICA

# Zonas por territorio
ZONAS_POR_TERRITORIO: Dict[Territorio, Tuple[Zona, ...]] = {
    t: tuple(z for z in Zona if ZONA_TERRITORIO[z] == t) for t in Territorio
}


# =============================================================================
# PATRONES DE TOKENS POR ZONA
# =============================================================================

# PRINCIPIO: cada token tiene UNA zona primaria.
# LLAVES busca en orden, la primera coincidencia gana.
# Zonas ESTRUCTURALES y LOGICO usan patrones regex + contexto
# (vía context_conv en el Tálamo), NO duplican tokens de SINTAXIS.

ZONAS: Dict[Zona, Set[str]] = {
    # =========================================================================
    # SINTAXIS (15 zonas) — keywords y delimitadores del lenguaje
    # Cada keyword pertenece a UNA sola zona primaria.
    # Multi-lenguaje: Python, JavaScript/TypeScript, Rust, SQL, Bash
    # =========================================================================
    Zona.B01_KW_DEF: {
        "def",
        "lambda",  # Python
        "function",
        "fn",  # JS, Rust
    },
    Zona.B02_KW_CLASS: {
        "class",  # Python, JS
        "struct",
        "enum",  # Rust, C, TS
        "interface",
        "trait",  # TS, Rust
        "impl",  # Rust
        "extends",
        "implements",  # JS/TS, Java
    },
    Zona.B03_KW_IMPORT: {
        "import",  # Python, JS
        "require",
        "export",  # Node.js, JS modules
        "use",
        "mod",
        "crate",  # Rust
        "include",
        "package",  # C, Go
        "source",  # Bash
    },
    Zona.B04_KW_RETURN: {"return", "yield"},
    Zona.B05_KW_CONTROL: {
        "if",
        "else",
        "elif",  # Python
        "match",
        "case",  # Python 3.10+, Rust, SQL
        "switch",
        "default",  # JS, C
        "then",
        "fi",
        "esac",  # Bash
        "when",
        "end",  # SQL CASE/WHEN, Ruby
    },
    Zona.B06_KW_LOOP: {
        "for",
        "while",  # Universal
        "do",
        "loop",  # JS do-while, Rust infinite loop
        "done",
        "foreach",  # Bash, PHP/Perl
    },
    Zona.B07_KW_EXCEPT: {
        "try",
        "except",  # Python
        "finally",
        "raise",  # Python
        "catch",
        "throw",  # JS, Java, C++, Rust
    },
    Zona.B08_KW_ASYNC: {"async", "await"},
    Zona.B09_KW_MOD: {
        # Python
        "global",
        "nonlocal",
        "del",
        "with",
        "as",
        "staticmethod",
        "classmethod",
        "property",
        # OOP modifiers (JS/TS, Java, C#)
        "public",
        "private",
        "protected",
        "static",
        "abstract",
        "final",
        "override",
        "readonly",
        "new",  # JS/Java object creation
        # Rust modifiers
        "mut",
        "pub",
        "unsafe",
        "extern",
        # SQL DDL/DML actions & modifiers
        "select",
        "insert",
        "update",
        "delete",
        "create",
        "drop",
        "alter",
        "truncate",
        "order",
        "group",
        "limit",
        "offset",
        "distinct",
        "primary",
        "foreign",
        "key",
        "references",
        "constraint",
        "unique",
        # Bash
        "alias",
    },
    Zona.B10_KW_VAR: {
        "assert",
        "pass",
        "break",
        "continue",  # Python
        "let",
        "const",
        "var",  # JS, Rust
        "exit",  # Bash/shell
    },
    # Incluye tokens combinados paren+quote del tokenizer (SentencePiece).
    Zona.B11_DELIM_PAREN: {"(", ")", "('", '("', "')", '")'},
    Zona.B12_DELIM_BRACK: {"[", "]"},
    Zona.B13_DELIM_BRACE: {"{", "}"},
    Zona.B14_PUNCT: {",", ";", ":", "..."},
    Zona.B15_COMMENT: {
        "#",  # Python, Bash
        "//",  # JS, Rust, C
        "/*",
        "*/",  # JS, C, Rust block comments
        "--",  # SQL
    },
    # =========================================================================
    # SEMANTICA (15 zonas) — significado: identificadores, literales, tipos
    # =========================================================================
    Zona.B16_ID_VAR: {
        "self",
        "cls",
        "_",  # Python
        "this",  # JS, Java
    },
    Zona.B17_ID_FUNC: {},  # Detectado por regex (snake_case seguido de "(")
    Zona.B18_ID_CLASS: {},  # Detectado por regex (CamelCase, UPPER_CASE)
    Zona.B19_ID_PARAM: {"args", "kwargs"},
    Zona.B20_ID_ATTR: {},  # Detectado por contexto (después de ".")
    Zona.B21_LIT_INT: {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"},
    Zona.B22_LIT_FLOAT: {
        "0.0",
        "1.0",
        "0.5",
        "0.1",
        "3.14",
        "1e-5",
        "NaN",
        "Infinity",  # JS globals
    },
    Zona.B23_LIT_STR: {
        "'",
        '"',  # Universal
        "f'",
        'f"',
        "r'",
        'r"',
        "b'",
        'b"',  # Python
        "`",  # JS template literals
    },
    Zona.B24_LIT_BOOL: {
        "True",
        "False",  # Python
        "true",
        "false",  # JS, Rust, JSON, SQL
    },
    Zona.B25_LIT_NONE: {
        "None",  # Python
        "null",
        "undefined",  # JS, SQL
        "nil",  # Go, Ruby, Lua
    },
    Zona.B26_TYPE_PRIM: {
        # Python
        "int",
        "str",
        "float",
        "bool",
        "bytes",
        "complex",
        # JS/TS primitives
        "number",
        "string",
        "boolean",
        "bigint",
        "symbol",
        "void",
        # Rust primitives
        "i8",
        "i16",
        "i32",
        "i64",
        "i128",
        "isize",
        "u8",
        "u16",
        "u32",
        "u64",
        "u128",
        "usize",
        "f32",
        "f64",
        "char",
        # SQL types
        "integer",
        "varchar",
        "text",
        "decimal",
        "numeric",
        "timestamp",
        "serial",
        "bigint",
        "smallint",
    },
    Zona.B27_TYPE_COLL: {
        # Python
        "list",
        "dict",
        "set",
        "tuple",
        "frozenset",
        "deque",
        # JS/TS built-in objects
        "Array",
        "Object",
        "Map",
        "Set",
        "WeakMap",
        "WeakSet",
        # Rust collections
        "Vec",
        "HashMap",
        "HashSet",
        "BTreeMap",
        "BTreeSet",
        "VecDeque",
        "String",  # Rust String (heap-allocated)
        # SQL structural
        "table",
        "view",
        "index",
        "schema",
        "database",
        "column",
    },
    Zona.B28_TYPE_GEN: {
        # Python typing
        "Optional",
        "List",
        "Dict",
        "Tuple",
        "Set",
        "Union",
        "Any",
        "Callable",
        "Iterator",
        "Generator",
        "Iterable",
        "Sequence",
        "Mapping",
        # Rust generics/wrappers
        "Option",
        "Result",
        "Box",
        "Arc",
        "Rc",
        "Ref",
        "RefCell",
        "Mutex",
        "Some",
        "Ok",
        "Err",  # Rust enum variants (used as type constructors)
        # TS utility types
        "Partial",
        "Readonly",
        "Record",
        "Pick",
        "Omit",
        "Exclude",
        "Extract",
        "ReturnType",
        "Promise",
    },
    Zona.B29_BUILTIN: {
        # Python built-ins
        "print",
        "len",
        "range",
        "open",
        "input",
        "type",
        "isinstance",
        "issubclass",
        "hasattr",
        "getattr",
        "setattr",
        "delattr",
        "abs",
        "min",
        "max",
        "sum",
        "sorted",
        "reversed",
        "enumerate",
        "zip",
        "map",
        "filter",
        "any",
        "all",
        "round",
        "pow",
        "repr",
        "hash",
        "id",
        "iter",
        "next",
        "callable",
        "super",
        "object",
        "format",
        "chr",
        "ord",
        "hex",
        "bin",
        "oct",
        "ValueError",
        "TypeError",
        "KeyError",
        "IndexError",
        "AttributeError",
        "RuntimeError",
        "StopIteration",
        "FileNotFoundError",
        "IOError",
        "Exception",
        "BaseException",
        "NotImplementedError",
        "ZeroDivisionError",
        # JS built-ins
        "console",
        "JSON",
        "Math",
        "Date",
        "RegExp",
        "Symbol",
        "parseInt",
        "parseFloat",
        "isNaN",
        "isFinite",
        "setTimeout",
        "setInterval",
        "clearTimeout",
        "clearInterval",
        "fetch",
        "Error",
        "Promise",
        # Rust macros (sin !, el tokenizer separa el !)
        "println",
        "eprintln",
        "dbg",
        "vec",
        "panic",
        "assert_eq",
        "assert_ne",
        "todo",
        "unimplemented",
        # SQL aggregate/scalar functions
        "count",
        "avg",
        "coalesce",
        "cast",
        "convert",
        "exists",
        "between",
        "like",
        "ilike",
        # Bash built-in commands
        "echo",
        "read",
        "cd",
        "ls",
        "grep",
        "sed",
        "awk",
        "cat",
        "mv",
        "cp",
        "rm",
        "mkdir",
        "chmod",
        "chown",
        "find",
        "xargs",
        "curl",
        "wget",
        "tar",
        "ssh",
        "git",
        "docker",
    },
    Zona.B30_MAGIC: {
        "__init__",
        "__str__",
        "__repr__",
        "__len__",
        "__call__",
        "__enter__",
        "__exit__",
        "__iter__",
        "__next__",
        "__getitem__",
        "__setitem__",
        "__delitem__",
        "__contains__",
        "__eq__",
        "__lt__",
        "__gt__",
        "__le__",
        "__ge__",
        "__ne__",
        "__hash__",
        "__add__",
        "__sub__",
        "__mul__",
        "__truediv__",
        "__floordiv__",
        "__mod__",
        "__pow__",
        "__and__",
        "__or__",
        "__xor__",
        "__bool__",
        "__int__",
        "__float__",
        "__index__",
        "__new__",
        "__del__",
        "__slots__",
        "__dict__",
        "__class__",
        "__name__",
        "__doc__",
        "__module__",
        "__file__",
        "__all__",
    },
    # =========================================================================
    # LOGICO (12 zonas) — operadores y razonamiento
    # Sin duplicados de SINTAXIS. Keywords como "and", "or", "not", "in", "is"
    # pertenecen aquí porque su función primaria es lógica.
    # =========================================================================
    Zona.B31_OP_ARITH: {"+", "-", "*", "/", "%", "**", "//"},
    Zona.B32_OP_COMP: {
        "==",
        "!=",
        "<",
        ">",
        "<=",
        ">=",
        "is",
        "in",
        "not",
        "===",
        "!==",  # JS strict equality
        "typeof",
        "instanceof",  # JS type operators
    },
    Zona.B33_OP_LOGIC: {
        "and",
        "or",  # Python
        "&&",
        "||",  # JS, Rust, C, Bash
        "union",
        "intersect",  # SQL set operations
    },
    Zona.B34_OP_BIT: {"&", "|", "^", "~", "<<", ">>", ">>>"},
    Zona.B35_OP_ASSIGN: {
        "=",
        "+=",
        "-=",
        "*=",
        "/=",
        ":=",
        "//=",
        "**=",
        "%=",
        "&=",
        "|=",
        "^=",
        "<<=",
        ">>=",  # Bitwise assigns (JS, Rust, C)
        "??=",  # JS nullish assign
    },
    Zona.B36_OP_MEMBER: {
        ".",  # Universal
        "::",  # Rust path separator
        "?.",  # JS optional chaining
    },
    Zona.B37_OP_TERNARY: {
        "??",  # JS nullish coalescing
    },
    Zona.B38_FLOW_BRANCH: {},  # Delegado a B05 + context_conv detecta branching
    Zona.B39_FLOW_LOOP: {},  # Delegado a B06 + context_conv detecta iteración
    Zona.B40_FLOW_JUMP: {},  # break/continue ya en B10
    Zona.B41_FLOW_CALL: {},  # Detectado por contexto: id + "("
    Zona.B42_FLOW_EXCEPT: {},  # Delegado a B07
    # =========================================================================
    # ESTRUCTURAL (10 zonas) — patrones y formato
    # Formato/whitespace puro. No duplica keywords.
    # =========================================================================
    Zona.B43_BLOCK_FUNC: {
        "->",  # Python return type, Rust return type
        "=>",  # JS arrow function
    },
    Zona.B44_BLOCK_CLASS: {},  # class ya en B02
    Zona.B45_BLOCK_LOOP: {},  # for/while ya en B06
    Zona.B46_BLOCK_COND: {},  # if/elif/else ya en B05
    Zona.B47_INDENT: {"\t", "    "},
    Zona.B48_NEWLINE: {"\n", "\r\n"},
    Zona.B49_SPACE: {" ", "  "},
    Zona.B50_PATTERN_LIST: {},  # Detectado por contexto: "[" + "for" + "in"
    Zona.B51_PATTERN_DICT: {},  # Detectado por contexto: "{" + ":" + "}"
    Zona.B52_PATTERN_CALL: {"from"},  # Structural framing: from X import Y, SQL FROM
}
