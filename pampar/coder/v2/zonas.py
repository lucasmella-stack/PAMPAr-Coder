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
    B01_KW_DEF = auto()       # def, function, fn
    B02_KW_CLASS = auto()     # class, struct, interface
    B03_KW_IMPORT = auto()    # import, from, require
    B04_KW_RETURN = auto()    # return, yield
    B05_KW_CONTROL = auto()   # if, else, elif, switch
    B06_KW_LOOP = auto()      # for, while, loop
    B07_KW_EXCEPT = auto()    # try, except, catch, finally
    B08_KW_ASYNC = auto()     # async, await
    B09_KW_MOD = auto()       # public, private, static
    B10_KW_VAR = auto()       # let, const, var
    B11_DELIM_PAREN = auto()  # ( )
    B12_DELIM_BRACK = auto()  # [ ]
    B13_DELIM_BRACE = auto()  # { }
    B14_PUNCT = auto()        # , ; :
    B15_COMMENT = auto()      # # // /* */
    
    # =========================================================================
    # SEMANTICA (15 zonas) - Significado
    # =========================================================================
    B16_ID_VAR = auto()       # variables locales
    B17_ID_FUNC = auto()      # nombres de funciones
    B18_ID_CLASS = auto()     # nombres de clases
    B19_ID_PARAM = auto()     # parámetros
    B20_ID_ATTR = auto()      # atributos .attr
    B21_LIT_INT = auto()      # enteros
    B22_LIT_FLOAT = auto()    # decimales
    B23_LIT_STR = auto()      # strings
    B24_LIT_BOOL = auto()     # True, False
    B25_LIT_NONE = auto()     # None, null, nil
    B26_TYPE_PRIM = auto()    # int, str, float
    B27_TYPE_COLL = auto()    # list, dict, set
    B28_TYPE_GEN = auto()     # Optional, List[T]
    B29_BUILTIN = auto()      # print, len, range
    B30_MAGIC = auto()        # __init__, __str__
    
    # =========================================================================
    # LOGICO (12 zonas) - Razonamiento
    # =========================================================================
    B31_OP_ARITH = auto()     # + - * / % **
    B32_OP_COMP = auto()      # == != < > <= >=
    B33_OP_LOGIC = auto()     # and or not
    B34_OP_BIT = auto()       # & | ^ ~ << >>
    B35_OP_ASSIGN = auto()    # = += -= *=
    B36_OP_MEMBER = auto()    # . ->
    B37_OP_TERNARY = auto()   # ? :
    B38_FLOW_BRANCH = auto()  # decisiones if/else
    B39_FLOW_LOOP = auto()    # iteraciones
    B40_FLOW_JUMP = auto()    # break, continue
    B41_FLOW_CALL = auto()    # llamadas a función
    B42_FLOW_EXCEPT = auto()  # manejo de excepciones
    
    # =========================================================================
    # ESTRUCTURAL (10 zonas) - Patrones
    # =========================================================================
    B43_BLOCK_FUNC = auto()   # cuerpo de función
    B44_BLOCK_CLASS = auto()  # cuerpo de clase
    B45_BLOCK_LOOP = auto()   # cuerpo de loop
    B46_BLOCK_COND = auto()   # cuerpo de condicional
    B47_INDENT = auto()       # indentación
    B48_NEWLINE = auto()      # saltos de línea
    B49_SPACE = auto()        # espacios
    B50_PATTERN_LIST = auto() # comprehensions
    B51_PATTERN_DICT = auto() # dict literals
    B52_PATTERN_CALL = auto() # f(x, y, z)


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
ZONA_TERRITORIO: Dict[Zona, Territorio] = {
    z: _zona_a_territorio(z) for z in Zona
}

# Zonas por territorio
ZONAS_POR_TERRITORIO: Dict[Territorio, Tuple[Zona, ...]] = {
    t: tuple(z for z in Zona if ZONA_TERRITORIO[z] == t)
    for t in Territorio
}


# =============================================================================
# PATRONES DE TOKENS POR ZONA
# =============================================================================

ZONAS: Dict[Zona, Set[str]] = {
    # SINTAXIS
    Zona.B01_KW_DEF: {"def", "function", "fn", "func", "lambda"},
    Zona.B02_KW_CLASS: {"class", "struct", "interface", "trait", "enum"},
    Zona.B03_KW_IMPORT: {"import", "from", "require", "include", "use"},
    Zona.B04_KW_RETURN: {"return", "yield", "raise", "throw"},
    Zona.B05_KW_CONTROL: {"if", "else", "elif", "switch", "case", "match"},
    Zona.B06_KW_LOOP: {"for", "while", "loop", "do", "foreach"},
    Zona.B07_KW_EXCEPT: {"try", "except", "catch", "finally", "throw"},
    Zona.B08_KW_ASYNC: {"async", "await", "spawn"},
    Zona.B09_KW_MOD: {"public", "private", "protected", "static", "final"},
    Zona.B10_KW_VAR: {"let", "const", "var", "mut"},
    Zona.B11_DELIM_PAREN: {"(", ")"},
    Zona.B12_DELIM_BRACK: {"[", "]"},
    Zona.B13_DELIM_BRACE: {"{", "}"},
    Zona.B14_PUNCT: {",", ";", ":", ".", "..."},
    Zona.B15_COMMENT: {"#", "//", "/*", "*/", "'''", '"""'},
    
    # SEMANTICA
    Zona.B24_LIT_BOOL: {"True", "False", "true", "false"},
    Zona.B25_LIT_NONE: {"None", "null", "nil", "undefined"},
    Zona.B26_TYPE_PRIM: {"int", "str", "float", "bool", "char", "byte"},
    Zona.B27_TYPE_COLL: {"list", "dict", "set", "tuple", "array", "map"},
    Zona.B29_BUILTIN: {"print", "len", "range", "open", "input", "type"},
    
    # LOGICO
    Zona.B31_OP_ARITH: {"+", "-", "*", "/", "%", "**", "//"},
    Zona.B32_OP_COMP: {"==", "!=", "<", ">", "<=", ">=", "is", "in"},
    Zona.B33_OP_LOGIC: {"and", "or", "not", "&&", "||", "!"},
    Zona.B34_OP_BIT: {"&", "|", "^", "~", "<<", ">>"},
    Zona.B35_OP_ASSIGN: {"=", "+=", "-=", "*=", "/=", ":="},
    Zona.B36_OP_MEMBER: {".", "->", "::"},
    Zona.B40_FLOW_JUMP: {"break", "continue", "pass", "goto"},
    
    # ESTRUCTURAL
    Zona.B47_INDENT: {"\t", "    "},
    Zona.B48_NEWLINE: {"\n", "\r\n"},
}
