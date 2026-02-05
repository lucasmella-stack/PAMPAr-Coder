# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Zonas de Brodmann para PAMPAr-Coder v2.

Inspirado en las 52 áreas de Brodmann del cerebro humano,
expandimos a 52 zonas especializadas para código.

Arquitectura:
- 4 Macro-Territorios (Lóbulos)
- 52 Zonas de Brodmann (Áreas especializadas)
- Cuantización INT4/INT8 para eficiencia

El cerebro humano:
- Área 4: Corteza motora → Nosotros: ACCION (return, yield, break)
- Área 17: Visual primaria → Nosotros: LITERALES (strings, números)
- Área 22: Wernicke (comprensión) → Nosotros: SEMANTICA_COMPRENSION
- Área 44-45: Broca (producción) → Nosotros: SINTAXIS_PRODUCCION
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from enum import Enum, auto
import re


# =============================================================================
# DEFINICIÓN DE LAS 52 ZONAS DE BRODMANN PARA CÓDIGO
# =============================================================================

class MacroTerritorio(Enum):
    """Los 4 macro-territorios (como los lóbulos cerebrales)."""
    SINTAXIS = auto()      # Frontal: producción, keywords, estructura
    SEMANTICA = auto()     # Temporal: comprensión, tipos, contexto
    LOGICO = auto()        # Parietal: razonamiento, control flow
    ESTRUCTURAL = auto()   # Occipital: patrones visuales, indentación


class ZonaBrodmann(Enum):
    """
    52 Zonas especializadas para código.
    
    Nomenclatura: B{número}_{nombre}
    Siguiendo la convención de Brodmann pero adaptada a código.
    """
    # =========================================================================
    # TERRITORIO SINTAXIS (Lóbulo Frontal) - 15 zonas
    # Producción de código, keywords, estructura gramatical
    # =========================================================================
    
    # Área motora primaria → Acciones/Keywords de acción
    B04_KEYWORDS_DEF = auto()         # def, fn, function, func
    B05_KEYWORDS_CLASS = auto()       # class, struct, enum, trait, interface
    B06_KEYWORDS_IMPORT = auto()      # import, from, use, require, include
    
    # Área premotora → Modificadores y decoradores
    B08_MODIFICADORES = auto()        # async, await, static, const, mut, pub
    B09_DECORADORES = auto()          # @decorator, #[attr], annotations
    
    # Área de Broca (producción) → Sintaxis core
    B44_OPERADORES_ARIT = auto()      # +, -, *, /, %, **
    B45_OPERADORES_COMP = auto()      # ==, !=, <, >, <=, >=
    B46_OPERADORES_LOGIC = auto()     # &&, ||, !, and, or, not
    B47_OPERADORES_BIT = auto()       # &, |, ^, ~, <<, >>
    
    # Área prefrontal → Asignación y binding
    B10_ASIGNACION = auto()           # =, :=, <-, +=, -=
    B11_BINDING = auto()              # let, var, const, val
    
    # Delimitadores sintácticos
    B12_DELIM_PAREN = auto()          # (, )
    B13_DELIM_BRACKET = auto()        # [, ]
    B14_DELIM_BRACE = auto()          # {, }
    B15_DELIM_OTROS = auto()          # ,, ;, :, .
    
    # =========================================================================
    # TERRITORIO SEMÁNTICA (Lóbulo Temporal) - 15 zonas
    # Comprensión, tipos, nombres, literales
    # =========================================================================
    
    # Área de Wernicke (comprensión) → Tipos
    B22_TIPOS_PRIMITIVOS = auto()     # int, float, str, bool, char
    B21_TIPOS_NUMERICOS = auto()      # i32, u64, f64, Number
    B20_TIPOS_TEXTO = auto()          # str, String, &str, string
    B37_TIPOS_BOOL = auto()           # bool, boolean, True, False
    
    # Área de asociación temporal → Colecciones
    B38_TIPOS_LISTA = auto()          # list, List, Vec, Array, []
    B39_TIPOS_DICT = auto()           # dict, Dict, HashMap, Map, {}
    B40_TIPOS_SET = auto()            # set, Set, HashSet
    B41_TIPOS_TUPLE = auto()          # tuple, Tuple, ()
    B42_TIPOS_OPTION = auto()         # Option, Optional, Maybe, Result
    
    # Área auditiva → Literales (lo que "escuchamos" del código)
    B17_LITERAL_STRING = auto()       # "hello", 'world', `template`
    B18_LITERAL_NUMERO = auto()       # 42, 3.14, 0xFF, 1e-5
    B19_LITERAL_ESPECIAL = auto()     # None, null, undefined, nil
    
    # Nombres e identificadores
    B35_IDENTIFICADOR_VAR = auto()    # variables: myVar, count, data
    B36_IDENTIFICADOR_FUNC = auto()   # funciones: calculate, getData
    B34_IDENTIFICADOR_CLASE = auto()  # clases: MyClass, UserService
    
    # =========================================================================
    # TERRITORIO LÓGICO (Lóbulo Parietal) - 12 zonas
    # Razonamiento, control flow, decisiones
    # =========================================================================
    
    # Área somatosensorial → Condicionales (sentir el flujo)
    B01_COND_IF = auto()              # if, elif, else if
    B02_COND_ELSE = auto()            # else
    B03_COND_SWITCH = auto()          # switch, match, case, default
    
    # Área de asociación parietal → Loops
    B07_LOOP_FOR = auto()             # for, foreach, for...in, for...of
    B23_LOOP_WHILE = auto()           # while, do...while, loop
    B24_LOOP_ITER = auto()            # iter, map, filter, reduce
    
    # Área de integración → Excepciones
    B25_EXCEPT_TRY = auto()           # try
    B26_EXCEPT_CATCH = auto()         # catch, except, rescue
    B27_EXCEPT_FINALLY = auto()       # finally, ensure, defer
    B28_EXCEPT_THROW = auto()         # throw, raise, panic
    
    # Área motora suplementaria → Jumps
    B29_JUMP_RETURN = auto()          # return, yield
    B30_JUMP_BREAK = auto()           # break, continue, goto
    
    # =========================================================================
    # TERRITORIO ESTRUCTURAL (Lóbulo Occipital) - 10 zonas
    # Patrones visuales, estructura, formato
    # =========================================================================
    
    # Área visual primaria → Bloques
    B31_BLOQUE_PYTHON = auto()        # : + indentación
    B32_BLOQUE_C = auto()             # { }
    B33_BLOQUE_ML = auto()            # begin/end, do/end
    
    # Área visual secundaria → Indentación
    B48_INDENT_SPACE = auto()         # espacios (2, 4, 8)
    B49_INDENT_TAB = auto()           # tabs
    
    # Área de asociación visual → Comentarios
    B50_COMMENT_LINE = auto()         # //, #, --
    B51_COMMENT_BLOCK = auto()        # /* */, """ """, =begin =end
    B52_COMMENT_DOC = auto()          # ///, /**, #', docstrings
    
    # Patrones de formato
    B43_FORMATO_NEWLINE = auto()      # \n, líneas en blanco
    B16_FORMATO_WHITESPACE = auto()   # espacios, tabs entre tokens


# =============================================================================
# MAPEO ZONA → TERRITORIO
# =============================================================================

ZONA_A_TERRITORIO: Dict[ZonaBrodmann, MacroTerritorio] = {
    # SINTAXIS (15 zonas)
    ZonaBrodmann.B04_KEYWORDS_DEF: MacroTerritorio.SINTAXIS,
    ZonaBrodmann.B05_KEYWORDS_CLASS: MacroTerritorio.SINTAXIS,
    ZonaBrodmann.B06_KEYWORDS_IMPORT: MacroTerritorio.SINTAXIS,
    ZonaBrodmann.B08_MODIFICADORES: MacroTerritorio.SINTAXIS,
    ZonaBrodmann.B09_DECORADORES: MacroTerritorio.SINTAXIS,
    ZonaBrodmann.B44_OPERADORES_ARIT: MacroTerritorio.SINTAXIS,
    ZonaBrodmann.B45_OPERADORES_COMP: MacroTerritorio.SINTAXIS,
    ZonaBrodmann.B46_OPERADORES_LOGIC: MacroTerritorio.SINTAXIS,
    ZonaBrodmann.B47_OPERADORES_BIT: MacroTerritorio.SINTAXIS,
    ZonaBrodmann.B10_ASIGNACION: MacroTerritorio.SINTAXIS,
    ZonaBrodmann.B11_BINDING: MacroTerritorio.SINTAXIS,
    ZonaBrodmann.B12_DELIM_PAREN: MacroTerritorio.SINTAXIS,
    ZonaBrodmann.B13_DELIM_BRACKET: MacroTerritorio.SINTAXIS,
    ZonaBrodmann.B14_DELIM_BRACE: MacroTerritorio.SINTAXIS,
    ZonaBrodmann.B15_DELIM_OTROS: MacroTerritorio.SINTAXIS,
    
    # SEMÁNTICA (15 zonas)
    ZonaBrodmann.B22_TIPOS_PRIMITIVOS: MacroTerritorio.SEMANTICA,
    ZonaBrodmann.B21_TIPOS_NUMERICOS: MacroTerritorio.SEMANTICA,
    ZonaBrodmann.B20_TIPOS_TEXTO: MacroTerritorio.SEMANTICA,
    ZonaBrodmann.B37_TIPOS_BOOL: MacroTerritorio.SEMANTICA,
    ZonaBrodmann.B38_TIPOS_LISTA: MacroTerritorio.SEMANTICA,
    ZonaBrodmann.B39_TIPOS_DICT: MacroTerritorio.SEMANTICA,
    ZonaBrodmann.B40_TIPOS_SET: MacroTerritorio.SEMANTICA,
    ZonaBrodmann.B41_TIPOS_TUPLE: MacroTerritorio.SEMANTICA,
    ZonaBrodmann.B42_TIPOS_OPTION: MacroTerritorio.SEMANTICA,
    ZonaBrodmann.B17_LITERAL_STRING: MacroTerritorio.SEMANTICA,
    ZonaBrodmann.B18_LITERAL_NUMERO: MacroTerritorio.SEMANTICA,
    ZonaBrodmann.B19_LITERAL_ESPECIAL: MacroTerritorio.SEMANTICA,
    ZonaBrodmann.B35_IDENTIFICADOR_VAR: MacroTerritorio.SEMANTICA,
    ZonaBrodmann.B36_IDENTIFICADOR_FUNC: MacroTerritorio.SEMANTICA,
    ZonaBrodmann.B34_IDENTIFICADOR_CLASE: MacroTerritorio.SEMANTICA,
    
    # LÓGICO (12 zonas)
    ZonaBrodmann.B01_COND_IF: MacroTerritorio.LOGICO,
    ZonaBrodmann.B02_COND_ELSE: MacroTerritorio.LOGICO,
    ZonaBrodmann.B03_COND_SWITCH: MacroTerritorio.LOGICO,
    ZonaBrodmann.B07_LOOP_FOR: MacroTerritorio.LOGICO,
    ZonaBrodmann.B23_LOOP_WHILE: MacroTerritorio.LOGICO,
    ZonaBrodmann.B24_LOOP_ITER: MacroTerritorio.LOGICO,
    ZonaBrodmann.B25_EXCEPT_TRY: MacroTerritorio.LOGICO,
    ZonaBrodmann.B26_EXCEPT_CATCH: MacroTerritorio.LOGICO,
    ZonaBrodmann.B27_EXCEPT_FINALLY: MacroTerritorio.LOGICO,
    ZonaBrodmann.B28_EXCEPT_THROW: MacroTerritorio.LOGICO,
    ZonaBrodmann.B29_JUMP_RETURN: MacroTerritorio.LOGICO,
    ZonaBrodmann.B30_JUMP_BREAK: MacroTerritorio.LOGICO,
    
    # ESTRUCTURAL (10 zonas)
    ZonaBrodmann.B31_BLOQUE_PYTHON: MacroTerritorio.ESTRUCTURAL,
    ZonaBrodmann.B32_BLOQUE_C: MacroTerritorio.ESTRUCTURAL,
    ZonaBrodmann.B33_BLOQUE_ML: MacroTerritorio.ESTRUCTURAL,
    ZonaBrodmann.B48_INDENT_SPACE: MacroTerritorio.ESTRUCTURAL,
    ZonaBrodmann.B49_INDENT_TAB: MacroTerritorio.ESTRUCTURAL,
    ZonaBrodmann.B50_COMMENT_LINE: MacroTerritorio.ESTRUCTURAL,
    ZonaBrodmann.B51_COMMENT_BLOCK: MacroTerritorio.ESTRUCTURAL,
    ZonaBrodmann.B52_COMMENT_DOC: MacroTerritorio.ESTRUCTURAL,
    ZonaBrodmann.B43_FORMATO_NEWLINE: MacroTerritorio.ESTRUCTURAL,
    ZonaBrodmann.B16_FORMATO_WHITESPACE: MacroTerritorio.ESTRUCTURAL,
}


# =============================================================================
# LLAVES PARA CADA ZONA DE BRODMANN
# =============================================================================

@dataclass
class LlavesZona:
    """LLAVES específicas para una zona de Brodmann."""
    zona: ZonaBrodmann
    tokens: Set[str] = field(default_factory=set)
    patrones: List[str] = field(default_factory=list)  # Regex patterns
    peso: float = 1.0
    bits: int = 4  # Cuantización recomendada


# Definición completa de LLAVES para las 52 zonas
LLAVES_BRODMANN: Dict[ZonaBrodmann, LlavesZona] = {
    # =========================================================================
    # SINTAXIS
    # =========================================================================
    ZonaBrodmann.B04_KEYWORDS_DEF: LlavesZona(
        zona=ZonaBrodmann.B04_KEYWORDS_DEF,
        tokens={'def', 'fn', 'function', 'func', 'fun', 'proc', 'sub', 'method'},
        peso=1.0, bits=4
    ),
    ZonaBrodmann.B05_KEYWORDS_CLASS: LlavesZona(
        zona=ZonaBrodmann.B05_KEYWORDS_CLASS,
        tokens={'class', 'struct', 'enum', 'trait', 'interface', 'protocol', 
                'type', 'record', 'union', 'data'},
        peso=1.0, bits=4
    ),
    ZonaBrodmann.B06_KEYWORDS_IMPORT: LlavesZona(
        zona=ZonaBrodmann.B06_KEYWORDS_IMPORT,
        tokens={'import', 'from', 'use', 'require', 'include', 'using', 
                'extern', 'module', 'package', 'export', 'crate', 'mod'},
        peso=1.0, bits=4
    ),
    ZonaBrodmann.B08_MODIFICADORES: LlavesZona(
        zona=ZonaBrodmann.B08_MODIFICADORES,
        tokens={'async', 'await', 'static', 'const', 'mut', 'pub', 'private',
                'public', 'protected', 'final', 'abstract', 'virtual', 'override',
                'readonly', 'volatile', 'transient', 'synchronized', 'native',
                'extern', 'inline', 'unsafe', 'ref', 'move'},
        peso=0.9, bits=4
    ),
    ZonaBrodmann.B09_DECORADORES: LlavesZona(
        zona=ZonaBrodmann.B09_DECORADORES,
        tokens={'@', '#[', '#', '[[', ']]'},
        patrones=[r'^@\w+', r'^#\[\w+', r'^#\w+'],
        peso=0.8, bits=4
    ),
    ZonaBrodmann.B44_OPERADORES_ARIT: LlavesZona(
        zona=ZonaBrodmann.B44_OPERADORES_ARIT,
        tokens={'+', '-', '*', '/', '%', '**', '//', '÷', '×'},
        peso=0.9, bits=4
    ),
    ZonaBrodmann.B45_OPERADORES_COMP: LlavesZona(
        zona=ZonaBrodmann.B45_OPERADORES_COMP,
        tokens={'==', '!=', '<', '>', '<=', '>=', '===', '!==', '<>', '<=>', 
                'is', 'is not', 'in', 'not in'},
        peso=0.9, bits=4
    ),
    ZonaBrodmann.B46_OPERADORES_LOGIC: LlavesZona(
        zona=ZonaBrodmann.B46_OPERADORES_LOGIC,
        tokens={'&&', '||', '!', 'and', 'or', 'not', '?', '?:', '??', '?.'},
        peso=0.9, bits=4
    ),
    ZonaBrodmann.B47_OPERADORES_BIT: LlavesZona(
        zona=ZonaBrodmann.B47_OPERADORES_BIT,
        tokens={'&', '|', '^', '~', '<<', '>>', '>>>', '&=', '|=', '^=', '<<=', '>>='},
        peso=0.8, bits=4
    ),
    ZonaBrodmann.B10_ASIGNACION: LlavesZona(
        zona=ZonaBrodmann.B10_ASIGNACION,
        tokens={'=', ':=', '<-', '+=', '-=', '*=', '/=', '%=', '**=', '//=', 
                '++', '--', '?='},
        peso=0.9, bits=4
    ),
    ZonaBrodmann.B11_BINDING: LlavesZona(
        zona=ZonaBrodmann.B11_BINDING,
        tokens={'let', 'var', 'const', 'val', 'auto', 'dim', 'my', 'local', 'global'},
        peso=1.0, bits=4
    ),
    ZonaBrodmann.B12_DELIM_PAREN: LlavesZona(
        zona=ZonaBrodmann.B12_DELIM_PAREN,
        tokens={'(', ')'},
        peso=0.7, bits=4
    ),
    ZonaBrodmann.B13_DELIM_BRACKET: LlavesZona(
        zona=ZonaBrodmann.B13_DELIM_BRACKET,
        tokens={'[', ']'},
        peso=0.7, bits=4
    ),
    ZonaBrodmann.B14_DELIM_BRACE: LlavesZona(
        zona=ZonaBrodmann.B14_DELIM_BRACE,
        tokens={'{', '}'},
        peso=0.9, bits=4
    ),
    ZonaBrodmann.B15_DELIM_OTROS: LlavesZona(
        zona=ZonaBrodmann.B15_DELIM_OTROS,
        tokens={',', ';', '.', '::', '->', '=>', '|>', '<|', '..', '...', '~>'},
        peso=0.7, bits=4
    ),
    
    # =========================================================================
    # SEMÁNTICA
    # =========================================================================
    ZonaBrodmann.B22_TIPOS_PRIMITIVOS: LlavesZona(
        zona=ZonaBrodmann.B22_TIPOS_PRIMITIVOS,
        tokens={'int', 'float', 'str', 'bool', 'char', 'byte', 'short', 'long',
                'double', 'void', 'any', 'object', 'dynamic'},
        peso=1.0, bits=4
    ),
    ZonaBrodmann.B21_TIPOS_NUMERICOS: LlavesZona(
        zona=ZonaBrodmann.B21_TIPOS_NUMERICOS,
        tokens={'i8', 'i16', 'i32', 'i64', 'i128', 'isize',
                'u8', 'u16', 'u32', 'u64', 'u128', 'usize',
                'f32', 'f64', 'f128', 'Int', 'Float', 'Double', 'Number',
                'BigInt', 'BigDecimal', 'Decimal', 'Complex'},
        peso=1.0, bits=4
    ),
    ZonaBrodmann.B20_TIPOS_TEXTO: LlavesZona(
        zona=ZonaBrodmann.B20_TIPOS_TEXTO,
        tokens={'str', 'String', '&str', 'string', 'Text', 'Str', 'char', 
                'Character', 'Rune', 'bytes', 'Bytes', 'ByteString'},
        peso=1.0, bits=4
    ),
    ZonaBrodmann.B37_TIPOS_BOOL: LlavesZona(
        zona=ZonaBrodmann.B37_TIPOS_BOOL,
        tokens={'bool', 'boolean', 'Bool', 'Boolean', 'True', 'False', 
                'true', 'false', 'TRUE', 'FALSE'},
        peso=1.0, bits=4
    ),
    ZonaBrodmann.B38_TIPOS_LISTA: LlavesZona(
        zona=ZonaBrodmann.B38_TIPOS_LISTA,
        tokens={'list', 'List', 'Vec', 'Array', 'array', 'ArrayList',
                'LinkedList', 'Deque', 'Stack', 'Queue', 'Slice', 'slice'},
        peso=1.0, bits=4
    ),
    ZonaBrodmann.B39_TIPOS_DICT: LlavesZona(
        zona=ZonaBrodmann.B39_TIPOS_DICT,
        tokens={'dict', 'Dict', 'HashMap', 'Map', 'map', 'Dictionary',
                'BTreeMap', 'OrderedDict', 'DefaultDict', 'Object', 'Record'},
        peso=1.0, bits=4
    ),
    ZonaBrodmann.B40_TIPOS_SET: LlavesZona(
        zona=ZonaBrodmann.B40_TIPOS_SET,
        tokens={'set', 'Set', 'HashSet', 'BTreeSet', 'FrozenSet', 'frozenset'},
        peso=1.0, bits=4
    ),
    ZonaBrodmann.B41_TIPOS_TUPLE: LlavesZona(
        zona=ZonaBrodmann.B41_TIPOS_TUPLE,
        tokens={'tuple', 'Tuple', 'pair', 'Pair', 'Triple'},
        peso=1.0, bits=4
    ),
    ZonaBrodmann.B42_TIPOS_OPTION: LlavesZona(
        zona=ZonaBrodmann.B42_TIPOS_OPTION,
        tokens={'Option', 'Optional', 'Maybe', 'Result', 'Either', 
                'Future', 'Promise', 'Async', 'Task', 'IO'},
        peso=1.0, bits=4
    ),
    ZonaBrodmann.B17_LITERAL_STRING: LlavesZona(
        zona=ZonaBrodmann.B17_LITERAL_STRING,
        tokens={'"', "'", '`', '"""', "'''", 'r"', "r'", 'f"', "f'", 
                'b"', "b'", '@"', '$"'},
        patrones=[r'^["\'].*["\']$', r'^`.*`$'],
        peso=0.8, bits=8
    ),
    ZonaBrodmann.B18_LITERAL_NUMERO: LlavesZona(
        zona=ZonaBrodmann.B18_LITERAL_NUMERO,
        tokens={'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'},
        patrones=[r'^\d+\.?\d*$', r'^0x[0-9a-fA-F]+$', r'^0b[01]+$', r'^0o[0-7]+$',
                  r'^\d+e[+-]?\d+$', r'^\d+_\d+$'],
        peso=0.7, bits=4
    ),
    ZonaBrodmann.B19_LITERAL_ESPECIAL: LlavesZona(
        zona=ZonaBrodmann.B19_LITERAL_ESPECIAL,
        tokens={'None', 'null', 'nil', 'undefined', 'NaN', 'Inf', 'infinity',
                'NULL', 'nullptr', 'void', 'unit', '()'},
        peso=1.0, bits=4
    ),
    ZonaBrodmann.B35_IDENTIFICADOR_VAR: LlavesZona(
        zona=ZonaBrodmann.B35_IDENTIFICADOR_VAR,
        tokens=set(),  # Detectado por patrón
        patrones=[r'^[a-z_][a-zA-Z0-9_]*$'],  # camelCase, snake_case
        peso=0.5, bits=8
    ),
    ZonaBrodmann.B36_IDENTIFICADOR_FUNC: LlavesZona(
        zona=ZonaBrodmann.B36_IDENTIFICADOR_FUNC,
        tokens=set(),
        patrones=[r'^[a-z_][a-zA-Z0-9_]*$'],  # Similar a var pero contexto diferente
        peso=0.5, bits=8
    ),
    ZonaBrodmann.B34_IDENTIFICADOR_CLASE: LlavesZona(
        zona=ZonaBrodmann.B34_IDENTIFICADOR_CLASE,
        tokens=set(),
        patrones=[r'^[A-Z][a-zA-Z0-9_]*$'],  # PascalCase
        peso=0.6, bits=8
    ),
    
    # =========================================================================
    # LÓGICO
    # =========================================================================
    ZonaBrodmann.B01_COND_IF: LlavesZona(
        zona=ZonaBrodmann.B01_COND_IF,
        tokens={'if', 'elif', 'elsif', 'elseif', 'else if', 'If', 'ElseIf', 'unless'},
        peso=1.0, bits=4
    ),
    ZonaBrodmann.B02_COND_ELSE: LlavesZona(
        zona=ZonaBrodmann.B02_COND_ELSE,
        tokens={'else', 'Else', 'otherwise'},
        peso=1.0, bits=4
    ),
    ZonaBrodmann.B03_COND_SWITCH: LlavesZona(
        zona=ZonaBrodmann.B03_COND_SWITCH,
        tokens={'switch', 'match', 'case', 'default', 'when', 'Select', 
                'Case', 'Default', 'cond', 'given'},
        peso=1.0, bits=4
    ),
    ZonaBrodmann.B07_LOOP_FOR: LlavesZona(
        zona=ZonaBrodmann.B07_LOOP_FOR,
        tokens={'for', 'foreach', 'For', 'ForEach', 'each'},
        peso=1.0, bits=4
    ),
    ZonaBrodmann.B23_LOOP_WHILE: LlavesZona(
        zona=ZonaBrodmann.B23_LOOP_WHILE,
        tokens={'while', 'do', 'loop', 'While', 'Do', 'Loop', 'until', 'repeat'},
        peso=1.0, bits=4
    ),
    ZonaBrodmann.B24_LOOP_ITER: LlavesZona(
        zona=ZonaBrodmann.B24_LOOP_ITER,
        tokens={'iter', 'map', 'filter', 'reduce', 'fold', 'collect', 'flatMap',
                'forEach', 'find', 'any', 'all', 'zip', 'enumerate', 'range'},
        peso=0.8, bits=4
    ),
    ZonaBrodmann.B25_EXCEPT_TRY: LlavesZona(
        zona=ZonaBrodmann.B25_EXCEPT_TRY,
        tokens={'try', 'Try', 'begin', 'attempt'},
        peso=1.0, bits=4
    ),
    ZonaBrodmann.B26_EXCEPT_CATCH: LlavesZona(
        zona=ZonaBrodmann.B26_EXCEPT_CATCH,
        tokens={'catch', 'except', 'rescue', 'Catch', 'Except', 'on', 'handle'},
        peso=1.0, bits=4
    ),
    ZonaBrodmann.B27_EXCEPT_FINALLY: LlavesZona(
        zona=ZonaBrodmann.B27_EXCEPT_FINALLY,
        tokens={'finally', 'ensure', 'defer', 'Finally', 'cleanup', 'always'},
        peso=1.0, bits=4
    ),
    ZonaBrodmann.B28_EXCEPT_THROW: LlavesZona(
        zona=ZonaBrodmann.B28_EXCEPT_THROW,
        tokens={'throw', 'raise', 'panic', 'Throw', 'Raise', 'error', 'fail',
                'assert', 'Assert', 'unreachable'},
        peso=1.0, bits=4
    ),
    ZonaBrodmann.B29_JUMP_RETURN: LlavesZona(
        zona=ZonaBrodmann.B29_JUMP_RETURN,
        tokens={'return', 'yield', 'Return', 'Yield', 'emit'},
        peso=1.0, bits=4
    ),
    ZonaBrodmann.B30_JUMP_BREAK: LlavesZona(
        zona=ZonaBrodmann.B30_JUMP_BREAK,
        tokens={'break', 'continue', 'goto', 'Break', 'Continue', 'pass', 
                'next', 'last', 'redo', 'fallthrough'},
        peso=1.0, bits=4
    ),
    
    # =========================================================================
    # ESTRUCTURAL
    # =========================================================================
    ZonaBrodmann.B31_BLOQUE_PYTHON: LlavesZona(
        zona=ZonaBrodmann.B31_BLOQUE_PYTHON,
        tokens={':'},
        patrones=[r':\s*$', r'^:\s*#'],
        peso=0.9, bits=4
    ),
    ZonaBrodmann.B32_BLOQUE_C: LlavesZona(
        zona=ZonaBrodmann.B32_BLOQUE_C,
        tokens={'{', '}', '};'},
        peso=1.0, bits=4
    ),
    ZonaBrodmann.B33_BLOQUE_ML: LlavesZona(
        zona=ZonaBrodmann.B33_BLOQUE_ML,
        tokens={'begin', 'end', 'do', 'done', 'then', 'in', 'where', 'let'},
        peso=0.8, bits=4
    ),
    ZonaBrodmann.B48_INDENT_SPACE: LlavesZona(
        zona=ZonaBrodmann.B48_INDENT_SPACE,
        tokens={'  ', '    ', '        '},  # 2, 4, 8 espacios
        patrones=[r'^[ ]+'],
        peso=0.6, bits=4
    ),
    ZonaBrodmann.B49_INDENT_TAB: LlavesZona(
        zona=ZonaBrodmann.B49_INDENT_TAB,
        tokens={'\t', '\t\t'},
        patrones=[r'^\t+'],
        peso=0.6, bits=4
    ),
    ZonaBrodmann.B50_COMMENT_LINE: LlavesZona(
        zona=ZonaBrodmann.B50_COMMENT_LINE,
        tokens={'//', '#', '--', ';', '%', "'"},
        patrones=[r'^//.*$', r'^#.*$', r'^--.*$'],
        peso=0.5, bits=4
    ),
    ZonaBrodmann.B51_COMMENT_BLOCK: LlavesZona(
        zona=ZonaBrodmann.B51_COMMENT_BLOCK,
        tokens={'/*', '*/', '"""', "'''", '{-', '-}', '=begin', '=end'},
        peso=0.5, bits=4
    ),
    ZonaBrodmann.B52_COMMENT_DOC: LlavesZona(
        zona=ZonaBrodmann.B52_COMMENT_DOC,
        tokens={'///', '/**', '*/', '//!', '#\'', '##'},
        patrones=[r'^///.*$', r'^/\*\*', r'^\s*\*\s'],
        peso=0.6, bits=4
    ),
    ZonaBrodmann.B43_FORMATO_NEWLINE: LlavesZona(
        zona=ZonaBrodmann.B43_FORMATO_NEWLINE,
        tokens={'\n', '\r\n', '\r', '\\n'},
        peso=0.3, bits=4
    ),
    ZonaBrodmann.B16_FORMATO_WHITESPACE: LlavesZona(
        zona=ZonaBrodmann.B16_FORMATO_WHITESPACE,
        tokens={' ', '\t', '▁'},  # El ▁ de SentencePiece
        peso=0.2, bits=4
    ),
}


# =============================================================================
# CLASIFICADOR DE ZONAS DE BRODMANN
# =============================================================================

def normalizar_token_brodmann(token: str) -> str:
    """Normaliza token para clasificación."""
    if not token:
        return token
    # Quitar prefijos de tokenizers
    token = token.replace('▁', '').replace('Ġ', '').replace('Ċ', '')
    token = token.replace('##', '')  # BERT WordPiece
    return token.strip()


class ClasificadorBrodmann:
    """
    Clasificador de tokens a zonas de Brodmann.
    
    Usa LLAVES (reglas explícitas) para clasificación O(1).
    Cada token puede activar múltiples zonas con diferentes pesos.
    """
    
    def __init__(self):
        self.llaves = LLAVES_BRODMANN
        self._cache: Dict[str, Dict[ZonaBrodmann, float]] = {}
        
    def clasificar(self, token: str) -> Dict[ZonaBrodmann, float]:
        """
        Clasifica un token en zonas de Brodmann.
        
        Returns:
            Dict[ZonaBrodmann, float]: Zona -> peso de activación
        """
        # Cache hit
        if token in self._cache:
            return self._cache[token]
        
        token_norm = normalizar_token_brodmann(token)
        token_lower = token_norm.lower() if token_norm else ""
        
        activaciones: Dict[ZonaBrodmann, float] = {}
        
        for zona, llaves in self.llaves.items():
            peso = 0.0
            
            # Coincidencia exacta en tokens
            if token_norm in llaves.tokens or token in llaves.tokens:
                peso = llaves.peso
            elif token_lower in {t.lower() for t in llaves.tokens if isinstance(t, str)}:
                peso = llaves.peso * 0.9  # Pequeña penalización por case mismatch
            
            # Coincidencia por patrón regex
            if peso == 0 and llaves.patrones:
                for patron in llaves.patrones:
                    try:
                        if re.match(patron, token_norm) or re.match(patron, token):
                            peso = llaves.peso * 0.8  # Patterns tienen menos peso
                            break
                    except:
                        continue
            
            if peso > 0:
                activaciones[zona] = peso
        
        # Si no hay activaciones, usar catch-all semántico
        if not activaciones:
            # Verificar si parece identificador
            if token_norm and re.match(r'^[a-z_][a-zA-Z0-9_]*$', token_norm):
                activaciones[ZonaBrodmann.B35_IDENTIFICADOR_VAR] = 0.4
            elif token_norm and re.match(r'^[A-Z][a-zA-Z0-9_]*$', token_norm):
                activaciones[ZonaBrodmann.B34_IDENTIFICADOR_CLASE] = 0.5
            else:
                # Último fallback: whitespace/formato
                activaciones[ZonaBrodmann.B16_FORMATO_WHITESPACE] = 0.2
        
        # Cache
        self._cache[token] = activaciones
        return activaciones
    
    def clasificar_a_territorio(self, token: str) -> Dict[MacroTerritorio, float]:
        """
        Clasifica token y agrega a macro-territorios.
        """
        zonas = self.clasificar(token)
        territorios: Dict[MacroTerritorio, float] = {t: 0.0 for t in MacroTerritorio}
        
        for zona, peso in zonas.items():
            territorio = ZONA_A_TERRITORIO[zona]
            territorios[territorio] = max(territorios[territorio], peso)
        
        return territorios
    
    def zona_principal(self, token: str) -> Tuple[ZonaBrodmann, float]:
        """Retorna la zona principal para un token."""
        zonas = self.clasificar(token)
        if not zonas:
            return ZonaBrodmann.B16_FORMATO_WHITESPACE, 0.1
        zona = max(zonas, key=zonas.get)
        return zona, zonas[zona]


# =============================================================================
# REGISTRO PARA TOKENIZER
# =============================================================================

class RegistroBrodmann:
    """
    Registro que mapea IDs de tokens a zonas de Brodmann.
    
    Pre-computa todas las clasificaciones al registrar el tokenizer
    para O(1) lookup durante inferencia.
    """
    
    def __init__(self, vocab_size: int = 16000):
        self.vocab_size = vocab_size
        self.clasificador = ClasificadorBrodmann()
        
        # Lookup tables
        self._zona_principal: Dict[int, ZonaBrodmann] = {}
        self._peso_principal: Dict[int, float] = {}
        self._territorio_principal: Dict[int, MacroTerritorio] = {}
        self._todas_zonas: Dict[int, Dict[ZonaBrodmann, float]] = {}
        
    def registrar_tokenizer(self, tokenizer) -> Dict[str, int]:
        """
        Registra tokenizer y pre-computa clasificaciones.
        
        Returns:
            Estadísticas de clasificación
        """
        print("\n[BRODMANN] Registrando tokenizer en 52 zonas...")
        
        stats_zonas = {z.name: 0 for z in ZonaBrodmann}
        stats_territorios = {t.name: 0 for t in MacroTerritorio}
        
        vocab_size = tokenizer.vocab_size() if hasattr(tokenizer, 'vocab_size') else self.vocab_size
        
        for token_id in range(min(vocab_size, self.vocab_size)):
            try:
                piece = tokenizer.id_to_piece(token_id) if hasattr(tokenizer, 'id_to_piece') else str(token_id)
                
                # Clasificar
                zonas = self.clasificador.clasificar(piece)
                zona_p, peso_p = self.clasificador.zona_principal(piece)
                territorio_p = ZONA_A_TERRITORIO[zona_p]
                
                # Guardar
                self._zona_principal[token_id] = zona_p
                self._peso_principal[token_id] = peso_p
                self._territorio_principal[token_id] = territorio_p
                self._todas_zonas[token_id] = zonas
                
                # Stats
                stats_zonas[zona_p.name] += 1
                stats_territorios[territorio_p.name] += 1
                
            except Exception as e:
                # Token especial
                self._zona_principal[token_id] = ZonaBrodmann.B16_FORMATO_WHITESPACE
                self._peso_principal[token_id] = 0.1
                self._territorio_principal[token_id] = MacroTerritorio.ESTRUCTURAL
                self._todas_zonas[token_id] = {}
        
        print(f"[OK] {vocab_size} tokens clasificados")
        self._print_stats(stats_zonas, stats_territorios)
        
        return {'zonas': stats_zonas, 'territorios': stats_territorios}
    
    def _print_stats(self, stats_zonas: Dict, stats_territorios: Dict):
        """Imprime estadísticas de distribución."""
        total = sum(stats_territorios.values())
        
        print("\n[STATS] Distribución por Territorio:")
        for territorio in MacroTerritorio:
            count = stats_territorios[territorio.name]
            pct = count / total * 100 if total > 0 else 0
            bar = "#" * int(pct / 2)
            print(f"   {territorio.name:12} {bar:30} {pct:5.1f}% ({count})")
        
        print("\n[STATS] Top 10 Zonas más activas:")
        sorted_zonas = sorted(stats_zonas.items(), key=lambda x: x[1], reverse=True)[:10]
        for nombre, count in sorted_zonas:
            pct = count / total * 100 if total > 0 else 0
            print(f"   {nombre:25} {pct:5.1f}% ({count})")
    
    def get_zona(self, token_id: int) -> Tuple[ZonaBrodmann, float]:
        """Obtiene zona principal para un token ID."""
        return (
            self._zona_principal.get(token_id, ZonaBrodmann.B16_FORMATO_WHITESPACE),
            self._peso_principal.get(token_id, 0.1)
        )
    
    def get_territorio(self, token_id: int) -> MacroTerritorio:
        """Obtiene territorio principal para un token ID."""
        return self._territorio_principal.get(token_id, MacroTerritorio.ESTRUCTURAL)
    
    def get_todas_zonas(self, token_id: int) -> Dict[ZonaBrodmann, float]:
        """Obtiene todas las zonas activas para un token ID."""
        return self._todas_zonas.get(token_id, {})


# =============================================================================
# DEMO
# =============================================================================

def demo_brodmann():
    """Demo del sistema de 52 zonas de Brodmann."""
    print("=" * 70)
    print("🧠 PAMPAr-Coder: Sistema de 52 Zonas de Brodmann")
    print("   Inspirado en las áreas funcionales del cerebro humano")
    print("=" * 70)
    
    # Mostrar distribución de zonas por territorio
    print("\n📊 Distribución de Zonas por Territorio:")
    print("-" * 60)
    
    for territorio in MacroTerritorio:
        zonas = [z for z in ZonaBrodmann if ZONA_A_TERRITORIO[z] == territorio]
        print(f"\n{territorio.name} ({len(zonas)} zonas):")
        for zona in zonas:
            llaves = LLAVES_BRODMANN[zona]
            n_tokens = len(llaves.tokens) if llaves.tokens else "patrón"
            print(f"   {zona.name:25} | {n_tokens:>6} | INT{llaves.bits}")
    
    print("\n" + "-" * 60)
    print(f"Total: {len(ZonaBrodmann)} zonas en {len(MacroTerritorio)} territorios")
    
    # Test de clasificación
    print("\n" + "=" * 70)
    print("🔑 Test de Clasificación")
    print("=" * 70)
    
    clasificador = ClasificadorBrodmann()
    
    test_tokens = [
        # Sintaxis
        'def', 'class', 'import', 'async', '@property',
        '+', '==', '&&', '|', '=',
        'let', '(', '[', '{', ',',
        # Semántica
        'int', 'f64', 'String', 'True', 'Vec',
        'HashMap', 'Option', '"hello"', '42', 'None',
        'myVariable', 'getData', 'MyClass',
        # Lógico
        'if', 'else', 'match', 'for', 'while',
        'map', 'try', 'catch', 'finally', 'throw',
        'return', 'break',
        # Estructural
        ':', '{', '}', '    ', '\t',
        '//', '/*', '///', '\n',
        # Con prefijo SentencePiece
        '▁def', '▁if', '▁class', '▁myVar',
    ]
    
    print(f"\n{'Token':20} {'Zona Principal':30} {'Peso':5} {'Territorio':12}")
    print("-" * 70)
    
    for token in test_tokens:
        zona, peso = clasificador.zona_principal(token)
        territorio = ZONA_A_TERRITORIO[zona]
        print(f"{repr(token):20} {zona.name:30} {peso:5.2f} {territorio.name:12}")


if __name__ == "__main__":
    demo_brodmann()
