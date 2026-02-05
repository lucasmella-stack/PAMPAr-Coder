# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Micro-Zonas: Arquitectura jerárquica inspirada en áreas de Brodmann.

El cerebro humano tiene 52 áreas de Brodmann especializadas.
Nosotros usamos una versión simplificada:
- 4 Macro-Territorios (como los lóbulos)
- 16-32 Micro-Zonas (como áreas de Brodmann)

Las micro-zonas son CUANTIZABLES porque:
1. Sus LLAVES son determinísticas (reglas fijas)
2. Solo necesitan reconocer un subconjunto pequeño de tokens
3. Pueden usar INT4/INT8 sin pérdida de calidad

Beneficios:
- Más especialización → Mejor routing
- Cuantización → Menos memoria
- Sparse activation → Más velocidad
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from enum import Enum, auto


# =============================================================================
# DEFINICIÓN DE MICRO-ZONAS
# =============================================================================

class MacroTerritorio(Enum):
    """Los 4 macro-territorios (como lóbulos cerebrales)."""
    SINTAXIS = auto()
    SEMANTICA = auto()
    LOGICO = auto()
    ESTRUCTURAL = auto()


class MicroZona(Enum):
    """
    Las 20 micro-zonas especializadas (como áreas de Brodmann).
    
    Cada micro-zona pertenece a un macro-territorio y maneja
    un aspecto muy específico del código.
    """
    # === SINTAXIS (6 zonas) ===
    KEYWORDS_PYTHON = auto()      # def, class, return, import...
    KEYWORDS_JS = auto()          # function, const, let, export...
    KEYWORDS_RUST = auto()        # fn, let, mut, struct, impl...
    OPERADORES_ARIT = auto()      # +, -, *, /, %, **
    OPERADORES_LOGIC = auto()     # ==, !=, <, >, &&, ||
    OPERADORES_ASSIGN = auto()    # =, +=, -=, *=
    
    # === SEMÁNTICA (5 zonas) ===
    TIPOS_PRIMITIVOS = auto()     # int, str, float, bool
    TIPOS_COLECCIONES = auto()    # list, dict, Vec, HashMap
    IDENTIFICADORES = auto()      # variables, funciones, clases
    LITERALES_STRING = auto()     # "hello", 'world', `template`
    LITERALES_NUMERO = auto()     # 42, 3.14, 0xFF
    
    # === LÓGICO (5 zonas) ===
    CONDICIONALES = auto()        # if, else, elif, match, switch
    LOOPS = auto()                # for, while, loop, do
    EXCEPCIONES = auto()          # try, except, catch, finally
    JUMPS = auto()                # break, continue, return, yield
    ASSERTIONS = auto()           # assert, raise, throw
    
    # === ESTRUCTURAL (4 zonas) ===
    BLOQUES_PYTHON = auto()       # :, indentación
    BLOQUES_C_STYLE = auto()      # {, }, ;
    PARENTESIS = auto()           # (, ), [, ]
    DECORADORES = auto()          # @, #, //


# Mapeo de micro-zona a macro-territorio
MICROZONA_A_TERRITORIO: Dict[MicroZona, MacroTerritorio] = {
    # Sintaxis
    MicroZona.KEYWORDS_PYTHON: MacroTerritorio.SINTAXIS,
    MicroZona.KEYWORDS_JS: MacroTerritorio.SINTAXIS,
    MicroZona.KEYWORDS_RUST: MacroTerritorio.SINTAXIS,
    MicroZona.OPERADORES_ARIT: MacroTerritorio.SINTAXIS,
    MicroZona.OPERADORES_LOGIC: MacroTerritorio.SINTAXIS,
    MicroZona.OPERADORES_ASSIGN: MacroTerritorio.SINTAXIS,
    # Semántica
    MicroZona.TIPOS_PRIMITIVOS: MacroTerritorio.SEMANTICA,
    MicroZona.TIPOS_COLECCIONES: MacroTerritorio.SEMANTICA,
    MicroZona.IDENTIFICADORES: MacroTerritorio.SEMANTICA,
    MicroZona.LITERALES_STRING: MacroTerritorio.SEMANTICA,
    MicroZona.LITERALES_NUMERO: MacroTerritorio.SEMANTICA,
    # Lógico
    MicroZona.CONDICIONALES: MacroTerritorio.LOGICO,
    MicroZona.LOOPS: MacroTerritorio.LOGICO,
    MicroZona.EXCEPCIONES: MacroTerritorio.LOGICO,
    MicroZona.JUMPS: MacroTerritorio.LOGICO,
    MicroZona.ASSERTIONS: MacroTerritorio.LOGICO,
    # Estructural
    MicroZona.BLOQUES_PYTHON: MacroTerritorio.ESTRUCTURAL,
    MicroZona.BLOQUES_C_STYLE: MacroTerritorio.ESTRUCTURAL,
    MicroZona.PARENTESIS: MacroTerritorio.ESTRUCTURAL,
    MicroZona.DECORADORES: MacroTerritorio.ESTRUCTURAL,
}


# =============================================================================
# LLAVES PARA MICRO-ZONAS
# =============================================================================

@dataclass
class LlavesMicroZona:
    """
    LLAVES específicas para cada micro-zona.
    
    Cada micro-zona tiene un conjunto pequeño y específico de tokens
    que reconoce. Esto permite cuantización extrema (INT4).
    """
    zona: MicroZona
    tokens: Set[str] = field(default_factory=set)
    peso_activacion: float = 1.0
    
    # Para cuantización
    puede_cuantizar: bool = True
    bits_recomendados: int = 4  # INT4 por defecto


# Definición de todas las LLAVES por micro-zona
LLAVES_MICROZONAS: Dict[MicroZona, LlavesMicroZona] = {
    # === SINTAXIS ===
    MicroZona.KEYWORDS_PYTHON: LlavesMicroZona(
        zona=MicroZona.KEYWORDS_PYTHON,
        tokens={
            'def', 'class', 'return', 'yield', 'async', 'await',
            'import', 'from', 'as', 'with', 'lambda', 'global', 'nonlocal',
            'True', 'False', 'None', 'pass', 'del', 'in', 'is', 'not', 'and', 'or',
        },
        peso_activacion=1.0,
        bits_recomendados=4,
    ),
    MicroZona.KEYWORDS_JS: LlavesMicroZona(
        zona=MicroZona.KEYWORDS_JS,
        tokens={
            'function', 'const', 'let', 'var', 'return', 'yield',
            'async', 'await', 'import', 'export', 'from', 'default',
            'class', 'extends', 'super', 'this', 'new', 'delete', 'typeof', 'instanceof',
            'true', 'false', 'null', 'undefined', 'void',
        },
        peso_activacion=1.0,
        bits_recomendados=4,
    ),
    MicroZona.KEYWORDS_RUST: LlavesMicroZona(
        zona=MicroZona.KEYWORDS_RUST,
        tokens={
            'fn', 'let', 'mut', 'const', 'static', 'return',
            'struct', 'enum', 'impl', 'trait', 'type', 'where',
            'pub', 'crate', 'mod', 'use', 'as', 'self', 'super',
            'move', 'unsafe', 'dyn', 'ref',
        },
        peso_activacion=1.0,
        bits_recomendados=4,
    ),
    MicroZona.OPERADORES_ARIT: LlavesMicroZona(
        zona=MicroZona.OPERADORES_ARIT,
        tokens={'+', '-', '*', '/', '%', '**', '//', '@'},
        peso_activacion=0.9,
        bits_recomendados=4,
    ),
    MicroZona.OPERADORES_LOGIC: LlavesMicroZona(
        zona=MicroZona.OPERADORES_LOGIC,
        tokens={
            '==', '!=', '<', '>', '<=', '>=',
            '&&', '||', '!', '&', '|', '^', '~',
            '<<', '>>', '>>>',
        },
        peso_activacion=0.9,
        bits_recomendados=4,
    ),
    MicroZona.OPERADORES_ASSIGN: LlavesMicroZona(
        zona=MicroZona.OPERADORES_ASSIGN,
        tokens={
            '=', '+=', '-=', '*=', '/=', '%=', '**=', '//=',
            '&=', '|=', '^=', '<<=', '>>=',
        },
        peso_activacion=0.9,
        bits_recomendados=4,
    ),
    
    # === SEMÁNTICA ===
    MicroZona.TIPOS_PRIMITIVOS: LlavesMicroZona(
        zona=MicroZona.TIPOS_PRIMITIVOS,
        tokens={
            # Python
            'int', 'float', 'str', 'bool', 'bytes', 'None',
            # TypeScript/JS
            'number', 'string', 'boolean', 'any', 'void', 'never', 'unknown',
            # Rust
            'i8', 'i16', 'i32', 'i64', 'i128', 'isize',
            'u8', 'u16', 'u32', 'u64', 'u128', 'usize',
            'f32', 'f64', 'bool', 'char',
        },
        peso_activacion=1.0,
        bits_recomendados=4,
    ),
    MicroZona.TIPOS_COLECCIONES: LlavesMicroZona(
        zona=MicroZona.TIPOS_COLECCIONES,
        tokens={
            # Python
            'list', 'dict', 'set', 'tuple', 'frozenset',
            # TypeScript/JS
            'Array', 'Map', 'Set', 'Object',
            # Rust
            'Vec', 'HashMap', 'HashSet', 'BTreeMap', 'BTreeSet',
            'Option', 'Result', 'Box', 'Rc', 'Arc', 'String',
        },
        peso_activacion=1.0,
        bits_recomendados=4,
    ),
    MicroZona.IDENTIFICADORES: LlavesMicroZona(
        zona=MicroZona.IDENTIFICADORES,
        tokens=set(),  # Se detecta por patrón, no por lista
        peso_activacion=0.6,
        puede_cuantizar=False,  # Necesita embeddings completos
        bits_recomendados=8,
    ),
    MicroZona.LITERALES_STRING: LlavesMicroZona(
        zona=MicroZona.LITERALES_STRING,
        tokens={'"', "'", '`', '"""', "'''", 'f"', "f'", 'r"', "r'"},
        peso_activacion=0.7,
        bits_recomendados=8,
    ),
    MicroZona.LITERALES_NUMERO: LlavesMicroZona(
        zona=MicroZona.LITERALES_NUMERO,
        tokens={'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '0x', '0b', '0o'},
        peso_activacion=0.5,
        bits_recomendados=4,
    ),
    
    # === LÓGICO ===
    MicroZona.CONDICIONALES: LlavesMicroZona(
        zona=MicroZona.CONDICIONALES,
        tokens={
            'if', 'else', 'elif', 'switch', 'case', 'default', 'match',
            'If', 'Else', 'Switch', 'Case', 'Default', 'Match',
        },
        peso_activacion=1.0,
        bits_recomendados=4,
    ),
    MicroZona.LOOPS: LlavesMicroZona(
        zona=MicroZona.LOOPS,
        tokens={
            'for', 'while', 'do', 'loop', 'foreach',
            'For', 'While', 'Do', 'Loop',
        },
        peso_activacion=1.0,
        bits_recomendados=4,
    ),
    MicroZona.EXCEPCIONES: LlavesMicroZona(
        zona=MicroZona.EXCEPCIONES,
        tokens={
            'try', 'except', 'catch', 'finally', 'throw', 'raise',
            'Try', 'Catch', 'Finally', 'Throw', 'Raise',
        },
        peso_activacion=1.0,
        bits_recomendados=4,
    ),
    MicroZona.JUMPS: LlavesMicroZona(
        zona=MicroZona.JUMPS,
        tokens={
            'break', 'continue', 'return', 'yield', 'goto',
            'Break', 'Continue', 'Return', 'Yield',
        },
        peso_activacion=1.0,
        bits_recomendados=4,
    ),
    MicroZona.ASSERTIONS: LlavesMicroZona(
        zona=MicroZona.ASSERTIONS,
        tokens={'assert', 'raise', 'throw', 'panic', 'unreachable'},
        peso_activacion=0.8,
        bits_recomendados=4,
    ),
    
    # === ESTRUCTURAL ===
    MicroZona.BLOQUES_PYTHON: LlavesMicroZona(
        zona=MicroZona.BLOQUES_PYTHON,
        tokens={':', '    ', '\t', 'pass'},  # Colon e indentación
        peso_activacion=0.9,
        bits_recomendados=4,
    ),
    MicroZona.BLOQUES_C_STYLE: LlavesMicroZona(
        zona=MicroZona.BLOQUES_C_STYLE,
        tokens={'{', '}', ';'},
        peso_activacion=1.0,
        bits_recomendados=4,
    ),
    MicroZona.PARENTESIS: LlavesMicroZona(
        zona=MicroZona.PARENTESIS,
        tokens={'(', ')', '[', ']', '<', '>'},
        peso_activacion=0.7,
        bits_recomendados=4,
    ),
    MicroZona.DECORADORES: LlavesMicroZona(
        zona=MicroZona.DECORADORES,
        tokens={'@', '#', '//', '/*', '*/', '///', '//!'},
        peso_activacion=0.6,
        bits_recomendados=4,
    ),
}


# =============================================================================
# CUANTIZACIÓN DE MICRO-ZONAS
# =============================================================================

class MicroZonaCuantizada(nn.Module):
    """
    Una micro-zona cuantizada a INT4/INT8.
    
    La idea clave:
    - Las LLAVES son lookup tables → Perfecto para INT4
    - Solo guardamos los pesos de los tokens que esta zona reconoce
    - Activación sparse: solo esta zona se activa para sus tokens
    
    Ejemplo:
    - MicroZona.KEYWORDS_PYTHON tiene ~25 tokens
    - En lugar de 16000 x dim embeddings, solo necesitamos 25 x dim
    - Cuantizado a INT4: 25 x dim x 0.5 bytes
    """
    
    def __init__(
        self,
        zona: MicroZona,
        dim: int,
        vocab_size: int = 16000,
        bits: int = 4,
    ):
        super().__init__()
        self.zona = zona
        self.dim = dim
        self.vocab_size = vocab_size
        self.bits = bits
        
        llaves = LLAVES_MICROZONAS[zona]
        self.tokens = llaves.tokens
        self.peso_activacion = llaves.peso_activacion
        self.territorio = MICROZONA_A_TERRITORIO[zona]
        
        # Lookup table: token_id -> es_reconocido (bool)
        # Esta tabla es BINARIA, no necesita cuantización
        self.register_buffer('token_mask', torch.zeros(vocab_size, dtype=torch.bool))
        
        # Pesos de procesamiento (cuantizados)
        # Solo para tokens reconocidos
        n_tokens = len(self.tokens) if self.tokens else 100  # fallback
        
        if bits == 4:
            # Simular INT4 con FP16 escalado (PyTorch no tiene INT4 nativo)
            self.scale = nn.Parameter(torch.ones(1))
            self.weights = nn.Parameter(torch.randn(n_tokens, dim) * 0.02)
        elif bits == 8:
            self.scale = nn.Parameter(torch.ones(1))
            self.weights = nn.Parameter(torch.randn(n_tokens, dim) * 0.02)
        else:
            self.weights = nn.Parameter(torch.randn(n_tokens, dim) * 0.02)
            self.scale = None
        
        # Mapping interno de token_id -> índice en weights
        self.register_buffer('token_to_idx', torch.full((vocab_size,), -1, dtype=torch.long))
        
    def registrar_tokenizer(self, tokenizer) -> int:
        """
        Registra el tokenizer y llena las tablas de lookup.
        
        Returns:
            Número de tokens reconocidos por esta zona.
        """
        from .llaves_codigo import normalizar_token
        
        idx = 0
        for token_id in range(self.vocab_size):
            try:
                piece = tokenizer.id_to_piece(token_id)
                normalizado = normalizar_token(piece)
                
                if normalizado in self.tokens or piece in self.tokens:
                    self.token_mask[token_id] = True
                    if idx < self.weights.shape[0]:
                        self.token_to_idx[token_id] = idx
                        idx += 1
            except:
                continue
        
        return idx
    
    def forward(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Procesa tokens y retorna activaciones sparse.
        
        Args:
            token_ids: (B, L) IDs de tokens
            
        Returns:
            activacion: (B, L) peso de activación [0, 1]
            mask: (B, L) bool - qué tokens esta zona reconoce
        """
        B, L = token_ids.shape
        
        # Mask de tokens reconocidos
        mask = self.token_mask[token_ids]  # (B, L)
        
        # Activación: peso_activacion donde mask=True, 0 otherwise
        activacion = mask.float() * self.peso_activacion
        
        return activacion, mask
    
    def memory_footprint(self) -> Dict[str, int]:
        """Calcula memoria usada por esta zona."""
        bytes_weights = self.weights.numel() * (self.bits / 8)
        bytes_mask = self.token_mask.numel() / 8  # bits to bytes
        bytes_idx = self.token_to_idx.numel() * 8  # int64
        
        return {
            'weights': int(bytes_weights),
            'mask': int(bytes_mask),
            'idx': int(bytes_idx),
            'total': int(bytes_weights + bytes_mask + bytes_idx),
        }


# =============================================================================
# GESTOR DE MICRO-ZONAS
# =============================================================================

class GestorMicroZonas(nn.Module):
    """
    Gestor de todas las micro-zonas.
    
    Orquesta:
    1. Routing a micro-zonas (sparse)
    2. Agregación a macro-territorios
    3. Comunicación entre zonas del mismo territorio
    
    Beneficios vs arquitectura plana:
    - Más especialización (20 zonas vs 4 territorios)
    - Activación sparse (solo 2-4 zonas por token)
    - Cuantizable (la mayoría INT4)
    - Escalable (agregar nuevas zonas sin reentrenar todo)
    """
    
    def __init__(
        self,
        dim: int,
        vocab_size: int = 16000,
        usar_cuantizacion: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.usar_cuantizacion = usar_cuantizacion
        
        # Crear todas las micro-zonas
        self.microzonas = nn.ModuleDict()
        for zona in MicroZona:
            llaves = LLAVES_MICROZONAS[zona]
            bits = llaves.bits_recomendados if usar_cuantizacion else 16
            
            self.microzonas[zona.name] = MicroZonaCuantizada(
                zona=zona,
                dim=dim,
                vocab_size=vocab_size,
                bits=bits,
            )
        
        # Matriz de agregación: micro-zonas -> macro-territorios
        # Shape: (n_microzonas, n_territorios)
        n_micro = len(MicroZona)
        n_macro = len(MacroTerritorio)
        agregacion = torch.zeros(n_micro, n_macro)
        
        for i, zona in enumerate(MicroZona):
            territorio = MICROZONA_A_TERRITORIO[zona]
            j = list(MacroTerritorio).index(territorio)
            agregacion[i, j] = 1.0
        
        self.register_buffer('matriz_agregacion', agregacion)
        
        # Normalización por territorio
        self.norm_territorios = nn.ModuleDict({
            t.name: nn.LayerNorm(dim) for t in MacroTerritorio
        })
    
    def registrar_tokenizer(self, tokenizer) -> Dict[str, int]:
        """Registra tokenizer en todas las micro-zonas."""
        stats = {}
        total = 0
        
        print("\n[MICRO-ZONAS] Registrando tokenizer...")
        for name, zona in self.microzonas.items():
            n = zona.registrar_tokenizer(tokenizer)
            stats[name] = n
            total += n
        
        print(f"[OK] {total} tokens clasificados en {len(self.microzonas)} micro-zonas")
        return stats
    
    def forward(
        self, 
        token_ids: torch.Tensor
    ) -> Tuple[Dict[MacroTerritorio, torch.Tensor], Dict[MicroZona, torch.Tensor]]:
        """
        Calcula activaciones de micro-zonas y agrega a macro-territorios.
        
        Args:
            token_ids: (B, L)
            
        Returns:
            macro_activaciones: Dict[MacroTerritorio, (B, L)]
            micro_activaciones: Dict[MicroZona, (B, L)]
        """
        B, L = token_ids.shape
        device = token_ids.device
        
        # 1. Calcular activaciones de cada micro-zona
        micro_acts = {}
        micro_tensor = torch.zeros(B, L, len(MicroZona), device=device)
        
        for i, zona in enumerate(MicroZona):
            act, mask = self.microzonas[zona.name](token_ids)
            micro_acts[zona] = act
            micro_tensor[:, :, i] = act
        
        # 2. Agregar a macro-territorios
        # (B, L, n_micro) @ (n_micro, n_macro) -> (B, L, n_macro)
        macro_tensor = torch.matmul(micro_tensor, self.matriz_agregacion)
        
        # Normalizar para que sume ~1 por token
        macro_tensor = macro_tensor / (macro_tensor.sum(dim=-1, keepdim=True) + 1e-8)
        
        macro_acts = {}
        for j, territorio in enumerate(MacroTerritorio):
            macro_acts[territorio] = macro_tensor[:, :, j]
        
        return macro_acts, micro_acts
    
    def memory_report(self) -> Dict[str, any]:
        """Genera reporte de memoria de todas las zonas."""
        total = 0
        by_zona = {}
        by_territorio = {t.name: 0 for t in MacroTerritorio}
        
        for zona in MicroZona:
            mem = self.microzonas[zona.name].memory_footprint()
            by_zona[zona.name] = mem['total']
            total += mem['total']
            
            territorio = MICROZONA_A_TERRITORIO[zona]
            by_territorio[territorio.name] += mem['total']
        
        return {
            'total_bytes': total,
            'total_mb': total / (1024 * 1024),
            'by_zona': by_zona,
            'by_territorio': by_territorio,
        }


# =============================================================================
# DEMO
# =============================================================================

def demo_microzonas():
    """Demo del sistema de micro-zonas."""
    print("=" * 70)
    print("🧠 PAMPAr-Coder: Sistema de Micro-Zonas")
    print("   Inspirado en las 52 Áreas de Brodmann del cerebro humano")
    print("=" * 70)
    
    print("\n📊 Distribución de Micro-Zonas:")
    print("-" * 50)
    
    for territorio in MacroTerritorio:
        zonas = [z for z in MicroZona if MICROZONA_A_TERRITORIO[z] == territorio]
        print(f"\n{territorio.name}:")
        for zona in zonas:
            llaves = LLAVES_MICROZONAS[zona]
            n_tokens = len(llaves.tokens) if llaves.tokens else "∞"
            bits = llaves.bits_recomendados
            print(f"  ├── {zona.name:20} | {n_tokens:>4} tokens | INT{bits}")
    
    print("\n" + "-" * 50)
    print(f"Total: {len(MicroZona)} micro-zonas en {len(MacroTerritorio)} territorios")
    
    # Crear gestor y mostrar memoria
    print("\n💾 Estimación de Memoria (cuantizado):")
    gestor = GestorMicroZonas(dim=256, vocab_size=16000, usar_cuantizacion=True)
    
    report = gestor.memory_report()
    print(f"  Total: {report['total_mb']:.2f} MB")
    print("\n  Por territorio:")
    for t, mem in report['by_territorio'].items():
        print(f"    {t:12}: {mem / 1024:.1f} KB")


if __name__ == "__main__":
    demo_microzonas()
