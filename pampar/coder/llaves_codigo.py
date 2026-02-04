# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
LLAVES especializadas para código.

Las LLAVES son el corazón del routing en PAMPAr.
Para código, definimos patrones específicos por lenguaje que activan
los territorios correctos instantáneamente.

Sistema 70-80% reglas + 20-30% atención aprendida.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Set, Optional, Tuple
import re


class TipoTerritorioCoder(Enum):
    """Territorios especializados para código."""
    SINTAXIS = auto()      # Keywords, operadores, delimitadores
    SEMANTICA = auto()     # Nombres, strings, comentarios, contexto
    LOGICO = auto()        # Control flow, condiciones, loops
    ESTRUCTURAL = auto()   # Indentación, bloques, scopes, patterns


@dataclass
class LlavesCodigo:
    """
    LLAVES (Linguistic Lexical Anchoring for Vectorized Entry Selection)
    especializado para tokens de código.
    
    Cada token puede activar 1 o más territorios con diferentes pesos.
    """
    
    # =========================================================================
    # TERRITORIO SINTAXIS - Keywords y operadores
    # =========================================================================
    
    # Keywords Python
    keywords_python: Set[str] = field(default_factory=lambda: {
        'def', 'class', 'return', 'yield', 'async', 'await',
        'import', 'from', 'as', 'with', 'lambda', 'global', 'nonlocal',
        'True', 'False', 'None', 'pass', 'break', 'continue',
        'raise', 'assert', 'del', 'in', 'is', 'not', 'and', 'or',
    })
    
    # Keywords JavaScript/TypeScript
    keywords_js: Set[str] = field(default_factory=lambda: {
        'function', 'const', 'let', 'var', 'return', 'yield',
        'async', 'await', 'import', 'export', 'from', 'default',
        'class', 'extends', 'super', 'this', 'new', 'delete', 'typeof', 'instanceof',
        'true', 'false', 'null', 'undefined', 'void',
        'throw', 'try', 'catch', 'finally',
    })
    
    # Keywords Rust
    keywords_rust: Set[str] = field(default_factory=lambda: {
        'fn', 'let', 'mut', 'const', 'static', 'return',
        'struct', 'enum', 'impl', 'trait', 'type', 'where',
        'pub', 'crate', 'mod', 'use', 'as', 'self', 'super',
        'match', 'if', 'else', 'loop', 'while', 'for', 'in', 'break', 'continue',
        'async', 'await', 'move', 'unsafe', 'dyn', 'ref',
    })
    
    # Operadores comunes (todos los lenguajes)
    operadores: Set[str] = field(default_factory=lambda: {
        '+', '-', '*', '/', '%', '**', '//',           # Aritméticos
        '=', '+=', '-=', '*=', '/=', '%=',             # Asignación
        '==', '!=', '<', '>', '<=', '>=',             # Comparación
        '&&', '||', '!', '&', '|', '^', '~',          # Lógicos/Bitwise
        '<<', '>>', '>>>', '&=', '|=', '^=',          # Bitwise assign
        '->', '=>', '::', '.', '..', '...',           # Especiales
        '?', '??', '?.', '?:',                        # Ternario/nullish
    })
    
    # Delimitadores
    delimitadores: Set[str] = field(default_factory=lambda: {
        '(', ')', '[', ']', '{', '}',
        ',', ';', ':', '@', '#',
        '"', "'", '`', '"""', "'''",
    })
    
    # =========================================================================
    # TERRITORIO LOGICO - Control flow
    # =========================================================================
    
    control_flow: Set[str] = field(default_factory=lambda: {
        # Condicionales
        'if', 'else', 'elif', 'switch', 'case', 'default', 'match',
        # Loops
        'for', 'while', 'do', 'loop', 'foreach',
        # Excepciones
        'try', 'except', 'catch', 'finally', 'throw', 'raise',
        # Jumps
        'break', 'continue', 'return', 'yield', 'goto',
    })
    
    # =========================================================================
    # TERRITORIO ESTRUCTURAL - Patrones y estructura
    # =========================================================================
    
    # Patrones de indentación (Python style)
    patrones_indent: List[str] = field(default_factory=lambda: [
        r'^    ',      # 4 espacios
        r'^\t',        # Tab
        r'^        ',  # 8 espacios
    ])
    
    # Patrones de bloques
    patrones_bloque: List[str] = field(default_factory=lambda: [
        r':\s*$',           # Python: fin de línea con :
        r'\{\s*$',          # C-style: abrir bloque
        r'^\s*\}',          # C-style: cerrar bloque
        r'^(def|class|fn)\s+\w+',  # Definición función/clase
    ])
    
    # =========================================================================
    # TERRITORIO SEMANTICA - Contexto y significado
    # =========================================================================
    
    # Patrones semánticos
    patron_string: str = r'["\'].*?["\']|""".*?"""|\'\'\'.*?\'\'\''
    patron_comentario: str = r'#.*$|//.*$|/\*.*?\*/|""".*?"""|\'\'\'.*?\'\'\''
    patron_numero: str = r'\b\d+\.?\d*\b'
    patron_nombre: str = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
    
    # Tipos built-in (semántica fuerte)
    tipos_builtin: Set[str] = field(default_factory=lambda: {
        # Python
        'int', 'float', 'str', 'bool', 'list', 'dict', 'set', 'tuple',
        'bytes', 'None', 'type', 'object',
        # TypeScript/JS
        'number', 'string', 'boolean', 'object', 'array', 'any', 'void',
        'null', 'undefined', 'never', 'unknown',
        # Rust
        'i8', 'i16', 'i32', 'i64', 'i128', 'isize',
        'u8', 'u16', 'u32', 'u64', 'u128', 'usize',
        'f32', 'f64', 'bool', 'char', 'str', 'String',
        'Vec', 'Option', 'Result', 'Box', 'Rc', 'Arc',
    })

    def activar_territorios(self, token: str) -> Dict[TipoTerritorioCoder, float]:
        """
        Determina qué territorios activa un token y con qué peso.
        
        Returns:
            Dict con territorio -> peso de activación [0, 1]
        """
        activaciones = {
            TipoTerritorioCoder.SINTAXIS: 0.0,
            TipoTerritorioCoder.SEMANTICA: 0.0,
            TipoTerritorioCoder.LOGICO: 0.0,
            TipoTerritorioCoder.ESTRUCTURAL: 0.0,
        }
        
        token_lower = token.lower() if token else ""
        
        # =====================================================================
        # SINTAXIS: Keywords y operadores
        # =====================================================================
        if token in self.keywords_python or token in self.keywords_js or token in self.keywords_rust:
            activaciones[TipoTerritorioCoder.SINTAXIS] = 1.0
            
            # Keywords de control flow también activan LOGICO
            if token_lower in self.control_flow:
                activaciones[TipoTerritorioCoder.LOGICO] = 0.8
        
        if token in self.operadores:
            activaciones[TipoTerritorioCoder.SINTAXIS] = 0.9
            # Operadores lógicos activan LOGICO
            if token in {'&&', '||', '!', 'and', 'or', 'not'}:
                activaciones[TipoTerritorioCoder.LOGICO] = 0.7
        
        if token in self.delimitadores:
            activaciones[TipoTerritorioCoder.SINTAXIS] = 0.7
            # Brackets activan ESTRUCTURAL
            if token in {'(', ')', '[', ']', '{', '}'}:
                activaciones[TipoTerritorioCoder.ESTRUCTURAL] = 0.6
        
        # =====================================================================
        # LOGICO: Control flow específico
        # =====================================================================
        if token_lower in self.control_flow:
            activaciones[TipoTerritorioCoder.LOGICO] = 1.0
        
        # =====================================================================
        # ESTRUCTURAL: Indentación y patrones
        # =====================================================================
        for patron in self.patrones_indent:
            if re.match(patron, token):
                activaciones[TipoTerritorioCoder.ESTRUCTURAL] = 0.9
                break
        
        for patron in self.patrones_bloque:
            if re.search(patron, token):
                activaciones[TipoTerritorioCoder.ESTRUCTURAL] = 0.8
                break
        
        # =====================================================================
        # SEMANTICA: Nombres, tipos, literales
        # =====================================================================
        if token in self.tipos_builtin:
            activaciones[TipoTerritorioCoder.SEMANTICA] = 1.0
        
        # Nombres de variables/funciones
        if re.match(self.patron_nombre, token) and token not in self.keywords_python:
            if activaciones[TipoTerritorioCoder.SEMANTICA] == 0.0:
                activaciones[TipoTerritorioCoder.SEMANTICA] = 0.6
        
        # Números
        if re.match(self.patron_numero, token):
            activaciones[TipoTerritorioCoder.SEMANTICA] = 0.5
        
        # Si ningún territorio se activó, default a semántica (catch-all)
        if sum(activaciones.values()) == 0:
            activaciones[TipoTerritorioCoder.SEMANTICA] = 0.3
        
        return activaciones
    
    def clasificar_token(self, token: str) -> Tuple[TipoTerritorioCoder, float]:
        """
        Retorna el territorio principal y su peso.
        """
        activaciones = self.activar_territorios(token)
        territorio_principal = max(activaciones, key=activaciones.get)
        return territorio_principal, activaciones[territorio_principal]


@dataclass
class LlavesCodigoRegistry:
    """
    Registry para mapear tokens de tokenizer a activaciones de territorios.
    
    Se inicializa una vez con el tokenizer y pre-computa las activaciones
    para todos los tokens del vocabulario.
    """
    
    llaves: LlavesCodigo = field(default_factory=LlavesCodigo)
    _cache: Dict[int, Dict[TipoTerritorioCoder, float]] = field(default_factory=dict)
    _tokens: Dict[int, str] = field(default_factory=dict)
    
    def registrar_tokenizer(self, tokenizer) -> None:
        """
        Pre-computa activaciones para todos los tokens del vocabulario.
        """
        print("[LLAVES] Registrando LLAVES para codigo...")
        
        vocab_size = tokenizer.vocab_size() if hasattr(tokenizer, 'vocab_size') else len(tokenizer)
        
        for token_id in range(vocab_size):
            try:
                token_str = tokenizer.id_to_piece(token_id) if hasattr(tokenizer, 'id_to_piece') else tokenizer.decode([token_id])
                self._tokens[token_id] = token_str
                self._cache[token_id] = self.llaves.activar_territorios(token_str)
            except:
                # Token especial o no decodificable
                self._tokens[token_id] = "<unk>"
                self._cache[token_id] = {t: 0.25 for t in TipoTerritorioCoder}
        
        print(f"[OK] {len(self._cache)} tokens registrados")
        self._print_stats()
    
    def _print_stats(self) -> None:
        """Muestra estadísticas de distribución."""
        counts = {t: 0 for t in TipoTerritorioCoder}
        
        for activaciones in self._cache.values():
            principal = max(activaciones, key=activaciones.get)
            counts[principal] += 1
        
        total = len(self._cache)
        print("\n[STATS] Distribucion de territorios:")
        for territorio, count in counts.items():
            pct = count / total * 100
            bar = "#" * int(pct / 5)
            print(f"   {territorio.name:12} {bar:20} {pct:.1f}%")
    
    def get_activaciones(self, token_id: int) -> Dict[TipoTerritorioCoder, float]:
        """Obtiene activaciones pre-computadas para un token."""
        return self._cache.get(token_id, {t: 0.25 for t in TipoTerritorioCoder})
    
    def get_activaciones_batch(self, token_ids: List[int]) -> List[Dict[TipoTerritorioCoder, float]]:
        """Obtiene activaciones para un batch de tokens."""
        return [self.get_activaciones(tid) for tid in token_ids]


# =============================================================================
# Demo
# =============================================================================

def demo_llaves():
    """Demo del sistema de LLAVES."""
    llaves = LlavesCodigo()
    
    test_tokens = [
        'def', 'function', 'fn',           # Keywords
        'if', 'while', 'for', 'match',     # Control flow
        '+', '==', '->', '=>',             # Operadores
        '(', '{', '[', ':',                # Delimitadores
        'int', 'str', 'Vec', 'Option',     # Tipos
        'myVariable', 'calculate_sum',      # Nombres
        '42', '3.14',                        # Números
    ]
    
    print("\n" + "=" * 70)
    print("🔑 PAMPAr-Coder LLAVES Demo")
    print("=" * 70)
    
    for token in test_tokens:
        activaciones = llaves.activar_territorios(token)
        principal, peso = llaves.clasificar_token(token)
        
        # Formatear activaciones
        acts_str = " | ".join(
            f"{t.name[:4]}:{v:.1f}" 
            for t, v in activaciones.items() if v > 0
        )
        
        print(f"'{token:15}' → {principal.name:12} ({peso:.1f})  [{acts_str}]")


if __name__ == "__main__":
    demo_llaves()
