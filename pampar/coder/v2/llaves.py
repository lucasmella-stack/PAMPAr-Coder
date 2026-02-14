# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Sistema LLAVES v2: Routing de tokens a zonas.

LLAVES = "Lookup tables + Learned Attention + Vectorized Evaluation System"

80% basado en reglas (lookup tables) + 20% atención aprendida.
Las tablas se cuantizan a INT4 para ahorrar memoria.
"""

import re
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .zonas import Zona, Territorio, ZONAS, ZONA_TERRITORIO


# =============================================================================
# NORMALIZACIÓN DE TOKENS
# =============================================================================

# Prefijos de tokenizers (SentencePiece, BPE, etc.)
_PREFIXES = ("▁", "Ġ", "Ċ", "##", "Ã", "Â")

# Patrones para clasificación
_PAT_INT = re.compile(r"^-?\d+$")
_PAT_FLOAT = re.compile(r"^-?\d+\.\d*$")
_PAT_UPPER = re.compile(r"^[A-Z][A-Z0-9_]*$")
_PAT_CAMEL = re.compile(r"^[A-Z][a-z]+")
_PAT_ID = re.compile(r"^[a-z_][a-z0-9_]*$", re.IGNORECASE)


def normalizar(token: str) -> str:
    """
    Normaliza token removiendo prefijos de tokenizer.
    
    Args:
        token: Token crudo del tokenizer
        
    Returns:
        Token limpio para clasificación
    """
    t = token
    for p in _PREFIXES:
        if t.startswith(p):
            t = t[len(p):]
    return t


# Patrones adicionales para Python
_PAT_DUNDER = re.compile(r"^__[a-z][a-z0-9_]*__$")
_PAT_SNAKE = re.compile(r"^[a-z][a-z0-9_]*$")
_PAT_DECORATOR = re.compile(r"^@[a-zA-Z]")
_PAT_FSTRING = re.compile(r"^[fFrRbB]['\"]")


def clasificar_token(token: str) -> Tuple[Zona, float]:
    """
    Clasifica un token en su zona correspondiente.
    
    Orden de prioridad:
    1. Exact match en ZONAS (keywords, operadores, delimitadores)
    2. Regex patterns (números, magic methods, strings)
    3. Convenciones de naming (CamelCase, snake_case, UPPER_CASE)
    4. Default: variable genérica
    
    Args:
        token: Token a clasificar
        
    Returns:
        (zona, confianza) donde confianza ∈ [0, 1]
    """
    t = normalizar(token)
    
    if not t:
        return Zona.B49_SPACE, 0.5
    
    # — Whitespace puro —
    if t in ("\n", "\r\n"):
        return Zona.B48_NEWLINE, 1.0
    if t in ("\t", "    "):
        return Zona.B47_INDENT, 1.0
    if t.strip() == "":
        return Zona.B49_SPACE, 0.8
    
    # 1. Búsqueda exacta en lookup tables (solo zonas no vacías)
    for zona, patrones in ZONAS.items():
        if not patrones:
            continue
        if t in patrones or t.lower() in patrones:
            return zona, 1.0
    
    # 2. Dunder / magic methods
    if _PAT_DUNDER.match(t):
        return Zona.B30_MAGIC, 0.95
    
    # 3. Números
    if _PAT_INT.match(t):
        return Zona.B21_LIT_INT, 0.95
    if _PAT_FLOAT.match(t):
        return Zona.B22_LIT_FLOAT, 0.95
    
    # 4. Strings (comillas, f-strings)
    if _PAT_FSTRING.match(t):
        return Zona.B23_LIT_STR, 0.9
    if t.startswith(('"', "'", '`')):
        return Zona.B23_LIT_STR, 0.9
    
    # 5. Decorators (@)
    if _PAT_DECORATOR.match(t):
        return Zona.B29_BUILTIN, 0.7
    
    # 6. Identificadores por convención de naming
    if _PAT_UPPER.match(t) and len(t) > 1:
        # ALL_CAPS = constante o excepción (ej: HTTP_ERROR)
        return Zona.B18_ID_CLASS, 0.7
    
    if _PAT_CAMEL.match(t) and len(t) > 1:
        # CamelCase = clase
        return Zona.B18_ID_CLASS, 0.85
    
    if _PAT_SNAKE.match(t):
        # snake_case = variable o función (contexto distingue)
        return Zona.B16_ID_VAR, 0.6
    
    if _PAT_ID.match(t):
        return Zona.B16_ID_VAR, 0.5
    
    # 7. Default: semántica general con baja confianza
    return Zona.B16_ID_VAR, 0.3


# =============================================================================
# LLAVES CON LOOKUP CUANTIZADO
# =============================================================================

class LlavesV2(nn.Module):
    """
    Sistema LLAVES v2 con lookup tables cuantizadas.
    
    Usa INT4 para almacenar las activaciones de zona por token,
    reduciendo memoria 8x vs FP32.
    """
    
    def __init__(
        self,
        vocab_size: int,
        n_zonas: int = 52,
        usar_cuant: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_zonas = n_zonas
        self.usar_cuant = usar_cuant
        
        # Tabla de lookup: vocab_size x n_zonas
        # Almacena la activación de cada zona para cada token
        if usar_cuant:
            # Cuantizado: 8 bits por valor (INT8)
            # 52 zonas = 52 bytes por token (vs 208 bytes en FP32)
            # Resolución: 256 niveles (error <0.4% vs INT4 ~11%)
            self.register_buffer(
                "tabla_cuant",
                torch.zeros(vocab_size, n_zonas, dtype=torch.uint8)
            )
        else:
            self.register_buffer(
                "tabla",
                torch.zeros(vocab_size, n_zonas)
            )
        
        # Estadísticas
        self.tokens_registrados = 0
    
    def registrar_tokenizer(self, tokenizer) -> int:
        """
        Llena la tabla de lookup con el vocabulario del tokenizer.
        
        Args:
            tokenizer: SentencePiece tokenizer
            
        Returns:
            Número de tokens registrados
        """
        count = 0
        
        for token_id in range(min(self.vocab_size, tokenizer.GetPieceSize())):
            token = tokenizer.IdToPiece(token_id)
            zona, conf = clasificar_token(token)
            
            # Guardar activación
            if self.usar_cuant:
                self._set_cuant(token_id, zona.value - 1, conf)
            else:
                self.tabla[token_id, zona.value - 1] = conf
            
            count += 1
        
        self.tokens_registrados = count
        return count
    
    def _set_cuant(self, token_id: int, zona_idx: int, valor: float):
        """Guarda valor cuantizado a INT8 (256 niveles, error <0.4%)."""
        v_int = int(min(255, max(0, valor * 255)))
        self.tabla_cuant[token_id, zona_idx] = v_int
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Obtiene activaciones de zona para tokens.
        
        Args:
            token_ids: [B, L] tensor de IDs
            
        Returns:
            [B, L, n_zonas] activaciones por zona
        """
        B, L = token_ids.shape
        
        if self.usar_cuant:
            # INT8 → float: simple y preciso (256 niveles)
            acts = self.tabla_cuant[token_ids].float() / 255.0  # [B, L, n_zonas]
        else:
            acts = self.tabla[token_ids]
        
        return acts


# =============================================================================
# FUNCIONES DE UTILIDAD
# =============================================================================

def zona_a_territorio(zona: Zona) -> Territorio:
    """Convierte zona a territorio."""
    return ZONA_TERRITORIO[zona]


def agregar_zonas_a_territorios(
    zona_acts: torch.Tensor,
) -> torch.Tensor:
    """
    Agrega activaciones de 52 zonas a 4 territorios.
    
    Args:
        zona_acts: [B, L, 52] activaciones por zona
        
    Returns:
        [B, L, 4] activaciones por territorio
    """
    B, L, Z = zona_acts.shape
    
    # Índices de zonas por territorio
    indices = [
        list(range(0, 15)),   # SINTAXIS: B01-B15
        list(range(15, 30)),  # SEMANTICA: B16-B30
        list(range(30, 42)),  # LOGICO: B31-B42
        list(range(42, 52)),  # ESTRUCTURAL: B43-B52
    ]
    
    terr_acts = torch.zeros(B, L, 4, device=zona_acts.device)
    
    for t, idx in enumerate(indices):
        terr_acts[:, :, t] = zona_acts[:, :, idx].mean(dim=-1)
    
    return terr_acts
