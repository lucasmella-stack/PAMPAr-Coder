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
_PAT_MAGIC = re.compile(r"^__\w+__$")
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


def clasificar_token(token: str) -> Tuple[Zona, float]:
    """
    Clasifica un token en su zona correspondiente.
    
    Args:
        token: Token a clasificar
        
    Returns:
        (zona, confianza) donde confianza ∈ [0, 1]
    """
    t = normalizar(token)
    
    if not t:
        return Zona.B49_SPACE, 0.5
    
    # 1. Búsqueda exacta en lookup tables
    for zona, patrones in ZONAS.items():
        if t in patrones or t.lower() in patrones:
            return zona, 1.0
    
    # 2. Patrones regex
    if _PAT_INT.match(t):
        return Zona.B21_LIT_INT, 0.95
    
    if _PAT_FLOAT.match(t):
        return Zona.B22_LIT_FLOAT, 0.95
    
    if _PAT_MAGIC.match(t):
        return Zona.B30_MAGIC, 0.9
    
    # 3. Strings (comillas)
    if t.startswith(('"', "'", '`', 'f"', "f'", 'r"', "r'")):
        return Zona.B23_LIT_STR, 0.9
    
    # 4. Identificadores por convención
    if _PAT_UPPER.match(t):
        return Zona.B18_ID_CLASS, 0.7  # CONSTANTE o Clase
    
    if _PAT_CAMEL.match(t):
        return Zona.B18_ID_CLASS, 0.8  # ClassName
    
    if _PAT_ID.match(t):
        return Zona.B16_ID_VAR, 0.6  # variable genérica
    
    # 5. Default: semántica general
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
            # Cuantizado: 4 bits por valor, empaquetados en int8
            # 52 zonas = 26 bytes por token (vs 208 bytes en FP32)
            n_bytes = (n_zonas + 1) // 2
            self.register_buffer(
                "tabla_cuant",
                torch.zeros(vocab_size, n_bytes, dtype=torch.uint8)
            )
            self.register_buffer(
                "escala",
                torch.ones(1)  # Factor de escala para INT4 -> FP
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
        """Guarda valor cuantizado a INT4."""
        # Cuantizar a 4 bits (0-15)
        v_int = int(min(15, max(0, valor * 15)))
        
        byte_idx = zona_idx // 2
        if zona_idx % 2 == 0:
            # Nibble bajo
            self.tabla_cuant[token_id, byte_idx] = (
                (self.tabla_cuant[token_id, byte_idx] & 0xF0) | v_int
            )
        else:
            # Nibble alto
            self.tabla_cuant[token_id, byte_idx] = (
                (self.tabla_cuant[token_id, byte_idx] & 0x0F) | (v_int << 4)
            )
    
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
            # Desempaquetar INT4 a FP
            packed = self.tabla_cuant[token_ids]  # [B, L, n_bytes]
            
            # Separar nibbles
            lo = (packed & 0x0F).float() / 15.0
            hi = ((packed >> 4) & 0x0F).float() / 15.0
            
            # Intercalar
            acts = torch.stack([lo, hi], dim=-1)  # [B, L, n_bytes, 2]
            acts = acts.view(B, L, -1)[:, :, :self.n_zonas]
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
