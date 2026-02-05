# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Tálamo: Orquestador central del modelo.

Combina:
- LLAVES (80%): Routing basado en reglas
- Atención (20%): Routing aprendido

El tálamo decide qué territorios y zonas procesar cada token.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ConfigV2
from .llaves import LlavesV2, agregar_zonas_a_territorios


class Talamo(nn.Module):
    """
    Tálamo v2: Router híbrido reglas + atención.
    
    Arquitectura:
    1. LLAVES clasifica tokens en 52 zonas
    2. Atención aprende routing dinámico
    3. Se combinan: 80% LLAVES + 20% atención
    4. Se agregan zonas a 4 territorios
    """
    
    def __init__(self, config: ConfigV2):
        super().__init__()
        self.config = config
        self.peso_llaves = config.peso_llaves
        
        # LLAVES: lookup tables cuantizadas
        self.llaves = LlavesV2(
            vocab_size=config.vocab_size,
            n_zonas=config.n_zonas,
            usar_cuant=True,
        )
        
        # Atención aprendida: embedding -> zonas
        self.attn_proj = nn.Sequential(
            nn.Linear(config.dim, config.dim // 2),
            nn.GELU(),
            nn.Linear(config.dim // 2, config.n_zonas),
        )
        
        # Proyección final a territorios
        self.terr_gate = nn.Linear(config.n_territorios, config.n_territorios)
    
    def registrar_tokenizer(self, tokenizer) -> int:
        """Registra vocabulario en LLAVES."""
        return self.llaves.registrar_tokenizer(tokenizer)
    
    def forward(
        self,
        x: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calcula routing a territorios y zonas.
        
        Args:
            x: [B, L, D] embeddings
            token_ids: [B, L] IDs de tokens
            
        Returns:
            terr_acts: [B, L, 4] activaciones de territorio
            zona_acts: [B, L, 52] activaciones de zona
        """
        # 1. LLAVES: lookup basado en reglas
        llaves_acts = self.llaves(token_ids)  # [B, L, 52]
        
        # 2. Atención: routing aprendido
        attn_acts = torch.sigmoid(self.attn_proj(x))  # [B, L, 52]
        
        # 3. Combinar: 80% LLAVES + 20% atención
        zona_acts = (
            self.peso_llaves * llaves_acts +
            (1 - self.peso_llaves) * attn_acts
        )
        
        # 4. Agregar a territorios
        terr_acts = agregar_zonas_a_territorios(zona_acts)  # [B, L, 4]
        
        # 5. Gate para territorios
        terr_acts = torch.sigmoid(self.terr_gate(terr_acts))
        
        return terr_acts, zona_acts
