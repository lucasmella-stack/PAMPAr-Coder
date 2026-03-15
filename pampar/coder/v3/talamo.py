# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Tálamo inicial de PamparV3.

El Tálamo hace el routing de entrada: dado un token, determina
qué streams (territorios) y zonas deben procesar ese token con qué
peso.

Dos componentes:
  TalamoInicial — routing completo al inicio del forward
      LLAVES (80%, reglas INT8) + attn_proj (20%, aprendido)
      + context_conv causal (contextualización del vecindario)

  TalamoNivel   — re-routing ligero en cada NivelProfundo
      (definido en bloques.py, solo usa el estado actual x)

La función agregar_zonas_a_territorios es compartida con v2 —
no tiene sentido duplicarla, el mapeo Zona → Territorio es el mismo.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .config import ConfigV3
from .llaves import LlavesV2, agregar_zonas_a_territorios


class TalamoInicial(nn.Module):
    """
    Routing inicial de tokens a zonas y territorios.

    Igual que en v2 pero ajustado a dim=640 de v3.

    80% reglas (LLAVES: lookup tables INT8 sobre vocabulario) +
    20% aprendido (attn_proj: proyección lineal sobre embeddings).

    El context_conv causal evita routing "ciego":
    ve la ventana de 32 tokens anteriores para saber si "for" es
    un loop o parte de "formula".

    La función agregar_fn() está expuesta como método para poder
    pasarla a TalamoNivel en cada NivelProfundo sin hardcoding.
    """

    def __init__(self, config: ConfigV3):
        super().__init__()
        self.config = config
        self.peso_llaves = config.peso_llaves
        self.ventana = config.ventana_contexto

        # LLAVES: sistema de lookup rules sobre token_ids (sin parámetros)
        self.llaves = LlavesV2(config.vocab_size, config.n_zonas)

        # Routing aprendido: embeddings → zonas
        # Bottleneck para mantener params bajos
        mid = config.dim // 2  # 320 para dim=640
        self.attn_proj = nn.Sequential(
            nn.Linear(config.dim, mid, bias=False),
            nn.GELU(),
            nn.Linear(mid, config.n_zonas, bias=False),
        )

        # Contextualización causal: cada token ve su vecindario anterior
        # Groups=n_zonas → depthwise conv (muy barato en params)
        self.context_conv = nn.Conv1d(
            in_channels=config.n_zonas,
            out_channels=config.n_zonas,
            kernel_size=config.ventana_contexto,
            groups=config.n_zonas,
            bias=False,
        )

        # Gate final de territorios
        self.terr_gate = nn.Linear(
            config.n_territorios, config.n_territorios, bias=False
        )

        # Tokenizer (registrado después de crear el modelo)
        self._tokenizer: Optional[object] = None

    def registrar_tokenizer(self, tokenizer: object) -> None:
        """Registra el tokenizer para que LLAVES pueda usarlo."""
        self._tokenizer = tokenizer
        if hasattr(self.llaves, "registrar_tokenizer"):
            self.llaves.registrar_tokenizer(tokenizer)

    @staticmethod
    def agregar_fn(zona_acts: torch.Tensor) -> torch.Tensor:
        """
        Wrapper estático de agregar_zonas_a_territorios.
        Se pasa como callable a TalamoNivel en cada nivel.
        """
        return agregar_zonas_a_territorios(zona_acts)

    def forward(
        self,
        x: torch.Tensor,         # [B, L, D] embeddings
        token_ids: torch.Tensor,  # [B, L] IDs de tokens
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computa activaciones de zonas y territorios para la entrada.

        Returns:
            terr_acts: [B, L, n_territorios]
            zona_acts: [B, L, n_zonas]
        """
        # 1. LLAVES: lookup basado en reglas sobre token_ids
        llaves_acts = self.llaves(token_ids)          # [B, L, 52]

        # 2. Atención aprendida: embeddings → zonas
        attn_acts = torch.sigmoid(self.attn_proj(x))  # [B, L, 52]

        # 3. Combinar: 80% LLAVES (estables desde epoch 0) + 20% aprendido
        zona_acts = self.peso_llaves * llaves_acts + (1 - self.peso_llaves) * attn_acts

        # 4. Contextualización causal
        #    Pad izquierda (causal: solo ve pasado)
        zona_ctx = F.pad(
            zona_acts.transpose(1, 2),   # [B, 52, L]
            (self.ventana - 1, 0),
        )
        zona_acts = self.context_conv(zona_ctx).transpose(1, 2)  # [B, L, 52]

        # 5. Agregar zonas → territorios
        terr_acts = self.agregar_fn(zona_acts)  # [B, L, 4]

        # 6. Gate sobre territorios (aprende escala por territorio)
        terr_acts = torch.sigmoid(self.terr_gate(terr_acts))

        return terr_acts, zona_acts
