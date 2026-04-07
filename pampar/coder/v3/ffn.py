# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
FFN y modulación contextual para PamparV3.

Componentes:
  StreamFFN        — Feed-forward SwiGLU
  ContextModulator — Modulación FiLM de selectividad mixta (63 indicadores)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ConfigV3


class StreamFFN(nn.Module):
    """
    Feed-forward SwiGLU especializado por stream territorial.

    SwiGLU = SiLU(gate) ⊙ up → down.
    Cada stream (SINTAXIS/SEMANTICA/LOGICO/ESTRUCTURAL) tiene su propio
    conjunto de pesos — como neuronas de áreas corticales distintas.

    hidden_dim = 2/3 × dim × ffn_mult (compensa la gate extra de SwiGLU).
    """

    def __init__(self, config: ConfigV3):
        super().__init__()
        hidden = config.ffn_hidden

        self.gate = nn.Linear(config.dim, hidden, bias=False)
        self.up = nn.Linear(config.dim, hidden, bias=False)
        self.down = nn.Linear(hidden, config.dim, bias=False)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU: SiLU(gate(x)) ⊙ up(x) → down."""
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class ContextModulator(nn.Module):
    """
    Modulador de selectividad mixta — inspirado en neurociencia cortical.

    Un solo FFN compartido codifica el conocimiento (2.2M params).
    Este modulador genera gamma/beta por token usando un vector de contexto
    rico (zona_acts + terr_acts + depth + conf = 63 indicadores) para que
    la MISMA memoria se lea de formas diferentes según el contexto.

    Reemplaza 4 StreamFFN independientes (8.8M/nivel) por:
      1 SharedFFN (2.2M) + 4 modulaciones (gamma, beta) desde contexto (200K)
      = 2.4M/nivel → ahorro de ~6.4M/nivel → ~32M total

    Basado en:
      - FiLM (Perez et al., 2018): Feature-wise Linear Modulation
      - Mixed Selectivity (Rigotti et al., 2013): misma neurona, múltiples roles
      - Superposition (Anthropic, 2022): más conceptos que dimensiones

    El contexto de 63d se compone de:
      zona_acts [52]  — tipo de token (keyword, variable, string...)
      terr_acts [4]   — área dominante
      depth     [1]   — nivel actual (0..n_levels-1), normalizado
      conf      [1]   — confianza del modelo en este punto
      n_levels  [1]   — total de niveles (para normalización)
      stream_id [4]   — one-hot del stream que se está modulando

    Total: 52 + 4 + 1 + 1 + 1 + 4 = 63 indicadores.
    """

    # Dimensión fija del vector de contexto
    CONTEXT_DIM: int = 63

    def __init__(self, config: ConfigV3):
        super().__init__()
        self.dim = config.dim
        mid = config.modulator_bottleneck

        # Contexto (63d) → bottleneck → gamma + beta (dim + dim)
        self.proj = nn.Sequential(
            nn.Linear(self.CONTEXT_DIM, mid, bias=False),
            nn.SiLU(),
            nn.Linear(mid, config.dim * 2, bias=False),
        )

        # Inicializar cerca de identidad: gamma≈1, beta≈0
        nn.init.zeros_(self.proj[2].weight)

    def forward(
        self,
        ffn_out: torch.Tensor,
        zona_acts: torch.Tensor,
        terr_acts: torch.Tensor,
        stream_idx: int,
        nivel_idx: int,
        n_levels: int,
        conf: float,
    ) -> torch.Tensor:
        """
        Modula la salida del FFN compartido según contexto completo.

        Args:
            ffn_out:    [B, L, D] salida del FFN compartido
            zona_acts:  [B, L, 52] activaciones por zona
            terr_acts:  [B, L, 4] activaciones territoriales
            stream_idx: índice del stream actual (0-3)
            nivel_idx:  índice del nivel actual (0-4)
            n_levels:   total de niveles
            conf:       confianza actual (0.0-1.0)

        Returns:
            [B, L, D] salida modulada para este stream/contexto
        """
        B, L, _ = ffn_out.shape
        device = ffn_out.device

        # Construir vector de contexto [B, L, 63]
        depth = torch.full(
            (B, L, 1),
            nivel_idx / max(n_levels - 1, 1),
            device=device,
            dtype=ffn_out.dtype,
        )

        conf_t = torch.full(
            (B, L, 1),
            conf,
            device=device,
            dtype=ffn_out.dtype,
        )

        nl_t = torch.full(
            (B, L, 1),
            n_levels / 10.0,
            device=device,
            dtype=ffn_out.dtype,
        )

        # Stream ID one-hot [B, L, 4]
        stream_oh = torch.zeros(
            B,
            L,
            4,
            device=device,
            dtype=ffn_out.dtype,
        )
        stream_oh[:, :, stream_idx] = 1.0

        # Concatenar: [52] + [4] + [1] + [1] + [1] + [4] = 63
        ctx = torch.cat(
            [zona_acts, terr_acts, depth, conf_t, nl_t, stream_oh],
            dim=-1,
        )  # [B, L, 63]

        # Proyectar a gamma + beta
        modulation = self.proj(ctx)  # [B, L, dim*2]
        gamma, beta = modulation.chunk(2, dim=-1)  # [B, L, dim] cada uno

        # FiLM: gamma modula la escala, beta desplaza
        # +1 para que gamma inicie en identidad (init de proj es zeros)
        return (1.0 + gamma) * ffn_out + beta
