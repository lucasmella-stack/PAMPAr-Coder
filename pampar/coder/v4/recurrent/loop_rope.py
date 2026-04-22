# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Loop-Index Embedding (sinusoidal) para el recurrent loop.

Cada iteración del loop recibe una "marca temporal" sinusoidal, equivalente
a un RoPE/positional embedding pero indexado por el paso del loop t en
lugar de la posición de secuencia.

Por qué: sin esto, el bloque recurrente con peso compartido no sabe en qué
iteración está. El loop_idx también va al ContextModulatorV4 (slot
`loop_idx[1]`) pero ese contexto solo afecta gamma/beta del FiLM. El
embedding sinusoidal aporta una señal aditiva más rica directamente sobre h.

Implementación: idéntica a Vaswani 2017 sinusoidal positional encoding,
proyectada a `dim`. No requiere parámetros si `project=False`; con
`project=True` agrega un Linear(dim→dim) (init zeros, identidad inicial).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoopIndexEmbedding(nn.Module):
    """
    Embedding sinusoidal del índice de loop, opcionalmente proyectado.

    Args:
        dim:        dimensión del modelo.
        max_loops:  máximo de iteraciones esperadas (controla la base de
                    la sinusoidal — análogo a max_seq_len en RoPE).
        project:    si True, agrega Linear(dim, dim) init zeros para que
                    el modelo aprenda cómo modular el efecto del loop_idx.
                    Si False, el embedding se suma directo (sin parámetros).

    Forward:
        Args:
            h:        [B, L, dim] estado oculto.
            loop_idx: int, paso actual del loop (0..max_loops-1).
        Returns:
            [B, L, dim] estado con embedding del loop sumado.
    """

    def __init__(self, dim: int, max_loops: int = 16, project: bool = True):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim debe ser par para sinusoidal embedding, got {dim}")

        self.dim = dim
        self.max_loops = max_loops
        self.project = project

        # Tabla precomputada [max_loops, dim] de sinusoidal embeddings
        pe = self._build_table(max_loops, dim)
        self.register_buffer("pe", pe, persistent=False)

        if project:
            self.proj = nn.Linear(dim, dim, bias=False)
            # Init zeros → modelo recién creado ignora el loop_idx
            # (puede aprender a usarlo durante entrenamiento)
            nn.init.zeros_(self.proj.weight)
        else:
            self.proj = None

    @staticmethod
    def _build_table(max_loops: int, dim: int) -> torch.Tensor:
        """Construye tabla sinusoidal estilo Vaswani 2017."""
        pe = torch.zeros(max_loops, dim)
        position = torch.arange(0, max_loops, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, h: torch.Tensor, loop_idx: int) -> torch.Tensor:
        """
        Suma el embedding del paso `loop_idx` a `h`.

        Args:
            h:        [B, L, dim] estado oculto.
            loop_idx: paso del loop. Debe estar en [0, max_loops).

        Returns:
            [B, L, dim] = h + embedding (proyectado si project=True).
        """
        if not (0 <= loop_idx < self.max_loops):
            raise ValueError(
                f"loop_idx={loop_idx} fuera de rango [0, {self.max_loops})"
            )

        emb = self.pe[loop_idx]  # [dim]
        if self.proj is not None:
            emb = self.proj(emb)  # [dim]

        # Broadcast a [B, L, dim]
        return h + emb.unsqueeze(0).unsqueeze(0)
