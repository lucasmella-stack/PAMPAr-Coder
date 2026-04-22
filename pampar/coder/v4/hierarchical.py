# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Modulator jerárquico para PAMPAr v4 (Fase 2).

Diferencia clave con `ContextModulatorV4`:
  - v4 base: cada modulator es independiente. Con 5 niveles × 4 streams =
    20 modulators × 2 capas cada uno = 40 matrices.
  - jerárquico: 1 backbone compartido entre TODOS los (nivel, stream) +
    20 cabezas pequeñas (nivel × stream). El backbone aprende la
    "representación contextual" universal; las cabezas la proyectan a
    (gamma, beta) específicas.

Beneficios:
  - Menos params (1 backbone en vez de N×S backbones).
  - Consistencia entre profundidades: el cerebro entrena UNA representación
    contextual y la reusa, en vez de aprender 20 versiones casi-iguales.
  - Mejor inducción cross-level: los gradientes del backbone se acumulan
    desde todos los niveles, así que aprende patrones más robustos.

Init: las cabezas (último Linear) arrancan en zeros → gamma=0, beta=0 →
identidad numérica. Garantiza estabilidad de entrenamiento al activar
`use_hierarchical_modulators=True`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import ConfigV4
from .ffn import CONTEXT_DIM, build_context_v4
from .modalities import ModalityId


class HierarchicalModulator(nn.Module):
    """
    Modulator FiLM con backbone compartido entre (nivel, stream).

    Arquitectura:
        ctx[71] → Linear(71→bottleneck) → SiLU       # backbone compartido
                ↓
        head[level][stream]: Linear(bottleneck→dim*2)   # head específica

    Total params:
        backbone:  71 × bottleneck                    (compartido)
        heads:     n_levels × n_streams × bottleneck × dim × 2
    vs ContextModulatorV4 independiente (n_levels × n_streams instancias):
        each:      71 × bottleneck + bottleneck × dim × 2
        total:     n_levels × n_streams × (71 × bottleneck + bottleneck × dim × 2)

    Ahorro: (n_levels × n_streams - 1) × 71 × bottleneck params.
    Para defaults v3 (5 × 4 = 20 pares, bottleneck=128):
        ahorro = 19 × 71 × 128 = 172_672 params (~0.27% del modelo)
    El beneficio real no es ahorro de params sino consistencia inductiva.
    """

    def __init__(self, config: ConfigV4):
        super().__init__()
        self.dim = config.dim
        self.n_levels = config.n_levels
        self.n_streams = config.n_streams
        self.bottleneck = config.modulator_bottleneck

        # Backbone compartido: ctx → bottleneck (con SiLU)
        self.backbone = nn.Sequential(
            nn.Linear(CONTEXT_DIM, self.bottleneck, bias=False),
            nn.SiLU(),
        )

        # Cabezas: una Linear(bottleneck → dim*2) por (nivel, stream).
        # Aplanado en n_levels * n_streams para indexar por
        # `level_idx * n_streams + stream_idx`.
        self.heads = nn.ModuleList(
            [
                nn.Linear(self.bottleneck, config.dim * 2, bias=False)
                for _ in range(self.n_levels * self.n_streams)
            ]
        )

        # Init en zeros para identidad inicial (gamma=0, beta=0)
        for head in self.heads:
            nn.init.zeros_(head.weight)

    def _head_idx(self, level_idx: int, stream_idx: int) -> int:
        """Mapea (level, stream) al índice plano de la ModuleList."""
        if not (0 <= level_idx < self.n_levels):
            raise ValueError(f"level_idx {level_idx} fuera de [0, {self.n_levels})")
        if not (0 <= stream_idx < self.n_streams):
            raise ValueError(f"stream_idx {stream_idx} fuera de [0, {self.n_streams})")
        return level_idx * self.n_streams + stream_idx

    def forward(
        self,
        ffn_out: torch.Tensor,
        zona_acts: torch.Tensor,
        terr_acts: torch.Tensor,
        stream_idx: int,
        nivel_idx: int,
        n_levels: int,
        conf: float,
        loop_idx: int = 0,
        max_loops: int = 1,
        modality_id: int = ModalityId.TEXT,
    ) -> torch.Tensor:
        """
        Modula `ffn_out` con FiLM, usando el backbone compartido y la cabeza
        específica de (nivel_idx, stream_idx).

        Args:
            ffn_out:     [B, L, dim] salida del FFN compartido.
            zona_acts:   [B, L, 52] activaciones de zonas.
            terr_acts:   [B, L, 4]  pesos territoriales.
            stream_idx:  stream que se modula (0..n_streams-1).
            nivel_idx:   nivel de profundidad actual (0..n_levels-1).
            n_levels:    total de niveles del modelo (debe coincidir con config).
            conf:        confianza del exit_head (0..1).
            loop_idx:    iteración actual del recurrent loop (0 si N/A).
            max_loops:   total de loops (1 si no hay loop).
            modality_id: tipo de modalidad de los tokens actuales.

        Returns:
            [B, L, dim] salida modulada: (1 + gamma) * ffn_out + beta.
        """
        ctx = build_context_v4(
            zona_acts=zona_acts,
            terr_acts=terr_acts,
            stream_idx=stream_idx,
            nivel_idx=nivel_idx,
            n_levels=n_levels,
            conf=conf,
            loop_idx=loop_idx,
            max_loops=max_loops,
            modality_id=modality_id,
        )

        hidden = self.backbone(ctx)  # [B, L, bottleneck]
        head = self.heads[self._head_idx(nivel_idx, stream_idx)]
        modulation = head(hidden)  # [B, L, 2*dim]

        gamma, beta = modulation.chunk(2, dim=-1)
        return (1.0 + gamma) * ffn_out + beta
