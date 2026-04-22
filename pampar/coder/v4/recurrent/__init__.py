# SPDX-License-Identifier: BUSL-1.1
"""
Recurrent-Depth components para PAMPAr v4 (Fase 4).

Subpaquete con la maquinaria del loop recurrente:
  - LTIInjection      — input injection LTI-stable (ρ(A)<1 por construcción)
  - LoopIndexEmbedding — embedding sinusoidal del índice de loop
  - ACTHalting        — adaptive computation time per-token
  - RecurrentBlock    — orquestador Prelude → step×T → Coda con early exit

Hoy son componentes independientes con tests propios. Se cablean al modelo
en Fase 5 vía flag `ConfigV4.use_recurrent_loop=True`.
"""

from .act import ACTHalting, ACTOutput
from .block import RecurrentBlock, RecurrentOutput
from .loop_rope import LoopIndexEmbedding
from .lti import LTIInjection

__all__ = [
    "ACTHalting",
    "ACTOutput",
    "LoopIndexEmbedding",
    "LTIInjection",
    "RecurrentBlock",
    "RecurrentOutput",
]
