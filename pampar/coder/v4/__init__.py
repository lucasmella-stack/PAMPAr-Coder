# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
PAMPAr-Coder v4 — Recurrent-Depth Multimodal Transformer.

Evolución de v3 con:
  - Recurrent loop con peso compartido + LTI-stable injection (futuro)
  - Loop-index RoPE para diferenciar iteraciones (futuro)
  - ACT halting per-token (futuro)
  - Modulators jerárquicos sobre Mixed Selectivity v3
  - Embeddings polimórficos vía ModalityRouter (rieles tendidos para imagen/audio/etc.)

Diseñado para ser multimodal-ready desde el día uno. La versión texto-only
no agrega params respecto a v3; los slots para otras modalidades están
reservados pero no activos.

Estado: scaffold + Fase 1 (contexto enriquecido). v3 sigue siendo producción.
Ver `docs/V4_ARCHITECTURE.md` para el plan completo de evolución.
"""

from .config import ConfigV4
from .ffn import CONTEXT_DIM, ContextModulatorV4, StreamFFN, build_context_v4
from .hierarchical import HierarchicalModulator
from .modalities import (
    NUM_MODALITIES,
    ModalityEncoder,
    ModalityId,
    ModalityRouter,
    TextEncoder,
)
from .modelo import PamparV4
from .recurrent import (
    ACTHalting,
    ACTOutput,
    LoopIndexEmbedding,
    LTIInjection,
    RecurrentBlock,
    RecurrentOutput,
)
from .recurrent_nivel import RecurrentNivelAdapter

__all__ = [
    "ConfigV4",
    "ContextModulatorV4",
    "HierarchicalModulator",
    "ModalityEncoder",
    "ModalityId",
    "ModalityRouter",
    "NUM_MODALITIES",
    "PamparV4",
    "StreamFFN",
    "TextEncoder",
    "CONTEXT_DIM",
    "build_context_v4",
    # Recurrent (Fase 4)
    "ACTHalting",
    "ACTOutput",
    "LoopIndexEmbedding",
    "LTIInjection",
    "RecurrentBlock",
    "RecurrentOutput",
    "RecurrentNivelAdapter",
]
