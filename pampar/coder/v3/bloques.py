# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Bloques de la arquitectura 2D de PamparV3 — Hub de re-exportación.

Los componentes viven en módulos separados:
  norm.py  — RMSNorm
  rope.py  — RoPE
  attn.py  — BloqueAttn
  ffn.py   — StreamFFN, ContextModulator
  nivel.py — TalamoNivel, LateralGate, NivelProfundo

Este archivo re-exporta todo para backward compatibility.
"""

from .attn import BloqueAttn
from .ffn import ContextModulator, StreamFFN
from .nivel import LateralGate, NivelProfundo, TalamoNivel
from .norm import RMSNorm
from .rope import RoPE

__all__ = [
    "RMSNorm",
    "RoPE",
    "BloqueAttn",
    "StreamFFN",
    "ContextModulator",
    "TalamoNivel",
    "LateralGate",
    "NivelProfundo",
]
