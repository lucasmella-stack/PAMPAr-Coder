# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
PAMPAr-Coder: Motor de razonamiento puro.

Arquitectura activa: PamparV3 — 108.3M params, vocab 48K.
  - Grilla 2D: 4 streams × 5 niveles
  - TalamoInicial: LLAVES (80% reglas) + atención (20%)
  - GQA 4:1, SwiGLU, lateral gates
  - Early exit (umbral 90%)

Uso:
    from pampar.coder import PamparV3, PRESET_V3

    model = PamparV3(PRESET_V3)
"""

# === Arquitectura activa (v3) ===
from .v3 import (
    PRESET_V3,
    PRESET_V3_LARGE,
    PRESET_V3_SMALL,
    ConfigV3,
    PamparV3,
    crear_modelo_v3,
)

__all__ = [
    # Config v3
    "ConfigV3",
    "PRESET_V3",
    "PRESET_V3_SMALL",
    "PRESET_V3_LARGE",
    # Modelo v3
    "PamparV3",
    "crear_modelo_v3",
]

__version__ = "3.0.0"
