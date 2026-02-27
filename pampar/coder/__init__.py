# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
PAMPAr-Coder: Modelo de lenguaje cerebral para código.

Arquitectura activa: PamparV3 — 108.3M params, vocab 48K.
  - Grilla 2D: 4 streams × 5 niveles
  - TalamoInicial: LLAVES (80% reglas) + atención (20%)
  - GQA 4:1, SwiGLU, lateral gates
  - Early exit (umbral 90%)

Uso:
    from pampar.coder import PamparV3, PRESET_V3

    model = PamparV3(PRESET_V3)

Arquitectura legacy (v2, 42M params, vocab 16K):
    from pampar.coder.deprecated import PampaRCoderV2, ConfigV2, PRESET_4GB
"""

# === Arquitectura activa (v3) ===
from .v3 import (
    ConfigV3,
    PRESET_V3,
    PRESET_V3_SMALL,
    PRESET_V3_LARGE,
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
