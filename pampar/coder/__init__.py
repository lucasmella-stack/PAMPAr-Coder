# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
PAMPAr-Coder: Modelo de lenguaje cerebral para código.

Arquitectura v2 con 52 Zonas de Brodmann:
- 4 Territorios: SINTAXIS, SEMANTICA, LOGICO, ESTRUCTURAL
- 52 Zonas especializadas con routing LLAVES (80% reglas + 20% atención)
- RoPE embeddings + SwiGLU FFN
- Early Exit para inferencia rápida

Uso:
    from pampar.coder import PampaRCoderV2, crear_modelo, PRESET_4GB

    # Crear modelo para GTX 1650
    model = crear_modelo(PRESET_4GB)

    # Con config custom
    from pampar.coder import ConfigV2
    config = ConfigV2(dim=512, n_heads=8, n_capas=8)
    model = PampaRCoderV2(config)
"""

# === Arquitectura canónica (v2 — 52 Zonas de Brodmann) ===
from .v2 import (
    # Config
    ConfigV2,
    PRESET_4GB,
    PRESET_8GB,
    PRESET_24GB,
    PRESET_1_5B,
    # Zonas y territorios
    Zona,
    Territorio,
    ZONAS,
    # LLAVES (routing por reglas)
    LlavesV2,
    clasificar_token,
    # Componentes
    Talamo,
    RMSNorm,
    RoPE,
    BloqueAttn,
    BloqueFFN,
    BloqueTerritorial,
    # Modelo
    PampaRCoderV2,
    crear_modelo,
)

# === Distillation ===
from .distillation import (
    DistillationConfig,
    TeacherAPI,
    DistillationDataCollector,
    DistillationTrainer,
)

__all__ = [
    # Config
    "ConfigV2",
    "PRESET_4GB",
    "PRESET_8GB",
    "PRESET_24GB",
    "PRESET_1_5B",
    # Zonas
    "Zona",
    "Territorio",
    "ZONAS",
    # LLAVES
    "LlavesV2",
    "clasificar_token",
    # Componentes
    "Talamo",
    "RMSNorm",
    "RoPE",
    "BloqueAttn",
    "BloqueFFN",
    "BloqueTerritorial",
    # Modelo
    "PampaRCoderV2",
    "crear_modelo",
    # Distillation
    "DistillationConfig",
    "TeacherAPI",
    "DistillationDataCollector",
    "DistillationTrainer",
]

__version__ = "2.0.0"
