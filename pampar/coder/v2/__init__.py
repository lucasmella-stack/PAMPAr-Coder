# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
PAMPAr-Coder v2: Arquitectura con 52 Zonas de Brodmann.

Módulos:
- config: Configuración del modelo
- zonas: Definición de 52 zonas especializadas
- llaves: Sistema de routing basado en reglas
- talamo: Orquestador central
- bloques: Bloques de procesamiento territorial
- modelo: Modelo completo
- distill: Knowledge Distillation

Uso:
    from pampar.coder.v2 import crear_modelo, PRESET_4GB
    modelo = crear_modelo(PRESET_4GB)
"""

from .config import ConfigV2, PRESET_4GB, PRESET_8GB, PRESET_24GB
from .zonas import Zona, Territorio, ZONAS
from .llaves import LlavesV2, clasificar_token
from .talamo import Talamo
from .bloques import BloqueAttn, BloqueFFN, BloqueTerritorial
from .modelo import PampaRCoderV2, crear_modelo

__all__ = [
    # Config
    "ConfigV2", "PRESET_4GB", "PRESET_8GB", "PRESET_24GB",
    # Zonas
    "Zona", "Territorio", "ZONAS",
    # LLAVES
    "LlavesV2", "clasificar_token",
    # Componentes
    "Talamo", "BloqueAttn", "BloqueFFN", "BloqueTerritorial",
    # Modelo
    "PampaRCoderV2", "crear_modelo",
]

__version__ = "2.0.0"
