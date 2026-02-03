# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
PAMPAr-Coder: Módulo especializado para código.

Arquitectura territorial adaptada para programación:
- Territorios: SINTAXIS, SEMANTICA, LOGICO, ESTRUCTURAL
- LLAVES específicas para tokens de código (keywords, operadores, etc.)
- Fronteras optimizadas para el flujo de información en código
- Early Exit para inferencia ultra-rápida

Uso:
    from pampar.coder import PampaRCoder, crear_modelo, CODER_4GB
    
    # Crear modelo para GTX 1650
    model = crear_modelo("4GB")
    
    # O con config custom
    model = PampaRCoder(CODER_4GB)
"""

from .config import (
    ConfigPampaRCoder, 
    CODER_4GB,
    CODER_4GB_MAX,
    CODER_8GB, 
    CODER_24GB,
    print_coder_configs
)
from .llaves_codigo import (
    LlavesCodigo,
    LlavesCodigoRegistry, 
    TipoTerritorioCoder
)
from .territorios_codigo import (
    TerritorioCoder,
    FronteraCoder,
    GestorTerritoriosCoder
)
from .model import (
    PampaRCoder,
    TalamoCoder,
    crear_modelo
)

__all__ = [
    # Config
    'ConfigPampaRCoder',
    'CODER_4GB',
    'CODER_8GB', 
    'CODER_24GB',
    'print_coder_configs',
    # LLAVES
    'LlavesCodigo',
    'LlavesCodigoRegistry',
    'TipoTerritorioCoder',
    # Territorios
    'TerritorioCoder',
    'FronteraCoder',
    'GestorTerritoriosCoder',
    # Modelo
    'PampaRCoder',
    'TalamoCoder',
    'crear_modelo',
]

__version__ = "0.1.0"
