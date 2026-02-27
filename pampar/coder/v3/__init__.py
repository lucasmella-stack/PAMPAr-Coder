# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""PAMPAr-Coder v3 — Arquitectura 2D con 4 streams × 5 niveles de profundidad."""

from .config import ConfigV3, PRESET_V3
from .modelo import PamparV3, crear_modelo_v3

__all__ = ["ConfigV3", "PRESET_V3", "PamparV3", "crear_modelo_v3"]
