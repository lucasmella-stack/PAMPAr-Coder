# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Skills — Capacidades externas del sistema PAMPAr.

Las skills son las "partes del cuerpo" que permiten al cerebro
(PamparV3) interactuar con el mundo real:

    LectorArchivos  → ojos (leer código, archivos, directorios)
    EjecutorCodigo  → manos (ejecutar Python, ver output y errores)
    BuscadorWeb     → biblioteca externa (búsqueda online, opcional)

El runtime/agente.py decide cuándo y cómo llamar a cada skill
basándose en el output del modelo.
"""

from .base import Skill, ResultadoSkill
from .lector_archivos import LectorArchivos
from .ejecutar_codigo import EjecutorCodigo

__all__ = [
    "Skill",
    "ResultadoSkill",
    "LectorArchivos",
    "EjecutorCodigo",
]
