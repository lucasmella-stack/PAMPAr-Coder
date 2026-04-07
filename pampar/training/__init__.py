# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
pampar.training — Sistema de aprendizaje autónomo para PamparV3.

  MotorCuriosidad     : selección de temas por ZPD (Vygotsky)
  LectorBiblioteca    : carga y tokeniza datos de biblioteca/
  MemoriaJerarquica   : memoria L0/L1/L2 para replay anti-olvido
"""

from .curiosidad import MotorCuriosidad, PerfilTema
from .lector import LectorBiblioteca
from .memoria_jerarquica import MemoriaJerarquica

__all__ = ["MotorCuriosidad", "PerfilTema", "LectorBiblioteca", "MemoriaJerarquica"]
