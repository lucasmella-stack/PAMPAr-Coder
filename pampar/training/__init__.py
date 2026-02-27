# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
pampar.training — Sistema de aprendizaje autónomo para PamparV3.

  MotorCuriosidad : selección de temas por ZPD (Vygotsky)
  LectorBiblioteca: carga y tokeniza datos de biblioteca/
"""

from .curiosidad import MotorCuriosidad, PerfilTema
from .lector import LectorBiblioteca

__all__ = ["MotorCuriosidad", "PerfilTema", "LectorBiblioteca"]
