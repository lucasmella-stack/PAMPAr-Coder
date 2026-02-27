# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Skill base — Interfaz común para todas las capacidades externas.

Todas las skills heredan de Skill e implementan execute().
El runtime/agente.py las invoca por nombre sin saber su implementación.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ResultadoSkill:
    """Resultado estandarizado de cualquier skill."""
    exito: bool
    contenido: str                    # Output legible por el modelo
    datos: Dict[str, Any] = field(default_factory=dict)  # Datos estructurados extras
    error: str = ""                   # Mensaje de error si exito=False

    def __str__(self) -> str:
        if self.exito:
            return self.contenido
        return f"[ERROR] {self.error}"


class Skill(ABC):
    """
    Interfaz base para todas las skills externas de PAMPAr.

    Una skill representa una capacidad del modelo para interactuar
    con el mundo real: leer archivos, ejecutar código, buscar web, etc.

    Las skills son síncronas por defecto. El agente las llama en el loop
    de razonamiento del modelo cuando detecta una intención de acción.
    """

    name: str = "skill_base"
    description: str = "Skill base"

    @abstractmethod
    def execute(self, **kwargs) -> ResultadoSkill:
        """
        Ejecuta la skill con los argumentos dados.

        Args:
            **kwargs: Argumentos específicos de cada skill
        Returns:
            ResultadoSkill con el output y estado
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
