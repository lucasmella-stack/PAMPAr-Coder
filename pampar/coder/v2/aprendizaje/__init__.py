# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Aprendizaje Cerebral: Paradigma de entrenamiento bio-inspirado.

7 fases que imitan cómo aprende el cerebro humano:
1. Infancia    — Curriculum Learning (simple → complejo)
2. Experimentar — Self-Play con ejecución de código
3. Filosofar   — Knowledge Distillation (profesor → alumno)
4. Sueño       — Consolidación Hebbiana
5. Curiosidad  — Active Learning metacognitivo
6. Social      — Online Learning de interacciones con usuarios
7. Memoria     — Compresión Pareto jerárquica (recordar lo esencial)
"""

from .curriculum import (
    NivelDificultad,
    CurriculumManager,
    clasificar_dificultad,
    crear_curriculum_desde_jsonl,
)

from .self_play import (
    SelfPlayEngine,
    ResultadoEjecucion,
    ejecutar_codigo_seguro,
)

from .neuroplasticidad import (
    ConsolidacionHebbiana,
    ajustar_fronteras,
    podar_pesos,
)

from .metacognicion import (
    MetacognitiveLoss,
    ActiveLearner,
    calcular_calibracion,
)

from .destilacion import (
    ConfigProfesor,
    ClienteProfesor,
    GeneradorDestilacion,
    distillation_loss,
    territory_aware_distillation,
    PROFESORES,
)

from .aprendizaje_online import (
    Interaccion,
    BufferExperiencia,
    EntrenadorOnline,
    crear_servidor,
)

from .memoria_errores import (
    MemoriaErrores,
    EntradaError,
)

from .memoria_jerarquica import (
    MemoriaJerarquica,
    NivelMemoria,
    EntradaMemoria,
)

from .curiosidad import (
    MotorCuriosidad,
    PerfilTema,
)

__all__ = [
    # Curriculum
    "NivelDificultad",
    "CurriculumManager",
    "clasificar_dificultad",
    "crear_curriculum_desde_jsonl",
    # Self-Play
    "SelfPlayEngine",
    "ResultadoEjecucion",
    "ejecutar_codigo_seguro",
    # Neuroplasticidad
    "ConsolidacionHebbiana",
    "ajustar_fronteras",
    "podar_pesos",
    # Metacognición
    "MetacognitiveLoss",
    "ActiveLearner",
    "calcular_calibracion",
    # Destilación
    "ConfigProfesor",
    "ClienteProfesor",
    "GeneradorDestilacion",
    "distillation_loss",
    "territory_aware_distillation",
    "PROFESORES",
    # Online Learning
    "Interaccion",
    "BufferExperiencia",
    "EntrenadorOnline",
    "crear_servidor",
    # Memoria de Errores
    "MemoriaErrores",
    "EntradaError",
    # Memoria Jerárquica (Pareto)
    "MemoriaJerarquica",
    "NivelMemoria",
    "EntradaMemoria",
    # Motor de Curiosidad
    "MotorCuriosidad",
    "PerfilTema",
]
