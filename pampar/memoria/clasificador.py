# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Clasificador de importancia con Ley de Pareto.

Principio:
    Del total de interacciones del usuario, solo el 20% contiene
    información suficientemente valuable para retener.
    De ese 20%, solo el 20% (4% total) merece ir a fine-tune.

El clasificador NO usa el modelo PAMPAr para hacer el análisis —
es un pipeline de heurísticas rápidas + embeddings ligeros.

Scoring de importancia:
    - Errores del modelo en ese fragmento (loss alta → más importante)
    - Novedad semántica vs lo que ya está en el RAG
    - Densidad de patrones (cuántos conceptos nuevos por línea)
    - Frecuencia de acceso (si el usuario pregunta lo mismo → importante)
    - Territorio dominante (código con múltiples territorios = más rico)

Umbrales Pareto:
    L1: importancia > 0.3  → guardar en RAG (20% de entradas)
    L2: importancia > 0.6  → marcar como "alta prioridad" (4% de entradas)
    L3: importancia > 0.85 + frecuencia >= 3 → cola de fine-tune (0.8%)
"""

import hashlib
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# =============================================================================
# ESTRUCTURA DE ENTRADA DE MEMORIA
# =============================================================================

@dataclass
class EntradaMemoria:
    """Una unidad de información clasificada para el RAG."""

    # Identificación
    id: str = ""                        # Hash del contenido
    timestamp: float = field(default_factory=time.time)

    # Contenido
    texto: str = ""                     # Fragmento de código/texto
    tipo: str = "codigo"                # "codigo", "error", "concepto", "dialogo"

    # Scoring Pareto
    importancia: float = 0.0           # [0, 1] — umbral L1: 0.3
    novedad: float = 0.0               # Qué tan diferente de lo existente
    densidad: float = 0.0              # Patrones/conceptos por token
    frecuencia: int = 1                # Cuántas veces visto/pedido
    loss_modelo: float = 0.0           # Loss del modelo en este fragmento

    # Nivel Pareto
    nivel: int = 0                      # 0=ignorar, 1=RAG, 2=alta prioridad, 3=finetune

    # Territorio dominante (del Tálamo)
    territorio_dominante: str = ""      # "SINTAXIS", "SEMANTICA", "LOGICO", "ESTRUCTURAL"

    def __post_init__(self):
        if not self.id:
            self.id = hashlib.sha256(self.texto.encode()).hexdigest()[:16]


# =============================================================================
# CLASIFICADOR PARETO
# =============================================================================

class ClasificadorPareto:
    """
    Clasifica fragmentos de código/texto por importancia.

    No usa GPU ni el modelo PAMPAr — es intencional.
    El clasificador debe ser rápido y operable incluso si el modelo
    está ocupado generando tokens.

    Args:
        umbral_l1: Importancia mínima para guardar en RAG (default 0.3)
        umbral_l2: Importancia mínima para alta prioridad (default 0.6)
        umbral_l3: Importancia mínima para cola de fine-tune (default 0.85)
        freq_l3:   Frecuencia mínima para ir a L3 (default 3 veces visto)
    """

    def __init__(
        self,
        umbral_l1: float = 0.30,
        umbral_l2: float = 0.60,
        umbral_l3: float = 0.85,
        freq_l3: int = 3,
    ):
        self.umbral_l1 = umbral_l1
        self.umbral_l2 = umbral_l2
        self.umbral_l3 = umbral_l3
        self.freq_l3 = freq_l3

        # Heurísticas para detectar patrones importantes en código
        self._pat_class = re.compile(r"\bclass\s+\w+")
        self._pat_def = re.compile(r"\bdef\s+\w+\s*\(")
        self._pat_decorator = re.compile(r"@\w+")
        self._pat_import = re.compile(r"\bimport\b|\bfrom\b.*\bimport\b")
        self._pat_exception = re.compile(r"\btry\b|\bexcept\b|\braise\b")
        self._pat_type_hint = re.compile(r":\s*\w+\s*[,\)=]|\s*->\s*\w+")
        self._pat_comprehension = re.compile(r"\[.*\bfor\b.*\bin\b")
        self._pat_lambda = re.compile(r"\blambda\b")
        self._pat_async = re.compile(r"\basync\b|\bawait\b")

    def clasificar(
        self,
        texto: str,
        tipo: str = "codigo",
        loss_modelo: float = 0.0,
        fragmentos_existentes: Optional[List[str]] = None,
    ) -> EntradaMemoria:
        """
        Analiza un fragmento y retorna una EntradaMemoria con scores.

        Args:
            texto:                 El fragmento a clasificar
            tipo:                  Tipo de contenido
            loss_modelo:           Si disponible, la loss del modelo en este fragmento
            fragmentos_existentes: Lista de textos ya en el RAG (para calcular novedad)

        Returns:
            EntradaMemoria clasificada con nivel Pareto asignado
        """
        if not texto.strip():
            return EntradaMemoria(texto=texto, tipo=tipo, nivel=0, importancia=0.0)

        densidad = self._calcular_densidad(texto)
        novedad = self._calcular_novedad(texto, fragmentos_existentes or [])
        riqueza_loss = min(1.0, loss_modelo / 5.0)  # Normalizar loss (5.0 = alta)

        # Score ponderado
        importancia = (
            0.35 * densidad       # ¿Qué tan rico es el código?
            + 0.30 * novedad      # ¿Es algo nuevo para el modelo?
            + 0.25 * riqueza_loss # ¿El modelo tuvo dificultad aquí?
            + 0.10 * min(1.0, len(texto) / 500)  # Fragmentos más largos = más info
        )
        importancia = round(min(1.0, importancia), 4)

        territorio = self._detectar_territorio(texto)

        entrada = EntradaMemoria(
            texto=texto,
            tipo=tipo,
            importancia=importancia,
            novedad=novedad,
            densidad=densidad,
            loss_modelo=loss_modelo,
            territorio_dominante=territorio,
        )
        entrada.nivel = self._asignar_nivel(entrada)
        return entrada

    def _calcular_densidad(self, texto: str) -> float:
        """
        Densidad de patrones avanzados por línea de código.

        Más patrones avanzados = más conceptos = más valioso para aprender.
        """
        lineas = [l for l in texto.splitlines() if l.strip()]
        if not lineas:
            return 0.0

        patrones = [
            self._pat_class, self._pat_def, self._pat_decorator,
            self._pat_exception, self._pat_type_hint,
            self._pat_comprehension, self._pat_lambda, self._pat_async,
        ]

        hits = sum(
            1 for p in patrones for _ in p.finditer(texto)
        )

        # Normalizar: 8+ hits en 10 líneas = densidad máxima
        densidad = min(1.0, hits / max(1, len(lineas)) / 0.8)
        return round(densidad, 4)

    def _calcular_novedad(self, texto: str, existentes: List[str]) -> float:
        """
        Qué tan diferente es este fragmento de lo que ya está en el RAG.

        Heurística simple: overlap de n-gramas de palabras.
        No usa embeddings para mantenerse rápido y offline.
        """
        if not existentes:
            return 1.0

        def ngrams(t: str, n: int = 3) -> set:
            words = re.findall(r"\w+", t.lower())
            return {tuple(words[i:i+n]) for i in range(len(words) - n + 1)}

        ng_nuevo = ngrams(texto)
        if not ng_nuevo:
            return 0.5

        # Overlap promedio con los últimos 20 fragmentos (ventana deslizante)
        muestra = existentes[-20:]
        overlaps = []
        for ex in muestra:
            ng_ex = ngrams(ex)
            if not ng_ex:
                continue
            inter = len(ng_nuevo & ng_ex)
            union = len(ng_nuevo | ng_ex)
            overlaps.append(inter / union if union else 0)

        jaccard_promedio = sum(overlaps) / len(overlaps) if overlaps else 0
        novedad = 1.0 - jaccard_promedio
        return round(novedad, 4)

    def _detectar_territorio(self, texto: str) -> str:
        """Detecta el territorio dominante del fragmento."""
        scores = {
            "SINTAXIS": 0,
            "SEMANTICA": 0,
            "LOGICO": 0,
            "ESTRUCTURAL": 0,
        }
        scores["SINTAXIS"] += len(self._pat_def.findall(texto))
        scores["SINTAXIS"] += len(self._pat_class.findall(texto))
        scores["SINTAXIS"] += len(self._pat_import.findall(texto))
        scores["LOGICO"] += len(self._pat_exception.findall(texto))
        scores["LOGICO"] += len(self._pat_comprehension.findall(texto))
        scores["SEMANTICA"] += len(self._pat_type_hint.findall(texto))
        scores["SEMANTICA"] += len(self._pat_lambda.findall(texto))
        scores["ESTRUCTURAL"] += len(self._pat_decorator.findall(texto))
        scores["ESTRUCTURAL"] += len(self._pat_async.findall(texto))

        return max(scores, key=lambda k: scores[k])

    def _asignar_nivel(self, entrada: EntradaMemoria) -> int:
        """Asigna nivel Pareto: 0=ignorar, 1=RAG, 2=alta prio, 3=finetune."""
        if entrada.importancia < self.umbral_l1:
            return 0
        if entrada.importancia >= self.umbral_l3 and entrada.frecuencia >= self.freq_l3:
            return 3
        if entrada.importancia >= self.umbral_l2:
            return 2
        return 1

    def actualizar_frecuencia(self, entrada: EntradaMemoria) -> EntradaMemoria:
        """
        Incrementa la frecuencia cuando el usuario vuelve a pedir algo similar.

        Puede elevar el nivel Pareto de L1 a L3 si se accede suficientemente.
        """
        entrada.frecuencia += 1
        # Re-score con boost por frecuencia
        freq_boost = min(0.2, math.log(entrada.frecuencia + 1) * 0.05)
        entrada.importancia = min(1.0, entrada.importancia + freq_boost)
        entrada.nivel = self._asignar_nivel(entrada)
        return entrada
