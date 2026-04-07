# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Motor de Curiosidad — "El modelo sabe lo que no sabe".

Concepto: Zona de Desarrollo Próximo de Vygotsky aplicada a IA.
  - Loss baja   → ya lo sabe → aburrimiento → curiosidad baja
  - Loss media  → zona óptima → curiosidad MÁXIMA
  - Loss muy alta → demasiado difícil aún → curiosidad media

curiosidad(tema) = zona_proximal × novedad × urgencia_temporal × bonus_mejora
"""

import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# =============================================================================
# PERFIL DE TEMA
# =============================================================================

@dataclass
class PerfilTema:
    """Estado de aprendizaje del modelo para un tema específico."""

    nombre: str
    categoria: str
    nivel_dificultad: int       # 1-6

    historial_loss: deque = field(default_factory=lambda: deque(maxlen=20))
    primera_vez: float = field(default_factory=time.time)
    ultima_vez: float = field(default_factory=time.time)
    n_sesiones: int = 0
    loss_media: float = 99.0
    tasa_mejora: float = 0.0
    dominado: bool = False
    curiosidad: float = 1.0

    def registrar_sesion(self, loss: float) -> None:
        """Registra una sesión y actualiza estadísticas."""
        self.historial_loss.append(loss)
        self.ultima_vez = time.time()
        self.n_sesiones += 1

        reciente = list(self.historial_loss)[-5:]
        self.loss_media = sum(reciente) / len(reciente)

        # Tasa de mejora: pendiente lineal del historial
        if len(self.historial_loss) >= 3:
            hist = list(self.historial_loss)
            n = len(hist)
            xs = list(range(n))
            x_mean = sum(xs) / n
            y_mean = sum(hist) / n
            num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, hist))
            den = sum((x - x_mean) ** 2 for x in xs) or 1e-8
            self.tasa_mejora = -(num / den)  # Negativo = mejora

        # Dominado si las últimas 5 sesiones están por debajo del umbral de dominio
        if len(self.historial_loss) >= 5:
            self.dominado = all(l < 1.3 for l in list(self.historial_loss)[-5:])

    def tiempo_sin_ver(self) -> float:
        """Horas desde la última sesión."""
        return (time.time() - self.ultima_vez) / 3600.0

    def to_dict(self) -> dict:
        return {
            "nombre": self.nombre,
            "categoria": self.categoria,
            "nivel_dificultad": self.nivel_dificultad,
            "historial_loss": list(self.historial_loss),
            "primera_vez": self.primera_vez,
            "ultima_vez": self.ultima_vez,
            "n_sesiones": self.n_sesiones,
            "loss_media": self.loss_media,
            "tasa_mejora": self.tasa_mejora,
            "dominado": self.dominado,
            "curiosidad": self.curiosidad,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PerfilTema":
        p = cls(
            nombre=d["nombre"],
            categoria=d["categoria"],
            nivel_dificultad=d["nivel_dificultad"],
        )
        p.historial_loss = deque(d.get("historial_loss", []), maxlen=20)
        p.primera_vez = d.get("primera_vez", time.time())
        p.ultima_vez = d.get("ultima_vez", time.time())
        p.n_sesiones = d.get("n_sesiones", 0)
        p.loss_media = d.get("loss_media", 99.0)
        p.tasa_mejora = d.get("tasa_mejora", 0.0)
        p.dominado = d.get("dominado", False)
        p.curiosidad = d.get("curiosidad", 1.0)
        return p


# =============================================================================
# MOTOR DE CURIOSIDAD
# =============================================================================

class MotorCuriosidad:
    """
    Decide qué estudiar a continuación usando curiosidad intrínseca.

    Algoritmo por tema:
      curiosidad = zona_proximal × novedad × temporal × bonus_mejora

    zona_proximal: campana gaussiana centrada en LOSS_OPTIMA (1.5).
    novedad:       1 / (1 + n_sesiones × 0.08) — temas nuevos son más ricos.
    temporal:      spacing effect — temas olvidados suben de prioridad.
    bonus_mejora:  refuerzo si el modelo está mejorando rápido.
    """

    LOSS_OPTIMA: float = 1.5       # Centro de la zona de máxima curiosidad
    LOSS_DOMINIO: float = 1.3       # Bajo este umbral el modelo ya dominó el tema
    LOSS_MUY_DIFICIL: float = 5.0  # Sobre este umbral el tema está fuera del ZPD

    def __init__(
        self,
        ruta_estado: Optional[Path] = None,
        nivel_actual: int = 1,
    ) -> None:
        self.ruta_estado = ruta_estado
        self.nivel_actual = nivel_actual
        self.temas: Dict[str, PerfilTema] = {}
        self.cola_reciente: deque = deque(maxlen=5)
        self.sesiones_totales: int = 0
        self.temas_dominados: int = 0

        if ruta_estado and Path(ruta_estado).exists():
            self.cargar(ruta_estado)

    # ── Registro ──────────────────────────────────────────────────────────────

    def registrar_tema(self, nombre: str, categoria: str, nivel: int) -> None:
        """Registra un tema nuevo si no existe."""
        if nombre not in self.temas:
            self.temas[nombre] = PerfilTema(
                nombre=nombre,
                categoria=categoria,
                nivel_dificultad=nivel,
            )

    def registrar_temas_desde_indice(self, indice: dict) -> int:
        """Carga todos los temas del índice JSON de biblioteca/. Devuelve nº nuevos."""
        nuevos = 0
        for categoria, temas in indice.items():
            if not isinstance(temas, list):
                continue
            for tema in temas:
                nombre = tema["nombre"]
                if nombre not in self.temas:
                    self.registrar_tema(nombre, categoria, tema.get("nivel", 1))
                    nuevos += 1
        return nuevos

    # ── Cálculo de curiosidad ─────────────────────────────────────────────────

    def _zpd(self, loss: float) -> float:
        """Campana gaussiana en LOSS_OPTIMA."""
        if loss < self.LOSS_DOMINIO:
            return 0.1
        if loss > self.LOSS_MUY_DIFICIL:
            return 0.2
        sigma = 0.8
        z = (loss - self.LOSS_OPTIMA) / sigma
        return math.exp(-0.5 * z * z)

    def calcular_curiosidad(self, tema: PerfilTema) -> float:
        """Score de curiosidad [0, ∞). Mayor = más prioritario."""
        if tema.n_sesiones <= 5:
            zpd = max(0.18, 0.6 - tema.n_sesiones * 0.07)
            nivel_ok = 0.5
        else:
            zpd = self._zpd(tema.loss_media)
            nivel_ok = 1.0 if tema.nivel_dificultad <= self.nivel_actual + 1 else 0.3

        novedad = 1.0 / (1.0 + tema.n_sesiones * 0.08)
        temporal = 1.0 + math.log1p(tema.tiempo_sin_ver() / 12.0) * 0.5

        if tema.tasa_mejora > 0.05:
            bonus = 1.3
        elif tema.tasa_mejora < -0.02:
            bonus = 0.8
        else:
            bonus = 1.0

        curiosidad = zpd * novedad * temporal * bonus * nivel_ok
        tema.curiosidad = curiosidad
        return curiosidad

    def actualizar_todos(self) -> None:
        for tema in self.temas.values():
            self.calcular_curiosidad(tema)

    # ── Selección ─────────────────────────────────────────────────────────────

    def siguiente_tema(self, excluir_recientes: bool = True) -> Optional[str]:
        """Devuelve el tema más curioso. Usa muestreo probabilístico top-3."""
        self.actualizar_todos()

        candidatos = list(self.temas.items())
        if excluir_recientes:
            candidatos = [(n, t) for n, t in candidatos if n not in self.cola_reciente]
        if not candidatos:
            candidatos = list(self.temas.items())
        if not candidatos:
            return None

        candidatos.sort(key=lambda x: x[1].curiosidad, reverse=True)
        top = candidatos[:min(3, len(candidatos))]
        scores = [t.curiosidad for _, t in top]
        total = sum(scores) or 1.0

        rand = torch.rand(1).item() * total
        acum, elegido = 0.0, top[0][0]
        for nombre, tema in top:
            acum += tema.curiosidad
            if rand <= acum:
                elegido = nombre
                break

        self.cola_reciente.append(elegido)
        return elegido

    def tops(self, n: int = 5) -> List[Tuple[str, float]]:
        """Top N temas por curiosidad."""
        self.actualizar_todos()
        ordenados = sorted(self.temas.items(), key=lambda x: x[1].curiosidad, reverse=True)
        return [(nombre, tema.curiosidad) for nombre, tema in ordenados[:n]]

    # ── Retroalimentación ─────────────────────────────────────────────────────

    def retroalimentar(self, nombre_tema: str, loss: float) -> dict:
        """Actualiza el perfil de un tema tras una sesión."""
        if nombre_tema not in self.temas:
            return {}

        tema = self.temas[nombre_tema]
        loss_anterior = tema.loss_media
        era_dominado = tema.dominado
        tema.registrar_sesion(loss)
        self.sesiones_totales += 1

        recien_dominado = tema.dominado and not era_dominado
        if recien_dominado:
            self.temas_dominados += 1
            self._verificar_avance_nivel()

        return {
            "tema": nombre_tema,
            "loss_anterior": loss_anterior,
            "loss_actual": tema.loss_media,
            "mejora": loss_anterior - tema.loss_media,
            "dominado": tema.dominado,
            "recien_dominado": recien_dominado,
            "nivel_actual": self.nivel_actual,
        }

    def _verificar_avance_nivel(self) -> bool:
        """Sube de nivel si ≥70% de los temas del nivel actual están dominados."""
        temas_nivel = [t for t in self.temas.values() if t.nivel_dificultad == self.nivel_actual]
        if not temas_nivel:
            return False
        dominados = sum(1 for t in temas_nivel if t.dominado)
        if dominados / len(temas_nivel) >= 0.70 and self.nivel_actual < 6:
            self.nivel_actual += 1
            return True
        return False

    # ── Resumen ───────────────────────────────────────────────────────────────

    def resumen(self) -> dict:
        total = len(self.temas)
        dominados = sum(1 for t in self.temas.values() if t.dominado)
        en_progreso = sum(1 for t in self.temas.values() if not t.dominado and t.n_sesiones > 0)
        loss_global = sum(t.loss_media for t in self.temas.values()) / total if total else 0.0
        return {
            "nivel_actual": self.nivel_actual,
            "sesiones_totales": self.sesiones_totales,
            "temas_total": total,
            "temas_dominados": dominados,
            "temas_en_progreso": en_progreso,
            "temas_intactos": total - dominados - en_progreso,
            "porcentaje_dominio": (dominados / total * 100) if total else 0.0,
            "loss_promedio_global": loss_global,
            "tops_curiosidad": self.tops(3),
        }

    def __repr__(self) -> str:
        r = self.resumen()
        return (
            f"MotorCuriosidad(nivel={r['nivel_actual']}, "
            f"dominados={r['temas_dominados']}/{r['temas_total']}, "
            f"loss={r['loss_promedio_global']:.2f})"
        )

    # ── Persistencia ──────────────────────────────────────────────────────────

    def guardar(self, ruta: Optional[Path] = None) -> None:
        """Guarda el estado completo en JSON."""
        ruta = Path(ruta or self.ruta_estado)
        if ruta is None:
            return
        ruta.parent.mkdir(parents=True, exist_ok=True)
        estado = {
            "nivel_actual": self.nivel_actual,
            "sesiones_totales": self.sesiones_totales,
            "temas_dominados": self.temas_dominados,
            "cola_reciente": list(self.cola_reciente),
            "temas": {n: t.to_dict() for n, t in self.temas.items()},
        }
        ruta.write_text(json.dumps(estado, indent=2, ensure_ascii=False))

    def cargar(self, ruta: Optional[Path] = None) -> None:
        """Carga el estado desde JSON."""
        ruta = Path(ruta or self.ruta_estado)
        if not ruta.exists():
            return
        estado = json.loads(ruta.read_text())
        self.nivel_actual = estado.get("nivel_actual", 1)
        self.sesiones_totales = estado.get("sesiones_totales", 0)
        self.temas_dominados = estado.get("temas_dominados", 0)
        self.cola_reciente = deque(estado.get("cola_reciente", []), maxlen=5)
        self.temas = {n: PerfilTema.from_dict(d) for n, d in estado.get("temas", {}).items()}
