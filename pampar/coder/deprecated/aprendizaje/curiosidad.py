# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Motor de Curiosidad — "El modelo sabe lo que no sabe".

Concepto central: Zona de Desarrollo Próximo (Vygotsky aplicado a IA).
  - Demasiado fácil  → pérdida baja → el modelo ya lo sabe → aburrimiento
  - Demasiado difícil → pérdida muy alta → el modelo no entiende nada → frustración
  - ZONA ÓPTIMA      → pérdida media-alta → puede aprender → CURIOSIDAD MÁXIMA

Curiosidad(tema) = f(dificultad_actual, tasa_de_mejora, novedad, tiempo_sin_ver)

El motor:
  1. Mide la pérdida del modelo por TEMA en la biblioteca
  2. Calcula la "curiosidad" de cada tema (zona de desarrollo próximo)
  3. Prioriza qué estudiar a continuación (como un estudiante inteligente)
  4. Detecta cuando un tema está "dominado" y avanza al siguiente nivel
  5. Nunca olvida los temas dominados (replay periódico desde MemoriaJerarquica)

Inspirado en:
  - Intrinsic Motivation y Curiosity-driven learning (Schmidhuber, 1991)
  - Zona de Desarrollo Próximo (Vygotsky, 1978)
  - Competence-based Intrinsic Motivation (Oudeyer & Kaplan, 2007)
"""

import json
import math
import time
from collections import defaultdict, deque
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
    """
    Estado de aprendizaje del modelo para un tema específico.

    Un tema puede ser: "funciones recursivas", "árboles", "decoradores", etc.
    """
    nombre: str
    categoria: str              # "algoritmos", "python", "patrones", etc.
    nivel_dificultad: int       # 1-6 (del CurriculumManager)

    # Historial de pérdida (rolling window de últimas N sesiones)
    historial_loss: deque = field(default_factory=lambda: deque(maxlen=20))

    # Tiempo
    primera_vez: float = field(default_factory=time.time)
    ultima_vez: float = field(default_factory=time.time)
    n_sesiones: int = 0

    # Estado actual
    loss_media: float = 99.0    # Alta al inicio (nunca visto)
    tasa_mejora: float = 0.0    # Derivada de loss (+ = mejorando, - = empeorando)
    dominado: bool = False       # True cuando loss < umbral por N sesiones consecutivas

    # Curiosidad calculada
    curiosidad: float = 1.0     # Score de interés (0-1)

    def registrar_sesion(self, loss: float) -> None:
        """Registra una sesión de entrenamiento y actualiza estadísticas."""
        self.historial_loss.append(loss)
        self.ultima_vez = time.time()
        self.n_sesiones += 1

        # Loss media (últimas 5 sesiones)
        reciente = list(self.historial_loss)[-5:]
        self.loss_media = sum(reciente) / len(reciente)

        # Tasa de mejora: pendiente lineal del historial
        if len(self.historial_loss) >= 3:
            hist = list(self.historial_loss)
            n = len(hist)
            xs = list(range(n))
            x_mean = sum(xs) / n
            y_mean = sum(hist) / n
            numerador = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, hist))
            denominador = sum((x - x_mean) ** 2 for x in xs) or 1e-8
            self.tasa_mejora = -(numerador / denominador)  # Negativo = mejora

        # Dominado si pérdida baja por 5 sesiones consecutivas
        if len(self.historial_loss) >= 5:
            ultimas = list(self.historial_loss)[-5:]
            self.dominado = all(l < 0.8 for l in ultimas)

    def tiempo_sin_ver(self) -> float:
        """Horas desde la última vez que se estudió este tema."""
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
    Decide qué estudiar a continuación basado en curiosidad intrínseca.

    El modelo aprende SIN supervisión externa — solo guiado por su propia
    incertidumbre sobre el mundo. Como un científico que sigue sus preguntas.

    Algoritmo:
      Para cada tema en la biblioteca:
        curiosidad = zona_proximal * novedad * urgencia_temporal

      zona_proximal = f(loss_actual):
        - loss < 0.5  → el modelo ya sabe esto → curiosidad baja (aburrido)
        - loss 0.5-2.0 → zona óptima de aprendizaje → curiosidad alta
        - loss > 4.0  → demasiado difícil aún → curiosidad media (espera)

      novedad = 1 / (1 + n_sesiones * 0.1):
        - Tema nunca visto → novedad alta
        - Tema visto muchas veces → novedad baja

      urgencia_temporal = 1 + tiempo_sin_ver / 24:
        - Tema no visto en 24h → urgencia × 2 (spacing effect)

      tasa_mejora_bonus:
        - Si el modelo mejora rápido → refuerzo positivo (≥ 0)
        - Si el modelo no mejora → señal de que necesita más práctica
    """

    # Zona de desarrollo próximo: loss ideal para aprendizaje máximo
    LOSS_OPTIMA = 1.5        # Ni muy fácil ni imposible
    LOSS_DOMINIO = 0.7       # Debajo de esto = dominado
    LOSS_MUY_DIFICIL = 5.0  # Arriba de esto = demasiado pronto

    def __init__(
        self,
        ruta_estado: Optional[Path] = None,
        nivel_actual: int = 1,
    ):
        """
        Args:
            ruta_estado: Dónde guardar/cargar el estado del motor.
            nivel_actual: Nivel de dificultad inicial (1-6).
        """
        self.ruta_estado = ruta_estado
        self.nivel_actual = nivel_actual

        # Perfiles de todos los temas conocidos
        self.temas: Dict[str, PerfilTema] = {}

        # Historial de qué se estudió (para evitar repetir inmediatamente)
        self.cola_reciente: deque = deque(maxlen=5)

        # Métricas globales
        self.sesiones_totales: int = 0
        self.temas_dominados: int = 0

        if ruta_estado and ruta_estado.exists():
            self.cargar(ruta_estado)

    # -------------------------------------------------------------------------
    # REGISTRO DE TEMAS
    # -------------------------------------------------------------------------

    def registrar_tema(
        self,
        nombre: str,
        categoria: str,
        nivel_dificultad: int,
    ) -> None:
        """Registra un tema nuevo en la biblioteca de conocimiento."""
        if nombre not in self.temas:
            self.temas[nombre] = PerfilTema(
                nombre=nombre,
                categoria=categoria,
                nivel_dificultad=nivel_dificultad,
            )

    def registrar_temas_desde_indice(self, indice: dict) -> int:
        """
        Carga todos los temas desde el índice de la biblioteca.

        Args:
            indice: Dict con estructura {"categoria": [{"nombre", "nivel", "archivo"}]}

        Returns:
            Número de temas nuevos registrados.
        """
        nuevos = 0
        for categoria, temas in indice.items():
            if not isinstance(temas, list):
                continue  # Ignorar meta-keys como "version", "descripcion"
            for tema in temas:
                nombre = tema["nombre"]
                if nombre not in self.temas:
                    self.registrar_tema(
                        nombre=nombre,
                        categoria=categoria,
                        nivel_dificultad=tema.get("nivel", 1),
                    )
                    nuevos += 1
        return nuevos

    # -------------------------------------------------------------------------
    # MEDICIÓN DE CURIOSIDAD
    # -------------------------------------------------------------------------

    def _curiosidad_zona_proximal(self, loss: float) -> float:
        """
        Función de curiosidad basada en la Zona de Desarrollo Próximo.

        Campana gaussiana centrada en LOSS_OPTIMA.
        """
        if loss < self.LOSS_DOMINIO:
            # Ya dominado — poco valor en seguir aquí
            return 0.1

        if loss > self.LOSS_MUY_DIFICIL:
            # Demasiado difícil aún — volver más tarde
            return 0.2

        # Campana gaussiana: máximo en LOSS_OPTIMA
        sigma = 0.8
        z = (loss - self.LOSS_OPTIMA) / sigma
        return math.exp(-0.5 * z * z)

    def _curiosidad_novedad(self, tema: PerfilTema) -> float:
        """Temas poco vistos son más interesantes."""
        return 1.0 / (1.0 + tema.n_sesiones * 0.08)

    def _curiosidad_temporal(self, tema: PerfilTema) -> float:
        """Spacing effect: temas no vistos en mucho tiempo suben de prioridad."""
        horas = tema.tiempo_sin_ver()
        # Logarítmico para no hacer crecer infinito
        return 1.0 + math.log1p(horas / 12.0) * 0.5

    def _bonus_mejora(self, tema: PerfilTema) -> float:
        """Refuerzo positivo si el modelo está mejorando rápido en este tema."""
        if tema.tasa_mejora > 0.05:
            return 1.3  # Está aprendiendo — seguir
        if tema.tasa_mejora < -0.02:
            return 0.8  # Retrocediendo — darle un descanso
        return 1.0

    def calcular_curiosidad(self, tema: PerfilTema) -> float:
        """
        Calcula el score de curiosidad total para un tema.

        Returns:
            Score [0, ∞) — mayor = más prioritario para estudiar.
        """
        # Exploración garantizada para temas poco visitados:
        # las primeras 6 sesiones no dependen del nivel ni del ZPD puro.
        # Esto evita que temas con loss alta o nivel alto queden en curiosidad=0.
        if tema.n_sesiones <= 5:
            # Curiosidad decreciente: 0.6 en sesión 0 → 0.18 en sesión 5
            zpd = max(0.18, 0.6 - tema.n_sesiones * 0.07)
            nivel_ok = 0.5  # no penalizar nivel en temas que apenas se conocen
        else:
            zpd = self._curiosidad_zona_proximal(tema.loss_media)
            # Penalizar temas de nivel superior al actual, pero suave (0.3 no 0.1)
            nivel_ok = 1.0 if tema.nivel_dificultad <= self.nivel_actual + 1 else 0.3

        novedad = self._curiosidad_novedad(tema)
        temporal = self._curiosidad_temporal(tema)
        bonus = self._bonus_mejora(tema)

        curiosidad = zpd * novedad * temporal * bonus * nivel_ok
        tema.curiosidad = curiosidad
        return curiosidad

    def actualizar_todos(self) -> None:
        """Recalcula curiosidad para todos los temas."""
        for tema in self.temas.values():
            self.calcular_curiosidad(tema)

    # -------------------------------------------------------------------------
    # SELECCIÓN DEL PRÓXIMO TEMA
    # -------------------------------------------------------------------------

    def siguiente_tema(self, excluir_recientes: bool = True) -> Optional[str]:
        """
        Devuelve el nombre del tema más interesante para estudiar ahora.

        Args:
            excluir_recientes: Si True, evita repetir los últimos 5 temas.

        Returns:
            Nombre del tema o None si no hay temas disponibles.
        """
        self.actualizar_todos()

        candidatos = list(self.temas.items())

        if excluir_recientes:
            candidatos = [
                (n, t) for n, t in candidatos
                if n not in self.cola_reciente
            ]

        if not candidatos:
            # Si todos están en recientes, ignorar esa restricción
            candidatos = list(self.temas.items())

        if not candidatos:
            return None

        # Ordenar por curiosidad descendente
        candidatos.sort(key=lambda x: x[1].curiosidad, reverse=True)

        # Sampling probabilístico (no siempre el más curioso, para exploración)
        # Los top-3 tienen probabilidad proporcional a su curiosidad
        top_n = min(3, len(candidatos))
        top = candidatos[:top_n]
        scores = [t.curiosidad for _, t in top]
        total = sum(scores) or 1.0

        # Selección por ruleta
        rand = torch.rand(1).item() * total
        acum = 0.0
        elegido = top[0][0]
        for nombre, tema in top:
            acum += tema.curiosidad
            if rand <= acum:
                elegido = nombre
                break

        self.cola_reciente.append(elegido)
        return elegido

    def tops(self, n: int = 5) -> List[Tuple[str, float]]:
        """Devuelve los N temas con mayor curiosidad."""
        self.actualizar_todos()
        ordenados = sorted(
            self.temas.items(),
            key=lambda x: x[1].curiosidad,
            reverse=True,
        )
        return [(nombre, tema.curiosidad) for nombre, tema in ordenados[:n]]

    # -------------------------------------------------------------------------
    # RETROALIMENTACIÓN
    # -------------------------------------------------------------------------

    def retroalimentar(self, nombre_tema: str, loss: float) -> dict:
        """
        Actualiza el perfil de un tema después de una sesión de entrenamiento.

        Args:
            nombre_tema: El tema que se acaba de estudiar.
            loss: Pérdida media de la sesión.

        Returns:
            Dict con información de progreso.
        """
        if nombre_tema not in self.temas:
            return {}

        tema = self.temas[nombre_tema]
        loss_anterior = tema.loss_media
        era_dominado = tema.dominado   # Estado ANTES de esta sesión
        tema.registrar_sesion(loss)

        self.sesiones_totales += 1

        # ¿Acaba de dominar este tema? (estaba sin dominar, ahora sí)
        recien_dominado = tema.dominado and not era_dominado
        if recien_dominado:
            self.temas_dominados += 1
            self._verificar_avance_nivel()

        return {
            "tema": nombre_tema,
            "loss_anterior": loss_anterior,
            "loss_actual": tema.loss_media,
            "mejora": loss_anterior - tema.loss_media,
            "tasa_mejora": tema.tasa_mejora,
            "dominado": tema.dominado,
            "recien_dominado": recien_dominado,
            "nivel_actual": self.nivel_actual,
        }

    def _verificar_avance_nivel(self) -> bool:
        """
        Avanza al siguiente nivel si el 70% de los temas del nivel actual
        están dominados.

        Returns:
            True si avanzó de nivel.
        """
        temas_nivel = [
            t for t in self.temas.values()
            if t.nivel_dificultad == self.nivel_actual
        ]
        if not temas_nivel:
            return False

        dominados = sum(1 for t in temas_nivel if t.dominado)
        porcentaje = dominados / len(temas_nivel)

        if porcentaje >= 0.70 and self.nivel_actual < 6:
            self.nivel_actual += 1
            return True
        return False

    # -------------------------------------------------------------------------
    # EVALUACIÓN CON EL MODELO (medición directa de pérdida)
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def medir_loss_tema(
        self,
        modelo: torch.nn.Module,
        tokens: torch.Tensor,
        device: str = "cpu",
    ) -> float:
        """
        Mide la pérdida del modelo en un batch de tokens del tema.

        Args:
            modelo: El modelo PAMPAr.
            tokens: Tensor [B, L] de token IDs.
            device: Dispositivo de cómputo.

        Returns:
            Loss media del modelo en este batch.
        """
        modelo.eval()
        tokens = tokens.to(device)

        if tokens.shape[1] < 2:
            return 99.0

        input_ids = tokens[:, :-1]
        targets = tokens[:, 1:]

        try:
            logits, _, _ = modelo(input_ids)
            B, L, V = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B * L, V),
                targets.reshape(B * L),
                ignore_index=0,
            )
            return loss.item()
        except Exception:
            return 99.0

    # -------------------------------------------------------------------------
    # RESUMEN Y ESTADO
    # -------------------------------------------------------------------------

    def resumen(self) -> dict:
        """Estado global del aprendizaje."""
        total = len(self.temas)
        dominados = sum(1 for t in self.temas.values() if t.dominado)
        en_progreso = sum(
            1 for t in self.temas.values()
            if not t.dominado and t.n_sesiones > 0
        )
        intactos = total - dominados - en_progreso

        loss_promedio = (
            sum(t.loss_media for t in self.temas.values()) / total
            if total > 0 else 0.0
        )

        return {
            "nivel_actual": self.nivel_actual,
            "sesiones_totales": self.sesiones_totales,
            "temas_total": total,
            "temas_dominados": dominados,
            "temas_en_progreso": en_progreso,
            "temas_intactos": intactos,
            "porcentaje_dominio": (dominados / total * 100) if total else 0.0,
            "loss_promedio_global": loss_promedio,
            "tops_curiosidad": self.tops(3),
        }

    def __repr__(self) -> str:
        r = self.resumen()
        return (
            f"MotorCuriosidad("
            f"nivel={r['nivel_actual']}, "
            f"dominados={r['temas_dominados']}/{r['temas_total']}, "
            f"loss_global={r['loss_promedio_global']:.2f})"
        )

    # -------------------------------------------------------------------------
    # PERSISTENCIA
    # -------------------------------------------------------------------------

    def guardar(self, ruta: Optional[Path] = None) -> None:
        """Guarda el estado completo del motor."""
        ruta = ruta or self.ruta_estado
        if ruta is None:
            return

        estado = {
            "nivel_actual": self.nivel_actual,
            "sesiones_totales": self.sesiones_totales,
            "temas_dominados": self.temas_dominados,
            "cola_reciente": list(self.cola_reciente),
            "temas": {n: t.to_dict() for n, t in self.temas.items()},
        }

        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(json.dumps(estado, indent=2, ensure_ascii=False))

    def cargar(self, ruta: Optional[Path] = None) -> None:
        """Carga el estado desde disco."""
        ruta = ruta or self.ruta_estado
        if ruta is None or not Path(ruta).exists():
            return

        estado = json.loads(Path(ruta).read_text())
        self.nivel_actual = estado.get("nivel_actual", 1)
        self.sesiones_totales = estado.get("sesiones_totales", 0)
        self.temas_dominados = estado.get("temas_dominados", 0)
        self.cola_reciente = deque(
            estado.get("cola_reciente", []), maxlen=5
        )
        self.temas = {
            n: PerfilTema.from_dict(d)
            for n, d in estado.get("temas", {}).items()
        }
