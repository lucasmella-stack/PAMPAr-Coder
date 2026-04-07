# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
MemoriaJerarquica — Memoria de entrenamiento con 3 niveles.

L0 (reciente):  FIFO ring buffer de las últimas secuencias vistas.
L1 (difícil):   secuencias con loss alta → se usan en replay anti-olvido.
L2 (dominado):  patrones ya aprendidos → consolidación periódica.

Principio anti-olvido:
    Durante el aprendizaje continuo, el modelo tiende a olvidar lo que
    ya aprendió cuando entrena en datos nuevos (catastrophic forgetting).
    La MemoriaJerarquica mitiga esto con:
      - Replay periódico de los patrones más difíciles (L1)
      - Consolidación de patrones dominados (L2)
      - FIFO de observaciones recientes para contexto (L0)

Usado por scripts/aprender_solo.py para el viaje intelectual autónomo.
"""

import json
import logging
import random
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Umbrales de clasificación por loss
UMBRAL_DIFICIL = 1.5  # loss > this → L1 (es difícil, necesita práctica)
UMBRAL_DOMINADO = 0.8  # loss < this → candidato a L2 (ya lo domina)
BATCH_REPLAY = 4  # Tamaño de batch por defecto para replay


@dataclass
class EntradaEntrenamiento:
    """Una secuencia almacenada en la memoria de entrenamiento."""

    tokens: list[int]
    loss_media: float = 0.0
    territorio: str = ""
    timestamp: float = field(default_factory=time.time)
    n_replay: int = 0

    def to_dict(self) -> dict:
        return {
            "tokens": self.tokens,
            "loss_media": self.loss_media,
            "territorio": self.territorio,
            "timestamp": self.timestamp,
            "n_replay": self.n_replay,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EntradaEntrenamiento":
        return cls(
            tokens=d["tokens"],
            loss_media=d.get("loss_media", 0.0),
            territorio=d.get("territorio", ""),
            timestamp=d.get("timestamp", 0.0),
            n_replay=d.get("n_replay", 0),
        )


class MemoriaJerarquica:
    """
    Memoria jerárquica de entrenamiento con 3 niveles.

    L0: Ring buffer FIFO — almacena las últimas N secuencias vistas.
    L1: Buffer de prioridad — secuencias difíciles (loss alta) para replay.
    L2: Buffer de consolidación — patrones dominados (loss baja consistente).

    Args:
        capacidad_l0: Máximo de entradas en L0 (FIFO).
        capacidad_l1: Máximo de entradas en L1 (las más fáciles se evictan).
        capacidad_l2: Máximo de entradas en L2 (las más viejas se evictan).
    """

    def __init__(
        self,
        capacidad_l0: int = 2048,
        capacidad_l1: int = 8000,
        capacidad_l2: int = 3000,
    ) -> None:
        self.capacidad_l0 = capacidad_l0
        self.capacidad_l1 = capacidad_l1
        self.capacidad_l2 = capacidad_l2

        self.l0: deque[EntradaEntrenamiento] = deque(maxlen=capacidad_l0)
        self.l1: list[EntradaEntrenamiento] = []
        self.l2: list[EntradaEntrenamiento] = []

        self._stats = {
            "total_procesados": 0,
            "promovidos_l1": 0,
            "promovidos_l2": 0,
            "consolidaciones": 0,
        }

    # ── Procesamiento ────────────────────────────────────────────────────────

    def procesar_batch(
        self,
        tokens: torch.Tensor,
        per_token_loss: Optional[torch.Tensor] = None,
        terr_acts: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Procesa un batch de entrenamiento y almacena las secuencias.

        Cada secuencia del batch se clasifica por su loss media:
          - Todas van a L0 (FIFO reciente).
          - Loss > UMBRAL_DIFICIL → también a L1 (necesita replay).
          - Loss < UMBRAL_DOMINADO → candidato a L2 en la próxima consolidación.

        Args:
            tokens:         [B, L] token IDs del batch.
            per_token_loss: [B, L] loss por token (None si no disponible).
            terr_acts:      [B, L, 4] activaciones de territorio (opcional).
        """
        B = tokens.shape[0]

        for i in range(B):
            seq = tokens[i].tolist()

            # Loss media (solo tokens no-padding)
            loss = 0.0
            if per_token_loss is not None and i < per_token_loss.shape[0]:
                mask = tokens[i] != 0
                valid_losses = per_token_loss[i][mask]
                if valid_losses.numel() > 0:
                    loss = valid_losses.mean().item()

            # Territorio dominante
            territorio = ""
            if terr_acts is not None and i < terr_acts.shape[0]:
                nombres = ["SINTAXIS", "SEMANTICA", "LOGICO", "ESTRUCTURAL"]
                dom_idx = terr_acts[i].mean(dim=0).argmax().item()
                if dom_idx < len(nombres):
                    territorio = nombres[dom_idx]

            entrada = EntradaEntrenamiento(
                tokens=seq,
                loss_media=loss,
                territorio=territorio,
            )

            # L0 siempre (FIFO)
            self.l0.append(entrada)
            self._stats["total_procesados"] += 1

            # L1 si es difícil
            if loss > UMBRAL_DIFICIL:
                self.l1.append(entrada)
                self._stats["promovidos_l1"] += 1

                # Evictar los más fáciles si L1 está llena
                if len(self.l1) > self.capacidad_l1:
                    self.l1.sort(key=lambda e: e.loss_media, reverse=True)
                    self.l1 = self.l1[: self.capacidad_l1]

    # ── Replay ───────────────────────────────────────────────────────────────

    def get_replay_batch(
        self,
        strategy: str = "hardest",
        batch_size: int = BATCH_REPLAY,
    ) -> Optional[torch.Tensor]:
        """
        Devuelve un batch de replay desde L1.

        Args:
            strategy: "hardest" (mayor loss) o "random" (aleatorio).
            batch_size: Número de secuencias en el batch.

        Returns:
            Tensor [B, L] listo para entrenamiento, o None si L1 está vacía.
        """
        if not self.l1:
            return None

        if strategy == "hardest":
            sorted_l1 = sorted(self.l1, key=lambda e: e.loss_media, reverse=True)
            selected = sorted_l1[:batch_size]
        else:
            selected = random.sample(self.l1, min(batch_size, len(self.l1)))

        # Marcar como "replayed"
        for entry in selected:
            entry.n_replay += 1

        # Pad al mismo largo y convertir a tensor
        max_len = max(len(e.tokens) for e in selected)
        padded = []
        for e in selected:
            trunc = e.tokens[:max_len]
            pad = [0] * (max_len - len(trunc))
            padded.append(trunc + pad)

        return torch.tensor(padded, dtype=torch.long)

    # ── Consolidación ────────────────────────────────────────────────────────

    def consolidar(self, modelo: torch.nn.Module) -> dict:
        """
        Reorganiza la memoria entre niveles.

        - L1 entries con loss baja (modelo ya las aprendió) → L2.
        - L2 se recorta por antigüedad si excede capacidad.
        - L1 entries con demasiados replays (>10) se retiran.

        Args:
            modelo: El modelo (por compatibilidad de interfaz; no se usa
                    para forward passes aquí para evitar overhead).

        Returns:
            dict con estadísticas de la consolidación.
        """
        self._stats["consolidaciones"] += 1

        # Mover patrones fáciles de L1 → L2
        mastered = [e for e in self.l1 if e.loss_media < UMBRAL_DOMINADO]
        still_hard = [e for e in self.l1 if e.loss_media >= UMBRAL_DOMINADO]

        # Retirar entries con demasiados replays (ya se practicaron suficiente)
        retired = [e for e in still_hard if e.n_replay > 10]
        active = [e for e in still_hard if e.n_replay <= 10]

        self.l2.extend(mastered)
        self.l1 = active
        self._stats["promovidos_l2"] += len(mastered)

        # Recortar L2 por antigüedad
        if len(self.l2) > self.capacidad_l2:
            self.l2.sort(key=lambda e: e.timestamp)
            self.l2 = self.l2[-self.capacidad_l2 :]

        result = {
            "a_l2": len(mastered),
            "retirados": len(retired),
            "l1_activo": len(self.l1),
            "l2_total": len(self.l2),
        }
        logger.debug("Consolidación: %s", result)
        return result

    # ── Persistencia ─────────────────────────────────────────────────────────

    def guardar(self, ruta: str) -> None:
        """Guarda el estado completo en JSON."""
        path = Path(ruta)
        path.parent.mkdir(parents=True, exist_ok=True)

        estado = {
            "capacidad_l0": self.capacidad_l0,
            "capacidad_l1": self.capacidad_l1,
            "capacidad_l2": self.capacidad_l2,
            "stats": self._stats,
            # L0 es FIFO efímero — solo guardar las últimas 256 para contexto
            "l0": [e.to_dict() for e in list(self.l0)[-256:]],
            "l1": [e.to_dict() for e in self.l1],
            "l2": [e.to_dict() for e in self.l2],
        }
        path.write_text(json.dumps(estado, ensure_ascii=False), encoding="utf-8")
        logger.info(
            "Memoria guardada en %s (L0=%d, L1=%d, L2=%d)",
            ruta,
            len(self.l0),
            len(self.l1),
            len(self.l2),
        )

    @classmethod
    def cargar(cls, ruta: str) -> "MemoriaJerarquica":
        """Carga el estado desde JSON."""
        path = Path(ruta)
        if not path.exists():
            raise FileNotFoundError(f"Estado de memoria no encontrado: {ruta}")

        estado = json.loads(path.read_text(encoding="utf-8"))

        mem = cls(
            capacidad_l0=estado.get("capacidad_l0", 2048),
            capacidad_l1=estado.get("capacidad_l1", 8000),
            capacidad_l2=estado.get("capacidad_l2", 3000),
        )
        mem._stats = estado.get("stats", mem._stats)

        for d in estado.get("l0", []):
            mem.l0.append(EntradaEntrenamiento.from_dict(d))
        mem.l1 = [EntradaEntrenamiento.from_dict(d) for d in estado.get("l1", [])]
        mem.l2 = [EntradaEntrenamiento.from_dict(d) for d in estado.get("l2", [])]

        logger.info(
            "Memoria cargada desde %s (L0=%d, L1=%d, L2=%d)",
            ruta,
            len(mem.l0),
            len(mem.l1),
            len(mem.l2),
        )
        return mem

    # ── Info ─────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"MemoriaJerarquica(L0={len(self.l0)}/{self.capacidad_l0}, "
            f"L1={len(self.l1)}/{self.capacidad_l1}, "
            f"L2={len(self.l2)}/{self.capacidad_l2})"
        )
