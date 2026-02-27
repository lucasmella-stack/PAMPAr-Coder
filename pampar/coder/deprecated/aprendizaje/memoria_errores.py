# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Memoria de Errores con Interiorización.

Buffer temporal que registra patrones de error del modelo.
Cuando el modelo deja de cometer un error (N éxitos consecutivos
en patrones similares), la entrada se borra — el conocimiento
ya está interiorizado en los pesos del modelo.

Analogía humana:
  - 2 años: te caés → recuerdo explícito "no inclinar tanto"
  - 4 años: caminar es automático → olvidaste la regla
  - 31 años: no recordás cómo aprendiste — está en tus músculos (pesos)

El ciclo:
  1. El modelo comete un error (loss alta en un token)
  2. Se guarda la ventana de tokens alrededor del error
  3. En los siguientes steps, si el mismo patrón aparece:
     - Si el modelo vuelve a fallar → penalización extra en la loss
     - Si el modelo acierta → cuenta como éxito
  4. Cuando acumula suficientes éxitos → se borra (interiorizado)

Diseño:
  - Ring buffer de tamaño fijo (10K entradas por defecto)
  - Hash table para búsqueda O(1) por patrón
  - Auto-limpieza: los patrones viejos se sobreescriben
  - Serializable: se guarda/carga con el checkpoint
"""

import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


# =============================================================================
# ENTRADA DE MEMORIA
# =============================================================================

@dataclass
class EntradaError:
    """Un patrón de error almacenado en la memoria."""
    patron: Tuple[int, ...]       # Ventana de token IDs
    loss_original: float          # Loss cuando ocurrió el error
    exitos: int = 0               # Éxitos consecutivos post-error
    timestamp: float = 0.0        # Cuándo se registró
    veces_penalizado: int = 0     # Cuántas veces se aplicó penalización

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


# =============================================================================
# MEMORIA DE ERRORES
# =============================================================================

class MemoriaErrores:
    """
    Buffer circular de patrones erróneos con auto-interiorización.

    Uso en el training loop:
        memoria = MemoriaErrores()

        for batch in dataloader:
            logits, loss, info = model(input_ids, targets)

            # Calcular loss per-token
            per_token_loss = F.cross_entropy(
                logits.view(-1, vocab), targets.view(-1),
                reduction='none', ignore_index=-100
            ).view(B, L)

            # 1. Penalizar patrones conocidos
            penalty = memoria.calcular_penalizacion(input_ids)
            loss_total = loss + (per_token_loss * penalty).mean()

            # 2. Registrar nuevos errores
            memoria.registrar_errores(input_ids, per_token_loss)

            # 3. Verificar interiorización
            memoria.verificar_interiorizacion(input_ids, per_token_loss)

            loss_total.backward()
            ...
    """

    def __init__(
        self,
        max_entries: int = 10000,
        hash_window: int = 16,
        umbral_interiorizacion: int = 5,
        umbral_error: float = 5.0,
        factor_penalizacion: float = 0.15,
    ):
        """
        Args:
            max_entries: Tamaño máximo del buffer
            hash_window: Tokens de contexto por patrón
            umbral_interiorizacion: Éxitos necesarios para borrar
            umbral_error: Loss mínima para considerar error
            factor_penalizacion: Peso extra en la loss (0.15 = +15%)
        """
        self.max_entries = max_entries
        self.hash_window = hash_window
        self.umbral_interiorizacion = umbral_interiorizacion
        self.umbral_error = umbral_error
        self.factor_penalizacion = factor_penalizacion

        # Hash table para búsqueda O(1)
        # key = tuple de token IDs, value = EntradaError
        self.memoria: OrderedDict[Tuple[int, ...], EntradaError] = OrderedDict()

        # Estadísticas
        self.total_registrados = 0
        self.total_interiorizados = 0
        self.total_penalizaciones = 0

    def _extraer_ventana(
        self,
        input_ids: torch.Tensor,
        batch_idx: int,
        pos: int,
    ) -> Optional[Tuple[int, ...]]:
        """Extrae ventana de tokens alrededor de una posición."""
        L = input_ids.shape[1]
        start = pos - self.hash_window + 1
        if start < 0:
            return None  # No hay suficiente contexto
        window = input_ids[batch_idx, start:pos + 1]
        return tuple(window.cpu().tolist())

    def registrar_errores(
        self,
        input_ids: torch.Tensor,  # [B, L]
        per_token_loss: torch.Tensor,  # [B, L]
    ) -> int:
        """
        Registra tokens con loss alta como patrones de error.

        Returns:
            Número de errores registrados
        """
        B, L = input_ids.shape
        count = 0

        # Encontrar posiciones con loss alta
        mask_error = per_token_loss > self.umbral_error  # [B, L]

        for b in range(B):
            error_positions = mask_error[b].nonzero(as_tuple=True)[0]

            for pos in error_positions:
                pos_int: int = int(pos.item())
                ventana = self._extraer_ventana(input_ids, b, pos_int)

                if ventana is None:
                    continue

                # No duplicar si ya existe
                if ventana in self.memoria:
                    continue

                # Si buffer lleno, eliminar el más viejo
                if len(self.memoria) >= self.max_entries:
                    self.memoria.popitem(last=False)  # FIFO: elimina el más antiguo

                self.memoria[ventana] = EntradaError(
                    patron=ventana,
                    loss_original=float(per_token_loss[b, pos_int].item()),
                )
                self.total_registrados += 1
                count += 1

        return count

    def verificar_interiorizacion(
        self,
        input_ids: torch.Tensor,  # [B, L]
        per_token_loss: torch.Tensor,  # [B, L]
    ) -> int:
        """
        Verifica si patrones erróneos ahora se predicen bien.
        Si acumulan suficientes éxitos, se interiorizan (borran).

        Returns:
            Número de patrones interiorizados
        """
        if not self.memoria:
            return 0

        B, L = input_ids.shape
        a_borrar = []
        interiorizados = 0

        for b in range(B):
            for pos in range(self.hash_window, L):
                ventana = self._extraer_ventana(input_ids, b, pos)
                if ventana is None:
                    continue

                if ventana in self.memoria:
                    entrada = self.memoria[ventana]

                    # ¿El modelo ahora acierta este patrón?
                    if per_token_loss[b, pos].item() < self.umbral_error * 0.4:
                        entrada.exitos += 1

                        # ¿Interiorizado?
                        if entrada.exitos >= self.umbral_interiorizacion:
                            a_borrar.append(ventana)
                            interiorizados += 1
                    else:
                        # Aún falla — resetear contador de éxitos
                        entrada.exitos = max(0, entrada.exitos - 1)

        # Borrar los interiorizados
        for key in a_borrar:
            if key in self.memoria:
                del self.memoria[key]
                self.total_interiorizados += 1

        return interiorizados

    def calcular_penalizacion(
        self,
        input_ids: torch.Tensor,  # [B, L]
    ) -> torch.Tensor:
        """
        Calcula penalización extra para tokens que matcheen patrones de error.

        Returns:
            [B, L] penalty weights (0 = sin penalización, >0 = penalización)
        """
        B, L = input_ids.shape
        penalty = torch.zeros(B, L, device=input_ids.device)

        if not self.memoria:
            return penalty

        for b in range(B):
            for pos in range(self.hash_window, L):
                ventana = self._extraer_ventana(input_ids, b, pos)
                if ventana is None:
                    continue

                if ventana in self.memoria:
                    entrada = self.memoria[ventana]
                    penalty[b, pos] = self.factor_penalizacion
                    entrada.veces_penalizado += 1
                    self.total_penalizaciones += 1

        return penalty

    # =========================================================================
    # PERSISTENCIA
    # =========================================================================

    def guardar(self, path: str):
        """Guarda la memoria a disco (JSON)."""
        data = {
            'config': {
                'max_entries': self.max_entries,
                'hash_window': self.hash_window,
                'umbral_interiorizacion': self.umbral_interiorizacion,
                'umbral_error': self.umbral_error,
                'factor_penalizacion': self.factor_penalizacion,
            },
            'stats': {
                'total_registrados': self.total_registrados,
                'total_interiorizados': self.total_interiorizados,
                'total_penalizaciones': self.total_penalizaciones,
            },
            'entries': [
                {
                    'patron': list(e.patron),
                    'loss_original': e.loss_original,
                    'exitos': e.exitos,
                    'timestamp': e.timestamp,
                    'veces_penalizado': e.veces_penalizado,
                }
                for e in self.memoria.values()
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def cargar(cls, path: str) -> 'MemoriaErrores':
        """Carga la memoria desde disco."""
        data = json.loads(Path(path).read_text())
        cfg = data['config']
        mem = cls(
            max_entries=cfg['max_entries'],
            hash_window=cfg['hash_window'],
            umbral_interiorizacion=cfg['umbral_interiorizacion'],
            umbral_error=cfg['umbral_error'],
            factor_penalizacion=cfg['factor_penalizacion'],
        )
        stats = data['stats']
        mem.total_registrados = stats['total_registrados']
        mem.total_interiorizados = stats['total_interiorizados']
        mem.total_penalizaciones = stats['total_penalizaciones']

        for entry in data['entries']:
            patron = tuple(entry['patron'])
            mem.memoria[patron] = EntradaError(
                patron=patron,
                loss_original=entry['loss_original'],
                exitos=entry['exitos'],
                timestamp=entry['timestamp'],
                veces_penalizado=entry['veces_penalizado'],
            )
        return mem

    # =========================================================================
    # ESTADÍSTICAS
    # =========================================================================

    def stats(self) -> Dict:
        """Estadísticas legibles de la memoria."""
        n_activos = len(self.memoria)
        return {
            'activos': n_activos,
            'capacidad': self.max_entries,
            'uso_pct': round(n_activos / self.max_entries * 100, 1),
            'total_registrados': self.total_registrados,
            'total_interiorizados': self.total_interiorizados,
            'total_penalizaciones': self.total_penalizaciones,
            'ratio_inter': (
                round(self.total_interiorizados / max(1, self.total_registrados) * 100, 1)
            ),
        }

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"MemoriaErrores("
            f"activos={s['activos']}/{s['capacidad']}, "
            f"interiorizados={s['total_interiorizados']}, "
            f"ratio={s['ratio_inter']}%)"
        )
