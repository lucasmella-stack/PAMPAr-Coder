# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
RecurrentNivelAdapter — puente entre `RecurrentBlock` (Fase 4) y un
`NivelProfundo` v3 (cerebro multi-stream).

Problema resuelto:
    `RecurrentBlock` espera `step_fn(h, e, t) -> h_next` con tensores
    [B, L, dim] (single-stream). `NivelProfundo` opera sobre
    `(streams: List[Tensor], terr_acts, zona_acts)` (multi-stream).

Estrategia "delta-based":
    Mantenemos los streams como estado interno del adapter. En cada step:
      1. delta = h - last_combined  (cuánto cambió el combined por LTI+RoPE)
      2. Aplicamos delta aditivamente a cada stream  → preserva la
         diferenciación inter-stream que ya tenían
      3. Corremos el NivelProfundo (body) con esos streams
      4. Recombinamos → nuevo h_combined que devolvemos

Esto preserva la propiedad de Mixed Selectivity: cada stream mantiene su
"interpretación" del token a lo largo del loop, mientras que la inyección
LTI y el Loop-RoPE actúan sobre la vista combinada (que es lo que el
cerebro "decide" a cada paso).

NOTA: el adapter es stateful entre llamadas a `step` dentro de un mismo
forward. SIEMPRE llamar `reset()` antes de cada forward del modelo.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from pampar.coder.v3.nivel import NivelProfundo
from pampar.coder.v3.talamo import TalamoInicial


class RecurrentNivelAdapter(nn.Module):
    """
    Adapta un `NivelProfundo` v3 a la interfaz `StepFn` de `RecurrentBlock`.

    Args:
        body_nivel: el `NivelProfundo` que se ejecutará en cada iteración
                    del loop (peso compartido entre iteraciones).
        n_streams:  cantidad de streams (debe coincidir con el config).

    Uso típico:
        adapter = RecurrentNivelAdapter(body_nivel, n_streams=4)
        adapter.reset(streams_init, terr_acts_init, zona_acts)
        out = recurrent_block(combined_init, step_fn=adapter.step)
        # Después del loop:
        final_streams = adapter.streams
        final_terr_acts = adapter.terr_acts
    """

    def __init__(self, body_nivel: NivelProfundo, n_streams: int):
        super().__init__()
        self.body_nivel = body_nivel
        self.n_streams = n_streams

        # Estado interno (no son nn.Parameters, no se serializan)
        self._streams: Optional[List[torch.Tensor]] = None
        self._terr_acts: Optional[torch.Tensor] = None
        self._zona_acts: Optional[torch.Tensor] = None
        self._last_combined: Optional[torch.Tensor] = None
        self._is_reset: bool = False

    # ────────────────────────────────────────────────────────────────────
    # API pública
    # ────────────────────────────────────────────────────────────────────

    def reset(
        self,
        streams: List[torch.Tensor],
        terr_acts: torch.Tensor,
        zona_acts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inicializa el estado del adapter para un nuevo forward.

        Args:
            streams:    [n_streams × [B, L, D]] estado por stream.
            terr_acts:  [B, L, n_streams] activaciones territoriales.
            zona_acts:  [B, L, 52] activaciones zonales.

        Returns:
            combined inicial [B, L, D] — sirve como `e` para RecurrentBlock.
        """
        if len(streams) != self.n_streams:
            raise ValueError(
                f"Esperaba {self.n_streams} streams, recibí {len(streams)}"
            )

        self._streams = list(streams)
        self._terr_acts = terr_acts
        self._zona_acts = zona_acts
        self._last_combined = self._combine(self._streams, self._terr_acts)
        self._is_reset = True
        return self._last_combined

    @property
    def streams(self) -> List[torch.Tensor]:
        """Streams actuales (después de N steps)."""
        if self._streams is None:
            raise RuntimeError("Llamar .reset() primero")
        return self._streams

    @property
    def terr_acts(self) -> torch.Tensor:
        """Activaciones territoriales actuales."""
        if self._terr_acts is None:
            raise RuntimeError("Llamar .reset() primero")
        return self._terr_acts

    def step(self, h: torch.Tensor, e: torch.Tensor, t: int) -> torch.Tensor:
        """
        StepFn compatible con `RecurrentBlock`.

        Args:
            h: [B, L, D] estado combinado actual (tras LTI + Loop-RoPE).
            e: [B, L, D] embedding original (no se usa aquí — los streams
               ya cargan esa información).
            t: índice de iteración actual.

        Returns:
            new_combined: [B, L, D] estado combinado tras correr body_nivel.
        """
        if not self._is_reset:
            raise RuntimeError("Llamar .reset() antes de step()")

        # 1) Delta inyectado por LTI/RoPE sobre la vista combinada
        delta = h - self._last_combined

        # 2) Aplicar delta a cada stream → preserva diferenciación
        streams_in = [s + delta for s in self._streams]

        # 3) Forward del nivel body (peso compartido entre iteraciones)
        new_streams, new_terr_acts, _conf = self.body_nivel(
            streams_in,
            self._terr_acts,
            TalamoInicial.agregar_fn,
            banco_engrama=None,
            zona_acts=self._zona_acts,
        )

        # 4) Actualizar estado y devolver nuevo combined
        self._streams = new_streams
        self._terr_acts = new_terr_acts
        new_combined = self._combine(new_streams, new_terr_acts)
        self._last_combined = new_combined
        return new_combined

    # ────────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────────

    @staticmethod
    def _combine(
        streams: List[torch.Tensor],
        terr_acts: torch.Tensor,
    ) -> torch.Tensor:
        """Combina los streams ponderados por terr_acts (sin softmax extra).

        Mantiene el mismo esquema que `NivelProfundo` usa internamente
        para `x_combined`."""
        n = len(streams)
        return sum(streams[t] * terr_acts[:, :, t : t + 1] for t in range(n))
