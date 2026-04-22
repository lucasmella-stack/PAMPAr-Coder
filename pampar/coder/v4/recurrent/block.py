# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
RecurrentBlock — orquestador del loop Prelude → step×T → Coda.

Recibe:
  - `step_fn`: una función `(h, e, loop_idx) -> h_next` que aplica UN paso
    del cerebro (típicamente un nivel v4 con peso compartido).
  - Componentes: LTIInjection, LoopIndexEmbedding, ACTHalting (todos opcionales).

Devuelve un `RecurrentOutput` con:
  - h_final:    [B, L, dim] estado final (output del Coda si lo hay)
  - n_steps:    int, pasos efectivamente usados
  - ponder_cost: scalar, cost de ACT (0 si no se usa)
  - halt_steps:  [B, L] long si ACT activado, sino None

Diseño deliberadamente **agnóstico del cerebro**: este módulo NO sabe nada
de niveles, streams, ni Mixed Selectivity. Solo orquesta el loop. Quien lo
usa pasa una `step_fn`. Eso permite testearlo en aislamiento con step_fn
sintéticas y reusarlo en distintos modelos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn

from .act import ACTHalting, ACTOutput
from .loop_rope import LoopIndexEmbedding
from .lti import LTIInjection

StepFn = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]
"""Firma: (h, e, loop_idx) -> h_next."""


@dataclass
class RecurrentOutput:
    """Resultado completo de un forward del RecurrentBlock."""

    h_final: torch.Tensor  # [B, L, dim]
    n_steps: int  # pasos efectivos
    ponder_cost: torch.Tensor  # scalar (0 si sin ACT)
    halt_steps: Optional[torch.Tensor]  # [B, L] long o None


class RecurrentBlock(nn.Module):
    """
    Loop recurrente con peso compartido + LTI + Loop-RoPE + ACT (todos opcionales).

    Args:
        dim:           dimensión del estado.
        max_loops:     máximo de iteraciones T.
        use_lti:       si True, aplica LTIInjection antes de step_fn cada paso.
        use_loop_rope: si True, suma LoopIndexEmbedding al estado cada paso.
        use_act:       si True, activa halting per-token via ACTHalting.
        threshold:     umbral de ACT (default 0.99).

    Uso típico:
        block = RecurrentBlock(dim=640, max_loops=8, use_lti=True,
                               use_loop_rope=True, use_act=True)

        def step_fn(h, e, t):
            # un nivel v4 con modality_id, loop_idx=t, etc.
            return nivel(h)

        out = block(e_input, step_fn=step_fn)
        # out.h_final, out.n_steps, out.ponder_cost
    """

    def __init__(
        self,
        dim: int,
        max_loops: int = 8,
        use_lti: bool = True,
        use_loop_rope: bool = True,
        use_act: bool = True,
        threshold: float = 0.99,
    ):
        super().__init__()
        if max_loops < 1:
            raise ValueError(f"max_loops debe ser ≥ 1, got {max_loops}")

        self.dim = dim
        self.max_loops = max_loops
        self.use_lti = use_lti
        self.use_loop_rope = use_loop_rope
        self.use_act = use_act

        self.lti = LTIInjection(dim) if use_lti else None
        self.loop_rope = (
            LoopIndexEmbedding(dim, max_loops=max_loops, project=True)
            if use_loop_rope
            else None
        )
        self.act = ACTHalting(dim, threshold=threshold) if use_act else None

    def forward(
        self,
        e: torch.Tensor,
        step_fn: StepFn,
        h_init: Optional[torch.Tensor] = None,
    ) -> RecurrentOutput:
        """
        Ejecuta el loop recurrente.

        Args:
            e:         [B, L, dim] embedding de entrada (constante en todo el loop).
            step_fn:   función que aplica un paso de cómputo.
            h_init:    [B, L, dim] estado inicial. Si None, arranca de e.

        Returns:
            RecurrentOutput con h_final, n_steps, ponder_cost, halt_steps.
        """
        if e.dim() != 3 or e.shape[-1] != self.dim:
            raise ValueError(f"e debe ser [B,L,{self.dim}], got {tuple(e.shape)}")

        B, L, _ = e.shape
        device = e.device
        dtype = e.dtype

        h = h_init if h_init is not None else e.clone()

        if self.act is not None:
            self.act.reset(B, L, device, dtype)

        for t in range(self.max_loops):
            # 1) LTI injection (estabilidad ρ(A)<1)
            if self.lti is not None:
                h = self.lti(h, e)

            # 2) Loop-index embedding sumado al estado
            if self.loop_rope is not None:
                h = self.loop_rope(h, loop_idx=t)

            # 3) Step de cómputo (cerebro v4)
            h = step_fn(h, e, t)

            # 4) ACT halting check
            if self.act is not None:
                done = self.act.update(h, step_idx=t)
                if done:
                    break

        if self.act is not None:
            act_out: ACTOutput = self.act.finalize()
            return RecurrentOutput(
                h_final=act_out.output,
                n_steps=act_out.n_steps_used,
                ponder_cost=act_out.ponder_cost,
                halt_steps=act_out.halt_steps,
            )

        # Sin ACT: simplemente devolvemos el último h
        return RecurrentOutput(
            h_final=h,
            n_steps=self.max_loops,
            ponder_cost=torch.zeros((), device=device, dtype=dtype),
            halt_steps=None,
        )
