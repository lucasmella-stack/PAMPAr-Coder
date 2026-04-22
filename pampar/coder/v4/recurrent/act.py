# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Adaptive Computation Time (ACT) — halting per-token.

Cada paso del loop, una head pequeña produce p_t ∈ (0, 1) por token —
"probabilidad de detenerse en este paso". Cuando la probabilidad
acumulada P_t = sum_{s≤t} p_s supera `threshold`, el token "se detuvo":

  - Su contribución al output final se acumula con peso p_t* (reparto justo)
  - Sus actualizaciones futuras de h se ignoran (mask)

Devolvemos también un `ponder_cost` (suma de los pasos efectivos por token,
escalar) que el training loop puede agregar al loss con un peso pequeño
para incentivar early exit donde sea posible.

Referencia: Graves 2017 "Adaptive Computation Time for Recurrent Neural
Networks", simplificado para el caso de step compartido.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ACTOutput:
    """Resultado de un loop con ACT halting."""

    output: torch.Tensor  # [B, L, dim] suma ponderada de los h_t
    halt_steps: torch.Tensor  # [B, L] long, paso en que cada token se detuvo
    ponder_cost: torch.Tensor  # scalar, sum de pasos / cantidad de tokens
    n_steps_used: int  # max_steps efectivos antes de salir todos


class ACTHalting(nn.Module):
    """
    Halting head per-token + acumulador ponderado.

    Args:
        dim:        dimensión del estado oculto.
        threshold:  umbral acumulativo de halting (default 0.99). Cuando la
                    suma de p_t supera 1-(1-threshold), el token se detiene.
                    Usar exactamente 0.99 = comportamiento Graves estándar.
        eps:        para numerical safety en el reparto del peso final.

    Uso (típico, dentro de un loop):
        act = ACTHalting(dim)
        act.reset(B, L, device, dtype)
        for t in range(max_steps):
            h = step(h, ...)
            done = act.update(h, t)   # bool, True si TODOS los tokens halted
            if done:
                break
        result = act.finalize()       # ACTOutput
    """

    def __init__(self, dim: int, threshold: float = 0.99, eps: float = 1e-3):
        super().__init__()
        if not (0.0 < threshold < 1.0):
            raise ValueError(f"threshold debe estar en (0,1), got {threshold}")

        self.dim = dim
        self.threshold = threshold
        self.eps = eps

        # Head pequeña: [dim] → escalar (logit del halting)
        self.halt_head = nn.Linear(dim, 1, bias=True)
        # Init bias negativo → al inicio prob halting baja → loop completo
        # En training se aprende a bajarlo si el cómputo sirve.
        nn.init.zeros_(self.halt_head.weight)
        nn.init.constant_(self.halt_head.bias, -2.0)  # sigmoid(-2) ≈ 0.12

        # Estado interno (se reinicia con .reset())
        self._reset_state()

    def _reset_state(self) -> None:
        self._cum_p: torch.Tensor | None = None  # [B, L] suma acumulada de p_t
        self._cum_out: torch.Tensor | None = None  # [B, L, dim] sum p_t*h_t
        self._halt_step: torch.Tensor | None = None  # [B, L] long
        self._halted: torch.Tensor | None = None  # [B, L] bool
        self._steps_done: int = 0

    def reset(
        self,
        B: int,
        L: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Inicializa el estado para un nuevo loop."""
        self._cum_p = torch.zeros(B, L, device=device, dtype=dtype)
        self._cum_out = torch.zeros(B, L, self.dim, device=device, dtype=dtype)
        self._halt_step = torch.full((B, L), -1, device=device, dtype=torch.long)
        self._halted = torch.zeros(B, L, device=device, dtype=torch.bool)
        self._steps_done = 0

    def update(self, h: torch.Tensor, step_idx: int) -> bool:
        """
        Actualiza acumuladores con la nueva h. Devuelve True si todos los
        tokens halted (early exit del loop).

        Args:
            h:        [B, L, dim] estado oculto del paso step_idx.
            step_idx: paso actual.

        Returns:
            True si ya halted todos los tokens.
        """
        if self._cum_p is None:
            raise RuntimeError("Llamar .reset() antes de .update()")

        # Probabilidad de halting per-token
        p = torch.sigmoid(self.halt_head(h)).squeeze(-1)  # [B, L]

        # Tokens ya detenidos: contribución 0 (mantienen su acumulado)
        not_halted = (~self._halted).to(p.dtype)
        p = p * not_halted

        # Detectar tokens que se detienen en este paso
        new_cum = self._cum_p + p
        will_halt = (new_cum > 1.0 - self.eps) & (~self._halted)

        # Para los que halt en este step: peso final = 1 - cum_p_anterior
        # (reparte el peso restante para que sumen exactamente 1)
        final_weight = torch.where(will_halt, 1.0 - self._cum_p, p)

        # Actualizar acumuladores
        self._cum_out = self._cum_out + final_weight.unsqueeze(-1) * h
        self._cum_p = self._cum_p + final_weight

        # Marcar nuevos halted
        self._halt_step = torch.where(
            will_halt,
            torch.full_like(self._halt_step, step_idx),
            self._halt_step,
        )
        self._halted = self._halted | will_halt

        self._steps_done = step_idx + 1
        return bool(self._halted.all())

    def finalize(self) -> ACTOutput:
        """
        Cierra el loop. Tokens que nunca halted reciben peso restante (1-cum_p)
        sobre la última h vista — pero como ya está sumada en _cum_out,
        ajustamos asumiendo que el loop terminó.

        En el caso "loop terminó por max_steps sin halting": el peso restante
        ya se incorporó cuando llamamos update con el último step (los
        tokens no-halted siguieron acumulando).

        Returns:
            ACTOutput con output final, halt_steps, ponder_cost.
        """
        if self._cum_p is None:
            raise RuntimeError("Llamar .reset() y .update() antes de .finalize()")

        # Tokens que nunca halted: se les fuerza halt step = último
        forced = ~self._halted
        last_step = max(self._steps_done - 1, 0)
        self._halt_step = torch.where(
            forced,
            torch.full_like(self._halt_step, last_step),
            self._halt_step,
        )

        # Si algún token no llegó a sumar peso 1, normalizamos para evitar
        # output con magnitud reducida.
        cum_p_safe = self._cum_p.clamp(min=self.eps)
        output = self._cum_out / cum_p_safe.unsqueeze(-1)

        # Ponder cost: media de halt_steps + 1 (cantidad de pasos usados)
        ponder = (self._halt_step.float() + 1.0).mean()

        result = ACTOutput(
            output=output,
            halt_steps=self._halt_step.clone(),
            ponder_cost=ponder,
            n_steps_used=self._steps_done,
        )
        # Limpiar para reutilizar el módulo
        self._reset_state()
        return result
