# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
LTI-stable Input Injection (port simplificado de OpenMythos).

Update rule del recurrent loop:
    h_{t+1} = A * h_t + B * e + delta_t

donde:
    h_t   ∈ R^{B,L,dim}  estado oculto en el paso t
    e     ∈ R^{B,L,dim}  embedding de entrada (constante en todo el loop)
    A     ∈ R^{dim}      diagonal, garantizada en (0, 1) por construcción
    B     ∈ R^{dim,dim}  proyección densa del embedding
    delta_t              salida del bloque recurrente (no es responsabilidad
                         de este módulo; viene de afuera)

Garantía de estabilidad:
    A = exp(-exp(log_A) * exp(log_dt))
    Como exp(.) > 0, los argumentos del segundo exp son negativos → A ∈ (0,1).
    El radio espectral ρ(A) = max(|A_i|) < 1 estrictamente.

Por qué importa: sin esta garantía, un loop recurrente con peso compartido
puede divergir (||h_t|| → ∞) o colapsar (||h_t|| → 0) según los pesos
aprendidos. La parametrización con doble-exp garantiza estabilidad sin
necesidad de gradient clipping ad-hoc ni regularización extra.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LTIInjection(nn.Module):
    """
    Input injection con estabilidad LTI garantizada por construcción.

    Args:
        dim:         dimensión del estado oculto.
        log_a_init:  init para log_A (controla decay base; por default
                     -1.0 → A ≈ exp(-exp(-1) * exp(0)) = exp(-0.368) ≈ 0.69).
        log_dt_init: init para log_dt (controla escala temporal; default 0).

    Args para forward:
        h:  [B, L, dim] estado actual.
        e:  [B, L, dim] embedding de entrada (input fijo del loop).

    Returns:
        h_next: [B, L, dim] = A*h + B*e   (sin delta — eso lo suma quien llame)
    """

    def __init__(
        self,
        dim: int,
        log_a_init: float = -1.0,
        log_dt_init: float = 0.0,
    ):
        super().__init__()
        self.dim = dim

        # Parámetros learnable que parametrizan A para garantizar A ∈ (0,1)
        self.log_A = nn.Parameter(torch.full((dim,), log_a_init))
        self.log_dt = nn.Parameter(torch.full((dim,), log_dt_init))

        # Proyección B del embedding al estado
        self.B = nn.Linear(dim, dim, bias=False)

        # Init B en zeros → al inicio el loop solo decae h sin inyectar e.
        # El entrenamiento aprende cuánto inyectar. Esto evita explosiones
        # iniciales si A es cercano a 1 y e tiene magnitud alta.
        nn.init.zeros_(self.B.weight)

    @property
    def A(self) -> torch.Tensor:
        """
        Diagonal de A, garantizada en (0, 1). Útil para inspección/tests.

        Construcción: A = exp(-exp(log_A) * exp(log_dt))
        """
        return torch.exp(-torch.exp(self.log_A) * torch.exp(self.log_dt))

    @property
    def spectral_radius(self) -> float:
        """ρ(A) = max(|A_i|). Siempre < 1 por construcción."""
        return float(self.A.abs().max())

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """
        Aplica un paso de input injection LTI-stable.

        Args:
            h: [B, L, dim] estado oculto actual.
            e: [B, L, dim] embedding de entrada (constante a lo largo del loop).

        Returns:
            [B, L, dim] estado actualizado: A*h + B*e.
        """
        if h.shape != e.shape:
            raise ValueError(
                f"Shapes incompatibles: h={tuple(h.shape)} vs e={tuple(e.shape)}"
            )
        if h.shape[-1] != self.dim:
            raise ValueError(f"Última dim debe ser {self.dim}, recibido {h.shape[-1]}")

        a = self.A  # [dim]
        return a * h + self.B(e)
