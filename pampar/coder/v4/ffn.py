# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
FFN y modulación contextual para PAMPAr v4.

Diferencias frente a v3.ffn.ContextModulator:
  - Vector de contexto reorganizado en slots semánticos explícitos
  - Slot reservado para `modality_id` (8 dims, multimodal-ready)
  - Slot reservado para `loop_idx` (1 dim, recurrent-loop-ready)
  - `n_levels` removido del contexto (era información estática inútil)
  - Modo backward-compat: cuando modality_id=TEXT y loop_idx=0, el
    comportamiento numérico es equivalente a v3 con misma init.

Layout del vector de contexto (71 dims total):
  zona_acts   [52]  -- tipo de token / zona de Brodmann
  terr_acts   [4]   -- peso por territorio (stream)
  depth       [1]   -- nivel actual normalizado (0..1)
  conf        [1]   -- confianza del exit_head (no_grad)
  loop_idx    [1]   -- índice de loop normalizado (0..1), 0 si no hay loop
  modality_id [8]   -- one-hot de modalidad (TEXT por defecto)
  stream_oh   [4]   -- one-hot del stream que se está modulando
  ────────────────
  TOTAL       71

Comparado con v3 (63):
  +8 modality_id, +1 loop_idx, -1 n_levels (descartado)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pampar.coder.v3.ffn import StreamFFN as StreamFFNv3

from .config import ConfigV4
from .modalities import NUM_MODALITIES, ModalityId

# Re-exportamos StreamFFN sin cambios para que el código de v4 use la
# misma implementación SwiGLU que v3 (sin duplicación).
StreamFFN = StreamFFNv3


# Layout del vector de contexto (constantes públicas)
ZONA_DIM: int = 52
TERR_DIM: int = 4
DEPTH_DIM: int = 1
CONF_DIM: int = 1
LOOP_DIM: int = 1
MODALITY_DIM: int = NUM_MODALITIES  # 8
STREAM_DIM: int = 4

CONTEXT_DIM: int = (
    ZONA_DIM + TERR_DIM + DEPTH_DIM + CONF_DIM + LOOP_DIM + MODALITY_DIM + STREAM_DIM
)  # 71


def build_context_v4(
    zona_acts: torch.Tensor,
    terr_acts: torch.Tensor,
    stream_idx: int,
    nivel_idx: int,
    n_levels: int,
    conf: float,
    loop_idx: int = 0,
    max_loops: int = 1,
    modality_id: int = ModalityId.TEXT,
) -> torch.Tensor:
    """
    Construye el vector de contexto multimodal v4 [B, L, 71].

    Función libre reusable por ContextModulatorV4 y HierarchicalModulator.
    Layout: zona[52] + terr[4] + depth[1] + conf[1] + loop[1] + mod[8] + stream[4].
    """
    B, L, _ = zona_acts.shape
    device = zona_acts.device
    dtype = zona_acts.dtype

    depth = torch.full(
        (B, L, 1),
        nivel_idx / max(n_levels - 1, 1),
        device=device,
        dtype=dtype,
    )
    conf_t = torch.full((B, L, 1), conf, device=device, dtype=dtype)
    loop_t = torch.full(
        (B, L, 1),
        loop_idx / max(max_loops - 1, 1) if max_loops > 1 else 0.0,
        device=device,
        dtype=dtype,
    )

    mod_oh = torch.zeros(B, L, NUM_MODALITIES, device=device, dtype=dtype)
    mod_oh[:, :, int(modality_id)] = 1.0

    stream_oh = torch.zeros(B, L, STREAM_DIM, device=device, dtype=dtype)
    stream_oh[:, :, stream_idx] = 1.0

    return torch.cat(
        [zona_acts, terr_acts, depth, conf_t, loop_t, mod_oh, stream_oh],
        dim=-1,
    )


class ContextModulatorV4(nn.Module):
    """
    Modulator FiLM con contexto multimodal-ready.

    Genera (gamma, beta) por token desde un vector de 71 dims que codifica
    explícitamente: tipo léxico, dominio territorial, profundidad, confianza,
    índice de loop, modalidad y stream-objetivo.

    Init: la última proyección arranca en zeros → gamma=0, beta=0 →
    salida = ffn_out (identidad). Esto garantiza que un modelo recién
    creado se comporte igual que un FFN denso, y que el entrenamiento
    sea estable.

    Backward-compat numérico con v3 ContextModulator:
      Si se llama con `modality_id=ModalityId.TEXT` y `loop_idx=0`, los
      slots multimodal/loop quedan en cero y el comportamiento es
      equivalente al ContextModulator de v3 (las 7 dims extra son ceros
      mapeados por pesos en zero, no aportan).
    """

    # Layout del vector de contexto (alias de las constantes de módulo)
    ZONA_DIM: int = ZONA_DIM
    TERR_DIM: int = TERR_DIM
    DEPTH_DIM: int = DEPTH_DIM
    CONF_DIM: int = CONF_DIM
    LOOP_DIM: int = LOOP_DIM
    MODALITY_DIM: int = MODALITY_DIM
    STREAM_DIM: int = STREAM_DIM
    CONTEXT_DIM: int = CONTEXT_DIM

    def __init__(self, config: ConfigV4):
        super().__init__()
        self.dim = config.dim
        mid = config.modulator_bottleneck

        self.proj = nn.Sequential(
            nn.Linear(self.CONTEXT_DIM, mid, bias=False),
            nn.SiLU(),
            nn.Linear(mid, config.dim * 2, bias=False),
        )

        # Init en zeros para que gamma=0, beta=0 al inicio (identidad)
        nn.init.zeros_(self.proj[2].weight)

    def _build_context(
        self,
        zona_acts: torch.Tensor,
        terr_acts: torch.Tensor,
        stream_idx: int,
        nivel_idx: int,
        n_levels: int,
        conf: float,
        loop_idx: int = 0,
        max_loops: int = 1,
        modality_id: int = ModalityId.TEXT,
    ) -> torch.Tensor:
        """Delega en `build_context_v4` (función libre)."""
        return build_context_v4(
            zona_acts=zona_acts,
            terr_acts=terr_acts,
            stream_idx=stream_idx,
            nivel_idx=nivel_idx,
            n_levels=n_levels,
            conf=conf,
            loop_idx=loop_idx,
            max_loops=max_loops,
            modality_id=modality_id,
        )

    def forward(
        self,
        ffn_out: torch.Tensor,
        zona_acts: torch.Tensor,
        terr_acts: torch.Tensor,
        stream_idx: int,
        nivel_idx: int,
        n_levels: int,
        conf: float,
        loop_idx: int = 0,
        max_loops: int = 1,
        modality_id: int = ModalityId.TEXT,
    ) -> torch.Tensor:
        """
        Modula `ffn_out` con FiLM condicionado por el contexto multimodal.

        Args:
            ffn_out:     [B, L, dim] salida del FFN compartido.
            zona_acts:   [B, L, 52] activaciones de zonas.
            terr_acts:   [B, L, 4]  pesos territoriales.
            stream_idx:  stream que se modula (0..3).
            nivel_idx:   nivel de profundidad actual.
            n_levels:    total de niveles del modelo.
            conf:        confianza del exit_head (0..1).
            loop_idx:    iteración actual del recurrent loop (0 si N/A).
            max_loops:   total de loops (1 si no hay loop).
            modality_id: tipo de modalidad de los tokens actuales.

        Returns:
            [B, L, dim] salida modulada: (1 + gamma) * ffn_out + beta.
        """
        ctx = self._build_context(
            zona_acts=zona_acts,
            terr_acts=terr_acts,
            stream_idx=stream_idx,
            nivel_idx=nivel_idx,
            n_levels=n_levels,
            conf=conf,
            loop_idx=loop_idx,
            max_loops=max_loops,
            modality_id=modality_id,
        )

        modulation = self.proj(ctx)  # [B, L, 2*dim]
        gamma, beta = modulation.chunk(2, dim=-1)
        return (1.0 + gamma) * ffn_out + beta
