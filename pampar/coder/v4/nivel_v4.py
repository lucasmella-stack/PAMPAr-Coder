# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
NivelProfundoV4 — nivel de profundidad con FFN/modulator V4.

Razón de existir:
    `NivelProfundo` v3 está congelado y construye `ContextModulator` v3
    (63 dims, sin slots `loop_idx` / `modality_id`). Para que el cerebro
    *sepa* en qué iteración del recurrent loop está y qué modalidad
    procesa, necesitamos modulators V4 (71 dims).

Dos modos de modulator (vía `ConfigV4.use_hierarchical_modulators`):

  - independent (default): un `ContextModulatorV4` por stream, propio
    de cada nivel. Equivalente directo a v3 pero con contexto enriquecido.

  - hierarchical: un único `HierarchicalModulator` (backbone compartido)
    inyectado externamente por `PamparV4`. Mismas cabezas (level, stream)
    pero un solo backbone para todo el modelo → consistencia inductiva.

`NivelProfundoV4.forward` acepta `loop_idx`, `max_loops`, `modality_id`
y los propaga a los modulators. En path A (stack secuencial) son
`loop_idx=0`, `max_loops=1`. En path B (recurrent), `RecurrentNivelAdapter`
los actualiza por iteración.

Compatibilidad:
    - Hereda de `NivelProfundo` v3 → reusa `attn`, `talamo_nivel`,
      `lateral`, `exit_head`, `norm_*`, `_stream_max_norm`, etc.
    - El __init__ de v3 construye `self.modulators` (v3) que se
      descartan inmediatamente. No quedan en el grafo de parámetros
      tras la sustitución.
    - Mixed selectivity: usa `ffn_shared` + modulators V4. Modo legacy
      (`use_mixed_selectivity=False`) hereda comportamiento v3 sin cambios.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pampar.coder.v3.nivel import NivelProfundo
from pampar.coder.v3.talamo import TalamoInicial

from .config import ConfigV4
from .ffn import ContextModulatorV4
from .hierarchical import HierarchicalModulator
from .modalities import ModalityId

if TYPE_CHECKING:
    from pampar.coder.v3.engrama_stream import BancoEngrama


class NivelProfundoV4(NivelProfundo):
    """
    Nivel V4: cerebro v3 + modulators V4 con `loop_idx` propagado.

    Args:
        config: `ConfigV4`.
        nivel_idx: índice del nivel (0..n_levels-1).
        hierarchical_modulator: si se provee, los modulators internos
            no se construyen y se delega en este modulator compartido
            (Fase 2 — `HierarchicalModulator`).

    Notas:
        - En modo legacy (`use_mixed_selectivity=False`) NO se usan
          modulators (ni v3 ni V4). El comportamiento es idéntico a v3.
        - `loop_idx`/`max_loops`/`modality_id` solo afectan el contexto
          de los modulators. Si están en sus defaults, el modulator V4
          es numéricamente equivalente al v3 (los slots extra son ceros).
    """

    def __init__(
        self,
        config: ConfigV4,
        nivel_idx: int = 0,
        hierarchical_modulator: Optional[HierarchicalModulator] = None,
    ):
        super().__init__(config, nivel_idx=nivel_idx)
        self._uses_hierarchical = hierarchical_modulator is not None
        # Guardamos la ref como atributo NO-Module para evitar que
        # PyTorch la registre como submódulo (sería duplicado: el
        # modulator vive en PamparV4 y se comparte entre todos los
        # niveles V4).
        object.__setattr__(self, "_hierarchical_modulator", hierarchical_modulator)

        if not self._use_mixed:
            # Legacy path: ningún modulator (ni v3 ni V4). Liberar los
            # módulos vacíos que pudieran haberse construido por v3
            # (en realidad v3 tampoco los construye en legacy, pero
            # explicitamos para mantener la simetría).
            return

        # Mixed selectivity path: reemplazar modulators v3 por V4 o
        # eliminar completamente si delegamos en el hierarchical.
        del self.modulators

        if self._uses_hierarchical:
            # Sentinel vacío para no romper consumers que iteren sobre
            # el atributo (no se usa en el forward).
            self.modulators = nn.ModuleList()
        else:
            self.modulators = nn.ModuleList(
                [ContextModulatorV4(config) for _ in range(config.n_streams)]
            )

    # ────────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────────

    def _modulate(
        self,
        h_base: torch.Tensor,
        zona_acts: torch.Tensor,
        terr_acts: torch.Tensor,
        stream_idx: int,
        conf_value: float,
        loop_idx: int,
        max_loops: int,
        modality_id: int,
    ) -> torch.Tensor:
        """Dispatcha al modulator apropiado (independiente o jerárquico).

        Si `config.disable_loop_idx_in_modulator` está activo (ablación),
        fuerza `loop_idx=0`/`max_loops=1` en el contexto. El recurrent
        loop sigue ejecutándose con T iteraciones reales — solo el
        modulator queda ciego a su posición temporal.
        """
        if self.config.disable_loop_idx_in_modulator:
            loop_idx = 0
            max_loops = 1
        if self._uses_hierarchical:
            return self._hierarchical_modulator(
                h_base,
                zona_acts=zona_acts,
                terr_acts=terr_acts,
                stream_idx=stream_idx,
                nivel_idx=self.nivel_idx,
                n_levels=self.config.n_levels,
                conf=conf_value,
                loop_idx=loop_idx,
                max_loops=max_loops,
                modality_id=modality_id,
            )
        return self.modulators[stream_idx](
            h_base,
            zona_acts=zona_acts,
            terr_acts=terr_acts,
            stream_idx=stream_idx,
            nivel_idx=self.nivel_idx,
            n_levels=self.config.n_levels,
            conf=conf_value,
            loop_idx=loop_idx,
            max_loops=max_loops,
            modality_id=modality_id,
        )

    # ────────────────────────────────────────────────────────────────────
    # Forward
    # ────────────────────────────────────────────────────────────────────

    def forward(
        self,
        streams: List[torch.Tensor],
        terr_acts: torch.Tensor,
        agregar_fn,
        banco_engrama: Optional["BancoEngrama"] = None,
        zona_acts: Optional[torch.Tensor] = None,
        loop_idx: int = 0,
        max_loops: int = 1,
        modality_id: int = ModalityId.TEXT,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, float]:
        """
        Forward del nivel V4.

        Idéntico a v3 excepto:
          - El bloque FFN (paso 4) usa modulators V4 con contexto extendido.
          - Acepta `loop_idx`, `max_loops`, `modality_id` y los propaga
            a `_modulate(...)`.
        """
        # 1. Combinado por activación territorial
        x_combined = sum(
            streams[t] * terr_acts[:, :, t : t + 1]
            for t in range(self.config.n_streams)
        )

        # 2. Atención compartida
        x_attn = self.drop(self.attn(self.norm_attn(x_combined)))

        # 2.5 Inyección de EngramaStream (idéntico a v3)
        if banco_engrama is not None and zona_acts is not None:
            with torch.no_grad():
                terr_dom = terr_acts[0].argmax(dim=-1)
                zona_dom = zona_acts[0].argmax(dim=-1)
                eng_vecs, eng_mask = banco_engrama.buscar_batch(
                    self.nivel_idx, terr_dom, zona_dom, x_attn.device
                )
            if eng_mask.any():
                alpha = 0.03 / (1.0 + 0.5 * self.nivel_idx)
                eng_residual = eng_vecs.unsqueeze(0)
                mask_f = eng_mask.float().unsqueeze(0).unsqueeze(-1)
                attn_norm = F.normalize(x_attn, dim=-1)
                eng_norm = F.normalize(eng_residual, dim=-1)
                cosine = (attn_norm * eng_norm).sum(dim=-1, keepdim=True)
                cosine_gate = torch.clamp(cosine - 0.3, min=0.0)
                attn_scale = x_attn.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                eng_scale = eng_residual.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                eng_normalized = eng_residual * (attn_scale / eng_scale)
                x_attn = x_attn + alpha * cosine_gate * mask_f * eng_normalized

        # 3. Re-routing
        terr_acts = self.talamo_nivel(x_combined + x_attn, terr_acts, agregar_fn)

        # 3.5 Confianza previa para modulación
        conf_value = 0.5
        if self._use_mixed:
            with torch.no_grad():
                _pre_conf = torch.sigmoid(self.exit_head(x_combined + x_attn)).squeeze(
                    -1
                )
                _k = max(1, int(_pre_conf.numel() * self.config.exit_percentile))
                conf_value = (
                    _pre_conf.reshape(-1).topk(_k, largest=False).values.mean().item()
                )

        # 4. FFN — V4 con loop_idx / modality_id propagados
        new_streams = []
        if self._use_mixed:
            zero_zona = (
                zona_acts
                if zona_acts is not None
                else torch.zeros(
                    streams[0].shape[0],
                    streams[0].shape[1],
                    self.config.n_zonas,
                    device=streams[0].device,
                    dtype=streams[0].dtype,
                )
            )
            for t in range(self.config.n_streams):
                h_normed = self.norm_streams[t](streams[t] + x_attn)
                h_base = self.ffn_shared(h_normed)
                h_mod = self._modulate(
                    h_base,
                    zona_acts=zero_zona,
                    terr_acts=terr_acts,
                    stream_idx=t,
                    conf_value=conf_value,
                    loop_idx=loop_idx,
                    max_loops=max_loops,
                    modality_id=modality_id,
                )
                h = h_mod * terr_acts[:, :, t : t + 1]
                new_streams.append(streams[t] + self.drop(h))
        else:
            for t in range(self.config.n_streams):
                h_normed = self.norm_streams[t](streams[t] + x_attn)
                h = self.ffns[t](h_normed) * terr_acts[:, :, t : t + 1]
                new_streams.append(streams[t] + self.drop(h))

        # 5. Lateral gates
        streams = self.lateral(new_streams, terr_acts)

        # 5.5 Norm clamping
        clamp_active = self._use_norm_clamp and (
            not self.training or self._train_norm_clamp
        )
        if clamp_active:
            max_norm = self._stream_max_norm
            for t in range(self.config.n_streams):
                norms = streams[t].norm(dim=-1, keepdim=True)
                scale = torch.clamp(max_norm / norms.clamp(min=1e-8), max=1.0)
                streams[t] = streams[t] * scale

        # 6. Confianza para Early Exit
        x_out = sum(
            streams[t] * terr_acts[:, :, t : t + 1]
            for t in range(self.config.n_streams)
        )
        per_token_conf = torch.sigmoid(self.exit_head(x_out)).squeeze(-1)
        k = max(1, int(per_token_conf.numel() * self.config.exit_percentile))
        conf = per_token_conf.reshape(-1).topk(k, largest=False).values.mean().item()

        return streams, terr_acts, conf
