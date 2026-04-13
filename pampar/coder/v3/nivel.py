# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Nivel profundo de la arquitectura 2D de PamparV3.

Componentes:
  TalamoNivel   — Re-routing ligero del Tálamo en cada nivel
  LateralGate   — Comunicación lateral entre streams (fibras blancas)
  NivelProfundo — Un nivel completo: atención + re-route + FFN + lateral + exit
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn import BloqueAttn
from .config import ConfigV3
from .ffn import ContextModulator, StreamFFN
from .norm import RMSNorm

if TYPE_CHECKING:
    from .engrama_stream import BancoEngrama


class TalamoNivel(nn.Module):
    """
    Re-routing ligero del Tálamo aplicado en cada nivel de profundidad.

    El Tálamo inicial hace el routing completo (LLAVES + attn_proj).
    En cada nivel subsiguiente, el modelo re-evalúa las activaciones
    territoriales basándose en el estado ACTUAL de los streams —
    no en los tokens originales.

    Esto permite que si `for` empieza como "control" (B06_KW_LOOP),
    pero en el nivel 3 el modelo detecta que es parte de un comprehension
    complejo, el routing puede deslizarse hacia "semántica".

    Parámetros: dim → 52 (Linear sin bias, muy barato ~33K params).
    """

    def __init__(self, config: ConfigV3):
        super().__init__()
        self.n_zonas = config.n_zonas
        self.n_territorios = config.n_territorios
        self.peso_previo = 0.7  # 70% routing previo, 30% re-evaluación

        # Proyección ligera: estado actual → zonas
        self.zone_proj = nn.Linear(config.dim, config.n_zonas, bias=False)

    def forward(
        self,
        x_combined: torch.Tensor,  # [B, L, D] estado combinado de streams
        terr_acts_prev: torch.Tensor,  # [B, L, 4] activaciones territoriales previas
        agregar_fn,  # función agregar_zonas_a_territorios del Tálamo inicial
    ) -> torch.Tensor:
        """
        Actualiza activaciones territoriales con el estado actual.

        Returns:
            terr_acts: [B, L, 4] activaciones actualizadas
        """
        # Nueva evaluación de zonas desde el estado actual
        zonas_nuevas = torch.sigmoid(self.zone_proj(x_combined))  # [B, L, 52]

        # Agregar zonas a territorios
        terr_nuevo = agregar_fn(zonas_nuevas)  # [B, L, 4]

        # Mezclar con el routing previo: suavidad para evitar oscilaciones
        terr_acts = self.peso_previo * terr_acts_prev + (
            1 - self.peso_previo
        ) * torch.sigmoid(terr_nuevo)
        return terr_acts


class LateralGate(nn.Module):
    """
    Comunicación lateral entre streams — las "fibras blancas" del cerebro.

    Cuando SINTAXIS está procesando fuertemente (activación 0.9), comparte
    parte de su representación con SEMANTICA (que está en 0.7), ayudándola
    a entender mejor el contexto estructural del token actual.

    La contribución de cada stream vecino se pondera por su activación
    territorial — streams muy activos aportan más a sus vecinos.

    Arquitectura del gate por stream:
        input: representaciones de los OTROS 3 streams [B, L, D×3]  →
        bottleneck: Linear(D×3, bottleneck) → SiLU → Linear(bottleneck, D)
        output: aporte lateral [B, L, D]

    Parámetros por stream: D×3→128 + 128→D ≈ 328K × 4 streams = 1.3M/nivel.
    """

    def __init__(self, config: ConfigV3):
        super().__init__()
        self.n_streams = config.n_streams
        bn = config.lateral_bottleneck

        # Un gate por stream (recibe de los otros n_streams-1)
        others = config.n_streams - 1
        self.gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.dim * others, bn, bias=False),
                    nn.SiLU(),
                    nn.Linear(bn, config.dim, bias=False),
                )
                for _ in range(config.n_streams)
            ]
        )

        # Escala de contribución lateral (learnable, inicia pequeño)
        self.scale = nn.Parameter(torch.full((config.n_streams,), 0.1))

    def forward(
        self,
        streams: List[torch.Tensor],  # [n_streams × [B, L, D]]
        terr_acts: torch.Tensor,  # [B, L, n_streams] peso de cada stream
    ) -> List[torch.Tensor]:
        """
        Permite que cada stream reciba aporte de sus peers.

        Returns:
            streams actualizados: [n_streams × [B, L, D]]
        """
        out = []
        for t in range(self.n_streams):
            # Recolectar representaciones de los OTROS streams
            others = [streams[k] for k in range(self.n_streams) if k != t]

            # Ponderar cada vecino por su activación territorial
            weighted_others = []
            other_idx = 0
            for k in range(self.n_streams):
                if k == t:
                    continue
                w = terr_acts[:, :, k : k + 1]  # [B, L, 1]
                weighted_others.append(others[other_idx] * w)
                other_idx += 1

            # Con n_streams=1 no hay peers — skip lateral
            if not weighted_others:
                out.append(streams[t])
                continue

            # Concatenar y proyectar
            lateral_input = torch.cat(weighted_others, dim=-1)  # [B, L, D*(n-1)]
            lateral_out = self.gates[t](lateral_input)  # [B, L, D]

            # Aporte lateral escalado
            streams_t_updated = streams[t] + self.scale[t] * lateral_out
            out.append(streams_t_updated)

        return out


class NivelProfundo(nn.Module):
    """
    Un nivel de profundidad de la arquitectura 2D.

    Cada nivel contiene:
      1. Atención GQA compartida — todos los streams ven el mismo contexto
      2. Re-routing del Tálamo — actualizar activaciones con estado actual
      3. FFN + modulación por stream (Mixed Selectivity o 4× FFN legacy)
      4. Lateral gates — streams se comunican entre sí
      5. Exit head — confianza para Early Exit

    Mixed Selectivity (use_mixed_selectivity=True):
      1 FFN compartido + 4 ContextModulator → misma memoria, 4 lecturas
      Ahorro: ~6.4M params/nivel vs 4 FFN separados.

    Legacy (use_mixed_selectivity=False):
      4 StreamFFN independientes (comportamiento original v3).

    Flujo por nivel:
        streams[t] previos
            ↓
        x_combined = suma ponderada de streams (por terr_acts)
            ↓ (atención compartida)
        x_attn = BloqueAttn(x_combined)
            ↓
        terr_acts = TalamoNivel(x_combined + x_attn, terr_acts_prev)
            ↓ (FFN compartido → modulación por stream)
        h_base = FFN(RMSNorm(streams[t] + x_attn))
        h[t] = ContextModulator(h_base, ctx) × terr_acts[t]
        streams[t] = streams[t] + h[t]
            ↓ (lateral gates)
        streams = LateralGate(streams, terr_acts)
            ↓
        confianza = exit_head(x_combined)
    """

    def __init__(self, config: ConfigV3, nivel_idx: int = 0):
        super().__init__()
        self.config = config
        self.nivel_idx = nivel_idx
        self._use_mixed = config.use_mixed_selectivity

        # Norm clamping adaptativo por nivel (previene explosión de activaciones)
        self._stream_max_norm = 50.0 * (
            2.0**nivel_idx
        )  # N0=50, N1=100, N2=200, N3=400, N4=800
        self._use_norm_clamp = True  # Clamp en inference
        self._train_norm_clamp = (
            False  # Clamp también en training (activar con set_train_clamp)
        )

        # Pre-norms
        self.norm_attn = RMSNorm(config.dim)
        self.norm_streams = nn.ModuleList(
            [RMSNorm(config.dim) for _ in range(config.n_streams)]
        )

        # Atención compartida (una por nivel)
        self.attn = BloqueAttn(config)

        # Re-routing ligero
        self.talamo_nivel = TalamoNivel(config)

        # ── FFN: Mixed Selectivity vs Legacy ─────────────────────────────────
        if self._use_mixed:
            # 1 FFN compartido + 4 moduladores contextuales
            self.ffn_shared = StreamFFN(config)
            self.modulators = nn.ModuleList(
                [ContextModulator(config) for _ in range(config.n_streams)]
            )
        else:
            # Legacy: 4 FFN independientes
            self.ffns = nn.ModuleList(
                [StreamFFN(config) for _ in range(config.n_streams)]
            )

        # Lateral gates
        self.lateral = LateralGate(config)

        # Dropout residual
        self.drop = nn.Dropout(config.dropout)

        # Exit head: confianza basada en el estado combinado
        self.exit_head = nn.Linear(config.dim, 1, bias=False)

    def forward(
        self,
        streams: List[torch.Tensor],  # [n_streams × [B, L, D]]
        terr_acts: torch.Tensor,  # [B, L, n_territorios]
        agregar_fn,  # función del Tálamo para agregar zonas
        banco_engrama: Optional[BancoEngrama] = None,
        zona_acts: Optional[torch.Tensor] = None,  # [B, L, 52]
    ) -> Tuple[List[torch.Tensor], torch.Tensor, float]:
        """
        Forward de un nivel de profundidad.

        Args:
            streams:        representaciones por stream [n_streams × [B, L, D]]
            terr_acts:      activaciones territoriales [B, L, 4]
            agregar_fn:     función del Tálamo para agregar zonas
            banco_engrama:  banco de engramas para inyección (None = sin inyección)
            zona_acts:      activaciones por zona para clave de búsqueda

        Returns:
            streams:   representaciones actualizadas [n_streams × [B, L, D]]
            terr_acts: activaciones actualizadas [B, L, 4]
            conf:      confianza para Early Exit (float)
        """
        # 1. Representación combinada ponderada por activación territorial
        x_combined = sum(
            streams[t] * terr_acts[:, :, t : t + 1]
            for t in range(self.config.n_streams)
        )  # [B, L, D]

        # 2. Atención compartida en el espacio combinado
        x_attn = self.drop(self.attn(self.norm_attn(x_combined)))

        # 2.5 Inyección de EngramaStream (si hay banco disponible)
        if banco_engrama is not None and zona_acts is not None:
            with torch.no_grad():
                terr_dom = terr_acts[0].argmax(dim=-1)  # [L]
                zona_dom = zona_acts[0].argmax(dim=-1)  # [L]
                eng_vecs, eng_mask = banco_engrama.buscar_batch(
                    self.nivel_idx, terr_dom, zona_dom, x_attn.device
                )
            if eng_mask.any():
                alpha = 0.03 / (1.0 + 0.5 * self.nivel_idx)
                eng_residual = eng_vecs.unsqueeze(0)  # [1, L, D]
                mask_f = eng_mask.float().unsqueeze(0).unsqueeze(-1)  # [1, L, 1]

                attn_norm = F.normalize(x_attn, dim=-1)
                eng_norm = F.normalize(eng_residual, dim=-1)
                cosine = (attn_norm * eng_norm).sum(dim=-1, keepdim=True)

                cosine_gate = torch.clamp(cosine - 0.3, min=0.0)

                attn_scale = x_attn.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                eng_scale = eng_residual.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                eng_normalized = eng_residual * (attn_scale / eng_scale)

                x_attn = x_attn + alpha * cosine_gate * mask_f * eng_normalized

        # 3. Re-routing del Tálamo con estado actual
        terr_acts = self.talamo_nivel(x_combined + x_attn, terr_acts, agregar_fn)

        # 3.5 Confianza previa para modulación
        conf_value = 0.5  # default neutro
        if self._use_mixed:
            with torch.no_grad():
                _pre_conf = torch.sigmoid(self.exit_head(x_combined + x_attn)).squeeze(
                    -1
                )
                _k = max(1, int(_pre_conf.numel() * self.config.exit_percentile))
                conf_value = (
                    _pre_conf.reshape(-1).topk(_k, largest=False).values.mean().item()
                )

        # 4. FFN — Mixed Selectivity o Legacy
        new_streams = []

        if self._use_mixed:
            for t in range(self.config.n_streams):
                h_normed = self.norm_streams[t](streams[t] + x_attn)
                h_base = self.ffn_shared(h_normed)

                h_mod = self.modulators[t](
                    h_base,
                    zona_acts=zona_acts
                    if zona_acts is not None
                    else torch.zeros(
                        h_base.shape[0],
                        h_base.shape[1],
                        52,
                        device=h_base.device,
                        dtype=h_base.dtype,
                    ),
                    terr_acts=terr_acts,
                    stream_idx=t,
                    nivel_idx=self.nivel_idx,
                    n_levels=self.config.n_levels,
                    conf=conf_value,
                )

                h = h_mod * terr_acts[:, :, t : t + 1]
                new_streams.append(streams[t] + self.drop(h))
        else:
            for t in range(self.config.n_streams):
                h_normed = self.norm_streams[t](streams[t] + x_attn)
                h = self.ffns[t](h_normed) * terr_acts[:, :, t : t + 1]
                new_streams.append(streams[t] + self.drop(h))

        # 5. Lateral gates — los streams se comunican
        streams = self.lateral(new_streams, terr_acts)

        # 5.5 Norm clamping — previene explosión de activaciones
        clamp_active = self._use_norm_clamp and (
            not self.training or self._train_norm_clamp
        )
        if clamp_active:
            max_norm = self._stream_max_norm
            for t in range(self.config.n_streams):
                norms = streams[t].norm(dim=-1, keepdim=True)
                scale = torch.clamp(max_norm / norms.clamp(min=1e-8), max=1.0)
                streams[t] = streams[t] * scale

        # 6. Confianza para Early Exit (percentil 10 de tokens más difíciles)
        x_out = sum(
            streams[t] * terr_acts[:, :, t : t + 1]
            for t in range(self.config.n_streams)
        )
        per_token_conf = torch.sigmoid(self.exit_head(x_out)).squeeze(-1)
        k = max(1, int(per_token_conf.numel() * self.config.exit_percentile))
        conf = per_token_conf.reshape(-1).topk(k, largest=False).values.mean().item()

        return streams, terr_acts, conf
