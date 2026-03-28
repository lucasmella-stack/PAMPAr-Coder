# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Bloques de la arquitectura 2D de PamparV3.

Componentes:
  RMSNorm      — Normalización eficiente (Llama-style)
  RoPE         — Rotary Position Embedding para generalización posicional
  BloqueAttn   — Atención GQA + Flash Attention (compartida entre streams)
  StreamFFN    — Feed-forward SwiGLU especializado por stream
  TalamoNivel  — Re-routing ligero del Tálamo en cada nivel de profundidad
  LateralGate  — Comunicación lateral entre streams (fibras blancas)
  NivelProfundo— Un nivel completo: atención + re-route + 4 FFN + lateral
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, TYPE_CHECKING, Tuple

from .config import ConfigV3

if TYPE_CHECKING:
    from .engrama_stream import BancoEngrama


# =============================================================================
# RMS NORMALIZATION
# =============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Más eficiente que LayerNorm: omite el centrado, solo normaliza por RMS.
    Usado en Llama, Qwen, Mistral.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).type_as(x) * self.weight


# =============================================================================
# ROTARY POSITION EMBEDDING (RoPE)
# =============================================================================

class RoPE(nn.Module):
    """
    Rotary Position Embedding (Su et al., 2021).

    Codifica posiciones como rotaciones complejas en Q y K.
    Zero parámetros extra (solo buffers pre-computados).
    Generaliza naturalmente a secuencias más largas que el training.
    """

    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        pos = torch.arange(max_seq_len)
        freqs = torch.outer(pos, inv_freq)
        self.register_buffer("cos_cache", freqs.cos())
        self.register_buffer("sin_cache", freqs.sin())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aplica RoPE a tensor [B, H, L, D]."""
        L = x.shape[2]
        cos = self.cos_cache[:L].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cache[:L].unsqueeze(0).unsqueeze(0)
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# =============================================================================
# ATENCIÓN GQA
# =============================================================================

class BloqueAttn(nn.Module):
    """
    Multi-head attention con GQA (Grouped Query Attention) y Flash Attention.

    GQA: 8 Q heads, 2 KV heads → ratio 4:1 → KV cache 4× más pequeño.
    Flash Attention vía F.scaled_dot_product_attention (PyTorch 2.0+).
    La máscara causal se aplica con is_causal=True sin materializar el tensor.

    Esta atención es COMPARTIDA: todos los streams la alimentan con
    una representación ponderada y reciben el output para contextualizarse.
    """

    def __init__(self, config: ConfigV3):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.kv_heads
        self.head_dim = config.head_dim
        self.dim = config.dim
        self.n_rep = config.n_rep
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.dim, self.n_heads * self.head_dim, bias=False)
        kv_dim = self.n_kv_heads * self.head_dim
        self.k_proj = nn.Linear(config.dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, kv_dim, bias=False)
        self.o_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.rope = RoPE(config.head_dim, config.max_seq_len)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """[B, n_kv, L, D] → [B, n_heads, L, D] para GQA."""
        if self.n_rep == 1:
            return x
        B, H, L, D = x.shape
        return (
            x.unsqueeze(2)
            .expand(B, H, self.n_rep, L, D)
            .reshape(B, H * self.n_rep, L, D)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] representación combinada de los streams
        Returns:
            [B, L, D] contexto enriquecido
        """
        B, L, _ = x.shape

        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        ).transpose(1, 2).reshape(B, L, self.dim)

        return self.o_proj(out)


# =============================================================================
# STREAM FFN (SWIGLU)
# =============================================================================

class StreamFFN(nn.Module):
    """
    Feed-forward SwiGLU especializado por stream territorial.

    SwiGLU = SiLU(gate) ⊙ up → down.
    Cada stream (SINTAXIS/SEMANTICA/LOGICO/ESTRUCTURAL) tiene su propio
    conjunto de pesos — como neuronas de áreas corticales distintas.

    hidden_dim = 2/3 × dim × ffn_mult (compensa la gate extra de SwiGLU).
    """

    def __init__(self, config: ConfigV3):
        super().__init__()
        hidden = config.ffn_hidden

        self.gate = nn.Linear(config.dim, hidden, bias=False)
        self.up = nn.Linear(config.dim, hidden, bias=False)
        self.down = nn.Linear(hidden, config.dim, bias=False)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU: SiLU(gate(x)) ⊙ up(x) → down."""
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


# =============================================================================
# TÁLAMO POR NIVEL (RE-ROUTING LIGERO)
# =============================================================================

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
        x_combined: torch.Tensor,   # [B, L, D] estado combinado de streams
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
        terr_acts = (
            self.peso_previo * terr_acts_prev
            + (1 - self.peso_previo) * torch.sigmoid(terr_nuevo)
        )
        return terr_acts


# =============================================================================
# LATERAL GATE (FIBRAS BLANCAS)
# =============================================================================

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
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.dim * others, bn, bias=False),
                nn.SiLU(),
                nn.Linear(bn, config.dim, bias=False),
            )
            for _ in range(config.n_streams)
        ])

        # Escala de contribución lateral (learnable, inicia pequeño)
        self.scale = nn.Parameter(torch.full((config.n_streams,), 0.1))

    def forward(
        self,
        streams: List[torch.Tensor],  # [n_streams × [B, L, D]]
        terr_acts: torch.Tensor,       # [B, L, n_streams] peso de cada stream
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
            # (streams más activos aportan más información)
            weighted_others = []
            other_idx = 0
            for k in range(self.n_streams):
                if k == t:
                    continue
                # terr_acts[:, :, k] = activación del stream k [B, L]
                w = terr_acts[:, :, k:k+1]  # [B, L, 1]
                weighted_others.append(others[other_idx] * w)
                other_idx += 1

            # Concatenar y proyectar
            lateral_input = torch.cat(weighted_others, dim=-1)  # [B, L, D*(n-1)]
            lateral_out = self.gates[t](lateral_input)           # [B, L, D]

            # Aporte lateral escalado (inicia en 0.1, el modelo aprende cuánto)
            streams_t_updated = streams[t] + self.scale[t] * lateral_out
            out.append(streams_t_updated)

        return out


# =============================================================================
# NIVEL PROFUNDO (UN NIVEL DE LA GRILLA 2D)
# =============================================================================

class NivelProfundo(nn.Module):
    """
    Un nivel de profundidad de la arquitectura 2D.

    Cada nivel contiene:
      1. Atención GQA compartida — todos los streams ven el mismo contexto
      2. Re-routing del Tálamo — actualizar activaciones con estado actual
      3. 4 × StreamFFN especializados — cada stream refina su representación
      4. Lateral gates — streams se comunican entre sí
      5. Exit head — confianza para Early Exit

    Flujo por nivel:
        streams[t] previos
            ↓
        x_combined = suma ponderada de streams (por terr_acts)
            ↓ (atención compartida)
        x_attn = BloqueAttn(x_combined)
            ↓
        terr_acts = TalamoNivel(x_combined + x_attn, terr_acts_prev)
            ↓ (4 FFN en paralelo, cada uno modulado por su territorio)
        h[t] = RMSNorm(streams[t] + x_attn) → StreamFFN → × terr_acts[t]
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

        # Norm clamping adaptativo por nivel (previene explosión de activaciones)
        # max_norm crece con el nivel para permitir acumulación controlada
        # En eval: clampea; en train: no interfiere (el modelo ya aprendió así)
        self._stream_max_norm = 50.0 * (2.0 ** nivel_idx)  # N0=50, N1=100, N2=200, N3=400, N4=800
        self._use_norm_clamp = True  # Activar en inference

        # Pre-norms
        self.norm_attn = RMSNorm(config.dim)
        self.norm_streams = nn.ModuleList([
            RMSNorm(config.dim) for _ in range(config.n_streams)
        ])

        # Atención compartida (una por nivel)
        self.attn = BloqueAttn(config)

        # Re-routing ligero (una por nivel, salvo el primero que usa TalamoInicial)
        self.talamo_nivel = TalamoNivel(config)

        # FFN especializados por stream
        self.ffns = nn.ModuleList([
            StreamFFN(config) for _ in range(config.n_streams)
        ])

        # Lateral gates
        self.lateral = LateralGate(config)

        # Dropout residual
        self.drop = nn.Dropout(config.dropout)

        # Exit head: confianza basada en el estado combinado
        self.exit_head = nn.Linear(config.dim, 1, bias=False)

    def forward(
        self,
        streams: List[torch.Tensor],   # [n_streams × [B, L, D]]
        terr_acts: torch.Tensor,        # [B, L, n_territorios]
        agregar_fn,                     # función del Tálamo para agregar zonas
        banco_engrama: Optional[BancoEngrama] = None,
        zona_acts: Optional[torch.Tensor] = None,  # [B, L, 52] para clave de búsqueda
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
        #    → el stream más activo domina el contexto que ve la atención
        x_combined = sum(
            streams[t] * terr_acts[:, :, t:t+1]
            for t in range(self.config.n_streams)
        )  # [B, L, D]

        # 2. Atención compartida en el espacio combinado
        x_attn = self.drop(self.attn(self.norm_attn(x_combined)))

        # 2.5 Inyección de EngramaStream (si hay banco disponible)
        if banco_engrama is not None and zona_acts is not None:
            with torch.no_grad():
                terr_dom = terr_acts[0].argmax(dim=-1)  # [L]
                zona_dom = zona_acts[0].argmax(dim=-1)   # [L]
                eng_vecs, eng_mask = banco_engrama.buscar_batch(
                    self.nivel_idx, terr_dom, zona_dom, x_attn.device
                )
            if eng_mask.any():
                # Filtrar por similitud coseno: solo inyectar si el engrama
                # apunta en una dirección similar al x_attn actual
                alpha = 0.1
                eng_residual = eng_vecs.unsqueeze(0)  # [1, L, D]
                mask_f = eng_mask.float().unsqueeze(0).unsqueeze(-1)  # [1, L, 1]

                # Normalizar ambos para calcular coseno per-token
                attn_norm = F.normalize(x_attn, dim=-1)
                eng_norm = F.normalize(eng_residual, dim=-1)
                cosine = (attn_norm * eng_norm).sum(dim=-1, keepdim=True)  # [B, L, 1]

                # Solo inyectar donde coseno > 0 (dirección compatible)
                cosine_gate = torch.clamp(cosine, min=0.0)  # [B, L, 1]

                # Normalizar engrama a escala de x_attn
                attn_scale = x_attn.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                eng_scale = eng_residual.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                eng_normalized = eng_residual * (attn_scale / eng_scale)

                x_attn = x_attn + alpha * cosine_gate * mask_f * eng_normalized

        # 3. Re-routing del Tálamo con estado actual
        terr_acts = self.talamo_nivel(
            x_combined + x_attn, terr_acts, agregar_fn
        )

        # 4. Cada stream procesa: su estado + aporte de atención → FFN
        new_streams = []
        for t in range(self.config.n_streams):
            # RMSNorm sobre el estado del stream enriquecido con atención
            h_normed = self.norm_streams[t](streams[t] + x_attn)
            # FFN especializado modulado por su activación territorial
            h = self.ffns[t](h_normed) * terr_acts[:, :, t:t+1]
            # Residual
            new_streams.append(streams[t] + self.drop(h))

        # 5. Lateral gates — los streams se comunican
        streams = self.lateral(new_streams, terr_acts)

        # 5.5 Norm clamping — previene explosión de activaciones en inference
        if self._use_norm_clamp and not self.training:
            max_norm = self._stream_max_norm
            for t in range(self.config.n_streams):
                norms = streams[t].norm(dim=-1, keepdim=True)  # [B, L, 1]
                scale = torch.clamp(max_norm / norms.clamp(min=1e-8), max=1.0)
                streams[t] = streams[t] * scale

        # 6. Confianza para Early Exit (percentil 10 de tokens más difíciles)
        x_out = sum(
            streams[t] * terr_acts[:, :, t:t+1]
            for t in range(self.config.n_streams)
        )
        per_token_conf = torch.sigmoid(self.exit_head(x_out)).squeeze(-1)  # [B, L]
        k = max(1, int(per_token_conf.numel() * self.config.exit_percentile))
        conf = per_token_conf.reshape(-1).topk(k, largest=False).values.mean().item()

        return streams, terr_acts, conf
