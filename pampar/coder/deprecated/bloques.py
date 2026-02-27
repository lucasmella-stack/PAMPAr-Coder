# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Bloques de procesamiento: Atención y FFN.

Componentes optimizados para escalar de 42M a 1.5B params:
- RoPE: Rotary Position Embedding para generalización posicional
- RMSNorm: Normalización eficiente (vs LayerNorm) — como Llama/Qwen
- BloqueAttn: Multi-head self-attention con RoPE y GQA
- BloqueFFN: Feed-forward con SwiGLU
- BloqueTerritorial: Combina atención + FFN modulados por territorios
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .config import ConfigV2


# =============================================================================
# RMS NORMALIZATION
# =============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Más eficiente que LayerNorm: omite el centrado (resta media),
    solo normaliza por RMS. Usado en Llama, Qwen, Mistral.
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

    Codifica posiciones como rotaciones complejas aplicadas a Q y K.
    Ventajas sobre posicional absoluto:
    - Codifica distancias relativas naturalmente
    - Generaliza a secuencias más largas que el training
    - Zero parámetros extra (solo buffers)
    """

    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Pre-computar cache para todas las posiciones
        pos = torch.arange(max_seq_len)
        freqs = torch.outer(pos, inv_freq)
        self.register_buffer("cos_cache", freqs.cos())
        self.register_buffer("sin_cache", freqs.sin())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aplica RoPE a tensor [B, H, L, D]."""
        L = x.shape[2]
        cos = self.cos_cache[:L].unsqueeze(0).unsqueeze(0)  # [1, 1, L, D/2]
        sin = self.sin_cache[:L].unsqueeze(0).unsqueeze(0)

        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos,
        ], dim=-1)


# =============================================================================
# ATENCIÓN CON GQA
# =============================================================================

class BloqueAttn(nn.Module):
    """
    Multi-head self-attention con RoPE, GQA y Flash Attention.

    GQA (Ainslie et al., 2023): Comparte KV heads entre múltiples Q heads.
    Reduce KV cache en inferencia sin perder calidad.

    Flash Attention (Dao et al., 2022, vía F.scaled_dot_product_attention):
    Implementación kernel-fused de softmax + dropout + matmul.
    Usa 2x menos VRAM y es 2-4x más rápida que la atención manual.
    El mask causal se aplica directamente con is_causal=True.

    Con config.n_kv_heads=0 → MHA estándar (backward compatible).
    Con config.n_kv_heads>0 → GQA con ese número de KV heads.

    Ejemplo PRESET_1_5B: 12 Q heads, 4 KV heads → ratio 3:1
    - KV cache es 3x menor que MHA
    - Calidad prácticamente igual que MHA
    """

    def __init__(self, config: ConfigV2):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.kv_heads  # Effective KV heads (MHA if 0)
        self.head_dim = config.head_dim
        self.dim = config.dim
        self.n_rep = self.n_heads // self.n_kv_heads  # Q groups per KV head
        self.dropout = config.dropout

        # Q projection: full n_heads
        self.q_proj = nn.Linear(config.dim, self.n_heads * self.head_dim, bias=False)
        # K,V projections: n_kv_heads (smaller if GQA)
        kv_dim = self.n_kv_heads * self.head_dim
        self.k_proj = nn.Linear(config.dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, kv_dim, bias=False)
        # Output projection: full dim
        self.o_proj = nn.Linear(config.dim, config.dim, bias=False)

        # RoPE
        self.rope = RoPE(config.head_dim, config.max_seq_len)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Repite KV heads para matchear Q heads (GQA → MHA-like).

        [B, n_kv_heads, L, D] → [B, n_heads, L, D]
        """
        if self.n_rep == 1:
            return x  # MHA: no repetir
        B, H, L, D = x.shape
        x = x.unsqueeze(2).expand(B, H, self.n_rep, L, D)
        return x.reshape(B, H * self.n_rep, L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward de atención con GQA y Flash Attention.

        Args:
            x: [B, L, D] input embeddings

        Returns:
            [B, L, D] output
        """
        B, L, _ = x.shape

        # Project Q (full heads), K and V (kv_heads)
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE on Q and K
        q = self.rope(q)
        k = self.rope(k)

        # Expand KV heads to match Q heads for attention
        k = self._repeat_kv(k)  # [B, n_heads, L, D]
        v = self._repeat_kv(v)  # [B, n_heads, L, D]

        # Flash Attention (PyTorch 2.0+): kernel-fused, 2x menos VRAM
        # is_causal=True aplica máscara causal sin materializar el tensor
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        ).transpose(1, 2).reshape(B, L, self.dim)

        return self.o_proj(out)


# =============================================================================
# FEED-FORWARD (SwiGLU)
# =============================================================================

class BloqueFFN(nn.Module):
    """
    Feed-forward con SwiGLU (Shazeer, 2020).

    SwiGLU = SiLU(x @ W_gate) ⊙ (x @ W_up), luego down-project.
    Superior a ReLU/GELU en benchmarks de LLM.
    Hidden dim = 2/3 * dim * mult (compensar la gate extra).
    """

    def __init__(self, config: ConfigV2):
        super().__init__()
        hidden = int(config.dim * config.ffn_mult * 2 / 3)

        self.gate = nn.Linear(config.dim, hidden, bias=False)
        self.up = nn.Linear(config.dim, hidden, bias=False)
        self.down = nn.Linear(hidden, config.dim, bias=False)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU forward: SiLU(gate) * up → down."""
        return self.drop(
            self.down(F.silu(self.gate(x)) * self.up(x))
        )


# =============================================================================
# BLOQUE TERRITORIAL
# =============================================================================

class BloqueTerritorial(nn.Module):
    """
    Bloque que procesa los 4 territorios en paralelo.

    Arquitectura por capa:
      x → RMSNorm → Atención(GQA+RoPE) → residual
      x → RMSNorm → [FFN_t0, FFN_t1, FFN_t2, FFN_t3] × terr_acts → mix → residual
      x → exit_head → confianza (para Early Exit)

    Cada FFN se modula por la activación de su territorio,
    logrando especialización sin separar los parámetros.
    """

    def __init__(self, config: ConfigV2):
        super().__init__()
        self.config = config

        # Pre-normalization (RMSNorm — más eficiente que LayerNorm)
        self.norm1 = RMSNorm(config.dim)
        self.norm2 = RMSNorm(config.dim)

        # Atención compartida (con GQA)
        self.attn = BloqueAttn(config)

        # FFN por territorio (4 FFN especializados)
        self.ffns = nn.ModuleList([
            BloqueFFN(config) for _ in range(config.n_territorios)
        ])

        # Mezcla de territorios: D*4 → D
        self.mix = nn.Linear(config.dim * config.n_territorios, config.dim, bias=False)

        # Relaciones simbióticas: apoyo complementario entre territorios
        # Como el cerebro: ninguna zona se "apaga", las secundarias aportan
        # contexto que refuerza al territorio dominante
        sym_dim = config.dim // config.sym_factor
        self.sym_proj = nn.Linear(config.dim * config.n_territorios, sym_dim, bias=False)
        self.sym_up = nn.Linear(sym_dim, config.dim, bias=False)

        # Dropout residual
        self.drop = nn.Dropout(config.dropout)

        # Cabeza de confianza para Early Exit
        self.exit_head = nn.Linear(config.dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        terr_acts: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """
        Forward del bloque territorial.

        Args:
            x: [B, L, D] input
            terr_acts: [B, L, 4] territorial activations from Tálamo

        Returns:
            x: [B, L, D] output
            confianza: float in [0,1] for Early Exit decision
        """
        # 1. Atención + residual (Flash Attention, causal)
        x = x + self.drop(self.attn(self.norm1(x)))

        # 2. FFN por territorio con relaciones simbióticas
        h = self.norm2(x)

        # Todos los FFN producen su salida
        ffn_outputs = [ffn(h) for ffn in self.ffns]

        # Principal: ponderado por activación territorial (dominante lidera)
        weighted = [ffn_outputs[t] * terr_acts[:, :, t:t+1]
                    for t in range(len(self.ffns))]
        main_out = self.mix(torch.cat(weighted, dim=-1))  # [B, L, D*4] → [B, L, D]

        # Relaciones simbióticas: todos los territorios aportan contexto
        # complementario — como el cerebro, ninguna zona se apaga, las
        # secundarias refuerzan al territorio dominante
        sym_input = torch.cat(ffn_outputs, dim=-1)  # [B, L, D*4]
        sym_support = self.sym_up(F.silu(self.sym_proj(sym_input)))  # [B, L, D]

        # 3. Combinación: principal + apoyo simbiótico
        mixed = main_out + sym_support

        # 4. Residual
        x = x + self.drop(mixed)

        # 5. Confianza para Early Exit (percentil 10 — foco en tokens difíciles)
        # En vez de promediar todos, miramos el 10% con menor confianza
        # Si un token duda, el modelo sigue procesando capas
        per_token_conf = torch.sigmoid(self.exit_head(x)).squeeze(-1)  # [B, L]
        k = max(1, int(per_token_conf.numel() * self.config.exit_percentile))
        conf = per_token_conf.reshape(-1).topk(k, largest=False).values.mean().item()

        return x, conf
