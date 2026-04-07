# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""Atención GQA + Flash Attention — compartida entre streams."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ConfigV3
from .rope import RoPE


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

        # KV cache state (managed by PamparV3._enable_kv_cache)
        self._use_kv_cache: bool = False
        self._kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._start_pos: int = 0

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

        q = self.rope(q, self._start_pos)
        k = self.rope(k, self._start_pos)

        # KV cache: append new K,V to past cache (inference only)
        if self._use_kv_cache and not self.training:
            if self._kv_cache is not None:
                k_past, v_past = self._kv_cache
                k = torch.cat([k_past, k], dim=2)
                v = torch.cat([v_past, v], dim=2)
            self._kv_cache = (k, v)

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # Causal mask: full causal for prefill/training,
        # not needed for single-token decode (L_q=1 attends to all)
        use_causal = not (self._use_kv_cache and L == 1 and not self.training)

        out = (
            F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=use_causal,
            )
            .transpose(1, 2)
            .reshape(B, L, self.dim)
        )

        return self.o_proj(out)
