# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""Rotary Position Embedding (RoPE) — Su et al., 2021."""

from __future__ import annotations

import torch
import torch.nn as nn


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

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """Aplica RoPE a tensor [B, H, L, D]. start_pos offsets positions for KV cache."""
        L = x.shape[2]
        cos = self.cos_cache[start_pos : start_pos + L].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cache[start_pos : start_pos + L].unsqueeze(0).unsqueeze(0)
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
