#!/usr/bin/env python3
# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Vanilla GPT baseline para ablación.

Transformer decoder estándar con ~62M parámetros para comparación justa
contra PAMPAr-V3 (sin LLAVES, sin streams, sin routing, sin FiLM).

Arquitectura:
  - Token + Positional Embedding
  - 12 capas TransformerBlock (Pre-Norm, GQA 8Q/2KV, SwiGLU FFN)
  - RMSNorm final + lm_head (weight-tied)

~62M params con dim=640, 12 layers, vocab=48K.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VanillaGPTConfig:
    """Configuración del GPT vanilla baseline."""

    vocab_size: int = 48_000
    dim: int = 640
    n_layers: int = 7  # ~60.8M params (≈ PamparV3 62.6M)
    n_heads: int = 8
    n_kv_heads: int = 2
    ffn_mult: float = 4.0
    max_seq_len: int = 4096
    dropout: float = 0.1

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def ffn_hidden(self) -> int:
        return int(self.dim * self.ffn_mult * 2 / 3)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).type_as(x) * self.w


class RotaryEmbedding(nn.Module):
    """RoPE — misma implementación que PAMPAr-V3 para comparación justa."""

    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + _rotate_half(x) * sin


class GQAttention(nn.Module):
    """Grouped-Query Attention con RoPE."""

    def __init__(self, cfg: VanillaGPTConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_heads // cfg.n_kv_heads

        self.q_proj = nn.Linear(cfg.dim, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.dim, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(L)
        cos = cos.to(x.device).unsqueeze(0).unsqueeze(0)
        sin = sin.to(x.device).unsqueeze(0).unsqueeze(0)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        mask = torch.triu(
            torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(out)


class SwiGLUFFN(nn.Module):
    """SwiGLU FFN — misma estructura que PAMPAr-V3."""

    def __init__(self, cfg: VanillaGPTConfig):
        super().__init__()
        h = cfg.ffn_hidden
        self.gate = nn.Linear(cfg.dim, h, bias=False)
        self.up = nn.Linear(cfg.dim, h, bias=False)
        self.down = nn.Linear(h, cfg.dim, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block."""

    def __init__(self, cfg: VanillaGPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.dim)
        self.attn = GQAttention(cfg)
        self.norm2 = RMSNorm(cfg.dim)
        self.ffn = SwiGLUFFN(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class VanillaGPT(nn.Module):
    """
    GPT decoder estándar para ablación.

    Componentes iguales a PAMPAr-V3 (GQA, SwiGLU, RoPE, RMSNorm)
    pero sin routing, sin streams, sin LLAVES, sin FiLM.
    """

    def __init__(self, cfg: VanillaGPTConfig):
        super().__init__()
        self.config = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.norm = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict]:
        """
        Forward pass compatible con la interfaz de PamparV3.

        Returns:
            (logits, loss, info_dict)
        """
        x = self.drop(self.tok_emb(input_ids))

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
            )

        info = {"exit_nivel": self.config.n_layers, "terr_acts": None}
        return logits, loss, info
