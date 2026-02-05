# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Bloques de procesamiento: Atención y FFN.

Componentes ligeros y optimizados:
- BloqueAttn: Multi-head self-attention con RoPE
- BloqueFFN: Feed-forward con SwiGLU
- BloqueTerritorial: Combina atención + FFN por territorio
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .config import ConfigV2


# =============================================================================
# ROTARY POSITION EMBEDDING (RoPE)
# =============================================================================

class RoPE(nn.Module):
    """Rotary Position Embedding para mejor generalización."""
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        # Frecuencias inversas
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Cache de posiciones
        pos = torch.arange(max_seq_len)
        freqs = torch.outer(pos, inv_freq)
        self.register_buffer("cos_cache", freqs.cos())
        self.register_buffer("sin_cache", freqs.sin())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aplica RoPE a tensor [B, H, L, D]."""
        L = x.shape[2]
        cos = self.cos_cache[:L].unsqueeze(0).unsqueeze(0)  # [1, 1, L, D/2]
        sin = self.sin_cache[:L].unsqueeze(0).unsqueeze(0)
        
        # Rotar pares de dimensiones
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos,
        ], dim=-1)


# =============================================================================
# ATENCIÓN
# =============================================================================

class BloqueAttn(nn.Module):
    """
    Multi-head self-attention con RoPE.
    
    Optimizaciones:
    - RoPE para posiciones
    - Grouped Query Attention (GQA) opcional
    - Flash Attention compatible
    """
    
    def __init__(self, config: ConfigV2):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.dim = config.dim
        
        # Proyecciones Q, K, V
        self.q_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.k_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.v_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.o_proj = nn.Linear(config.dim, config.dim, bias=False)
        
        # RoPE
        self.rope = RoPE(config.head_dim, config.max_seq_len)
        
        # Dropout
        self.attn_drop = nn.Dropout(config.dropout)
        
        # Escala
        self.scale = config.head_dim ** -0.5
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward de atención.
        
        Args:
            x: [B, L, D]
            mask: [L, L] máscara causal opcional
            
        Returns:
            [B, L, D] salida
        """
        B, L, D = x.shape
        
        # Proyectar Q, K, V
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Aplicar RoPE a Q y K
        q = self.rope(q)
        k = self.rope(k)
        
        # Atención
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask[:L, :L] == 0, float("-inf"))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        # Combinar
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.o_proj(out)


# =============================================================================
# FEED-FORWARD
# =============================================================================

class BloqueFFN(nn.Module):
    """
    Feed-forward con SwiGLU.
    
    SwiGLU = Swish(xW_gate) * (xW_up)
    Mejor que ReLU/GELU estándar.
    """
    
    def __init__(self, config: ConfigV2, mult: float = 4.0):
        super().__init__()
        hidden = int(config.dim * mult * 2 / 3)  # SwiGLU usa 2/3
        
        self.gate = nn.Linear(config.dim, hidden, bias=False)
        self.up = nn.Linear(config.dim, hidden, bias=False)
        self.down = nn.Linear(hidden, config.dim, bias=False)
        self.drop = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU forward."""
        return self.drop(
            self.down(F.silu(self.gate(x)) * self.up(x))
        )


# =============================================================================
# BLOQUE TERRITORIAL
# =============================================================================

class BloqueTerritorial(nn.Module):
    """
    Bloque que procesa los 4 territorios en paralelo.
    
    Cada territorio tiene su propio par atención+FFN,
    modulados por las activaciones del tálamo.
    """
    
    def __init__(self, config: ConfigV2):
        super().__init__()
        self.config = config
        
        # Pre-normalization
        self.norm1 = nn.LayerNorm(config.dim)
        self.norm2 = nn.LayerNorm(config.dim)
        
        # Un bloque de atención compartido
        self.attn = BloqueAttn(config)
        
        # FFN por territorio (4 territorios)
        self.ffns = nn.ModuleList([
            BloqueFFN(config) for _ in range(config.n_territorios)
        ])
        
        # Mezcla de territorios
        self.mix = nn.Linear(config.dim * config.n_territorios, config.dim)
        
        # Dropout residual
        self.drop = nn.Dropout(config.dropout)
        
        # Para early exit
        self.exit_head = nn.Linear(config.dim, 1)
    
    def forward(
        self,
        x: torch.Tensor,
        terr_acts: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Forward del bloque territorial.
        
        Args:
            x: [B, L, D] entrada
            terr_acts: [B, L, 4] activaciones de territorio
            mask: [L, L] máscara causal
            
        Returns:
            x: [B, L, D] salida
            confianza: float para early exit
        """
        B, L, D = x.shape
        
        # 1. Atención con residual
        x = x + self.drop(self.attn(self.norm1(x), mask))
        
        # 2. FFN por territorio
        h = self.norm2(x)
        outputs = []
        
        for t, ffn in enumerate(self.ffns):
            # Activación del territorio
            act = terr_acts[:, :, t:t+1]  # [B, L, 1]
            
            # FFN modulado
            out = ffn(h) * act
            outputs.append(out)
        
        # 3. Mezclar territorios
        concat = torch.cat(outputs, dim=-1)  # [B, L, D*4]
        mixed = self.mix(concat)
        
        # 4. Residual
        x = x + self.drop(mixed)
        
        # 5. Confianza para early exit
        conf = torch.sigmoid(self.exit_head(x.mean(dim=1))).mean().item()
        
        return x, conf
