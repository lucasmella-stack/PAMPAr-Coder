# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Configuración del modelo PAMPAr-Coder v2.

Presets optimizados para diferentes GPUs:
- PRESET_4GB: GTX 1650, RTX 3050 (4GB VRAM)
- PRESET_8GB: RTX 3060, RTX 4060 (8GB VRAM)
- PRESET_24GB: RTX 3090, RTX 4090 (24GB VRAM)
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfigV2:
    """Configuración inmutable del modelo."""
    
    # Dimensiones
    vocab_size: int = 16000
    dim: int = 384
    n_heads: int = 6
    n_capas: int = 6
    max_seq_len: int = 1024
    
    # Brodmann
    n_zonas: int = 52
    n_territorios: int = 4
    
    # Routing
    peso_llaves: float = 0.8  # 80% reglas, 20% atención
    
    # Regularización
    dropout: float = 0.1
    
    # Early Exit
    umbral_exit: float = 0.9
    capas_min: int = 2
    
    # Optimización
    use_amp: bool = True
    use_checkpoint: bool = True
    
    def __post_init__(self):
        """Valida configuración."""
        assert self.dim % self.n_heads == 0, "dim debe ser divisible por n_heads"
        assert 0 <= self.peso_llaves <= 1, "peso_llaves debe estar en [0, 1]"
    
    @property
    def head_dim(self) -> int:
        """Dimensión por cabeza de atención."""
        return self.dim // self.n_heads
    
    def memory_estimate_mb(self) -> float:
        """Estima memoria en MB (FP16)."""
        # Embeddings: vocab * dim + seq * dim
        emb = (self.vocab_size + self.max_seq_len) * self.dim
        
        # Por capa: 4 territorios * (attn + ffn)
        attn = 4 * self.dim * self.dim  # Q, K, V, O
        ffn = 3 * self.dim * (self.dim * 4)  # up, gate, down
        per_capa = (attn + ffn) * 4
        
        total = emb + per_capa * self.n_capas
        return total * 2 / (1024 ** 2)  # FP16 = 2 bytes


# =============================================================================
# PRESETS
# =============================================================================

PRESET_4GB = ConfigV2(
    vocab_size=16000,
    dim=384,
    n_heads=6,
    n_capas=6,
    max_seq_len=1024,
    dropout=0.1,
)

PRESET_8GB = ConfigV2(
    vocab_size=16000,
    dim=512,
    n_heads=8,
    n_capas=8,
    max_seq_len=2048,
    dropout=0.1,
)

PRESET_24GB = ConfigV2(
    vocab_size=16000,
    dim=768,
    n_heads=12,
    n_capas=12,
    max_seq_len=4096,
    dropout=0.05,
)
