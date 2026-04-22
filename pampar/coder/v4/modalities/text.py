# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Encoder de modalidad TEXTO.

Wrapper fino sobre `nn.Embedding` para mantener compatibilidad numérica
exacta con v3. Permite que `ModalityRouter` trate el texto igual que
cualquier otra modalidad futura.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import ModalityEncoder, ModalityId


class TextEncoder(ModalityEncoder):
    """
    Embedding clásico para tokens BPE.

    Numericamente idéntico a `nn.Embedding(vocab_size, dim)` de v3.
    El peso puede compartirse con `lm_head` desde fuera (weight tying).
    """

    def __init__(self, vocab_size: int, dim: int):
        super().__init__(modality=ModalityId.TEXT, dim=dim)
        self.embedding = nn.Embedding(vocab_size, dim)

    @property
    def weight(self) -> torch.Tensor:
        """Acceso al peso para weight-tying con lm_head."""
        return self.embedding.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: [B, L] long con IDs de tokens BPE.

        Returns:
            [B, L, dim] embeddings.
        """
        return self.embedding(token_ids)
