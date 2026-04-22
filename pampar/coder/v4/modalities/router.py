# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
ModalityRouter — orquesta múltiples ModalityEncoders.

Hoy solo cablea TEXT (idéntico a `nn.Embedding` de v3). Mañana se agregan
IMAGE, AUDIO, etc. sin tocar el cerebro: solo se registra un encoder más.

El router devuelve:
  - `embeds`:       [B, L, dim] tokens listos para el cerebro
  - `modality_ids`: [B, L] long con el `ModalityId` de cada token

El cerebro v3 ignora `modality_ids` (compatibilidad). El cerebro v4 (Fase
4/5, cuando use ContextModulatorV4 internamente) los lee para condicionar
los modulators FiLM.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .base import ModalityEncoder, ModalityId
from .text import TextEncoder


class ModalityRouter(nn.Module):
    """
    Registro de encoders de modalidad.

    Modo single-modality (hoy, solo texto):
        router(input_ids) → (embeds, modality_ids=[TEXT,...])

    Modo multi-modality (futuro):
        router(modality_inputs={TEXT: ids, IMAGE: pixels}, sequence=[...])
        Ese path se diseñará en Fase 6+ (cuando agreguemos image encoder).

    Hoy solo se requiere el path de single-modality para no romper el flujo
    de v3.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.encoders: Dict[ModalityId, ModalityEncoder] = {}
        # nn.ModuleDict requiere keys str → mapeo paralelo
        self._encoders_module = nn.ModuleDict()

    def register(self, encoder: ModalityEncoder) -> None:
        """
        Registra un encoder para una modalidad.

        Falla si ya hay un encoder para esa modalidad o si la dim no
        coincide con la del router (single embedding space para todas las
        modalidades).
        """
        if encoder.dim != self.dim:
            raise ValueError(
                f"Encoder dim {encoder.dim} != router dim {self.dim}. "
                "Todas las modalidades deben proyectar al mismo dim."
            )
        if encoder.modality in self.encoders:
            raise ValueError(
                f"Modalidad {encoder.modality.name} ya tiene encoder registrado."
            )
        self.encoders[encoder.modality] = encoder
        self._encoders_module[encoder.modality.name] = encoder

    def get(self, modality: ModalityId) -> ModalityEncoder:
        """Devuelve el encoder registrado para una modalidad. KeyError si no existe."""
        if modality not in self.encoders:
            raise KeyError(
                f"No hay encoder registrado para {modality.name}. "
                f"Registrados: {[m.name for m in self.encoders]}"
            )
        return self.encoders[modality]

    def has(self, modality: ModalityId) -> bool:
        return modality in self.encoders

    def encode_text(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Path rápido para texto puro (caso por defecto hoy).

        Args:
            token_ids: [B, L] long con IDs de tokens BPE.

        Returns:
            embeds:       [B, L, dim] embeddings del TextEncoder.
            modality_ids: [B, L] long, todos = ModalityId.TEXT.
        """
        text_enc = self.get(ModalityId.TEXT)
        embeds = text_enc(token_ids)  # [B, L, dim]
        modality_ids = torch.full(
            token_ids.shape,
            int(ModalityId.TEXT),
            dtype=torch.long,
            device=token_ids.device,
        )
        return embeds, modality_ids

    @property
    def text_encoder(self) -> TextEncoder:
        """Acceso tipado al TextEncoder (común para weight-tying con lm_head)."""
        enc = self.get(ModalityId.TEXT)
        if not isinstance(enc, TextEncoder):
            raise TypeError(
                f"Encoder de TEXT no es TextEncoder sino {type(enc).__name__}"
            )
        return enc
