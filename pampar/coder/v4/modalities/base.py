# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Interfaz base para encoders de modalidad.

Cualquier modalidad nueva (imagen, audio, video, etc.) implementa
`ModalityEncoder` y se registra en `ModalityRouter`. El cerebro no se
entera de qué modalidad procesa: solo recibe `[B, L, dim]` y un
`modality_id` por token para que el `ContextModulator` ajuste su
comportamiento.
"""

from __future__ import annotations

from enum import IntEnum

import torch
import torch.nn as nn


class ModalityId(IntEnum):
    """
    IDs de modalidad reservados.

    El orden NO debe cambiar entre versiones — los checkpoints dependen de
    estos índices para el one-hot del contexto. Agregar modalidades nuevas
    solo al final, antes de OTHER.
    """

    TEXT = 0  # Tokens de código/lenguaje natural (BPE actual)
    IMAGE = 1  # Patches ViT-style (16x16) — futuro
    AUDIO = 2  # Frames de mel-spectrogram — futuro
    VIDEO = 3  # Frames + temporal pooling — futuro
    CODE_AST = 4  # Nodos de AST con embedding propio — futuro
    DIAGRAM = 5  # Vector graphics / SVG tokenizados — futuro
    TABLE = 6  # Filas/columnas como tokens — futuro
    OTHER = 7  # Slot abierto para experimentación


NUM_MODALITIES: int = 8
"""Cantidad de slots de modalidad. Inmutable entre versiones."""


class ModalityEncoder(nn.Module):
    """
    Interfaz base para encoders de modalidad.

    Todo encoder produce `[B, L, dim]` donde `L` es la cantidad de tokens
    virtuales que esa modalidad inyecta en la secuencia. Para texto, L es
    la cantidad de tokens BPE. Para imagen, L será (H/patch * W/patch).

    Los encoders son responsables de:
      - Recibir input crudo (tensor int para texto, float para imagen, etc.)
      - Proyectarlo a `dim`
      - Devolver tokens listos para el cerebro

    El `modality_id` se inyecta DESPUÉS, en el `ModalityRouter`, no aquí.
    """

    modality: ModalityId

    def __init__(self, modality: ModalityId, dim: int):
        super().__init__()
        self.modality = modality
        self.dim = dim

    def forward(self, raw_input: torch.Tensor) -> torch.Tensor:
        """
        Convierte input crudo en tokens [B, L, dim].

        Subclases deben implementar este método.

        Args:
            raw_input: tensor con la representación cruda de la modalidad.
                Para TEXT: [B, L] long con IDs de tokens.
                Para IMAGE: [B, C, H, W] float con la imagen normalizada.
                Etc.

        Returns:
            [B, L, dim] tokens proyectados al espacio del cerebro.
        """
        raise NotImplementedError("ModalityEncoder.forward debe ser implementado")
