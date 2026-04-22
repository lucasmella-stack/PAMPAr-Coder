# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Encoders por modalidad para PAMPAr v4.

Cada encoder implementa la interfaz `ModalityEncoder` y proyecta su input
crudo al `dim` compartido del modelo. Eso permite que el cerebro
(recurrent loop, modulators, lateral gates) sea idéntico para todas las
modalidades.

Modalidades registradas (slots reservados en `ModalityId`):
  0. text     ✅ implementada (TextEncoder, wrapper sobre nn.Embedding)
  1. image    ⏳ futuro (PatchEncoder ViT-style)
  2. audio    ⏳ futuro (mel-spectrogram → linear)
  3. video    ⏳ futuro (frame patches + temporal pooling)
  4. code_ast ⏳ futuro (AST nodes con embedding propio)
  5. diagram  ⏳ futuro (vector graphics tokenizados)
  6. table    ⏳ futuro (fila/columna como tokens)
  7. other    ⏳ slot abierto para experimentación

Solo `text` está activa. El resto son rieles para no romper checkpoints
cuando se agreguen.
"""

from .base import NUM_MODALITIES, ModalityEncoder, ModalityId
from .router import ModalityRouter
from .text import TextEncoder

__all__ = [
    "ModalityEncoder",
    "ModalityId",
    "NUM_MODALITIES",
    "ModalityRouter",
    "TextEncoder",
]
