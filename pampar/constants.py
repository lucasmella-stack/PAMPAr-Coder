# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Constantes globales de PAMPAr-Coder.

Single source of truth para paths y valores compartidos entre módulos.
"""

# Tokenizer oficial (SentencePiece 48K vocab bilingual)
TOKENIZER_PATH = "data/tokenizer/pampar_48k.model"

# Tokenizer legacy (16K, solo para compatibilidad con v1/v2)
TOKENIZER_LEGACY_PATH = "data/tokenizer/code_tokenizer.model"
