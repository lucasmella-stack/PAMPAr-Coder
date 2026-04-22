# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Configuración PAMPAr-Coder v4 — Recurrent-Depth Multimodal Transformer.

Hereda toda la configuración de v3 y agrega los flags necesarios para:
  - Contexto enriquecido multimodal (loop_idx + modality_id)
  - Modulators jerárquicos (futuro)
  - Recurrent loop con LTI + ACT (futuro)
  - Tokens reservados para modalidades no-texto

Texto-only es 100% retrocompatible con v3 en cuanto a comportamiento;
los slots multimodal solo agregan dims al vector de contexto y no
cambian el conteo total de params en modo solo-texto.
"""

from __future__ import annotations

from dataclasses import dataclass

from pampar.coder.v3.config import ConfigV3

from .modalities import NUM_MODALITIES


@dataclass
class ConfigV4(ConfigV3):
    """
    Configuración v4. Hereda todo lo de v3 y agrega los rieles multimodal.

    Cambios respecto a v3:
      - `vocab_active`: cuántos IDs del vocab usa el texto (resto reservado
        para tokens especiales multimodal: <IMG>, </IMG>, <AUDIO>, etc.)
      - `n_zonas_reserved`: zonas adicionales reservadas para futuras
        modalidades (visuales, auditivas, cross-modal). NO se usan hoy.
      - `use_hierarchical_modulators`: cuando True, los modulators
        comparten un backbone común entre niveles (Fase 2). Default False
        para mantener compatibilidad numérica con v3 en Fase 1.
      - Recurrent loop / LTI / ACT: flags reservados, default off.
    """

    # ── Multimodal (rieles tendidos, no consumen params en texto-only) ──────
    vocab_active: int = 47_000
    """IDs de vocab usados por texto. Los IDs >= vocab_active quedan
    reservados para tokens especiales de modalidades no-texto. El embedding
    sigue siendo de `vocab_size` (48000) para no romper checkpoints, pero
    el tokenizer no debería emitir IDs >= vocab_active todavía."""

    n_zonas_reserved: int = 38
    """Zonas adicionales reservadas para visión (V1, V2, V4, IT),
    audición (A1, Wernicke), y cross-modal (parietal). 52+38=90 zonas
    totales en el techo conceptual. Hoy no se materializan."""

    # ── Modulators jerárquicos (Fase 2) ─────────────────────────────────────
    use_hierarchical_modulators: bool = False
    """Compartir backbone del modulator entre niveles. Reduce params y
    fuerza consistencia entre profundidades. Activar después de validar
    Fase 1."""

    # ── Recurrent loop (Fase 4 - futuro) ────────────────────────────────────
    use_recurrent_loop: bool = False
    """Reemplazar stack de N niveles secuenciales por
    Prelude → Recurrent×T → Coda. Default off."""

    max_loop_iters: int = 8
    """Iteraciones máximas del recurrent block. Solo aplica cuando
    use_recurrent_loop=True."""

    use_lti_injection: bool = False
    """LTI-stable input injection (ρ(A) < 1 garantizado)."""

    use_act_halting: bool = False
    """Adaptive Computation Time per-token."""

    act_loop_threshold: float = 0.99
    """Umbral acumulativo de halting para ACT."""

    use_loop_index_rope: bool = False
    """Inyectar embedding sinusoidal del índice de loop en el contexto."""

    # ── Sharing entre niveles del path B (Fase 5b) ──────────────────────────
    share_ffn_across_niveles: bool = False
    """Si True y `use_recurrent_loop=True`, los 3 niveles (Prelude, Body,
    Coda) comparten el bloque FFN (mismo módulo, mismos params). Reduce
    params totales ~33% del costo FFN. Los modulators y atención NO se
    comparten — cada nivel mantiene su selectividad propia.

    Solo aplica con `use_mixed_selectivity=True` (compartimos `ffn_shared`).
    En modo legacy se compartiría la lista `ffns` completa."""

    # ── Propiedades derivadas multimodal ────────────────────────────────────

    @property
    def n_modalities(self) -> int:
        """Cantidad de slots de modalidad (siempre 8, ver ModalityId)."""
        return NUM_MODALITIES

    @property
    def n_zonas_total(self) -> int:
        """Zonas totales (activas + reservadas) — solo para planificación."""
        return self.n_zonas + self.n_zonas_reserved
