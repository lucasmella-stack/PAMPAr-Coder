# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
PamparV4 — modelo principal v4 (Fase 3).

Estrategia de bajo riesgo:
  - Reutiliza TODO el cerebro v3 (TalamoInicial, NivelProfundo, lateral
    gates, attn, RMSNorm, etc.) sin modificarlo.
  - Reemplaza ÚNICAMENTE la capa de embedding: en vez de `nn.Embedding`
    directa, usa `ModalityRouter(TextEncoder)`.
  - Genera y propaga `modality_ids` por token, listos para que el cerebro
    v4 (Fase 4/5) los lea cuando los niveles usen ContextModulatorV4.

En modo solo-texto y con misma seed, **PamparV4 es numéricamente
equivalente a PamparV3**: el TextEncoder es un wrapper trivial sobre
nn.Embedding con la misma init.

Cuando se active Fase 4 (recurrent loop) o Fase 5 (FFN+modulator
jerárquico), los niveles internos se reemplazarán por versiones v4 que
consumen `modality_ids`. Por ahora, la información viaja pero no se usa
en el cerebro — los niveles v3 la ignoran.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from pampar.coder.v3.bloques import NivelProfundo, RMSNorm
from pampar.coder.v3.talamo import TalamoInicial

from .config import ConfigV4
from .modalities import ModalityRouter, TextEncoder

if TYPE_CHECKING:
    from pampar.coder.v3.engrama_stream import BancoEngrama


class PamparV4(nn.Module):
    """
    PAMPAr-Coder v4: cerebro v3 + ModalityRouter en la entrada.

    Arquitectura (Fase 3):
        ModalityRouter[TextEncoder] → embeds [B, L, dim], modality_ids [B, L]
            → TalamoInicial (v3, sin tocar)
            → NivelProfundo × n_levels (v3, sin tocar)
            → RMSNorm + lm_head (weight-tied con TextEncoder.weight)

    `modality_ids` se devuelve en `info` para que evals/debug puedan
    inspeccionarlo. El cerebro v3 no lo consume hoy.
    """

    def __init__(self, config: ConfigV4):
        super().__init__()
        self.config = config

        # Router multimodal con TextEncoder ya registrado
        self.modality_router = ModalityRouter(dim=config.dim)
        self.modality_router.register(
            TextEncoder(vocab_size=config.vocab_size, dim=config.dim)
        )

        self.emb_drop = nn.Dropout(config.dropout)

        # Cerebro v3 sin modificaciones
        self.talamo = TalamoInicial(config)
        self.niveles = nn.ModuleList(
            [NivelProfundo(config, nivel_idx=i) for i in range(config.n_levels)]
        )
        self.norm_f = RMSNorm(config.dim)

        # LM head + weight tying con el peso del TextEncoder
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.modality_router.text_encoder.weight

        self._init_weights()

    # ────────────────────────────────────────────────────────────────────
    # API helpers (mismas firmas que PamparV3 — drop-in compatible)
    # ────────────────────────────────────────────────────────────────────

    @property
    def tok_emb(self) -> nn.Embedding:
        """Compat con código v3 que accede a `model.tok_emb` directamente."""
        return self.modality_router.text_encoder.embedding

    def _init_weights(self) -> None:
        """Init estilo GPT-NeoX / Llama: N(0, 0.02). Idéntico a v3."""

        def _init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        self.apply(_init)

    def registrar_tokenizer(self, tokenizer: object) -> None:
        self.talamo.registrar_tokenizer(tokenizer)

    def _enable_kv_cache(self) -> None:
        for nivel in self.niveles:
            nivel.attn._use_kv_cache = True
            nivel.attn._kv_cache = None
            nivel.attn._start_pos = 0

    def _disable_kv_cache(self) -> None:
        for nivel in self.niveles:
            nivel.attn._use_kv_cache = False
            nivel.attn._kv_cache = None
            nivel.attn._start_pos = 0

    def _set_cache_pos(self, pos: int) -> None:
        for nivel in self.niveles:
            nivel.attn._start_pos = pos

    def set_train_norm_clamp(self, enabled: bool) -> None:
        for nivel in self.niveles:
            nivel._train_norm_clamp = enabled

    def _combinar_streams(
        self,
        streams: List[torch.Tensor],
        terr_acts: torch.Tensor,
    ) -> torch.Tensor:
        """Combina los n_streams streams ponderados por activación territorial."""
        weights = F.softmax(terr_acts, dim=-1)
        return sum(
            streams[t] * weights[:, :, t : t + 1] for t in range(self.config.n_streams)
        )

    # ────────────────────────────────────────────────────────────────────
    # Forward
    # ────────────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        use_early_exit: bool = False,
        banco_engrama: Optional["BancoEngrama"] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict]:
        """
        Forward pass de PamparV4.

        Args:
            input_ids:      [B, L] token IDs (texto, hoy).
            targets:        [B, L] labels (-100 = ignorar).
            use_early_exit: salir antes si la confianza supera el umbral.
            banco_engrama:  banco opcional de engramas.

        Returns:
            logits: [B, L, vocab_size]
            loss:   scalar si `targets` provisto, sino None.
            info:   dict con 'exit_nivel', 'terr_acts' y 'modality_ids'.
        """
        # 1. Embedding multimodal vía router (hoy = solo texto)
        embeds, modality_ids = self.modality_router.encode_text(input_ids)
        x = self.emb_drop(embeds)  # [B, L, D]

        # 2. Tálamo + cerebro v3 (sin tocar)
        terr_acts, zona_acts = self.talamo(x, input_ids)
        streams: List[torch.Tensor] = [x.clone() for _ in range(self.config.n_streams)]

        info: Dict = {
            "exit_nivel": self.config.n_levels,
            "terr_acts": terr_acts,
            "modality_ids": modality_ids,
        }

        for i, nivel in enumerate(self.niveles):
            if self.config.use_checkpoint and self.training and not use_early_exit:

                def create_checkpoint_fn(n):
                    def fn(*stream_tensors):
                        s_list = list(stream_tensors[:-2])
                        ta = stream_tensors[-2]
                        za = stream_tensors[-1]
                        new_s, new_ta, _ = n(
                            s_list,
                            ta,
                            TalamoInicial.agregar_fn,
                            zona_acts=za,
                        )
                        return (*new_s, new_ta)

                    return fn

                result = torch.utils.checkpoint.checkpoint(
                    create_checkpoint_fn(nivel),
                    *streams,
                    terr_acts,
                    zona_acts,
                    use_reentrant=False,
                )
                streams = list(result[: self.config.n_streams])
                terr_acts = result[self.config.n_streams]
                conf = 0.0
            else:
                streams, terr_acts, conf = nivel(
                    streams,
                    terr_acts,
                    TalamoInicial.agregar_fn,
                    banco_engrama=banco_engrama,
                    zona_acts=zona_acts,
                )

            if use_early_exit and conf > self.config.umbral_exit:
                if i >= self.config.capas_min - 1:
                    info["exit_nivel"] = i + 1
                    break

        x_final = self._combinar_streams(streams, terr_acts)
        x_final = self.norm_f(x_final)
        logits = self.lm_head(x_final)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
            )

        return logits, loss, info
