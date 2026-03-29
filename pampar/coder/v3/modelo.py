# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
PamparV3 — Modelo principal con arquitectura 2D.

Arquitectura:
  tok_emb [48K, 640]
    → TalamoInicial (LLAVES + attn_proj + context_conv)
        → terr_acts [B, L, 4], zona_acts [B, L, 52]
    → 4 streams inicializados desde tok_emb
    → [NivelProfundo × n_levels]
        cada nivel: attn compartida + re-routing + 4 FFN streams
                  + lateral gates (fibras blancas)
    → norm_f (RMSNorm)
    → lm_head (weight-tied con tok_emb)

4 streams × 5 niveles = grilla 2D donde:
  - La profundidad refina el significado (como capas corticales)
  - La anchura especializa por tipo de token (como áreas corticales)
  - Las lateral gates comunican áreas a cada nivel (como fibras blancas)
  - El re-routing adapta qué área lidera según el contexto acumulado
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .bloques import NivelProfundo, RMSNorm
from .config import PRESET_V3, ConfigV3
from .talamo import TalamoInicial

if TYPE_CHECKING:
    from .engrama_stream import BancoEngrama


class PamparV3(nn.Module):
    """
    PAMPAr-Coder v3: arquitectura 2D con streams especializados y profundidad.

    ~110M parámetros con PRESET_V3 (dim=640, 5 niveles, vocab=48K).
    """

    def __init__(self, config: ConfigV3 = PRESET_V3):
        super().__init__()
        self.config = config

        # Embedding de tokens (weight-tied con lm_head)
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.emb_drop = nn.Dropout(config.dropout)

        # Tálamo inicial: routing completo de tokens a zonas/territorios
        self.talamo = TalamoInicial(config)

        # Grilla 2D: n_levels niveles de profundidad
        self.niveles = nn.ModuleList(
            [NivelProfundo(config, nivel_idx=i) for i in range(config.n_levels)]
        )

        # Normalización final (antes de lm_head)
        self.norm_f = RMSNorm(config.dim)

        # LM head
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Weight tying: embedding y lm_head comparten pesos
        # Ahorra vocab_size × dim parámetros (48K × 640 = 30.7M fp16)
        self.lm_head.weight = self.tok_emb.weight

        # Inicialización
        self._init_weights()

    def _init_weights(self) -> None:
        """Inicialización estilo GPT-NeoX / Llama: N(0, 0.02)."""

        def _init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        self.apply(_init)

    def registrar_tokenizer(self, tokenizer: object) -> None:
        """Registra el tokenizer en el Tálamo para que LLAVES funcione."""
        self.talamo.registrar_tokenizer(tokenizer)

    def _enable_kv_cache(self) -> None:
        """Enable KV cache mode for generation (inference only)."""
        for nivel in self.niveles:
            nivel.attn._use_kv_cache = True
            nivel.attn._kv_cache = None
            nivel.attn._start_pos = 0

    def _disable_kv_cache(self) -> None:
        """Disable KV cache and free cached tensors."""
        for nivel in self.niveles:
            nivel.attn._use_kv_cache = False
            nivel.attn._kv_cache = None
            nivel.attn._start_pos = 0

    def _set_cache_pos(self, pos: int) -> None:
        """Set the position offset for RoPE in all attention layers."""
        for nivel in self.niveles:
            nivel.attn._start_pos = pos

    def set_train_norm_clamp(self, enabled: bool) -> None:
        """Activa/desactiva norm clamping durante training en todos los niveles."""
        for nivel in self.niveles:
            nivel._train_norm_clamp = enabled

    def _combinar_streams(
        self,
        streams: List[torch.Tensor],
        terr_acts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combina los 4 streams en una representación unificada.

        Ponderada por activación territorial: el stream más activo domina.
        Normalizada para que los pesos sumen 1.

        Args:
            streams:   [n_streams × [B, L, D]]
            terr_acts: [B, L, n_streams]
        Returns:
            x: [B, L, D]
        """
        # Normalizar pesos territoriales (softmax sobre streams)
        weights = F.softmax(terr_acts, dim=-1)  # [B, L, 4]
        return sum(
            streams[t] * weights[:, :, t : t + 1] for t in range(self.config.n_streams)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        use_early_exit: bool = False,
        banco_engrama: Optional[BancoEngrama] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict]:
        """
        Forward pass de PamparV3.

        Args:
            input_ids:      [B, L] token IDs
            targets:        [B, L] labels (-100 = ignorar)
            use_early_exit: salir antes si confianza suficiente
            banco_engrama:  banco de engramas para inyección (None = sin inyección)

        Returns:
            logits: [B, L, vocab_size]
            loss:   scalar (si targets provisto)
            info:   {'exit_nivel': int, 'terr_acts': tensor}
        """
        B, L = input_ids.shape

        # 1. Embedding
        x = self.emb_drop(self.tok_emb(input_ids))  # [B, L, D]

        # 2. Tálamo inicial: routing de tokens a zonas y territorios
        terr_acts, zona_acts = self.talamo(x, input_ids)  # [B,L,4], [B,L,52]

        # 3. Inicializar los 4 streams desde el mismo embedding
        #    Cada stream parte del mismo punto y se especializa a lo largo
        #    de los n_levels niveles
        streams: List[torch.Tensor] = [x.clone() for _ in range(self.config.n_streams)]

        # 4. Pasar por cada nivel de profundidad
        info: Dict = {"exit_nivel": self.config.n_levels, "terr_acts": terr_acts}

        for i, nivel in enumerate(self.niveles):
            if self.config.use_checkpoint and self.training and not use_early_exit:
                # Gradient checkpointing: ahorra VRAM no guardando activaciones
                # Se usan lambdas para pasar args no-tensor (agregar_fn)
                def create_checkpoint_fn(n):
                    def fn(*stream_tensors):
                        s_list = list(stream_tensors[:-1])
                        ta = stream_tensors[-1]
                        new_s, new_ta, _ = n(s_list, ta, TalamoInicial.agregar_fn)
                        return (*new_s, new_ta)

                    return fn

                result = torch.utils.checkpoint.checkpoint(
                    create_checkpoint_fn(nivel),
                    *streams,
                    terr_acts,
                    use_reentrant=False,
                )
                streams = list(result[: self.config.n_streams])
                terr_acts = result[self.config.n_streams]
                conf = 0.0  # No calculada durante checkpointing
            else:
                streams, terr_acts, conf = nivel(
                    streams,
                    terr_acts,
                    TalamoInicial.agregar_fn,
                    banco_engrama=banco_engrama,
                    zona_acts=zona_acts,
                )

            # Early Exit: si el nivel más difícil del 10% ya tiene confianza
            if use_early_exit and conf > self.config.umbral_exit:
                if i >= self.config.capas_min - 1:
                    info["exit_nivel"] = i + 1
                    break

        # 5. Combinar streams en representación final
        x_final = self._combinar_streams(streams, terr_acts)  # [B, L, D]

        # 6. Norm final + LM head
        x_final = self.norm_f(x_final)
        logits = self.lm_head(x_final)  # [B, L, vocab_size]

        # 7. Loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
            )

        return logits, loss, info

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        banco_engrama: Optional[BancoEngrama] = None,
    ) -> torch.Tensor:
        """
        Generación autoregresiva con KV cache y nucleus sampling.

        Usa prefill (procesa todo el prompt de una sola vez) +
        decode (un token por paso, reutilizando el KV cache).

        Args:
            prompt_ids:    [1, L] prompt tokenizado
            max_tokens:    máximo tokens a generar
            temperature:   diversidad (menor = más determinista)
            top_k:         Top-K sampling (0 = desactivado)
            top_p:         Nucleus sampling (1.0 = desactivado)
            banco_engrama: banco de engramas para inyección adaptativa

        Returns:
            [1, L+N] tokens generados
        """
        self.eval()
        generated = prompt_ids.clone()
        prompt_len = prompt_ids.shape[1]

        try:
            self._enable_kv_cache()

            # --- Prefill: procesar todo el prompt, poblar KV cache ---
            self._set_cache_pos(0)
            logits, _, _ = self.forward(
                prompt_ids,
                use_early_exit=False,
                banco_engrama=banco_engrama,
            )
            logits = logits[:, -1, :] / temperature

            for _ in range(max_tokens):
                # Top-K filtering
                if top_k > 0:
                    v, _ = logits.topk(top_k)
                    logits[logits < v[:, [-1]]] = float("-inf")

                # Nucleus (Top-P) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_idx = logits.sort(descending=True)
                    cumprobs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                    remove_mask = cumprobs - sorted_logits.softmax(dim=-1) > top_p
                    sorted_logits[remove_mask] = float("-inf")
                    logits = torch.zeros_like(logits).scatter(
                        1,
                        sorted_idx,
                        sorted_logits,
                    )

                # Guard contra NaN/Inf (frecuente en early training)
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

                probs = F.softmax(logits, dim=-1)
                if torch.isnan(probs).any() or (probs < 0).any():
                    probs = torch.ones_like(probs) / probs.shape[-1]

                next_tok = torch.multinomial(probs, 1)
                generated = torch.cat([generated, next_tok], dim=1)

                # Stop en EOS (token 0)
                if generated.shape[0] == 1 and next_tok.item() == 0:
                    break

                # --- Decode: un token a la vez, KV cache se reutiliza ---
                cur_pos = generated.shape[1] - 1
                if cur_pos >= self.config.max_seq_len:
                    break

                self._set_cache_pos(cur_pos)
                logits, _, _ = self.forward(
                    next_tok,
                    use_early_exit=False,
                    banco_engrama=banco_engrama,
                )
                logits = logits[:, -1, :] / temperature

        finally:
            self._disable_kv_cache()

        return generated

    def count_params(self) -> Dict[str, int]:
        """Cuenta parámetros por componente."""
        return {
            "embeddings": self.tok_emb.weight.numel(),
            "talamo_inicial": sum(p.numel() for p in self.talamo.parameters()),
            "niveles": sum(p.numel() for p in self.niveles.parameters()),
            "norm_f": sum(p.numel() for p in self.norm_f.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
            "total_sin_embedding": sum(
                p.numel()
                for name, p in self.named_parameters()
                if "tok_emb" not in name
            ),
        }

    def describe(self) -> str:
        """Descripción legible de la arquitectura."""
        p = self.count_params()
        cfg = self.config
        mem = cfg.memory_estimate_mb()
        return (
            f"PamparV3\n"
            f"  Arquitectura:  {cfg.n_streams} streams × {cfg.n_levels} niveles (2D)\n"
            f"  Dimensiones:   dim={cfg.dim}, heads={cfg.n_heads} "
            f"(GQA {cfg.n_kv_heads} KV), ffn_hidden={cfg.ffn_hidden}\n"
            f"  Vocab:         {cfg.vocab_size:,} tokens, seq_len={cfg.max_seq_len}\n"
            f"  Parámetros:    {p['total'] / 1e6:.1f}M total "
            f"({p['total_sin_embedding'] / 1e6:.1f}M sin embedding)\n"
            f"  VRAM model:    {mem['model_fp16_mb']}MB (fp16)\n"
            f"  VRAM training: {mem['training_total_mb']}MB (model+grad+Adam)\n"
            f"  KV cache:      {mem['kv_cache_inference_mb']}MB "
            f"(batch=1, seq=4096)\n"
        )


# =============================================================================
# FACTORY
# =============================================================================


def crear_modelo_v3(config: ConfigV3 = PRESET_V3) -> PamparV3:
    """Crea un modelo PamparV3 con la configuración dada."""
    return PamparV3(config)
