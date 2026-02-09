# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
PAMPAr-Coder v2: Modelo completo.

Integra todos los componentes en una arquitectura cerebral:
  Input → Embedding → Tálamo → [BloqueTerritorial × N] → RMSNorm → LM Head

Escalable de 42M (PRESET_4GB) a 1.54B (PRESET_1_5B).

Ventajas sobre transformers estándar (Qwen, Llama):
- LLAVES: routing determinista por tipo de token → eficiencia de datos
- 52 Brodmann Zones: especialización fina por territorio
- Early Exit: tokens simples salen antes → 2x throughput en inferencia
- GQA: KV cache 3x menor en inferencia batch
- Weight tying: embedding = LM head (sin posicional absoluto, usa RoPE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .config import ConfigV2, PRESET_4GB
from .talamo import Talamo
from .bloques import RMSNorm, BloqueTerritorial


class PampaRCoderV2(nn.Module):
    """
    Modelo PAMPAr-Coder v2 con 52 Zonas de Brodmann.

    Arquitectura cerebral para generación de código:
    - Input → Embedding (solo token, RoPE maneja posiciones)
    - Tálamo: LLAVES routing a 4 territorios
    - N × BloqueTerritorial: GQA + 4 FFN modulados
    - RMSNorm → LM Head (weight-tied con embedding)
    - Early Exit: si confianza > umbral, salir antes de la última capa
    """

    def __init__(self, config: ConfigV2 = PRESET_4GB):
        super().__init__()
        self.config = config

        # Token embedding (RoPE maneja posiciones — no learned pos_emb)
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.emb_drop = nn.Dropout(config.dropout)

        # Tálamo (router: LLAVES + atención aprendida)
        self.talamo = Talamo(config)

        # Bloques territoriales
        self.bloques = nn.ModuleList([
            BloqueTerritorial(config)
            for _ in range(config.n_capas)
        ])

        # Final RMSNorm + LM Head
        self.norm_f = RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Weight tying: embedding = LM head
        self.lm_head.weight = self.tok_emb.weight

        # Máscara causal (lower triangular)
        mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
        self.register_buffer("mask", mask)

        # Inicializar pesos
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """Inicialización scaled (estilo GPT-NeoX / Llama)."""
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def registrar_tokenizer(self, tokenizer) -> int:
        """Registra tokenizer en el tálamo para LLAVES."""
        return self.talamo.registrar_tokenizer(tokenizer)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        use_early_exit: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict]:
        """
        Forward pass.

        Args:
            input_ids: [B, L] token IDs
            targets: [B, L] labels (-100 = ignore)
            use_early_exit: salir antes si confianza alta

        Returns:
            logits: [B, L, vocab_size]
            loss: scalar if targets provided
            info: {'exit_capa': int} — which layer exited
        """
        B, L = input_ids.shape

        # 1. Embedding (solo token, RoPE da posiciones en atención)
        x = self.emb_drop(self.tok_emb(input_ids))

        # 2. Tálamo: routing a territorios y zonas
        terr_acts, zona_acts = self.talamo(x, input_ids)

        # 3. Bloques territoriales con Early Exit
        info = {"exit_capa": self.config.n_capas}
        mask = self.mask[:L, :L]

        for i, bloque in enumerate(self.bloques):
            if self.config.use_checkpoint and self.training and not use_early_exit:
                x, conf = torch.utils.checkpoint.checkpoint(
                    bloque, x, terr_acts, mask,
                    use_reentrant=False,
                )
            else:
                x, conf = bloque(x, terr_acts, mask)

            if use_early_exit and conf > self.config.umbral_exit:
                if i >= self.config.capas_min - 1:
                    info["exit_capa"] = i + 1
                    break

        # 4. Final norm + LM head
        x = self.norm_f(x)
        logits = self.lm_head(x)

        # 5. Loss (cross-entropy, ignore -100)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )

        return logits, loss, info

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        Genera tokens autoregressivamente con Early Exit.

        Args:
            prompt_ids: [1, L] prompt tokenizado
            max_tokens: máximo tokens a generar
            temperature: diversidad (menor = más determinista)
            top_k: Top-K sampling

        Returns:
            [1, L+N] tokens generados
        """
        self.eval()
        generated = prompt_ids.clone()

        for _ in range(max_tokens):
            ctx = generated[:, -self.config.max_seq_len:]
            logits, _, _ = self.forward(ctx, use_early_exit=True)
            logits = logits[:, -1, :] / temperature

            if top_k > 0:
                v, _ = logits.topk(top_k)
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_tok], dim=1)

            # Stop on EOS (token 0) — solo batch_size=1
            if generated.shape[0] == 1 and next_tok.item() == 0:
                break

        return generated

    def count_params(self) -> Dict[str, int]:
        """Cuenta parámetros por componente."""
        return {
            "embeddings": self.tok_emb.weight.numel(),
            "talamo": sum(p.numel() for p in self.talamo.parameters()),
            "bloques": sum(p.numel() for p in self.bloques.parameters()),
            "norm_f": sum(p.numel() for p in self.norm_f.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }


# =============================================================================
# FACTORY
# =============================================================================

def crear_modelo(config: ConfigV2 = PRESET_4GB) -> PampaRCoderV2:
    """Crea modelo PAMPAr-Coder v2."""
    return PampaRCoderV2(config)
