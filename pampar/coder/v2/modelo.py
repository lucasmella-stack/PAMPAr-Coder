# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
PAMPAr-Coder v2: Modelo completo.

Integra todos los componentes:
- Embeddings (token + posición)
- Tálamo (routing)
- Bloques territoriales
- LM Head

Optimizado para GTX 1650 (4GB VRAM).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .config import ConfigV2, PRESET_4GB
from .talamo import Talamo
from .bloques import BloqueTerritorial


class PampaRCoderV2(nn.Module):
    """
    Modelo PAMPAr-Coder v2 con 52 Zonas de Brodmann.
    
    Arquitectura cerebral para generación de código:
    - Input -> Embedding -> Tálamo -> Bloques -> LM Head
    - Early exit si confianza > umbral
    - Knowledge Distillation ready
    """
    
    def __init__(self, config: ConfigV2 = PRESET_4GB):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.dim)
        self.emb_drop = nn.Dropout(config.dropout)
        
        # Tálamo (router)
        self.talamo = Talamo(config)
        
        # Bloques territoriales
        self.bloques = nn.ModuleList([
            BloqueTerritorial(config)
            for _ in range(config.n_capas)
        ])
        
        # LM Head
        self.norm_f = nn.LayerNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.tok_emb.weight
        
        # Máscara causal
        mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
        self.register_buffer("mask", mask)
        
        # Inicializar pesos
        self.apply(self._init_weights)
    
    def _init_weights(self, m: nn.Module):
        """Inicialización estilo GPT-2."""
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def registrar_tokenizer(self, tokenizer) -> int:
        """Registra tokenizer en el tálamo."""
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
            targets: [B, L] para calcular loss
            use_early_exit: Salir antes si confianza alta
            
        Returns:
            logits: [B, L, vocab_size]
            loss: Escalar si targets dado
            info: Dict con métricas
        """
        B, L = input_ids.shape
        device = input_ids.device
        
        # 1. Embeddings
        pos = torch.arange(L, device=device).unsqueeze(0)
        x = self.emb_drop(
            self.tok_emb(input_ids) + self.pos_emb(pos)
        )
        
        # 2. Tálamo: routing
        terr_acts, zona_acts = self.talamo(x, input_ids)
        
        # 3. Bloques territoriales
        info = {"exit_capa": self.config.n_capas}
        
        for i, bloque in enumerate(self.bloques):
            x, conf = bloque(x, terr_acts, self.mask[:L, :L])
            
            # Early exit
            if use_early_exit and conf > self.config.umbral_exit:
                if i >= self.config.capas_min - 1:
                    info["exit_capa"] = i + 1
                    break
        
        # 4. LM Head
        x = self.norm_f(x)
        logits = self.lm_head(x)
        
        # 5. Loss
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
        Genera tokens autoregressivamente.
        
        Args:
            prompt_ids: [1, L] prompt tokenizado
            max_tokens: Máximo a generar
            temperature: Diversidad (menor = más determinista)
            top_k: Top-K sampling
            
        Returns:
            [1, L+N] tokens generados
        """
        self.eval()
        generated = prompt_ids.clone()
        
        for _ in range(max_tokens):
            # Truncar al contexto máximo
            ctx = generated[:, -self.config.max_seq_len:]
            
            # Forward con early exit
            logits, _, _ = self.forward(ctx, use_early_exit=True)
            
            # Último token
            logits = logits[:, -1, :] / temperature
            
            # Top-K
            if top_k > 0:
                v, _ = logits.topk(top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            
            generated = torch.cat([generated, next_tok], dim=1)
            
            # EOS check
            if next_tok.item() == 0:
                break
        
        return generated
    
    def count_params(self) -> Dict[str, int]:
        """Cuenta parámetros por componente."""
        return {
            "embeddings": sum(
                p.numel() for p in [self.tok_emb.weight, self.pos_emb.weight]
            ),
            "talamo": sum(p.numel() for p in self.talamo.parameters()),
            "bloques": sum(p.numel() for p in self.bloques.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }


# =============================================================================
# FACTORY
# =============================================================================

def crear_modelo(config: ConfigV2 = PRESET_4GB) -> PampaRCoderV2:
    """Crea modelo PAMPAr-Coder v2."""
    return PampaRCoderV2(config)
