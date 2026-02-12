# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
PAMPAr-Coder: Modelo de lenguaje territorial para código.

Arquitectura cerebral especializada en programación:
- 4 Territorios: SINTAXIS, SEMANTICA, LOGICO, ESTRUCTURAL
- LLAVES para routing instantáneo de tokens
- Fronteras bidireccionales para comunicación
- Early Exit para inferencia ultra-rápida
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .config import ConfigPampaRCoder, CODER_4GB
from .llaves_codigo import TipoTerritorioCoder, LlavesCodigo, LlavesCodigoRegistry
from .territorios_codigo import GestorTerritoriosCoder


class TalamoCoder(nn.Module):
    """
    Tálamo especializado para código.
    
    Combina:
    - LLAVES (reglas explícitas): 70-80% del routing
    - Atención aprendida: 20-30% para ajuste fino
    
    Recibe tokens y determina qué territorios procesar
    y con qué peso.
    """
    
    def __init__(
        self, 
        dim: int, 
        peso_llaves: float = 0.75,
        n_territorios: int = 4,
        vocab_size: int = 16000
    ):
        super().__init__()
        self.dim = dim
        self.peso_llaves = peso_llaves
        self.n_territorios = n_territorios
        self.vocab_size = vocab_size
        
        # LLAVES: reglas pre-definidas
        self.llaves = LlavesCodigo()
        self.llaves_registry: Optional[LlavesCodigoRegistry] = None
        
        # Lookup table para LLAVES (se llena al registrar tokenizer)
        # Shape: [vocab_size, 4] para los 4 territorios
        self.register_buffer('llaves_lookup', torch.full((vocab_size, 4), 0.25))
        
        # Atención aprendida para ajuste fino
        self.attn_routing = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, n_territorios),
            nn.Softmax(dim=-1)
        )
    
    def registrar_tokenizer(self, tokenizer) -> None:
        """Registra tokenizer para pre-computar LLAVES y crear lookup table."""
        self.llaves_registry = LlavesCodigoRegistry()
        self.llaves_registry.registrar_tokenizer(tokenizer)
        
        # Llenar lookup table vectorizada
        territorios = list(TipoTerritorioCoder)
        for token_id, acts in self.llaves_registry._cache.items():
            if token_id < self.vocab_size:
                for i, t in enumerate(territorios):
                    self.llaves_lookup[token_id, i] = acts.get(t, 0.25)
    
    def forward(
        self, 
        x: torch.Tensor, 
        token_ids: Optional[torch.Tensor] = None
    ) -> Dict[TipoTerritorioCoder, torch.Tensor]:
        """
        Calcula activaciones de territorios.
        
        Args:
            x: Embeddings (B, L, D)
            token_ids: IDs de tokens (B, L) para LLAVES
            
        Returns:
            Dict de territorio -> (B, L) pesos
        """
        B, L, D = x.shape
        device = x.device
        
        # 1. Activaciones desde LLAVES (reglas)
        if token_ids is not None and self.llaves_registry is not None:
            # Usar cache pre-computado
            llaves_acts = self._get_llaves_activations(token_ids, device)
        else:
            # Fallback: distribución uniforme
            llaves_acts = {
                t: torch.ones(B, L, device=device) * 0.25 
                for t in TipoTerritorioCoder
            }
        
        # 2. Activaciones desde atención aprendida
        attn_acts = self.attn_routing(x)  # (B, L, 4)
        
        # 3. Combinar: peso_llaves * LLAVES + (1-peso_llaves) * attn
        activaciones = {}
        for i, tipo in enumerate(TipoTerritorioCoder):
            llaves_t = llaves_acts[tipo]
            attn_t = attn_acts[:, :, i]
            
            combined = self.peso_llaves * llaves_t + (1 - self.peso_llaves) * attn_t
            activaciones[tipo] = combined
        
        return activaciones
    
    def _get_llaves_activations(
        self, 
        token_ids: torch.Tensor, 
        device: torch.device
    ) -> Dict[TipoTerritorioCoder, torch.Tensor]:
        """Obtiene activaciones de LLAVES - VECTORIZADO (sin loops Python)."""
        B, L = token_ids.shape
        
        # Clamp token_ids para evitar out of bounds
        safe_ids = token_ids.clamp(0, self.vocab_size - 1)
        
        # Lookup vectorizado: [B, L] -> [B, L, 4]
        lookup_result = self.llaves_lookup[safe_ids]  # [B, L, 4]
        
        # Separar por territorio
        territorios = list(TipoTerritorioCoder)
        activaciones = {}
        for i, tipo in enumerate(territorios):
            activaciones[tipo] = lookup_result[:, :, i]
        
        return activaciones


class PampaRCoder(nn.Module):
    """
    PAMPAr-Coder: Modelo de lenguaje territorial para código.
    
    Arquitectura:
    1. Embedding de tokens
    2. Tálamo determina routing a territorios
    3. Territorios procesan en paralelo
    4. Fronteras comunican entre territorios
    5. Early exit si confianza suficiente
    6. LM Head genera siguiente token
    """
    
    def __init__(self, config: ConfigPampaRCoder = None):
        super().__init__()
        self.config = config or CODER_4GB
        
        # =====================================================================
        # Embeddings
        # =====================================================================
        self.tok_emb = nn.Embedding(self.config.vocab_size, self.config.dim)
        self.pos_emb = nn.Embedding(self.config.max_seq_len, self.config.dim)
        self.emb_dropout = nn.Dropout(self.config.dropout)
        
        # =====================================================================
        # Tálamo (routing)
        # =====================================================================
        self.talamo = TalamoCoder(
            self.config.dim,
            self.config.peso_llaves,
            n_territorios=4,
            vocab_size=self.config.vocab_size
        )
        
        # =====================================================================
        # Bloques Territoriales
        # =====================================================================
        self.bloques = nn.ModuleList([
            GestorTerritoriosCoder(
                self.config.dim,
                self.config.n_heads,
                self.config.dropout,
                self.config.max_seq_len
            )
            for _ in range(self.config.n_capas)
        ])
        
        # =====================================================================
        # LM Head
        # =====================================================================
        self.ln_final = nn.LayerNorm(self.config.dim)
        self.lm_head = nn.Linear(self.config.dim, self.config.vocab_size, bias=False)
        
        # Weight tying (compartir pesos embedding ↔ lm_head)
        self.lm_head.weight = self.tok_emb.weight
        
        # =====================================================================
        # Early Exit
        # =====================================================================
        if self.config.usar_early_exit:
            # Heads intermedios para early exit
            self.early_heads = nn.ModuleList([
                nn.Linear(self.config.dim, self.config.vocab_size, bias=False)
                for _ in range(self.config.n_capas - 1)
            ])
            for head in self.early_heads:
                head.weight = self.tok_emb.weight  # Weight tying
        
        # Inicialización
        self.apply(self._init_weights)
        
        # Causal mask (registrada como buffer)
        mask = torch.tril(torch.ones(self.config.max_seq_len, self.config.max_seq_len))
        self.register_buffer('causal_mask', mask)
    
    def _init_weights(self, module):
        """Inicialización de pesos."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def registrar_tokenizer(self, tokenizer) -> None:
        """Registra tokenizer para LLAVES."""
        self.talamo.registrar_tokenizer(tokenizer)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        use_early_exit: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            input_ids: (B, L) token IDs
            targets: (B, L) targets para calcular loss
            use_early_exit: Usar early exit en inferencia
            
        Returns:
            logits: (B, L, vocab_size)
            loss: Scalar si targets provided
        """
        B, L = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        tok_emb = self.tok_emb(input_ids)
        pos = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_emb(pos)
        x = self.emb_dropout(tok_emb + pos_emb)
        
        # Mask causal
        mask = self.causal_mask[:L, :L].unsqueeze(0).unsqueeze(0)
        
        # Routing desde tálamo
        activaciones = self.talamo(x, input_ids)
        
        # Procesar bloques territoriales
        umbral = self.config.umbral_confianza_exit if use_early_exit else 1.0
        
        for i, bloque in enumerate(self.bloques):
            x, confianza, should_exit = bloque(
                x, activaciones, mask, 
                use_cache=(not self.training),
                umbral_early_exit=umbral
            )
            
            # Early exit en inferencia
            if use_early_exit and should_exit and i >= self.config.capas_minimas - 1:
                if i < len(self.early_heads):
                    logits = self.early_heads[i](self.ln_final(x))
                    return logits, None
        
        # LM Head final
        x = self.ln_final(x)
        logits = self.lm_head(x)
        
        # Calcular loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-100
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        use_early_exit: bool = True
    ) -> torch.Tensor:
        """
        Genera tokens autoregressivamente.
        
        Args:
            prompt_ids: (1, L) prompt tokenizado
            max_new_tokens: Máximo tokens a generar
            temperature: Temperatura de sampling
            top_k: Top-K filtering
            top_p: Nucleus sampling
            use_early_exit: Usar early exit
            
        Returns:
            generated: (1, L + new_tokens)
        """
        self.eval()
        device = prompt_ids.device
        
        # Limpiar caches
        for bloque in self.bloques:
            bloque.clear_caches()
        
        generated = prompt_ids.clone()
        
        for _ in range(max_new_tokens):
            # Solo usar últimos max_seq_len tokens
            context = generated[:, -self.config.max_seq_len:]
            
            # Forward pass
            logits, _ = self.forward(context, use_early_exit=use_early_exit)
            
            # Tomar logits del último token
            logits = logits[:, -1, :] / temperature
            
            # Top-K filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Top-P (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remover tokens con probabilidad acumulada > top_p
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            generated = torch.cat([generated, next_token], dim=1)
            
            # EOS check (si el tokenizer tiene EOS)
            # Por ahora, generar hasta max_new_tokens
        
        return generated
    
    def count_parameters(self) -> Dict[str, int]:
        """Cuenta parámetros por componente."""
        counts = {
            "embeddings": sum(p.numel() for p in self.tok_emb.parameters()) +
                         sum(p.numel() for p in self.pos_emb.parameters()),
            "talamo": sum(p.numel() for p in self.talamo.parameters()),
            "bloques": sum(p.numel() for bloque in self.bloques for p in bloque.parameters()),
            "lm_head": 0,  # Tied con embeddings
            "early_heads": sum(p.numel() for h in self.early_heads for p in h.parameters()) if self.config.usar_early_exit else 0,
        }
        counts["total"] = sum(counts.values())
        return counts


# =============================================================================
# Funciones de utilidad
# =============================================================================

def crear_modelo(preset: str = "4GB") -> PampaRCoder:
    """
    Crea modelo con preset.
    
    Args:
        preset: "4GB", "8GB", "24GB"
    """
    from .config import CODER_4GB, CODER_8GB, CODER_24GB
    
    presets = {
        "4GB": CODER_4GB,
        "8GB": CODER_8GB,
        "24GB": CODER_24GB,
    }
    
    config = presets.get(preset, CODER_4GB)
    return PampaRCoder(config)


# =============================================================================
# Demo
# =============================================================================

def demo_modelo():
    """Demo del modelo PAMPAr-Coder."""
    print("\n" + "=" * 70)
    print("🚀 PAMPAr-Coder - Modelo Demo")
    print("=" * 70)
    
    # Crear modelo
    model = crear_modelo("4GB")
    
    # Estadísticas
    params = model.count_parameters()
    print(f"\n📊 Parámetros:")
    for nombre, count in params.items():
        if count > 0:
            print(f"   {nombre:12}: {count:>12,}")
    
    # VRAM estimada
    vram_train = model.config.estimate_vram_gb(training=True)
    vram_infer = model.config.estimate_vram_gb(training=False)
    print(f"\n💾 VRAM estimada:")
    print(f"   Training:  {vram_train:.2f} GB")
    print(f"   Inference: {vram_infer:.2f} GB")
    
    # Test forward pass
    print("\n🧪 Test forward pass:")
    x = torch.randint(0, model.config.vocab_size, (2, 64))  # (batch=2, seq=64)
    targets = torch.randint(0, model.config.vocab_size, (2, 64))
    
    logits, loss = model(x, targets)
    print(f"   Input:  {x.shape}")
    print(f"   Logits: {logits.shape}")
    print(f"   Loss:   {loss.item():.4f}")
    
    # Test generation
    print("\n🎯 Test generation (early exit):")
    prompt = torch.randint(0, model.config.vocab_size, (1, 10))
    
    import time
    start = time.time()
    generated = model.generate(
        prompt, 
        max_new_tokens=50, 
        use_early_exit=True,
        temperature=0.8
    )
    elapsed = time.time() - start
    
    print(f"   Prompt:    {prompt.shape[1]} tokens")
    print(f"   Generated: {generated.shape[1]} tokens")
    print(f"   Time:      {elapsed:.3f}s ({50/elapsed:.1f} tokens/sec)")
    
    print("\n✅ PAMPAr-Coder funcionando!")


if __name__ == "__main__":
    demo_modelo()
