# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
PAMPAr-Coder v2: Modelo con 52 Zonas de Brodmann.

Arquitectura cerebral avanzada:
- 4 Macro-Territorios (Lóbulos)
- 52 Zonas de Brodmann (Áreas especializadas)
- LLAVES cuantizadas INT4
- Tálamo orquestador
- Knowledge Distillation ready

"El Linux de la IA" - Hacer más con menos.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from .zonas_brodmann import MacroTerritorio, ZonaBrodmann, ZONA_A_TERRITORIO
from .talamo_v2 import TalamoBrodmann, GestorTerritoriosV2


@dataclass
class ConfigPampaRCoderV2:
    """
    Configuración del modelo PAMPAr-Coder v2.
    
    Optimizado para:
    - Hardware consumer (GTX 1650, 4GB VRAM)
    - 52 zonas de Brodmann
    - Knowledge Distillation desde modelos grandes
    """
    # =========================================================================
    # MODELO
    # =========================================================================
    vocab_size: int = 16000
    dim: int = 384              # Aumentado para más capacidad
    n_heads: int = 6
    n_capas: int = 8            # Más capas, compensado por sparse activation
    dropout: float = 0.1
    max_seq_len: int = 2048     # Context más largo
    
    # =========================================================================
    # ARQUITECTURA BRODMANN
    # =========================================================================
    n_zonas: int = 52
    n_territorios: int = 4
    peso_llaves: float = 0.80   # 80% reglas, 20% atención
    usar_cuantizacion: bool = True
    
    # =========================================================================
    # EARLY EXIT
    # =========================================================================
    usar_early_exit: bool = True
    umbral_confianza: float = 0.90
    capas_minimas: int = 3
    
    # =========================================================================
    # KNOWLEDGE DISTILLATION
    # =========================================================================
    usar_distillation: bool = False
    temperatura_distill: float = 2.0
    peso_distill: float = 0.5   # Balance entre hard labels y soft labels
    
    # =========================================================================
    # EFICIENCIA
    # =========================================================================
    use_gradient_checkpointing: bool = True
    use_mixed_precision: bool = True
    
    def estimate_params(self) -> int:
        """Estima número de parámetros."""
        # Embeddings
        emb = self.vocab_size * self.dim + self.max_seq_len * self.dim
        
        # Tálamo (LLAVES + routing)
        talamo = self.vocab_size * self.n_zonas + self.dim * self.n_territorios
        
        # Por capa: 4 territorios x (attn + ffn)
        attn_per_territorio = 4 * self.dim * self.dim  # Q, K, V, O
        ffn_per_territorio = 3 * self.dim * self.dim * 4  # gate, up, down
        per_territorio = attn_per_territorio + ffn_per_territorio
        per_capa = per_territorio * 4 + self.dim * 4  # fusion
        
        capas = per_capa * self.n_capas
        
        # LM head (weight tied)
        lm_head = 0  # tied with embeddings
        
        return emb + talamo + capas + lm_head


# Presets optimizados
CODER_V2_4GB = ConfigPampaRCoderV2(
    vocab_size=16000,
    dim=384,
    n_heads=6,
    n_capas=6,
    max_seq_len=1024,
    usar_cuantizacion=True,
)

CODER_V2_8GB = ConfigPampaRCoderV2(
    vocab_size=16000,
    dim=512,
    n_heads=8,
    n_capas=8,
    max_seq_len=2048,
    usar_cuantizacion=True,
)

CODER_V2_24GB = ConfigPampaRCoderV2(
    vocab_size=16000,
    dim=768,
    n_heads=12,
    n_capas=12,
    max_seq_len=4096,
    usar_cuantizacion=False,  # Más VRAM disponible
)


class PampaRCoderV2(nn.Module):
    """
    PAMPAr-Coder v2 con 52 Zonas de Brodmann.
    
    Arquitectura:
    1. Embedding de tokens
    2. Tálamo con 52 zonas determina routing
    3. Bloques procesan según activaciones de territorio
    4. Early exit si confianza suficiente
    5. LM Head genera siguiente token
    
    Innovaciones v2:
    - 52 zonas vs 4 territorios (13x más granular)
    - LLAVES cuantizadas INT4 (ahorro 8x memoria)
    - Knowledge Distillation ready
    - Sparse activation por zonas
    """
    
    def __init__(self, config: ConfigPampaRCoderV2 = None):
        super().__init__()
        self.config = config or CODER_V2_4GB
        
        # =====================================================================
        # EMBEDDINGS
        # =====================================================================
        self.tok_emb = nn.Embedding(self.config.vocab_size, self.config.dim)
        self.pos_emb = nn.Embedding(self.config.max_seq_len, self.config.dim)
        self.emb_dropout = nn.Dropout(self.config.dropout)
        
        # =====================================================================
        # TÁLAMO v2 (52 Zonas de Brodmann)
        # =====================================================================
        self.talamo = TalamoBrodmann(
            dim=self.config.dim,
            vocab_size=self.config.vocab_size,
            peso_llaves=self.config.peso_llaves,
            usar_cuantizacion=self.config.usar_cuantizacion,
        )
        
        # =====================================================================
        # BLOQUES TERRITORIALES
        # =====================================================================
        self.bloques = nn.ModuleList([
            GestorTerritoriosV2(
                dim=self.config.dim,
                n_heads=self.config.n_heads,
                dropout=self.config.dropout,
                max_seq_len=self.config.max_seq_len,
            )
            for _ in range(self.config.n_capas)
        ])
        
        # =====================================================================
        # LM HEAD
        # =====================================================================
        self.ln_final = nn.LayerNorm(self.config.dim)
        self.lm_head = nn.Linear(self.config.dim, self.config.vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.tok_emb.weight
        
        # =====================================================================
        # EARLY EXIT HEADS
        # =====================================================================
        if self.config.usar_early_exit:
            self.early_heads = nn.ModuleList([
                nn.Linear(self.config.dim, self.config.vocab_size, bias=False)
                for _ in range(self.config.n_capas - 1)
            ])
            for head in self.early_heads:
                head.weight = self.tok_emb.weight  # Weight tying
        
        # =====================================================================
        # DISTILLATION HEAD (para aprender de modelos grandes)
        # =====================================================================
        if self.config.usar_distillation:
            self.distill_proj = nn.Linear(self.config.dim, self.config.dim * 2)
        
        # Inicialización
        self.apply(self._init_weights)
        
        # Causal mask
        mask = torch.tril(torch.ones(self.config.max_seq_len, self.config.max_seq_len))
        self.register_buffer('causal_mask', mask)
    
    def _init_weights(self, module):
        """Inicialización de pesos estilo GPT-2."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def registrar_tokenizer(self, tokenizer) -> Dict:
        """Registra tokenizer en el Tálamo."""
        return self.talamo.registrar_tokenizer(tokenizer)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        teacher_logits: Optional[torch.Tensor] = None,  # Para distillation
        use_early_exit: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict]]:
        """
        Forward pass.
        
        Args:
            input_ids: [B, L] token IDs
            targets: [B, L] para loss
            teacher_logits: [B, L, V] logits del modelo teacher
            use_early_exit: Usar early exit en inferencia
            
        Returns:
            logits: [B, L, vocab_size]
            loss: Scalar si targets provided
            info: Dict con métricas adicionales
        """
        B, L = input_ids.shape
        device = input_ids.device
        
        # =====================================================================
        # EMBEDDINGS
        # =====================================================================
        tok_emb = self.tok_emb(input_ids)
        pos = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_emb(pos)
        x = self.emb_dropout(tok_emb + pos_emb)
        
        # =====================================================================
        # TÁLAMO: Routing a territorios y zonas
        # =====================================================================
        territorio_acts, zona_acts = self.talamo(x, input_ids)
        
        # =====================================================================
        # BLOQUES TERRITORIALES
        # =====================================================================
        info = {'exit_layer': self.config.n_capas}
        
        for i, bloque in enumerate(self.bloques):
            x, confianza, should_exit = bloque(
                x, territorio_acts, zona_acts,
                umbral_early_exit=self.config.umbral_confianza if use_early_exit else 1.0
            )
            
            # Early exit
            if use_early_exit and should_exit and i >= self.config.capas_minimas - 1:
                if i < len(self.early_heads):
                    logits = self.early_heads[i](self.ln_final(x))
                    info['exit_layer'] = i + 1
                    return logits, None, info
        
        # =====================================================================
        # LM HEAD
        # =====================================================================
        x = self.ln_final(x)
        logits = self.lm_head(x)
        
        # =====================================================================
        # LOSS
        # =====================================================================
        loss = None
        if targets is not None:
            # Cross-entropy loss
            ce_loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-100
            )
            
            # Knowledge Distillation loss
            if teacher_logits is not None and self.config.usar_distillation:
                # Soft labels con temperatura
                T = self.config.temperatura_distill
                soft_targets = F.softmax(teacher_logits / T, dim=-1)
                soft_pred = F.log_softmax(logits / T, dim=-1)
                distill_loss = F.kl_div(soft_pred, soft_targets, reduction='batchmean') * (T ** 2)
                
                # Combinar
                loss = (1 - self.config.peso_distill) * ce_loss + self.config.peso_distill * distill_loss
                info['ce_loss'] = ce_loss.item()
                info['distill_loss'] = distill_loss.item()
            else:
                loss = ce_loss
        
        return logits, loss, info
    
    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        use_early_exit: bool = True,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Genera código autoregressivamente.
        
        Returns:
            generated: Tokens generados
            stats: Estadísticas de generación
        """
        self.eval()
        device = prompt_ids.device
        
        generated = prompt_ids.clone()
        stats = {
            'tokens_generated': 0,
            'early_exits': 0,
            'avg_exit_layer': 0,
        }
        
        exit_layers = []
        
        for _ in range(max_new_tokens):
            # Solo últimos max_seq_len tokens
            context = generated[:, -self.config.max_seq_len:]
            
            # Forward
            logits, _, info = self.forward(context, use_early_exit=use_early_exit)
            exit_layers.append(info.get('exit_layer', self.config.n_capas))
            
            # Último token
            logits = logits[:, -1, :] / temperature
            
            # Top-K
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Top-P (nucleus)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
            stats['tokens_generated'] += 1
            
            # Check EOS
            if next_token.item() == 0:  # Asumiendo 0 es EOS
                break
        
        stats['early_exits'] = sum(1 for e in exit_layers if e < self.config.n_capas)
        stats['avg_exit_layer'] = sum(exit_layers) / len(exit_layers) if exit_layers else 0
        
        return generated, stats
    
    def count_parameters(self) -> Dict[str, int]:
        """Cuenta parámetros por componente."""
        counts = {
            'embeddings': sum(p.numel() for p in [self.tok_emb.weight, self.pos_emb.weight]),
            'talamo': sum(p.numel() for p in self.talamo.parameters()),
            'bloques': sum(p.numel() for p in self.bloques.parameters()),
            'lm_head': 0,  # Weight tied
        }
        counts['total'] = sum(counts.values())
        return counts


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def crear_modelo_v2(preset: str = "4GB") -> PampaRCoderV2:
    """
    Crea modelo PAMPAr-Coder v2 según preset.
    
    Args:
        preset: "4GB", "8GB", "24GB"
    """
    presets = {
        "4GB": CODER_V2_4GB,
        "8GB": CODER_V2_8GB,
        "24GB": CODER_V2_24GB,
    }
    
    config = presets.get(preset.upper(), CODER_V2_4GB)
    return PampaRCoderV2(config)


# =============================================================================
# DEMO
# =============================================================================

def demo_modelo():
    """Demo del modelo PAMPAr-Coder v2."""
    print("=" * 70)
    print("🧠 PAMPAr-Coder v2: Modelo con 52 Zonas de Brodmann")
    print("   'El Linux de la IA' - Hacer más con menos")
    print("=" * 70)
    
    # Crear modelo
    model = crear_modelo_v2("4GB")
    
    # Contar parámetros
    params = model.count_parameters()
    print(f"\n📊 Parámetros:")
    for name, count in params.items():
        print(f"   {name:15}: {count:,}")
    
    print(f"\n💾 Memoria estimada: {params['total'] * 4 / 1024**2:.1f} MB (FP32)")
    print(f"                    {params['total'] * 2 / 1024**2:.1f} MB (FP16)")
    
    # Forward pass
    print("\n[TEST] Forward pass...")
    B, L = 2, 128
    input_ids = torch.randint(0, 16000, (B, L))
    targets = torch.randint(0, 16000, (B, L))
    
    logits, loss, info = model(input_ids, targets)
    
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Output shape: {logits.shape}")
    print(f"   Loss: {loss.item():.4f}")
    
    # Test early exit
    print("\n[TEST] Early exit inference...")
    model.eval()
    with torch.no_grad():
        logits, _, info = model(input_ids[:1], use_early_exit=True)
        print(f"   Exit layer: {info.get('exit_layer', 'N/A')}/{model.config.n_capas}")
    
    print("\n✅ Modelo v2 funcionando correctamente")


if __name__ == "__main__":
    demo_modelo()
