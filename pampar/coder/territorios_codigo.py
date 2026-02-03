# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Territorios especializados para código.

Cada territorio procesa un aspecto diferente del código:
- SINTAXIS: La gramática, keywords, operadores
- SEMANTICA: El significado, tipos, contexto
- LOGICO: El control flow, condiciones
- ESTRUCTURAL: La forma, indentación, patrones

Las fronteras permiten comunicación bidireccional entre territorios.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum, auto

from .llaves_codigo import TipoTerritorioCoder


@dataclass
class ConfigTerritorioCoder:
    """Configuración de un territorio."""
    tipo: TipoTerritorioCoder
    dim: int = 192
    n_heads: int = 6
    dim_ffn: int = 768
    dropout: float = 0.1
    peso_base: float = 0.25  # Peso por defecto en routing


class AttentionCoder(nn.Module):
    """
    Multi-head attention optimizada para código.
    
    Incluye:
    - Relative position encoding (código es muy sensible a posición)
    - Causal mask (autoregressive)
    - Fast path para inferencia
    """
    
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1, max_len: int = 1024):
        super().__init__()
        assert dim % n_heads == 0
        
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        # Proyecciones Q, K, V combinadas para eficiencia
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Relative position bias (importante para código)
        self.rel_pos_bias = nn.Parameter(torch.zeros(n_heads, max_len, max_len))
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)
        
        # Cache para inferencia
        self._cache_k: Optional[torch.Tensor] = None
        self._cache_v: Optional[torch.Tensor] = None
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = False
    ) -> torch.Tensor:
        B, L, D = x.shape
        
        # QKV projection
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # (B, L, n_heads, head_dim)
        
        q = q.transpose(1, 2)  # (B, n_heads, L, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # KV cache para inferencia incremental
        if use_cache:
            if self._cache_k is not None:
                k = torch.cat([self._cache_k, k], dim=2)
                v = torch.cat([self._cache_v, v], dim=2)
            self._cache_k = k
            self._cache_v = v
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Add relative position bias
        L_k = k.shape[2]
        scores = scores + self.rel_pos_bias[:, :L, :L_k]
        
        # Causal mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, L, D)
        
        return self.out(out)
    
    def clear_cache(self):
        """Limpia KV cache."""
        self._cache_k = None
        self._cache_v = None


class FFNGated(nn.Module):
    """
    Feed-forward con gating (SwiGLU style).
    Más expresivo que FFN simple, importante para código.
    """
    
    def __init__(self, dim: int, dim_ffn: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(dim, dim_ffn, bias=False)
        self.w2 = nn.Linear(dim_ffn, dim, bias=False)
        self.w3 = nn.Linear(dim, dim_ffn, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: (Swish(W1·x) ⊙ W3·x) · W2
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TerritorioCoder(nn.Module):
    """
    Un territorio especializado para procesar código.
    
    Cada territorio tiene:
    - Attention propia
    - FFN con gating
    - Buffer de memoria local
    - Conexiones a fronteras
    """
    
    def __init__(self, config: ConfigTerritorioCoder, max_len: int = 1024):
        super().__init__()
        self.tipo = config.tipo
        self.dim = config.dim
        self.peso_base = config.peso_base
        
        # Layers principales
        self.ln1 = nn.LayerNorm(config.dim)
        self.attn = AttentionCoder(
            config.dim, 
            config.n_heads, 
            config.dropout,
            max_len
        )
        self.ln2 = nn.LayerNorm(config.dim)
        self.ffn = FFNGated(config.dim, config.dim_ffn, config.dropout)
        
        # Buffer de contexto (memoria local del territorio)
        self.register_buffer('buffer', torch.zeros(1, 32, config.dim))
        self.buffer_gate = nn.Linear(config.dim, 1)
        
        # Early exit para este territorio
        self.exit_head = nn.Linear(config.dim, 1)
    
    def forward(
        self, 
        x: torch.Tensor, 
        activacion: torch.Tensor,  # (B, L) pesos de LLAVES
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (B, L, D)
            activacion: Peso de activación por token desde LLAVES
            mask: Causal mask
            use_cache: Usar KV cache para inferencia
            
        Returns:
            out: Output procesado
            confianza: Confianza para early exit
        """
        B, L, D = x.shape
        
        # Ponderar input por activación del territorio
        # Esto es clave: tokens que no pertenecen a este territorio
        # tienen menos influencia
        activacion_expanded = activacion.unsqueeze(-1)  # (B, L, 1)
        x_weighted = x * activacion_expanded
        
        # Self-attention
        h = self.ln1(x_weighted)
        h = x_weighted + self.attn(h, mask, use_cache)
        
        # FFN
        out = h + self.ffn(self.ln2(h))
        
        # Calcular confianza para early exit
        confianza = torch.sigmoid(self.exit_head(out.mean(dim=1)))  # (B, 1)
        
        return out, confianza
    
    def procesar_basal(self, x: torch.Tensor) -> torch.Tensor:
        """
        Procesamiento mínimo cuando la activación es muy baja.
        Solo aplica LayerNorm + FFN sin attention (más rápido).
        """
        h = self.ln2(x)
        return x + 0.1 * self.ffn(h)  # Contribución reducida


class FronteraCoder(nn.Module):
    """
    Frontera bidireccional entre dos territorios.
    
    Permite que la información fluya entre territorios relacionados:
    - SINTAXIS ↔ SEMANTICA (keywords informan significado)
    - LOGICO ↔ ESTRUCTURAL (control flow determina estructura)
    - etc.
    """
    
    def __init__(
        self, 
        origen: TipoTerritorioCoder,
        destino: TipoTerritorioCoder,
        dim: int,
        peso_inicial: float = 0.5
    ):
        super().__init__()
        self.origen = origen
        self.destino = destino
        
        # Gate aprendido para regular flujo
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        
        # Transform para adaptar representaciones
        self.transform = nn.Linear(dim, dim, bias=False)
        
        # Peso base de la conexión
        self.peso_base = nn.Parameter(torch.tensor(peso_inicial))
    
    def forward(
        self, 
        h_origen: torch.Tensor, 
        h_destino: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcula flujo de origen → destino.
        """
        # Gate basado en ambos estados
        combined = torch.cat([h_origen, h_destino], dim=-1)
        gate_value = self.gate(combined) * self.peso_base
        
        # Transform y mezcla
        h_transformed = self.transform(h_origen)
        return h_destino + gate_value * h_transformed


# =============================================================================
# Conexiones entre territorios
# =============================================================================

FRONTERAS_CODER = [
    # Sintaxis-Semántica: muy conectados (keywords ↔ significado)
    (TipoTerritorioCoder.SINTAXIS, TipoTerritorioCoder.SEMANTICA, 0.8),
    
    # Sintaxis-Lógico: keywords de control flow
    (TipoTerritorioCoder.SINTAXIS, TipoTerritorioCoder.LOGICO, 0.6),
    
    # Lógico-Estructural: control flow ↔ bloques
    (TipoTerritorioCoder.LOGICO, TipoTerritorioCoder.ESTRUCTURAL, 0.7),
    
    # Semántica-Estructural: nombres ↔ scopes
    (TipoTerritorioCoder.SEMANTICA, TipoTerritorioCoder.ESTRUCTURAL, 0.5),
    
    # Sintaxis-Estructural: delimitadores ↔ bloques
    (TipoTerritorioCoder.SINTAXIS, TipoTerritorioCoder.ESTRUCTURAL, 0.6),
    
    # Semántica-Lógico: tipos ↔ validación
    (TipoTerritorioCoder.SEMANTICA, TipoTerritorioCoder.LOGICO, 0.5),
]


class GestorTerritoriosCoder(nn.Module):
    """
    Gestiona todos los territorios y sus fronteras.
    
    Responsabilidades:
    - Crear territorios
    - Crear fronteras
    - Coordinar procesamiento
    - Early exit colectivo
    """
    
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1, max_len: int = 1024):
        super().__init__()
        self.dim = dim
        
        # Crear 4 territorios
        configs = {
            TipoTerritorioCoder.SINTAXIS: ConfigTerritorioCoder(
                TipoTerritorioCoder.SINTAXIS, dim, n_heads, dim * 4, dropout, 0.30
            ),
            TipoTerritorioCoder.SEMANTICA: ConfigTerritorioCoder(
                TipoTerritorioCoder.SEMANTICA, dim, n_heads, dim * 4, dropout, 0.25
            ),
            TipoTerritorioCoder.LOGICO: ConfigTerritorioCoder(
                TipoTerritorioCoder.LOGICO, dim, n_heads, dim * 4, dropout, 0.25
            ),
            TipoTerritorioCoder.ESTRUCTURAL: ConfigTerritorioCoder(
                TipoTerritorioCoder.ESTRUCTURAL, dim, n_heads, dim * 4, dropout, 0.20
            ),
        }
        
        self.territorios = nn.ModuleDict({
            t.name: TerritorioCoder(cfg, max_len) 
            for t, cfg in configs.items()
        })
        
        # Crear 6 fronteras bidireccionales
        self.fronteras = nn.ModuleList([
            FronteraCoder(origen, destino, dim, peso)
            for origen, destino, peso in FRONTERAS_CODER
        ])
        
        # Proyección de mezcla final
        self.mix_proj = nn.Linear(dim * 4, dim)
        
        # Early exit global
        self.exit_global = nn.Linear(dim, 1)
    
    def forward(
        self,
        x: torch.Tensor,
        activaciones: Dict[TipoTerritorioCoder, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        umbral_early_exit: float = 0.9
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Procesa input a través de todos los territorios.
        
        Args:
            x: Input (B, L, D)
            activaciones: Dict de territorio -> (B, L) pesos desde LLAVES
            mask: Causal mask
            use_cache: Usar cache para inferencia
            umbral_early_exit: Umbral para salida temprana
            
        Returns:
            out: Output procesado
            confianza: Confianza del modelo
            should_exit: Si debería terminar (early exit)
        """
        B, L, D = x.shape
        
        # Procesar cada territorio
        outputs = {}
        confianzas = []
        
        for tipo in TipoTerritorioCoder:
            territorio = self.territorios[tipo.name]
            activacion = activaciones.get(tipo, torch.ones(B, L, device=x.device) * 0.25)
            
            # Procesamiento completo o basal según activación
            if activacion.mean() > 0.3:
                out_t, conf_t = territorio(x, activacion, mask, use_cache)
            else:
                out_t = territorio.procesar_basal(x)
                conf_t = torch.zeros(B, 1, device=x.device)
            
            outputs[tipo] = out_t
            confianzas.append(conf_t)
        
        # Aplicar fronteras (comunicación entre territorios)
        for frontera in self.fronteras:
            h_origen = outputs[frontera.origen]
            h_destino = outputs[frontera.destino]
            
            # Bidireccional
            outputs[frontera.destino] = frontera(h_origen, h_destino)
            outputs[frontera.origen] = frontera(h_destino, h_origen)
        
        # Mezclar outputs de todos los territorios
        all_outputs = torch.cat([outputs[t] for t in TipoTerritorioCoder], dim=-1)
        mixed = self.mix_proj(all_outputs)
        
        # Confianza global para early exit
        confianza_global = torch.sigmoid(self.exit_global(mixed.mean(dim=1)))
        should_exit = confianza_global.mean() > umbral_early_exit
        
        return mixed, confianza_global, should_exit
    
    def clear_caches(self):
        """Limpia todos los KV caches."""
        for territorio in self.territorios.values():
            territorio.attn.clear_cache()


# =============================================================================
# Demo
# =============================================================================

def demo_territorios():
    """Demo de territorios."""
    print("\n" + "=" * 70)
    print("🏛️ PAMPAr-Coder Territorios Demo")
    print("=" * 70)
    
    # Crear gestor
    gestor = GestorTerritoriosCoder(dim=192, n_heads=6, max_len=512)
    
    # Contar parámetros
    total_params = sum(p.numel() for p in gestor.parameters())
    print(f"\n📊 Total parámetros: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Por territorio
    for nombre, territorio in gestor.territorios.items():
        params = sum(p.numel() for p in territorio.parameters())
        print(f"   {nombre:12}: {params:,}")
    
    # Fronteras
    frontera_params = sum(p.numel() for f in gestor.fronteras for p in f.parameters())
    print(f"   {'Fronteras':12}: {frontera_params:,}")
    
    # Test forward pass
    print("\n🧪 Test forward pass:")
    x = torch.randn(2, 64, 192)  # (batch=2, seq=64, dim=192)
    
    activaciones = {
        TipoTerritorioCoder.SINTAXIS: torch.rand(2, 64),
        TipoTerritorioCoder.SEMANTICA: torch.rand(2, 64),
        TipoTerritorioCoder.LOGICO: torch.rand(2, 64),
        TipoTerritorioCoder.ESTRUCTURAL: torch.rand(2, 64),
    }
    
    with torch.no_grad():
        out, confianza, should_exit = gestor(x, activaciones)
    
    print(f"   Input:  {x.shape}")
    print(f"   Output: {out.shape}")
    print(f"   Confianza: {confianza.mean().item():.3f}")
    print(f"   Early exit: {should_exit}")


if __name__ == "__main__":
    demo_territorios()
