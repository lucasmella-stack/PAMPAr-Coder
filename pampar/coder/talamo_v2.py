# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Tálamo v2 con 52 Zonas de Brodmann.

Arquitectura inspirada en el cerebro humano:
- 4 Macro-Territorios (Lóbulos cerebrales)
- 52 Zonas de Brodmann (Áreas funcionales especializadas)
- LLAVES cuantizadas INT4 para routing eficiente
- Atención aprendida para casos ambiguos

El tálamo:
- Recibe TODOS los tokens
- Usa LLAVES para determinar qué zonas activar
- Agrega zonas → territorios para procesamiento
- Combina reglas (80%) + atención (20%)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .zonas_brodmann import (
    MacroTerritorio, ZonaBrodmann, ZONA_A_TERRITORIO,
    LLAVES_BRODMANN, RegistroBrodmann, ClasificadorBrodmann
)


class TalamoBrodmann(nn.Module):
    """
    Tálamo con 52 Zonas de Brodmann.
    
    Innovaciones:
    1. Routing a nivel de ZONA (52) no solo territorio (4)
    2. LLAVES cuantizadas INT4 para eficiencia  
    3. Sparse activation - solo zonas relevantes
    4. Agregación jerárquica: zonas → territorios
    """
    
    def __init__(
        self,
        dim: int,
        vocab_size: int = 16000,
        peso_llaves: float = 0.80,
        usar_cuantizacion: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.peso_llaves = peso_llaves
        self.usar_cuantizacion = usar_cuantizacion
        
        self.n_zonas = len(ZonaBrodmann)        # 52
        self.n_territorios = len(MacroTerritorio)  # 4
        
        # Registro de Brodmann para clasificación
        self.registro = RegistroBrodmann(vocab_size)
        
        # =====================================================================
        # LLAVES: Lookup tables cuantizadas
        # =====================================================================
        # Para cada token → activaciones de zonas
        # Shape: [vocab_size, 52] - cuantizable a INT4
        if usar_cuantizacion:
            # Simular INT4 con escala + offset
            self.register_buffer('llaves_scale', torch.ones(1))
            self.register_buffer('llaves_offset', torch.zeros(1))
        
        self.register_buffer(
            'llaves_zonas', 
            torch.zeros(vocab_size, self.n_zonas)
        )
        
        # Matriz de agregación: zonas → territorios
        # Shape: [52, 4]
        agregacion = torch.zeros(self.n_zonas, self.n_territorios)
        for i, zona in enumerate(ZonaBrodmann):
            territorio = ZONA_A_TERRITORIO[zona]
            j = list(MacroTerritorio).index(territorio)
            agregacion[i, j] = 1.0
        self.register_buffer('zona_a_territorio', agregacion)
        
        # =====================================================================
        # ATENCIÓN APRENDIDA (20% del routing)
        # =====================================================================
        # Red pequeña que aprende a ajustar el routing
        self.attn_routing = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(dim // 4, self.n_territorios),
        )
        
        # Proyección para combinar zonas activas
        self.zona_proj = nn.Linear(self.n_zonas, dim // 4, bias=False)
        
        # =====================================================================
        # STATS para análisis
        # =====================================================================
        self.register_buffer('stats_zonas', torch.zeros(self.n_zonas))
        self.register_buffer('stats_territorios', torch.zeros(self.n_territorios))
        self.register_buffer('n_tokens_vistos', torch.tensor(0))
    
    def registrar_tokenizer(self, tokenizer) -> Dict:
        """
        Registra tokenizer y llena las LLAVES.
        
        Returns:
            Estadísticas de clasificación
        """
        stats = self.registro.registrar_tokenizer(tokenizer)
        
        # Llenar lookup table de LLAVES
        print("\n[TÁLAMO] Llenando lookup tables...")
        
        clasificador = ClasificadorBrodmann()
        
        for token_id in range(min(tokenizer.vocab_size(), self.vocab_size)):
            try:
                piece = tokenizer.id_to_piece(token_id)
                zonas = clasificador.clasificar(piece)
                
                for i, zona in enumerate(ZonaBrodmann):
                    if zona in zonas:
                        self.llaves_zonas[token_id, i] = zonas[zona]
                        
            except:
                continue
        
        # Cuantizar si está habilitado
        if self.usar_cuantizacion:
            self._cuantizar_llaves()
        
        print(f"[OK] Tálamo configurado con {self.n_zonas} zonas")
        
        return stats
    
    def _cuantizar_llaves(self):
        """Cuantiza LLAVES a INT4 (simulado con escala)."""
        # Encontrar min/max
        min_val = self.llaves_zonas.min()
        max_val = self.llaves_zonas.max()
        
        # INT4: 16 valores (0-15)
        self.llaves_scale = (max_val - min_val) / 15
        self.llaves_offset = min_val
        
        # Cuantizar y de-cuantizar (para mantener gradientes)
        if self.llaves_scale > 0:
            quantized = ((self.llaves_zonas - self.llaves_offset) / self.llaves_scale).round()
            quantized = quantized.clamp(0, 15)
            self.llaves_zonas.data = quantized * self.llaves_scale + self.llaves_offset
        
        # Calcular ahorro de memoria
        original_bytes = self.llaves_zonas.numel() * 4  # FP32
        quantized_bytes = self.llaves_zonas.numel() * 0.5  # INT4
        print(f"[CUANTIZACIÓN] {original_bytes/1024:.1f}KB → {quantized_bytes/1024:.1f}KB (INT4)")
    
    def forward(
        self,
        x: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[MacroTerritorio, torch.Tensor], Dict[ZonaBrodmann, torch.Tensor]]:
        """
        Calcula activaciones de zonas y territorios.
        
        Args:
            x: Embeddings [B, L, D]
            token_ids: IDs de tokens [B, L]
            
        Returns:
            territorio_acts: Dict[MacroTerritorio, [B, L]] 
            zona_acts: Dict[ZonaBrodmann, [B, L]]
        """
        B, L, D = x.shape
        device = x.device
        
        # =====================================================================
        # 1. LLAVES: Obtener activaciones de zonas desde lookup
        # =====================================================================
        if token_ids is not None:
            # Clamp para evitar out of bounds
            safe_ids = token_ids.clamp(0, self.vocab_size - 1)
            # Lookup: [B, L] → [B, L, 52]
            zonas_llaves = self.llaves_zonas[safe_ids]
        else:
            # Fallback: distribución uniforme
            zonas_llaves = torch.ones(B, L, self.n_zonas, device=device) / self.n_zonas
        
        # Agregar zonas → territorios: [B, L, 52] @ [52, 4] → [B, L, 4]
        territorios_llaves = torch.matmul(zonas_llaves, self.zona_a_territorio)
        
        # Normalizar territorios
        territorios_llaves = territorios_llaves / (territorios_llaves.sum(dim=-1, keepdim=True) + 1e-8)
        
        # =====================================================================
        # 2. ATENCIÓN APRENDIDA: Ajuste fino
        # =====================================================================
        # [B, L, D] → [B, L, 4]
        territorios_attn = self.attn_routing(x)
        territorios_attn = F.softmax(territorios_attn, dim=-1)
        
        # =====================================================================
        # 3. COMBINAR: LLAVES (80%) + Atención (20%)
        # =====================================================================
        territorios_final = (
            self.peso_llaves * territorios_llaves +
            (1 - self.peso_llaves) * territorios_attn
        )
        
        # =====================================================================
        # 4. CONSTRUIR DICTS DE SALIDA
        # =====================================================================
        territorio_acts = {}
        for i, territorio in enumerate(MacroTerritorio):
            territorio_acts[territorio] = territorios_final[:, :, i]
        
        zona_acts = {}
        for i, zona in enumerate(ZonaBrodmann):
            zona_acts[zona] = zonas_llaves[:, :, i]
        
        # =====================================================================
        # 5. STATS (solo en training)
        # =====================================================================
        if self.training:
            self._actualizar_stats(zonas_llaves, territorios_final)
        
        return territorio_acts, zona_acts
    
    def _actualizar_stats(self, zonas: torch.Tensor, territorios: torch.Tensor):
        """Actualiza estadísticas de activación."""
        # Promedio de activaciones
        self.stats_zonas += zonas.mean(dim=[0, 1])
        self.stats_territorios += territorios.mean(dim=[0, 1])
        self.n_tokens_vistos += zonas.shape[0] * zonas.shape[1]
    
    def get_stats(self) -> Dict:
        """Obtiene estadísticas de activación."""
        n = max(self.n_tokens_vistos.item(), 1)
        
        zonas_avg = self.stats_zonas / n
        territorios_avg = self.stats_territorios / n
        
        return {
            'zonas': {z.name: zonas_avg[i].item() for i, z in enumerate(ZonaBrodmann)},
            'territorios': {t.name: territorios_avg[i].item() for i, t in enumerate(MacroTerritorio)},
            'tokens_vistos': self.n_tokens_vistos.item(),
        }
    
    def reset_stats(self):
        """Resetea estadísticas."""
        self.stats_zonas.zero_()
        self.stats_territorios.zero_()
        self.n_tokens_vistos.zero_()


# =============================================================================
# GESTOR DE TERRITORIOS v2
# =============================================================================

class TerritorioV2(nn.Module):
    """
    Territorio con soporte para activaciones de Brodmann.
    
    Cada territorio agrupa múltiples zonas de Brodmann y procesa
    los tokens según sus activaciones.
    """
    
    def __init__(
        self,
        tipo: MacroTerritorio,
        dim: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.tipo = tipo
        self.dim = dim
        
        # Zonas que pertenecen a este territorio
        self.zonas = [z for z in ZonaBrodmann if ZONA_A_TERRITORIO[z] == tipo]
        self.n_zonas = len(self.zonas)
        
        # Attention
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        
        # FFN con gating (SwiGLU)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn_gate = nn.Linear(dim, dim * 4, bias=False)
        self.ffn_up = nn.Linear(dim, dim * 4, bias=False)
        self.ffn_down = nn.Linear(dim * 4, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Proyección de zonas (opcional, para modular por zona)
        self.zona_weights = nn.Parameter(torch.ones(self.n_zonas))
        
        # Causal mask
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer('causal_mask', mask)
    
    def forward(
        self,
        x: torch.Tensor,
        activacion: torch.Tensor,
        zona_acts: Optional[Dict[ZonaBrodmann, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Procesa tokens según activación del territorio.
        
        Args:
            x: [B, L, D] embeddings
            activacion: [B, L] peso del territorio para cada token
            zona_acts: Dict de activaciones por zona (opcional)
        """
        B, L, D = x.shape
        
        # Escalar por activación del territorio
        # [B, L, 1] * [B, L, D] → [B, L, D]
        x_scaled = x * activacion.unsqueeze(-1)
        
        # Self-attention con causal mask
        mask = self.causal_mask[:L, :L]
        attn_mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
        
        residual = x_scaled
        x_norm = self.norm1(x_scaled)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x_scaled = residual + self.dropout(attn_out)
        
        # FFN con SwiGLU
        residual = x_scaled
        x_norm = self.norm2(x_scaled)
        gate = F.silu(self.ffn_gate(x_norm))
        up = self.ffn_up(x_norm)
        x_scaled = residual + self.dropout(self.ffn_down(gate * up))
        
        return x_scaled


class GestorTerritoriosV2(nn.Module):
    """
    Gestor de territorios v2 con soporte Brodmann.
    
    Orquesta los 4 territorios y combina sus salidas
    según las activaciones del Tálamo.
    """
    
    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.dim = dim
        
        # Crear los 4 territorios
        self.territorios = nn.ModuleDict({
            territorio.name: TerritorioV2(
                tipo=territorio,
                dim=dim,
                n_heads=n_heads,
                dropout=dropout,
                max_seq_len=max_seq_len,
            )
            for territorio in MacroTerritorio
        })
        
        # Fusión de territorios
        self.fusion = nn.Linear(dim * 4, dim, bias=False)
        self.norm_fusion = nn.LayerNorm(dim)
        
        # Early exit: detector de confianza
        self.confianza = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        territorio_acts: Dict[MacroTerritorio, torch.Tensor],
        zona_acts: Optional[Dict[ZonaBrodmann, torch.Tensor]] = None,
        umbral_early_exit: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Procesa a través de todos los territorios.
        
        Returns:
            x: Output procesado
            confianza: Nivel de confianza
            should_exit: Si se puede hacer early exit
        """
        B, L, D = x.shape
        
        # Procesar cada territorio
        outputs = []
        for territorio in MacroTerritorio:
            activacion = territorio_acts[territorio]
            t_output = self.territorios[territorio.name](x, activacion, zona_acts)
            outputs.append(t_output)
        
        # Concatenar y fusionar
        # [B, L, D*4] → [B, L, D]
        concat = torch.cat(outputs, dim=-1)
        fused = self.fusion(concat)
        
        # Residual connection
        x = self.norm_fusion(x + fused)
        
        # Calcular confianza para early exit
        confianza = self.confianza(x).squeeze(-1)  # [B, L]
        mean_confianza = confianza.mean()
        should_exit = mean_confianza > umbral_early_exit
        
        return x, confianza, should_exit


# =============================================================================
# DEMO
# =============================================================================

def demo_talamo():
    """Demo del Tálamo v2 con Brodmann."""
    print("=" * 70)
    print("🧠 PAMPAr-Coder: Tálamo v2 con 52 Zonas de Brodmann")
    print("=" * 70)
    
    # Crear tálamo
    talamo = TalamoBrodmann(
        dim=256,
        vocab_size=16000,
        peso_llaves=0.80,
        usar_cuantizacion=True,
    )
    
    print(f"\nConfiguración:")
    print(f"  - Dimensión: 256")
    print(f"  - Zonas: {talamo.n_zonas}")
    print(f"  - Territorios: {talamo.n_territorios}")
    print(f"  - Peso LLAVES: {talamo.peso_llaves}")
    
    # Simular forward pass
    print("\n[TEST] Forward pass simulado...")
    B, L, D = 2, 64, 256
    x = torch.randn(B, L, D)
    token_ids = torch.randint(0, 16000, (B, L))
    
    territorio_acts, zona_acts = talamo(x, token_ids)
    
    print(f"\nActivaciones de Territorios:")
    for t, acts in territorio_acts.items():
        print(f"  {t.name:12}: mean={acts.mean():.3f}, std={acts.std():.3f}")
    
    print(f"\nTop 5 Zonas más activas:")
    zona_means = [(z, acts.mean().item()) for z, acts in zona_acts.items()]
    zona_means.sort(key=lambda x: x[1], reverse=True)
    for zona, mean in zona_means[:5]:
        print(f"  {zona.name:30}: {mean:.3f}")


if __name__ == "__main__":
    demo_talamo()
