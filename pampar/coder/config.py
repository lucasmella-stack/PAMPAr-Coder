# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Configuración de PAMPAr-Coder.

Arquitectura territorial especializada para código:
- Territorios: SINTAXIS, SEMANTICA, LOGICO, ESTRUCTURAL
- LLAVES adaptadas a tokens de programación
- Optimizado para inferencia rápida en hardware consumer
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class ConfigPampaRCoder:
    """
    Configuración del modelo PAMPAr-Coder.
    
    Diferencias con PAMPAr base:
    - Territorios especializados para código
    - LLAVES específicas para sintaxis de lenguajes
    - Mayor peso en estructural (código es muy estructurado)
    - Context window más grande (código necesita más contexto)
    """
    # ============================================
    # Modelo
    # ============================================
    vocab_size: int = 16000          # Más grande para tokens de código
    dim: int = 256                   # Dimensión de embeddings
    n_heads: int = 8                 # Heads de atención
    n_capas: int = 4                 # Bloques territoriales
    dropout: float = 0.1
    max_seq_len: int = 1024          # Código necesita contexto largo
    
    # ============================================
    # Sistema LLAVES para código
    # ============================================
    peso_llaves: float = 0.75        # Código es muy regular, más reglas
    
    # Lenguajes soportados (para LLAVES específicas)
    lenguajes: List[str] = field(default_factory=lambda: [
        'python', 'javascript', 'typescript', 'rust', 'go', 'java'
    ])
    
    # ============================================
    # Territorios especializados
    # ============================================
    # SINTAXIS: keywords, operadores, delimitadores
    # SEMANTICA: nombres, strings, comentarios
    # LOGICO: control flow, condiciones, loops
    # ESTRUCTURAL: indentación, bloques, scopes
    
    peso_territorio_sintaxis: float = 0.3
    peso_territorio_semantica: float = 0.25
    peso_territorio_logico: float = 0.25
    peso_territorio_estructural: float = 0.2
    
    # ============================================
    # Axiomas para código
    # ============================================
    usar_axiomas: bool = True
    # Axiomas de código:
    # - Tipo consistencia: if x: int entonces operaciones de int
    # - Scope resolution: variable definida antes de uso
    # - Pattern matching: if/else, try/except completos
    
    # ============================================
    # Memoria de patrones de código
    # ============================================
    usar_memoria: bool = True
    capacidad_memoria: int = 500     # Patrones de código comunes
    
    # ============================================
    # Eficiencia
    # ============================================
    use_gradient_checkpointing: bool = False
    use_mixed_precision: bool = True
    
    # ============================================
    # Entrenamiento
    # ============================================
    batch_size: int = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_epochs: int = 50
    
    # ============================================
    # Especial: Early Exit para inferencia rápida
    # ============================================
    usar_early_exit: bool = True
    umbral_confianza_exit: float = 0.92  # Salir si confianza > 92%
    capas_minimas: int = 2               # Mínimo 2 capas siempre
    
    def estimate_params(self) -> Dict[str, int]:
        """Estima parámetros del modelo."""
        # Embeddings (vocab + posicional)
        emb = self.vocab_size * self.dim + self.max_seq_len * self.dim
        
        # 4 territorios × n_capas bloques
        # Cada territorio: ~12 * dim^2 por attention + ffn
        territorios = 4 * self.n_capas * 12 * self.dim * self.dim
        
        # 6 fronteras bidireccionales
        fronteras = 6 * 3 * self.dim * self.dim
        
        # Tálamo (routing)
        talamo = self.n_capas * 4 * self.dim * self.dim
        
        # Axiomas
        axiomas = 4 * self.dim * self.dim if self.usar_axiomas else 0
        
        # Memoria
        memoria = self.capacidad_memoria * self.dim if self.usar_memoria else 0
        
        # LM Head (tied con embeddings = 0 extra)
        lm_head = 0
        
        total = emb + territorios + fronteras + talamo + axiomas + memoria
        
        return {
            "embeddings": emb,
            "territorios": territorios,
            "fronteras": fronteras,
            "talamo": talamo,
            "axiomas": axiomas,
            "memoria": memoria,
            "total": total
        }
    
    def estimate_vram_gb(self, training: bool = True) -> float:
        """Estima VRAM necesaria."""
        params = self.estimate_params()["total"]
        
        bytes_per_param = 2 if self.use_mixed_precision else 4
        model_bytes = params * bytes_per_param
        
        if training:
            # Modelo + gradientes + optimizer (AdamW 2x) + activaciones
            grad_bytes = params * bytes_per_param
            optim_bytes = params * 4 * 2  # Siempre FP32
            act_bytes = self.batch_size * self.max_seq_len * self.dim * self.n_capas * 4 * 10 * bytes_per_param
            total = model_bytes + grad_bytes + optim_bytes + act_bytes
        else:
            # Solo modelo + KV cache
            kv_cache = self.max_seq_len * self.dim * self.n_capas * 2 * bytes_per_param
            total = model_bytes + kv_cache
        
        return total / (1024**3)


# =============================================================================
# PRESETS PARA PAMPAr-Coder
# =============================================================================

# GTX 1650 4GB - Tu hardware
CODER_4GB = ConfigPampaRCoder(
    vocab_size=8000,      # BPE optimizado para código
    dim=192,              # Balance velocidad/calidad
    n_heads=6,
    n_capas=4,
    dropout=0.1,
    max_seq_len=512,      # Contexto razonable para funciones
    peso_llaves=0.8,      # Código es muy predecible
    usar_axiomas=True,
    usar_memoria=True,
    capacidad_memoria=200,
    use_gradient_checkpointing=False,
    use_mixed_precision=True,
    batch_size=8,
    learning_rate=2e-4,
    max_epochs=50,
    usar_early_exit=True,
    umbral_confianza_exit=0.90,
    capas_minimas=2,
)

# RTX 3060/3070 8GB
CODER_8GB = ConfigPampaRCoder(
    vocab_size=16000,
    dim=384,
    n_heads=8,
    n_capas=6,
    dropout=0.1,
    max_seq_len=1024,
    peso_llaves=0.75,
    usar_axiomas=True,
    usar_memoria=True,
    capacidad_memoria=500,
    use_gradient_checkpointing=True,
    use_mixed_precision=True,
    batch_size=16,
    learning_rate=1.5e-4,
    max_epochs=50,
    usar_early_exit=True,
    umbral_confianza_exit=0.92,
    capas_minimas=2,
)

# RTX 4090 24GB - Máximo local
CODER_24GB = ConfigPampaRCoder(
    vocab_size=32000,
    dim=512,
    n_heads=8,
    n_capas=8,
    dropout=0.1,
    max_seq_len=2048,
    peso_llaves=0.7,
    usar_axiomas=True,
    usar_memoria=True,
    capacidad_memoria=1000,
    use_gradient_checkpointing=True,
    use_mixed_precision=True,
    batch_size=32,
    learning_rate=1e-4,
    max_epochs=100,
    usar_early_exit=True,
    umbral_confianza_exit=0.93,
    capas_minimas=3,
)


def print_coder_configs():
    """Muestra comparación de configuraciones."""
    configs = [
        ("CODER_4GB (GTX 1650)", CODER_4GB),
        ("CODER_8GB (RTX 3060)", CODER_8GB),
        ("CODER_24GB (RTX 4090)", CODER_24GB),
    ]
    
    print("\n" + "=" * 70)
    print("PAMPAr-Coder - Configuraciones")
    print("=" * 70)
    
    for name, cfg in configs:
        params = cfg.estimate_params()
        vram_train = cfg.estimate_vram_gb(training=True)
        vram_infer = cfg.estimate_vram_gb(training=False)
        
        print(f"\n{name}:")
        print(f"  Parámetros: {params['total']:,} ({params['total']/1e6:.1f}M)")
        print(f"  VRAM train: {vram_train:.2f} GB | infer: {vram_infer:.2f} GB")
        print(f"  Config: dim={cfg.dim}, capas={cfg.n_capas}, ctx={cfg.max_seq_len}")
        print(f"  Early Exit: {cfg.usar_early_exit} (umbral={cfg.umbral_confianza_exit})")


if __name__ == "__main__":
    print_coder_configs()
