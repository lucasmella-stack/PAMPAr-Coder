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

# GTX 1650 4GB - Tu hardware (optimizado para entrenamiento de calidad)
CODER_4GB = ConfigPampaRCoder(
    vocab_size=16000,     # Coincide con tokenizer entrenado
    dim=256,              # Balance velocidad/calidad
    n_heads=8,
    n_capas=6,            # 6 capas para más profundidad
    dropout=0.1,
    max_seq_len=512,      # Contexto razonable para funciones
    peso_llaves=0.8,      # Código es muy predecible
    usar_axiomas=True,
    usar_memoria=True,
    capacidad_memoria=200,
    use_gradient_checkpointing=False,
    use_mixed_precision=True,
    batch_size=4,         # Batch pequeño para estabilidad
    learning_rate=1.5e-4,
    max_epochs=50,
    usar_early_exit=True,
    umbral_confianza_exit=0.90,
    capas_minimas=2,
)

# GTX 1650 4GB + 32GB RAM - MÁXIMA CALIDAD con CPU offloading
# Este es tu config óptimo: modelo más grande que usa RAM cuando necesita
CODER_4GB_MAX = ConfigPampaRCoder(
    vocab_size=16000,     # Vocabulario más rico
    dim=384,              # Dimensión mayor = mejor representación
    n_heads=8,
    n_capas=8,            # 8 capas territoriales
    dropout=0.1,
    max_seq_len=1024,     # Contexto largo para archivos completos
    peso_llaves=0.75,     # Balance reglas/atención
    usar_axiomas=True,
    usar_memoria=True,
    capacidad_memoria=500,
    use_gradient_checkpointing=True,  # Ahorra VRAM
    use_mixed_precision=True,
    batch_size=4,         # Batch pequeño para caber en VRAM
    learning_rate=1.5e-4,
    max_epochs=30,
    usar_early_exit=True,
    umbral_confianza_exit=0.92,
    capas_minimas=3,
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

# =============================================================================
# CONFIGURACIONES PROFESIONALES - Competir con modelos serios
# =============================================================================

# PAMPAr-Coder-1B (Multi-GPU / Cloud)
CODER_1B = ConfigPampaRCoder(
    vocab_size=50257,      # GPT-2 vocab size
    dim=1536,              # 1.5K hidden
    n_heads=16,
    n_capas=24,            # 24 bloques territoriales
    dropout=0.1,
    max_seq_len=4096,      # Archivos completos
    peso_llaves=0.65,      # Más atención aprendida
    usar_axiomas=True,
    usar_memoria=True,
    capacidad_memoria=5000,
    use_gradient_checkpointing=True,
    use_mixed_precision=True,
    batch_size=64,
    learning_rate=6e-5,
    max_epochs=3,
    usar_early_exit=True,
    umbral_confianza_exit=0.94,
    capas_minimas=6,
)

# PAMPAr-Coder-3B (Serio - Competidor de CodeLlama-3B)
CODER_3B = ConfigPampaRCoder(
    vocab_size=50257,
    dim=2560,              # 2.5K hidden
    n_heads=20,
    n_capas=32,            # 32 bloques
    dropout=0.1,
    max_seq_len=8192,      # Contexto largo
    peso_llaves=0.6,       # Balance reglas/atención
    usar_axiomas=True,
    usar_memoria=True,
    capacidad_memoria=10000,
    use_gradient_checkpointing=True,
    use_mixed_precision=True,
    batch_size=128,
    learning_rate=3e-5,
    max_epochs=2,
    usar_early_exit=True,
    umbral_confianza_exit=0.95,
    capas_minimas=8,
)

# PAMPAr-Coder-7B (Competidor de CodeLlama-7B / DeepSeek-Coder-7B)
CODER_7B = ConfigPampaRCoder(
    vocab_size=50257,
    dim=4096,              # 4K hidden (como Llama)
    n_heads=32,
    n_capas=32,
    dropout=0.1,
    max_seq_len=16384,     # Contexto muy largo
    peso_llaves=0.55,      # Más flexible
    usar_axiomas=True,
    usar_memoria=True,
    capacidad_memoria=20000,
    use_gradient_checkpointing=True,
    use_mixed_precision=True,
    batch_size=256,
    learning_rate=2e-5,
    max_epochs=1,
    usar_early_exit=True,
    umbral_confianza_exit=0.95,
    capas_minimas=10,
)

# PAMPAr-Coder-33B (Competidor de Kimi-style, CodeLlama-34B)
CODER_33B = ConfigPampaRCoder(
    vocab_size=50257,
    dim=6656,              # Grande
    n_heads=52,
    n_capas=60,            # 60 bloques territoriales
    dropout=0.05,
    max_seq_len=32768,     # Contexto enorme
    peso_llaves=0.5,       # 50/50 reglas y atención
    usar_axiomas=True,
    usar_memoria=True,
    capacidad_memoria=50000,
    use_gradient_checkpointing=True,
    use_mixed_precision=True,
    batch_size=512,
    learning_rate=1e-5,
    max_epochs=1,
    usar_early_exit=True,
    umbral_confianza_exit=0.96,
    capas_minimas=15,
)

# PAMPAr-Coder-72B (Competidor directo de Kimi-Dev-72B)
CODER_72B = ConfigPampaRCoder(
    vocab_size=50257,
    dim=8192,              # Masivo
    n_heads=64,
    n_capas=80,            # 80 bloques territoriales
    dropout=0.05,
    max_seq_len=65536,     # 64K contexto
    peso_llaves=0.45,      # Más atención para complejidad
    usar_axiomas=True,
    usar_memoria=True,
    capacidad_memoria=100000,
    use_gradient_checkpointing=True,
    use_mixed_precision=True,
    batch_size=1024,
    learning_rate=5e-6,
    max_epochs=1,
    usar_early_exit=True,
    umbral_confianza_exit=0.97,
    capas_minimas=20,
)


def print_coder_configs():
    """Muestra comparación de configuraciones."""
    configs = [
        ("CODER_4GB (GTX 1650)", CODER_4GB),
        ("CODER_8GB (RTX 3060)", CODER_8GB),
        ("CODER_24GB (RTX 4090)", CODER_24GB),
        ("CODER_1B (Cloud/Multi-GPU)", CODER_1B),
        ("CODER_3B (CodeLlama-3B tier)", CODER_3B),
        ("CODER_7B (CodeLlama-7B tier)", CODER_7B),
        ("CODER_33B (CodeLlama-34B tier)", CODER_33B),
        ("CODER_72B (Kimi-72B tier)", CODER_72B),
    ]
    
    print("\n" + "=" * 80)
    print("🐸 PAMPAr-Coder - Configuraciones (Arquitectura Territorial para Código)")
    print("=" * 80)
    
    for name, cfg in configs:
        params = cfg.estimate_params()
        vram_train = cfg.estimate_vram_gb(training=True)
        vram_infer = cfg.estimate_vram_gb(training=False)
        
        # Formato según tamaño
        if params['total'] >= 1e9:
            param_str = f"{params['total']/1e9:.1f}B"
        else:
            param_str = f"{params['total']/1e6:.1f}M"
        
        print(f"\n🔹 {name}:")
        print(f"   Parámetros: {params['total']:,} ({param_str})")
        print(f"   VRAM: {vram_train:.1f} GB (train) | {vram_infer:.1f} GB (infer)")
        print(f"   Arquitectura: dim={cfg.dim}, capas={cfg.n_capas}, ctx={cfg.max_seq_len:,}")
        print(f"   LLAVES: {int(cfg.peso_llaves*100)}% reglas + {int((1-cfg.peso_llaves)*100)}% atención")
        print(f"   Early Exit: capas_min={cfg.capas_minimas}, umbral={cfg.umbral_confianza_exit}")
    
    print("\n" + "=" * 80)
    print("💡 La arquitectura TERRITORIAL escala manteniendo:")
    print("   - 4 Territorios: SINTAXIS, SEMANTICA, LOGICO, ESTRUCTURAL")
    print("   - 6 Fronteras bidireccionales entre territorios")
    print("   - Sistema LLAVES: reglas explícitas + atención aprendida")
    print("   - Early Exit: inferencia adaptativa por confianza")
    print("=" * 80)


if __name__ == "__main__":
    print_coder_configs()
