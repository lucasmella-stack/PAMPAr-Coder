# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Configuración para modelo 3B en A40.

Optimizado para:
- GPU: A40 (48GB VRAM)
- Modelo: ~3B parámetros
- Entrenamiento eficiente con gradient checkpointing
"""

from dataclasses import dataclass


@dataclass
class Config3B:
    """Configuración para modelo de 3B parámetros."""
    
    # =========================================================================
    # MODELO (3B)
    # =========================================================================
    vocab_size: int = 16000
    dim: int = 2560              # Dimensión del modelo
    n_heads: int = 20            # Cabezas de atención
    n_capas: int = 32            # Capas transformer
    dropout: float = 0.1
    max_seq_len: int = 2048      # Contexto
    
    # =========================================================================
    # ARQUITECTURA BRODMANN
    # =========================================================================
    n_zonas: int = 52
    n_territorios: int = 4
    peso_llaves: float = 0.80
    usar_cuantizacion: bool = False  # No necesario con 48GB
    
    # =========================================================================
    # EFICIENCIA
    # =========================================================================
    use_gradient_checkpointing: bool = True  # Crítico para 3B
    use_mixed_precision: bool = True         # BF16 en A40
    
    # =========================================================================
    # ENTRENAMIENTO
    # =========================================================================
    batch_size: int = 4
    gradient_accumulation: int = 16
    effective_batch: int = 64    # batch_size * gradient_accumulation
    
    learning_rate: float = 2e-4
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    max_steps: int = 50000
    
    # =========================================================================
    # CHECKPOINTING
    # =========================================================================
    save_every_steps: int = 1000
    eval_every_steps: int = 500
    keep_checkpoints: int = 3
    
    def estimate_memory(self) -> dict:
        """Estima uso de memoria."""
        # Parámetros
        params = self.vocab_size * self.dim  # Embeddings
        params += self.max_seq_len * self.dim  # Pos embeddings
        
        # Por capa
        attn = 4 * self.dim * self.dim  # Q, K, V, O
        ffn = 3 * self.dim * (self.dim * 4)  # FFN con expansion 4x
        per_layer = attn + ffn + 2 * self.dim  # + layer norms
        params += per_layer * self.n_capas
        
        # Territorios
        params += self.dim * self.n_territorios * 4  # Procesamiento territorial
        
        # Total
        total_params = params
        
        # Memoria
        fp16_gb = total_params * 2 / 1024**3
        optimizer_gb = total_params * 8 / 1024**3  # Adam states
        gradients_gb = total_params * 2 / 1024**3
        activations_gb = 8  # Estimado con gradient checkpointing
        
        total_gb = fp16_gb + optimizer_gb + gradients_gb + activations_gb
        
        return {
            "params_billions": total_params / 1e9,
            "model_fp16_gb": fp16_gb,
            "optimizer_gb": optimizer_gb,
            "gradients_gb": gradients_gb,
            "activations_gb": activations_gb,
            "total_training_gb": total_gb,
            "fits_a40": total_gb < 48,
        }


@dataclass  
class Config1_5B:
    """Configuración para modelo de ~150M parámetros."""
    
    vocab_size: int = 32000      # Tokenizer tiene 32K
    dim: int = 512               # Más pequeño
    n_heads: int = 8
    n_capas: int = 12            # Menos capas
    dropout: float = 0.1
    max_seq_len: int = 512       # Contexto más corto
    
    n_zonas: int = 52
    n_territorios: int = 4
    peso_llaves: float = 0.80
    usar_cuantizacion: bool = True  # Cuantizar LLAVES
    
    use_gradient_checkpointing: bool = True
    use_mixed_precision: bool = True
    
    batch_size: int = 32         # Batch grande
    gradient_accumulation: int = 2
    effective_batch: int = 64
    
    learning_rate: float = 1e-3  # Más alto para modelo pequeño
    weight_decay: float = 0.1
    warmup_steps: int = 200
    max_steps: int = 30000
    
    save_every_steps: int = 500
    eval_every_steps: int = 250
    keep_checkpoints: int = 3


# Presets
CONFIGS = {
    "3B": Config3B(),
    "1.5B": Config1_5B(),
}


def print_config(name: str = "3B"):
    """Imprime configuración y estimación de memoria."""
    config = CONFIGS[name]
    
    print(f"\n{'='*60}")
    print(f"📦 Configuración: {name}")
    print(f"{'='*60}")
    
    print(f"\nModelo:")
    print(f"  Dimensión: {config.dim}")
    print(f"  Capas: {config.n_capas}")
    print(f"  Cabezas: {config.n_heads}")
    print(f"  Contexto: {config.max_seq_len}")
    
    print(f"\nEntrenamiento:")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Gradient accum: {config.gradient_accumulation}")
    print(f"  Effective batch: {config.effective_batch}")
    print(f"  Learning rate: {config.learning_rate}")
    
    mem = config.estimate_memory()
    print(f"\nMemoria estimada:")
    print(f"  Parámetros: {mem['params_billions']:.2f}B")
    print(f"  Modelo (FP16): {mem['model_fp16_gb']:.1f} GB")
    print(f"  Optimizer: {mem['optimizer_gb']:.1f} GB")
    print(f"  Gradients: {mem['gradients_gb']:.1f} GB")
    print(f"  Activations: {mem['activations_gb']:.1f} GB")
    print(f"  Total: {mem['total_training_gb']:.1f} GB")
    print(f"  ¿Cabe en A40? {'✅ Sí' if mem['fits_a40'] else '❌ No'}")


if __name__ == "__main__":
    print_config("3B")
    print_config("1.5B")
