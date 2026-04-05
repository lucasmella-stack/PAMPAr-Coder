# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Configuración PAMPAr-Coder v3.

Arquitectura 2D:
  4 STREAMS especializados (Sintaxis, Semántica, Lógico, Estructural)
  × N_LEVELS de profundidad cada uno
  + lateral gates entre streams en cada nivel (fibras blancas)
  + re-routing del Tálamo en cada nivel de profundidad

Cada stream acumula su propia representación a través de los niveles,
como áreas corticales distintas que refinan su propia información
y se comunican lateralmente entre sí.

PRESET_V3 (~110M params):
  dim=640, n_streams=4, n_levels=5
  GQA: 8 Q heads, 2 KV heads, head_dim=80
  vocab=48000, seq_len=4096
"""

from dataclasses import dataclass, field

from pampar.constants import TOKENIZER_PATH


@dataclass
class ConfigV3:
    """Configuración completa de PamparV3."""

    # ── Tokenizer ────────────────────────────────────────────────────────────
    vocab_size: int = 48_000  # pampar_48k.model
    tokenizer_path: str = TOKENIZER_PATH

    # ── Dimensiones ──────────────────────────────────────────────────────────
    dim: int = 640  # Dimensión compartida de todos los streams
    n_streams: int = 4  # Streams: SINTAXIS, SEMANTICA, LOGICO, ESTRUCTURAL
    n_levels: int = 5  # Niveles de profundidad por stream

    # ── Atención (GQA) ───────────────────────────────────────────────────────
    n_heads: int = 8  # Query heads
    n_kv_heads: int = 2  # KV heads (GQA ratio 4:1)
    # head_dim derivado: dim // n_heads = 80

    # ── Feed-forward (SwiGLU) ────────────────────────────────────────────────
    ffn_mult: float = 4.0  # Multiplicador hidden FFN

    # ── Tálamo ───────────────────────────────────────────────────────────────
    n_zonas: int = 52  # Zonas de Brodmann para código
    n_territorios: int = 4  # = n_streams (1:1)
    peso_llaves: float = 0.8  # 80% reglas, 20% aprendido
    ventana_contexto: int = 32  # Kernel conv causal para contextualizar

    # ── Lateral gates (fibras blancas) ───────────────────────────────────────
    # Cada stream recibe aporte de los demás, ponderado por su activación.
    # sym_factor controla el tamaño del bottleneck lateral.
    lateral_bottleneck: int = 128  # dim → 128 → dim para el gate lateral

    # ── Mixed Selectivity (Modulación FiLM) ──────────────────────────────────
    # 1 FFN compartido × n_streams moduladores (en vez de n_streams FFN)
    # El ContextModulator genera gamma/beta desde un vector de 63 indicadores
    # (zona_acts[52] + terr_acts[4] + depth[1] + conf[1] + n_levels[1] + stream_oh[4])
    use_mixed_selectivity: bool = True  # Activar FFN compartido + modulación
    modulator_bottleneck: int = 128  # 63 → 128 → dim×2 para gamma+beta

    # ── Secuencia ────────────────────────────────────────────────────────────
    max_seq_len: int = 4096

    # ── Regularización ───────────────────────────────────────────────────────
    dropout: float = 0.1

    # ── Early Exit ───────────────────────────────────────────────────────────
    umbral_exit: float = 0.90  # Confianza mínima para salir antes
    capas_min: int = 2  # Niveles mínimos antes de early exit
    exit_percentile: float = 0.10  # Foco en el 10% de tokens más difíciles

    # ── Training ─────────────────────────────────────────────────────────────
    use_checkpoint: bool = True  # Gradient checkpointing para ahorrar VRAM

    # ─────────────────────────────────────────────────────────────────────────
    # Propiedades derivadas
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def head_dim(self) -> int:
        """Dimensión por cabeza de atención."""
        return self.dim // self.n_heads

    @property
    def kv_heads(self) -> int:
        """KV heads efectivos (siempre ≥1)."""
        return max(1, self.n_kv_heads)

    @property
    def n_rep(self) -> int:
        """Cuántos Q heads comparten cada KV head."""
        return self.n_heads // self.kv_heads

    @property
    def ffn_hidden(self) -> int:
        """Hidden dim del FFN con SwiGLU (ajustado para la gate extra)."""
        return int(self.dim * self.ffn_mult * 2 / 3)

    def estimate_params(self) -> dict[str, int]:
        """Estima parámetros por componente."""
        # Embedding (weight-tied con lm_head)
        emb = self.vocab_size * self.dim

        # Tálamo inicial
        talamo = (
            self.dim * 192
            + 192  # attn_proj W + b (Linear → 192)
            + 192 * self.n_zonas  # → n_zonas
            + self.n_zonas * self.ventana_contexto  # context_conv depthwise
        )

        # Por nivel de profundidad
        # Atención GQA (compartida)
        attn = (
            self.dim * (self.n_heads * self.head_dim)  # q_proj
            + self.dim * (self.kv_heads * self.head_dim) * 2  # k+v_proj
            + self.dim * self.dim  # o_proj
        )

        # Re-routing ligero por nivel
        reroute = self.dim * self.n_zonas  # Linear(dim, n_zonas) sin bias

        # FFN por nivel: Mixed Selectivity o Legacy
        ffn_single = (
            self.dim * self.ffn_hidden  # gate
            + self.dim * self.ffn_hidden  # up
            + self.ffn_hidden * self.dim  # down
        )

        if self.use_mixed_selectivity:
            # 1 FFN compartido + n_streams moduladores
            modulator_single = (
                63 * self.modulator_bottleneck  # ctx → bottleneck
                + self.modulator_bottleneck * self.dim * 2  # bottleneck → gamma+beta
            )
            ffns = ffn_single + modulator_single * self.n_streams
        else:
            # Legacy: n_streams FFN independientes
            ffns = ffn_single * self.n_streams

        # Lateral gates (bottleneck): n_streams × (dim→bottleneck→dim)
        lateral = self.n_streams * (
            self.dim * self.lateral_bottleneck + self.lateral_bottleneck * self.dim
        )

        # RMSNorm × (2 attn + n_streams FFN + n_streams lateral) ≈ negligible
        norms = self.dim * (2 + self.n_streams * 2) * self.n_levels

        per_level = attn + reroute + ffns + lateral
        niveles = per_level * self.n_levels

        # Cabeza final + norm
        final = self.dim  # norm_f (lm_head weight-tied → no extra)

        total = emb + talamo + niveles + final + norms
        return {
            "embedding": emb,
            "talamo_inicial": talamo,
            "atencion_total": attn * self.n_levels,
            "ffn_total": ffns * self.n_levels,
            "modulators_total": (
                (modulator_single * self.n_streams * self.n_levels)
                if self.use_mixed_selectivity
                else 0
            ),
            "lateral_gates_total": lateral * self.n_levels,
            "rerouting_total": reroute * self.n_levels,
            "total": total,
        }

    def memory_estimate_mb(self, batch_size: int = 1, seq_len: int = 512) -> dict:
        """Estima uso de VRAM en MB para training e inferencia."""
        params = self.estimate_params()["total"]

        # Modelo en fp16
        model_mb = params * 2 / 1024**2

        # Gradientes (fp32) + optimizer Adam (2× fp32 momentums)
        grad_mb = params * 4 / 1024**2
        optim_mb = params * 8 / 1024**2

        # KV cache inferencia: 2 (K+V) × n_kv_heads × head_dim × seq_len × fp16
        kv_mb = (
            2
            * self.kv_heads
            * self.head_dim
            * self.n_levels
            * seq_len
            * batch_size
            * 2
            / 1024**2
        )

        return {
            "model_fp16_mb": round(model_mb, 1),
            "training_total_mb": round(model_mb + grad_mb + optim_mb, 1),
            "kv_cache_inference_mb": round(kv_mb, 1),
        }


# =============================================================================
# PRESETS
# =============================================================================

PRESET_V3 = ConfigV3(
    dim=640,
    n_streams=4,
    n_levels=5,
    n_heads=8,
    n_kv_heads=2,
    ffn_mult=4.0,
    vocab_size=48_000,
    max_seq_len=4096,
    dropout=0.1,
    umbral_exit=0.90,
    capas_min=2,
    exit_percentile=0.10,
    lateral_bottleneck=128,
    use_checkpoint=True,
)
"""~110M parámetros. Óptimo para GTX 1650 4GB con gradient checkpointing."""

PRESET_V3_SMALL = ConfigV3(
    dim=512,
    n_streams=4,
    n_levels=4,
    n_heads=8,
    n_kv_heads=2,
    ffn_mult=3.5,
    vocab_size=48_000,
    max_seq_len=2048,
    dropout=0.1,
    lateral_bottleneck=96,
    use_checkpoint=True,
)
"""~60M parámetros. Para experimentación rápida o hardware más limitado."""

PRESET_V3_LARGE = ConfigV3(
    dim=768,
    n_streams=4,
    n_levels=6,
    n_heads=12,
    n_kv_heads=3,
    ffn_mult=4.0,
    vocab_size=48_000,
    max_seq_len=4096,
    dropout=0.1,
    lateral_bottleneck=192,
    use_checkpoint=True,
)
"""~220M parámetros. Para cloud/RunPod con 24GB VRAM."""
