# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Harness A/B para comparar Path A (stack v3) vs Path B (recurrent loop)
en `PamparV4`. Reusable desde tests y desde scripts de benchmarking.

Métricas reportadas:
  - n_params: parámetros totales del modelo (sin contar shared dos veces)
  - loss: cross-entropy en input fijo
  - latency_ms: tiempo medio de un forward (eval, sin grad)
  - n_steps: pasos efectivos del loop (path B; -1 en path A)

Uso típico desde test:
    from pampar.coder.v4.ab_harness import build_pair, compare_forward

    cfg_base = ConfigV4(...)
    m_a, m_b = build_pair(cfg_base, max_loop_iters=4, share_ffn=True)
    report = compare_forward(m_a, m_b, input_ids)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch

from .config import ConfigV4
from .modelo import PamparV4


@dataclass
class ABReport:
    """Resultado de una corrida A/B."""

    n_params_a: int
    n_params_b: int
    loss_a: float
    loss_b: float
    latency_ms_a: float
    latency_ms_b: float
    n_steps_b: int
    output_diff_max: float = 0.0
    output_diff_mean: float = 0.0
    extra: dict = field(default_factory=dict)

    def summary(self) -> str:
        """Resumen humano de una página."""
        params_pct = (
            100.0 * (self.n_params_b - self.n_params_a) / max(self.n_params_a, 1)
        )
        speed_pct = (
            100.0
            * (self.latency_ms_b - self.latency_ms_a)
            / max(self.latency_ms_a, 1e-6)
        )
        return (
            f"Path A vs Path B comparison\n"
            f"  params:    A={self.n_params_a:,}  B={self.n_params_b:,}  "
            f"({params_pct:+.1f}%)\n"
            f"  loss:      A={self.loss_a:.4f}  B={self.loss_b:.4f}\n"
            f"  latency:   A={self.latency_ms_a:.2f}ms  B={self.latency_ms_b:.2f}ms  "
            f"({speed_pct:+.1f}%)\n"
            f"  B steps:   {self.n_steps_b}\n"
            f"  out diff:  max={self.output_diff_max:.4e}  "
            f"mean={self.output_diff_mean:.4e}\n"
        )


def count_params(model: torch.nn.Module) -> int:
    """Cuenta parámetros únicos (cada tensor compartido cuenta una sola vez)."""
    seen: set[int] = set()
    total = 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
    return total


def build_pair(
    cfg_base: ConfigV4,
    max_loop_iters: int = 4,
    share_ffn: bool = False,
    use_lti: bool = True,
    use_loop_rope: bool = True,
    use_act: bool = False,
    seed: int = 42,
) -> tuple[PamparV4, PamparV4]:
    """Construye el par (modelo_path_A, modelo_path_B) con misma seed.

    Args:
        cfg_base:        Config base. Se clona para cada path con flags
                         distintas; el path A fuerza `use_recurrent_loop=False`.
        max_loop_iters:  iteraciones para path B.
        share_ffn:       activa Fase 5b (sharing del FFN) en path B.
        use_lti, use_loop_rope, use_act: flags de RecurrentBlock para path B.
        seed:            misma seed para que ambos partan del mismo init
                         determinístico.
    """
    cfg_a = _replace(
        cfg_base,
        use_recurrent_loop=False,
    )
    cfg_b = _replace(
        cfg_base,
        use_recurrent_loop=True,
        max_loop_iters=max_loop_iters,
        use_lti_injection=use_lti,
        use_loop_index_rope=use_loop_rope,
        use_act_halting=use_act,
        share_ffn_across_niveles=share_ffn,
    )

    torch.manual_seed(seed)
    m_a = PamparV4(cfg_a).eval()
    torch.manual_seed(seed)
    m_b = PamparV4(cfg_b).eval()
    return m_a, m_b


def compare_forward(
    m_a: PamparV4,
    m_b: PamparV4,
    input_ids: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
    n_warmup: int = 1,
    n_runs: int = 3,
) -> ABReport:
    """Corre forward en ambos modelos y devuelve el reporte.

    Args:
        m_a:        modelo path A.
        m_b:        modelo path B.
        input_ids:  [B, L] tokens.
        targets:    opcional; si se provee se calcula loss.
        n_warmup:   forwards de calentamiento (no se cronometran).
        n_runs:     forwards cronometrados; se reporta la media.

    Returns:
        ABReport con todas las métricas.
    """
    if targets is None:
        targets = input_ids

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            m_a(input_ids, targets=targets)
            m_b(input_ids, targets=targets)

    # Cronometrar
    lat_a = _bench(m_a, input_ids, targets, n_runs)
    lat_b = _bench(m_b, input_ids, targets, n_runs)

    # Última pasada para capturar outputs/loss/info
    with torch.no_grad():
        logits_a, loss_a, info_a = m_a(input_ids, targets=targets)
        logits_b, loss_b, info_b = m_b(input_ids, targets=targets)

    diff = (logits_a - logits_b).abs()
    return ABReport(
        n_params_a=count_params(m_a),
        n_params_b=count_params(m_b),
        loss_a=float(loss_a),
        loss_b=float(loss_b),
        latency_ms_a=lat_a,
        latency_ms_b=lat_b,
        n_steps_b=int(info_b.get("recurrent_n_steps", -1)),
        output_diff_max=float(diff.max()),
        output_diff_mean=float(diff.mean()),
        extra={
            "info_a_exit_nivel": info_a.get("exit_nivel"),
            "info_b_ponder": float(info_b.get("recurrent_ponder_cost", 0.0)),
        },
    )


# =============================================================================
# Helpers internos
# =============================================================================


def _bench(
    model: PamparV4,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    n_runs: int,
) -> float:
    """Devuelve latencia media en milisegundos sobre n_runs forwards."""
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(input_ids, targets=targets)
            times.append((time.perf_counter() - t0) * 1000.0)
    return sum(times) / len(times)


def _replace(cfg: ConfigV4, **overrides) -> ConfigV4:
    """Clona un ConfigV4 cambiando solo los campos indicados.

    Usa `dataclasses.replace`-style sin importar el módulo, robusto frente
    a campos heredados de ConfigV3.
    """
    from dataclasses import replace as dc_replace

    return dc_replace(cfg, **overrides)
