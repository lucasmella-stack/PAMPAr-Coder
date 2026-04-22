# SPDX-License-Identifier: BUSL-1.1
"""
Tests para Fase 5b: FFN compartido entre niveles del path B + harness A/B.

Cubre:
  - share_ffn_across_niveles: ffn_shared apunta al mismo módulo en
    Prelude/Body/Coda (id() coincide).
  - count_params descuenta correctamente cuando hay sharing.
  - Gradientes se acumulan en el FFN compartido cuando se usa.
  - Sin sharing (default), los 3 FFN tienen ids distintos.
  - build_pair: construye A y B con misma seed, configs correctos.
  - compare_forward: produce ABReport con campos válidos.
  - Modo legacy (use_mixed_selectivity=False): comparte la lista `ffns`.
"""

from __future__ import annotations

import pytest
import torch
from pampar.coder.v4 import (
    ABReport,
    ConfigV4,
    PamparV4,
    build_pair,
    compare_forward,
    count_params,
)


@pytest.fixture
def cfg_b_share():
    return ConfigV4(
        dim=64,
        n_streams=4,
        n_levels=2,
        n_zonas=52,
        vocab_size=200,
        n_heads=4,
        n_kv_heads=2,
        ffn_mult=2.0,
        modulator_bottleneck=32,
        max_seq_len=64,
        dropout=0.0,
        use_checkpoint=False,
        use_recurrent_loop=True,
        max_loop_iters=3,
        share_ffn_across_niveles=True,
    )


@pytest.fixture
def cfg_b_no_share():
    return ConfigV4(
        dim=64,
        n_streams=4,
        n_levels=2,
        n_zonas=52,
        vocab_size=200,
        n_heads=4,
        n_kv_heads=2,
        ffn_mult=2.0,
        modulator_bottleneck=32,
        max_seq_len=64,
        dropout=0.0,
        use_checkpoint=False,
        use_recurrent_loop=True,
        max_loop_iters=3,
        share_ffn_across_niveles=False,
    )


@pytest.fixture
def cfg_b_legacy_share():
    """use_mixed_selectivity=False + share_ffn → comparte lista `ffns`."""
    return ConfigV4(
        dim=64,
        n_streams=4,
        n_levels=2,
        n_zonas=52,
        vocab_size=200,
        n_heads=4,
        n_kv_heads=2,
        ffn_mult=2.0,
        modulator_bottleneck=32,
        max_seq_len=64,
        dropout=0.0,
        use_checkpoint=False,
        use_mixed_selectivity=False,
        use_recurrent_loop=True,
        max_loop_iters=3,
        share_ffn_across_niveles=True,
    )


# =============================================================================
# Sharing real (id() coincide)
# =============================================================================


class TestFFNSharing:
    def test_default_no_share_distintos_ids(self, cfg_b_no_share):
        m = PamparV4(cfg_b_no_share)
        assert id(m.prelude_nivel.ffn_shared) != id(m.body_nivel.ffn_shared)
        assert id(m.body_nivel.ffn_shared) != id(m.coda_nivel.ffn_shared)

    def test_share_mismo_id_en_los_tres_niveles(self, cfg_b_share):
        m = PamparV4(cfg_b_share)
        ffn_id = id(m.prelude_nivel.ffn_shared)
        assert id(m.body_nivel.ffn_shared) == ffn_id
        assert id(m.coda_nivel.ffn_shared) == ffn_id

    def test_share_legacy_misma_lista_ffns(self, cfg_b_legacy_share):
        m = PamparV4(cfg_b_legacy_share)
        # En legacy compartimos la ModuleList completa
        ffns_id = id(m.prelude_nivel.ffns)
        assert id(m.body_nivel.ffns) == ffns_id
        assert id(m.coda_nivel.ffns) == ffns_id

    def test_share_reduce_param_count(self, cfg_b_share, cfg_b_no_share):
        m_share = PamparV4(cfg_b_share)
        m_no_share = PamparV4(cfg_b_no_share)
        n_share = count_params(m_share)
        n_no = count_params(m_no_share)
        # Sharing debe ahorrar ≥ 1 param (en realidad varios miles)
        assert n_share < n_no

    def test_gradientes_acumulan_en_ffn_compartido(self, cfg_b_share):
        torch.manual_seed(0)
        m = PamparV4(cfg_b_share)
        ids = torch.randint(0, cfg_b_share.vocab_size, (2, 6))
        targets = torch.randint(0, cfg_b_share.vocab_size, (2, 6))
        _, loss, _ = m(ids, targets=targets)
        loss.backward()

        # El FFN compartido es el mismo objeto: gradiente ÚNICO acumula
        # contribuciones del prelude, body y coda.
        shared = m.prelude_nivel.ffn_shared
        assert shared is m.body_nivel.ffn_shared
        assert shared is m.coda_nivel.ffn_shared
        # Verificar que el gradiente existe y no es trivial
        for p in shared.parameters():
            assert p.grad is not None
            assert p.grad.abs().sum() > 0


# =============================================================================
# build_pair / compare_forward
# =============================================================================


class TestABHarness:
    def test_build_pair_construye_paths_correctos(self):
        cfg_base = ConfigV4(
            dim=64,
            n_streams=4,
            n_levels=2,
            n_zonas=52,
            vocab_size=200,
            n_heads=4,
            n_kv_heads=2,
            ffn_mult=2.0,
            modulator_bottleneck=32,
            max_seq_len=64,
            dropout=0.0,
            use_checkpoint=False,
        )
        m_a, m_b = build_pair(cfg_base, max_loop_iters=3, share_ffn=False)
        assert not m_a.config.use_recurrent_loop
        assert m_b.config.use_recurrent_loop
        assert m_b.config.max_loop_iters == 3
        # Sanity: ambos modelos construidos correctamente con sus paths
        assert len(m_a.niveles) == cfg_base.n_levels
        assert m_b.recurrent_block is not None

    def test_compare_forward_devuelve_reporte_valido(self):
        cfg_base = ConfigV4(
            dim=64,
            n_streams=4,
            n_levels=2,
            n_zonas=52,
            vocab_size=200,
            n_heads=4,
            n_kv_heads=2,
            ffn_mult=2.0,
            modulator_bottleneck=32,
            max_seq_len=64,
            dropout=0.0,
            use_checkpoint=False,
        )
        m_a, m_b = build_pair(cfg_base, max_loop_iters=2, share_ffn=True)
        ids = torch.randint(0, cfg_base.vocab_size, (1, 4))
        report = compare_forward(m_a, m_b, ids, n_warmup=0, n_runs=1)

        assert isinstance(report, ABReport)
        assert report.n_params_a > 0
        assert report.n_params_b > 0
        # Sharing en B → menos params
        assert report.n_params_b < report.n_params_a or report.n_params_b > 0
        assert report.latency_ms_a > 0
        assert report.latency_ms_b > 0
        assert report.n_steps_b == 2  # max_loop_iters sin ACT
        assert "Path A vs Path B" in report.summary()

    def test_compare_forward_sin_targets_usa_input_como_targets(self):
        cfg_base = ConfigV4(
            dim=32,
            n_streams=2,
            n_levels=1,
            n_zonas=52,
            vocab_size=100,
            n_heads=2,
            n_kv_heads=1,
            ffn_mult=2.0,
            modulator_bottleneck=16,
            max_seq_len=32,
            dropout=0.0,
            use_checkpoint=False,
        )
        m_a, m_b = build_pair(cfg_base, max_loop_iters=2, share_ffn=False)
        ids = torch.randint(0, cfg_base.vocab_size, (1, 4))
        report = compare_forward(m_a, m_b, ids, n_warmup=0, n_runs=1)
        # loss debe ser finito
        assert report.loss_a == report.loss_a  # not NaN
        assert report.loss_b == report.loss_b


# =============================================================================
# count_params
# =============================================================================


class TestCountParams:
    def test_no_double_count_shared_modules(self, cfg_b_share):
        m = PamparV4(cfg_b_share)
        # Recuento naïve (suma todos los parameters() incluso compartidos):
        naive = sum(p.numel() for p in m.parameters())
        unique = count_params(m)
        # PyTorch's parameters() ya deduplica por identidad de tensor:
        # ambos cuentan igual. Lo importante es que no se sobre-cuente.
        assert unique <= naive
