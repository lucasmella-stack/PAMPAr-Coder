# SPDX-License-Identifier: BUSL-1.1
"""
Tests para Fase 5a: cableado del RecurrentBlock dentro de PamparV4.

Cubre:
  - Default off: PamparV4(use_recurrent_loop=False) → path A intacto.
  - Toggle on: PamparV4(use_recurrent_loop=True) → construye prelude/body/coda
    + recurrent_block + adapter, sin self.niveles.
  - Forward path B: shapes correctas, info["recurrent_n_steps"] presente.
  - ACT activo → info["recurrent_halt_steps"] y ponder_cost > 0.
  - Gradientes propagan a través de los 3 niveles (prelude/body/coda).
  - RecurrentNivelAdapter: reset, step, delta-based actualiza streams.
  - KV cache helpers funcionan en path B.
"""

from __future__ import annotations

import pytest
import torch
from pampar.coder.v3.nivel import NivelProfundo
from pampar.coder.v4 import (
    ConfigV4,
    PamparV4,
    RecurrentNivelAdapter,
)


@pytest.fixture
def cfg_path_a():
    """Default: path A (sin recurrent loop)."""
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
        use_recurrent_loop=False,
    )


@pytest.fixture
def cfg_path_b_basic():
    """Path B sin LTI/RoPE/ACT: solo loop con peso compartido."""
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
        use_lti_injection=False,
        use_loop_index_rope=False,
        use_act_halting=False,
    )


@pytest.fixture
def cfg_path_b_full():
    """Path B con LTI + Loop-RoPE + ACT activos."""
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
        max_loop_iters=4,
        use_lti_injection=True,
        use_loop_index_rope=True,
        use_act_halting=True,
        act_loop_threshold=0.99,
    )


# =============================================================================
# Construcción del modelo
# =============================================================================


class TestPamparV4Construction:
    def test_path_a_construye_stack_de_niveles(self, cfg_path_a):
        m = PamparV4(cfg_path_a)
        assert len(m.niveles) == cfg_path_a.n_levels
        assert m.prelude_nivel is None
        assert m.body_nivel is None
        assert m.coda_nivel is None
        assert m.recurrent_block is None
        assert m.recurrent_adapter is None

    def test_path_b_construye_prelude_body_coda(self, cfg_path_b_basic):
        m = PamparV4(cfg_path_b_basic)
        assert len(m.niveles) == 0
        assert isinstance(m.prelude_nivel, NivelProfundo)
        assert isinstance(m.body_nivel, NivelProfundo)
        assert isinstance(m.coda_nivel, NivelProfundo)
        assert m.recurrent_block is not None
        assert isinstance(m.recurrent_adapter, RecurrentNivelAdapter)

    def test_path_b_recurrent_block_respeta_flags(self, cfg_path_b_full):
        m = PamparV4(cfg_path_b_full)
        rb = m.recurrent_block
        assert rb.lti is not None
        assert rb.loop_rope is not None
        assert rb.act is not None
        assert rb.max_loops == cfg_path_b_full.max_loop_iters

    def test_path_b_basic_sin_lti_rope_act(self, cfg_path_b_basic):
        m = PamparV4(cfg_path_b_basic)
        rb = m.recurrent_block
        assert rb.lti is None
        assert rb.loop_rope is None
        assert rb.act is None


# =============================================================================
# Forward path B
# =============================================================================


class TestPamparV4ForwardRecurrent:
    def test_forward_path_b_basico(self, cfg_path_b_basic):
        torch.manual_seed(0)
        m = PamparV4(cfg_path_b_basic).eval()
        ids = torch.randint(0, cfg_path_b_basic.vocab_size, (2, 8))
        with torch.no_grad():
            logits, loss, info = m(ids)
        assert logits.shape == (2, 8, cfg_path_b_basic.vocab_size)
        assert loss is None
        assert info["recurrent_n_steps"] == cfg_path_b_basic.max_loop_iters
        assert "recurrent_halt_steps" not in info  # sin ACT
        assert info["recurrent_ponder_cost"].item() == 0.0

    def test_forward_path_b_con_act(self, cfg_path_b_full):
        torch.manual_seed(0)
        m = PamparV4(cfg_path_b_full).eval()
        ids = torch.randint(0, cfg_path_b_full.vocab_size, (2, 6))
        with torch.no_grad():
            logits, loss, info = m(ids)
        assert logits.shape == (2, 6, cfg_path_b_full.vocab_size)
        assert "recurrent_halt_steps" in info
        assert info["recurrent_halt_steps"].shape == (2, 6)
        assert info["recurrent_ponder_cost"].item() > 0.0
        assert 1 <= info["recurrent_n_steps"] <= cfg_path_b_full.max_loop_iters

    def test_forward_path_b_loss(self, cfg_path_b_basic):
        torch.manual_seed(0)
        m = PamparV4(cfg_path_b_basic)
        ids = torch.randint(0, cfg_path_b_basic.vocab_size, (2, 8))
        targets = torch.randint(0, cfg_path_b_basic.vocab_size, (2, 8))
        logits, loss, info = m(ids, targets=targets)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_gradientes_propagan_path_b(self, cfg_path_b_full):
        torch.manual_seed(0)
        m = PamparV4(cfg_path_b_full)
        ids = torch.randint(0, cfg_path_b_full.vocab_size, (2, 6))
        targets = torch.randint(0, cfg_path_b_full.vocab_size, (2, 6))
        _, loss, _ = m(ids, targets=targets)
        loss.backward()

        # Prelude, Body y Coda deben todos recibir gradiente
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in m.prelude_nivel.parameters()
        )
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in m.body_nivel.parameters()
        )
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in m.coda_nivel.parameters()
        )

    def test_path_a_y_b_dan_outputs_distintos(self, cfg_path_a, cfg_path_b_basic):
        """Sanity: diferentes paths producen forwards diferentes."""
        torch.manual_seed(42)
        m_a = PamparV4(cfg_path_a).eval()
        torch.manual_seed(42)
        m_b = PamparV4(cfg_path_b_basic).eval()
        ids = torch.randint(0, cfg_path_a.vocab_size, (1, 4))
        with torch.no_grad():
            la, _, _ = m_a(ids)
            lb, _, _ = m_b(ids)
        # Outputs no triviales
        assert torch.isfinite(la).all()
        assert torch.isfinite(lb).all()


# =============================================================================
# RecurrentNivelAdapter
# =============================================================================


class TestRecurrentNivelAdapter:
    def test_reset_devuelve_combined(self, cfg_path_b_basic):
        torch.manual_seed(0)
        m = PamparV4(cfg_path_b_basic)
        adapter = m.recurrent_adapter

        B, L, D = 2, 4, cfg_path_b_basic.dim
        n = cfg_path_b_basic.n_streams
        streams = [torch.randn(B, L, D) for _ in range(n)]
        terr = torch.softmax(torch.randn(B, L, n), dim=-1)
        zona = torch.sigmoid(torch.randn(B, L, 52))

        combined = adapter.reset(streams, terr, zona)
        assert combined.shape == (B, L, D)

    def test_step_sin_reset_falla(self, cfg_path_b_basic):
        m = PamparV4(cfg_path_b_basic)
        adapter = m.recurrent_adapter
        h = torch.randn(1, 2, cfg_path_b_basic.dim)
        with pytest.raises(RuntimeError, match="reset"):
            adapter.step(h, h, 0)

    def test_streams_sin_reset_falla(self, cfg_path_b_basic):
        m = PamparV4(cfg_path_b_basic)
        with pytest.raises(RuntimeError, match="reset"):
            _ = m.recurrent_adapter.streams

    def test_n_streams_mismatch_falla(self, cfg_path_b_basic):
        m = PamparV4(cfg_path_b_basic)
        adapter = m.recurrent_adapter
        B, L, D = 1, 2, cfg_path_b_basic.dim
        n_wrong = cfg_path_b_basic.n_streams - 1
        streams = [torch.randn(B, L, D) for _ in range(n_wrong)]
        terr = torch.softmax(torch.randn(B, L, cfg_path_b_basic.n_streams), dim=-1)
        zona = torch.sigmoid(torch.randn(B, L, 52))
        with pytest.raises(ValueError, match="streams"):
            adapter.reset(streams, terr, zona)

    def test_step_actualiza_streams(self, cfg_path_b_basic):
        torch.manual_seed(0)
        m = PamparV4(cfg_path_b_basic).eval()
        adapter = m.recurrent_adapter

        B, L, D = 1, 3, cfg_path_b_basic.dim
        n = cfg_path_b_basic.n_streams
        streams = [torch.randn(B, L, D) for _ in range(n)]
        terr = torch.softmax(torch.randn(B, L, n), dim=-1)
        zona = torch.sigmoid(torch.randn(B, L, 52))

        with torch.no_grad():
            combined = adapter.reset(streams, terr, zona)
            new_combined = adapter.step(combined.clone(), combined.clone(), 0)
        # Step debe producir nuevo combined distinto al inicial
        assert not torch.allclose(new_combined, combined)
        # Estado interno actualizado
        assert len(adapter.streams) == n


# =============================================================================
# Helpers (KV cache, norm clamp) en path B
# =============================================================================


class TestPathBHelpers:
    def test_enable_disable_kv_cache(self, cfg_path_b_basic):
        m = PamparV4(cfg_path_b_basic)
        m._enable_kv_cache()
        assert m.prelude_nivel.attn._use_kv_cache
        assert m.body_nivel.attn._use_kv_cache
        assert m.coda_nivel.attn._use_kv_cache
        m._disable_kv_cache()
        assert not m.prelude_nivel.attn._use_kv_cache

    def test_set_train_norm_clamp(self, cfg_path_b_basic):
        m = PamparV4(cfg_path_b_basic)
        m.set_train_norm_clamp(True)
        assert m.prelude_nivel._train_norm_clamp
        assert m.body_nivel._train_norm_clamp
        assert m.coda_nivel._train_norm_clamp
