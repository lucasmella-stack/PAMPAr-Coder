# SPDX-License-Identifier: BUSL-1.1
"""
Tests para Fase 4: componentes del recurrent loop.

Cubre:
  - LTIInjection: ρ(A)<1 garantizado, init B=0 (decay puro), shape, forward.
  - LoopIndexEmbedding: tabla sinusoidal correcta, init proj=0 (identidad),
    rangos válidos.
  - ACTHalting: reset/update/finalize, early-exit si todos halted, ponder
    cost, output normalizado.
  - RecurrentBlock: orquesta loop con step_fn fake, all-flags-on/off,
    early exit ACT, gradientes propagan.
"""

from __future__ import annotations

import pytest
import torch
from pampar.coder.v4 import (
    ACTHalting,
    LoopIndexEmbedding,
    LTIInjection,
    RecurrentBlock,
)

# =============================================================================
# LTIInjection
# =============================================================================


class TestLTIInjectionStability:
    def test_spectral_radius_menor_a_uno(self):
        m = LTIInjection(dim=32)
        assert m.spectral_radius < 1.0

    def test_a_diagonal_en_rango_valido(self):
        m = LTIInjection(dim=64)
        a = m.A
        assert a.shape == (64,)
        assert (a > 0).all()
        assert (a < 1).all()

    def test_a_estable_aun_con_pesos_aleatorios_grandes(self):
        """Aun si el optimizer empuja log_A y log_dt a valores grandes,
        A debe seguir en [0,1] (los bordes pueden aparecer por underflow
        float32, pero nunca explota)."""
        m = LTIInjection(dim=16)
        with torch.no_grad():
            m.log_A.normal_(std=10.0)
            m.log_dt.normal_(std=10.0)
        a = m.A
        assert torch.isfinite(a).all()
        assert (a >= 0).all()
        assert (a <= 1).all()


class TestLTIInjectionForward:
    def test_init_b_zeros_aplica_decay_puro(self):
        m = LTIInjection(dim=8)
        h = torch.randn(2, 4, 8)
        e = torch.randn(2, 4, 8)
        h_next = m(h, e)
        # Con B=0: h_next = A * h
        expected = m.A * h
        assert torch.allclose(h_next, expected)

    def test_shape_correcta(self):
        m = LTIInjection(dim=16)
        h = torch.randn(3, 7, 16)
        e = torch.randn(3, 7, 16)
        out = m(h, e)
        assert out.shape == (3, 7, 16)

    def test_shape_mismatch_falla(self):
        m = LTIInjection(dim=8)
        h = torch.randn(2, 4, 8)
        e = torch.randn(2, 5, 8)  # L distinto
        with pytest.raises(ValueError, match="Shapes incompatibles"):
            m(h, e)

    def test_dim_incorrecta_falla(self):
        m = LTIInjection(dim=8)
        h = torch.randn(2, 4, 16)
        e = torch.randn(2, 4, 16)
        with pytest.raises(ValueError, match="Última dim"):
            m(h, e)

    def test_loop_no_explota(self):
        """Iterar muchas veces con B no-cero debe converger, no explotar."""
        m = LTIInjection(dim=8)
        with torch.no_grad():
            m.B.weight.normal_(std=0.1)
        h = torch.randn(1, 1, 8)
        e = torch.randn(1, 1, 8)
        for _ in range(100):
            h = m(h, e)
        assert torch.isfinite(h).all()
        assert h.abs().max() < 100.0  # holgado, solo descarta divergencia


# =============================================================================
# LoopIndexEmbedding
# =============================================================================


class TestLoopIndexEmbedding:
    def test_dim_impar_falla(self):
        with pytest.raises(ValueError, match="par"):
            LoopIndexEmbedding(dim=7)

    def test_tabla_shape(self):
        m = LoopIndexEmbedding(dim=16, max_loops=8)
        assert m.pe.shape == (8, 16)

    def test_init_proj_zeros_identidad(self):
        m = LoopIndexEmbedding(dim=16, max_loops=8, project=True)
        h = torch.randn(2, 4, 16)
        out = m(h, loop_idx=3)
        # Con proj=0: out = h + 0 = h
        assert torch.allclose(out, h)

    def test_sin_proj_suma_directa(self):
        m = LoopIndexEmbedding(dim=16, max_loops=8, project=False)
        h = torch.zeros(1, 1, 16)
        out = m(h, loop_idx=2)
        # Con h=0 y sin proj: out = pe[2] broadcast
        expected = m.pe[2].unsqueeze(0).unsqueeze(0)
        assert torch.allclose(out, expected)

    def test_loop_idx_fuera_de_rango_falla(self):
        m = LoopIndexEmbedding(dim=8, max_loops=4)
        h = torch.randn(1, 1, 8)
        with pytest.raises(ValueError, match="fuera de rango"):
            m(h, loop_idx=5)
        with pytest.raises(ValueError, match="fuera de rango"):
            m(h, loop_idx=-1)

    def test_distintos_loop_idx_dan_distintos_outputs(self):
        m = LoopIndexEmbedding(dim=16, max_loops=8, project=False)
        h = torch.randn(1, 1, 16)
        out0 = m(h.clone(), loop_idx=0)
        out3 = m(h.clone(), loop_idx=3)
        assert not torch.allclose(out0, out3)


# =============================================================================
# ACTHalting
# =============================================================================


class TestACTHalting:
    def test_reset_inicializa_estado(self):
        act = ACTHalting(dim=8)
        act.reset(B=2, L=3, device=torch.device("cpu"), dtype=torch.float32)
        assert act._cum_p.shape == (2, 3)
        assert act._cum_out.shape == (2, 3, 8)
        assert (act._halt_step == -1).all()
        assert not act._halted.any()

    def test_update_sin_reset_falla(self):
        act = ACTHalting(dim=8)
        h = torch.randn(2, 3, 8)
        with pytest.raises(RuntimeError, match="reset"):
            act.update(h, step_idx=0)

    def test_finalize_sin_update_falla(self):
        act = ACTHalting(dim=8)
        with pytest.raises(RuntimeError, match="reset"):
            act.finalize()

    def test_loop_completo_acumula_y_normaliza(self):
        torch.manual_seed(0)
        act = ACTHalting(dim=4, threshold=0.99)
        act.reset(2, 3, torch.device("cpu"), torch.float32)
        for t in range(5):
            h = torch.randn(2, 3, 4)
            act.update(h, step_idx=t)
        out = act.finalize()
        assert out.output.shape == (2, 3, 4)
        assert torch.isfinite(out.output).all()
        assert out.n_steps_used == 5
        assert out.halt_steps.shape == (2, 3)

    def test_early_exit_cuando_todos_halted(self):
        """Forzando bias positivo grande, sigmoid≈1 → todos halt en step 1."""
        act = ACTHalting(dim=4, threshold=0.99)
        with torch.no_grad():
            act.halt_head.bias.fill_(10.0)  # sigmoid(10) ≈ 1
        act.reset(2, 3, torch.device("cpu"), torch.float32)

        h = torch.randn(2, 3, 4)
        # En el paso 0, p≈1 → cum_p=1 → todos halt
        done = act.update(h, step_idx=0)
        assert done is True

    def test_loop_no_halt_continua(self):
        """Bias muy negativo → sigmoid≈0 → nadie halt."""
        act = ACTHalting(dim=4, threshold=0.99)
        with torch.no_grad():
            act.halt_head.bias.fill_(-20.0)  # sigmoid(-20) ≈ 0
        act.reset(2, 3, torch.device("cpu"), torch.float32)

        h = torch.randn(2, 3, 4)
        done = act.update(h, step_idx=0)
        assert done is False
        assert not act._halted.any()

    def test_threshold_invalido_falla(self):
        with pytest.raises(ValueError, match="threshold"):
            ACTHalting(dim=4, threshold=1.5)
        with pytest.raises(ValueError, match="threshold"):
            ACTHalting(dim=4, threshold=0.0)


# =============================================================================
# RecurrentBlock
# =============================================================================


class TestRecurrentBlock:
    def test_construye_con_todos_los_flags(self):
        block = RecurrentBlock(
            dim=16, max_loops=4, use_lti=True, use_loop_rope=True, use_act=True
        )
        assert block.lti is not None
        assert block.loop_rope is not None
        assert block.act is not None

    def test_construye_sin_flags(self):
        block = RecurrentBlock(
            dim=16, max_loops=4, use_lti=False, use_loop_rope=False, use_act=False
        )
        assert block.lti is None
        assert block.loop_rope is None
        assert block.act is None

    def test_max_loops_invalido_falla(self):
        with pytest.raises(ValueError, match="max_loops"):
            RecurrentBlock(dim=16, max_loops=0)

    def test_e_shape_invalido_falla(self):
        block = RecurrentBlock(dim=16, max_loops=4, use_act=False)
        e = torch.randn(2, 4, 8)  # dim wrong
        with pytest.raises(ValueError, match="e debe ser"):
            block(e, step_fn=lambda h, e, t: h)

    def test_loop_sin_act_corre_max_loops(self):
        block = RecurrentBlock(
            dim=16, max_loops=5, use_lti=False, use_loop_rope=False, use_act=False
        )
        e = torch.randn(2, 4, 16)
        calls = []

        def step_fn(h, e_in, t):
            calls.append(t)
            return h + 0.1

        out = block(e, step_fn=step_fn)
        assert calls == [0, 1, 2, 3, 4]
        assert out.n_steps == 5
        assert out.h_final.shape == (2, 4, 16)
        assert out.halt_steps is None
        assert out.ponder_cost.item() == 0.0

    def test_loop_con_act_early_exit(self):
        """Con bias del halt_head muy positivo, debe salir en step 0."""
        block = RecurrentBlock(
            dim=16, max_loops=10, use_lti=False, use_loop_rope=False, use_act=True
        )
        with torch.no_grad():
            block.act.halt_head.bias.fill_(10.0)

        e = torch.randn(2, 4, 16)
        calls = []

        def step_fn(h, e_in, t):
            calls.append(t)
            return h

        out = block(e, step_fn=step_fn)
        # Sale después del step 0
        assert len(calls) == 1
        assert out.n_steps == 1
        assert out.halt_steps is not None
        assert (out.halt_steps == 0).all()

    def test_step_fn_recibe_loop_idx_correcto(self):
        block = RecurrentBlock(
            dim=8, max_loops=3, use_lti=False, use_loop_rope=False, use_act=False
        )
        e = torch.randn(1, 1, 8)
        loop_indices = []

        def step_fn(h, e_in, t):
            loop_indices.append(t)
            return h

        block(e, step_fn=step_fn)
        assert loop_indices == [0, 1, 2]

    def test_h_init_se_usa(self):
        block = RecurrentBlock(
            dim=8, max_loops=1, use_lti=False, use_loop_rope=False, use_act=False
        )
        e = torch.zeros(1, 1, 8)
        h0 = torch.full((1, 1, 8), 7.0)

        def step_fn(h, e_in, t):
            return h

        out = block(e, step_fn=step_fn, h_init=h0)
        assert torch.allclose(out.h_final, h0)

    def test_gradientes_propagan_a_step_fn(self):
        block = RecurrentBlock(
            dim=8, max_loops=3, use_lti=True, use_loop_rope=True, use_act=False
        )
        e = torch.randn(1, 2, 8, requires_grad=True)
        param = torch.nn.Parameter(torch.randn(8))

        def step_fn(h, e_in, t):
            return h + param

        out = block(e, step_fn=step_fn)
        out.h_final.sum().backward()
        assert param.grad is not None
        assert param.grad.abs().sum() > 0
        assert e.grad is not None

    def test_lti_se_aplica_antes_de_step(self):
        """Si use_lti=True con B=0 (init), step recibe h*A, no h."""
        block = RecurrentBlock(
            dim=4, max_loops=1, use_lti=True, use_loop_rope=False, use_act=False
        )
        # B ya es zeros por init de LTIInjection
        a = block.lti.A.detach().clone()  # [4]
        e = torch.full((1, 1, 4), 2.0)

        seen_h = []

        def step_fn(h, e_in, t):
            seen_h.append(h.clone())
            return h

        block(e, step_fn=step_fn, h_init=e.clone())
        # h_init = e = 2.0, después de LTI con B=0: h = A * 2
        expected = a * 2.0
        assert torch.allclose(seen_h[0][0, 0], expected)
