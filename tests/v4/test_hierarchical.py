# SPDX-License-Identifier: BUSL-1.1
"""
Tests para HierarchicalModulator (Fase 2 de PAMPAr v4).

Cubre:
  - Init en zeros → identidad numérica
  - Backbone compartido entre (nivel, stream) — mismo tensor
  - Cabezas independientes — gradientes separados
  - Validación de bounds en (level_idx, stream_idx)
  - Equivalencia funcional con ContextModulatorV4 cuando ambos arrancan
    con identidad (ambos devuelven ffn_out exacto)
  - Compatibilidad multimodal/loop (mismo contrato que ContextModulatorV4)
"""

from __future__ import annotations

import pytest
import torch
from pampar.coder.v4 import (
    CONTEXT_DIM,
    ConfigV4,
    ContextModulatorV4,
    HierarchicalModulator,
    ModalityId,
    build_context_v4,
)


@pytest.fixture
def cfg():
    return ConfigV4()  # defaults v3 (dim=640, n_streams=4, n_levels=5)


@pytest.fixture
def small_cfg():
    """Config pequeña para tests rápidos."""
    return ConfigV4(
        dim=32,
        n_streams=4,
        n_levels=3,
        modulator_bottleneck=16,
    )


@pytest.fixture
def sample_inputs(small_cfg):
    """Inputs sintéticos compatibles con el modulator."""
    B, L = 2, 7
    return {
        "ffn_out": torch.randn(B, L, small_cfg.dim),
        "zona_acts": torch.randn(B, L, 52),
        "terr_acts": torch.randn(B, L, 4),
    }


class TestHierarchicalInit:
    def test_backbone_unico_compartido(self, cfg):
        mod = HierarchicalModulator(cfg)
        # backbone es un nn.Sequential con (Linear, SiLU) — un solo Linear
        linears = [m for m in mod.backbone.modules() if isinstance(m, torch.nn.Linear)]
        assert len(linears) == 1
        assert linears[0].in_features == CONTEXT_DIM
        assert linears[0].out_features == cfg.modulator_bottleneck

    def test_n_heads_es_levels_x_streams(self, cfg):
        mod = HierarchicalModulator(cfg)
        assert len(mod.heads) == cfg.n_levels * cfg.n_streams

    def test_cada_head_proyecta_a_dim_x_2(self, cfg):
        mod = HierarchicalModulator(cfg)
        for head in mod.heads:
            assert head.in_features == cfg.modulator_bottleneck
            assert head.out_features == cfg.dim * 2

    def test_heads_init_zeros_garantiza_identidad(self, cfg):
        mod = HierarchicalModulator(cfg)
        for head in mod.heads:
            assert torch.allclose(head.weight, torch.zeros_like(head.weight))


class TestHierarchicalForwardIdentity:
    def test_init_zeros_devuelve_ffn_out_exacto(self, small_cfg, sample_inputs):
        mod = HierarchicalModulator(small_cfg)
        out = mod(
            ffn_out=sample_inputs["ffn_out"],
            zona_acts=sample_inputs["zona_acts"],
            terr_acts=sample_inputs["terr_acts"],
            stream_idx=1,
            nivel_idx=0,
            n_levels=small_cfg.n_levels,
            conf=0.5,
        )
        # Con heads=0 → gamma=0, beta=0 → out = (1+0)*ffn_out + 0 = ffn_out
        assert torch.allclose(out, sample_inputs["ffn_out"])

    def test_shape_correcta(self, small_cfg, sample_inputs):
        mod = HierarchicalModulator(small_cfg)
        out = mod(
            ffn_out=sample_inputs["ffn_out"],
            zona_acts=sample_inputs["zona_acts"],
            terr_acts=sample_inputs["terr_acts"],
            stream_idx=2,
            nivel_idx=1,
            n_levels=small_cfg.n_levels,
            conf=0.7,
        )
        assert out.shape == sample_inputs["ffn_out"].shape


class TestHierarchicalHeadIndexing:
    def test_head_idx_correcto(self, small_cfg):
        mod = HierarchicalModulator(small_cfg)
        # n_streams=4, n_levels=3
        assert mod._head_idx(0, 0) == 0
        assert mod._head_idx(0, 3) == 3
        assert mod._head_idx(1, 0) == 4
        assert mod._head_idx(2, 3) == 11

    def test_level_idx_fuera_de_rango_falla(self, small_cfg):
        mod = HierarchicalModulator(small_cfg)
        with pytest.raises(ValueError, match="level_idx"):
            mod._head_idx(99, 0)
        with pytest.raises(ValueError, match="level_idx"):
            mod._head_idx(-1, 0)

    def test_stream_idx_fuera_de_rango_falla(self, small_cfg):
        mod = HierarchicalModulator(small_cfg)
        with pytest.raises(ValueError, match="stream_idx"):
            mod._head_idx(0, 99)
        with pytest.raises(ValueError, match="stream_idx"):
            mod._head_idx(0, -1)


class TestHierarchicalGradientes:
    def test_gradientes_separados_por_head(self, small_cfg, sample_inputs):
        """Activar la cabeza (1,2) NO debe propagar gradientes a la cabeza (0,3)."""
        mod = HierarchicalModulator(small_cfg)
        # Romper la identidad inicial perturbando una head
        with torch.no_grad():
            mod.heads[mod._head_idx(1, 2)].weight.normal_(std=0.01)

        out = mod(
            ffn_out=sample_inputs["ffn_out"],
            zona_acts=sample_inputs["zona_acts"],
            terr_acts=sample_inputs["terr_acts"],
            stream_idx=2,
            nivel_idx=1,
            n_levels=small_cfg.n_levels,
            conf=0.5,
        )
        loss = out.sum()
        loss.backward()

        head_used = mod.heads[mod._head_idx(1, 2)]
        head_unused = mod.heads[mod._head_idx(0, 3)]

        assert head_used.weight.grad is not None
        assert head_used.weight.grad.abs().sum() > 0

        # La cabeza no usada no debe tener gradiente (o debe ser todo zeros)
        assert (
            head_unused.weight.grad is None or head_unused.weight.grad.abs().sum() == 0
        )

    def test_backbone_acumula_gradientes_de_multiples_heads(
        self, small_cfg, sample_inputs
    ):
        """El backbone es compartido → un forward por cada (level, stream) que
        usemos debe acumular gradientes en el backbone."""
        mod = HierarchicalModulator(small_cfg)
        with torch.no_grad():
            for h in mod.heads:
                h.weight.normal_(std=0.01)

        backbone_linear = [
            m for m in mod.backbone.modules() if isinstance(m, torch.nn.Linear)
        ][0]

        # Llamar con dos pares (level, stream) distintos
        out1 = mod(
            ffn_out=sample_inputs["ffn_out"],
            zona_acts=sample_inputs["zona_acts"],
            terr_acts=sample_inputs["terr_acts"],
            stream_idx=0,
            nivel_idx=0,
            n_levels=small_cfg.n_levels,
            conf=0.5,
        )
        out2 = mod(
            ffn_out=sample_inputs["ffn_out"],
            zona_acts=sample_inputs["zona_acts"],
            terr_acts=sample_inputs["terr_acts"],
            stream_idx=3,
            nivel_idx=2,
            n_levels=small_cfg.n_levels,
            conf=0.5,
        )
        loss = (out1 + out2).sum()
        loss.backward()

        assert backbone_linear.weight.grad is not None
        assert backbone_linear.weight.grad.abs().sum() > 0


class TestHierarchicalEquivalenciaConV4Plano:
    def test_init_identidad_da_misma_salida_que_modulator_plano(
        self, small_cfg, sample_inputs
    ):
        """Recién inicializados ambos devuelven ffn_out → output idéntico."""
        plain = ContextModulatorV4(small_cfg)
        hier = HierarchicalModulator(small_cfg)

        out_plain = plain(
            ffn_out=sample_inputs["ffn_out"],
            zona_acts=sample_inputs["zona_acts"],
            terr_acts=sample_inputs["terr_acts"],
            stream_idx=1,
            nivel_idx=0,
            n_levels=small_cfg.n_levels,
            conf=0.5,
        )
        out_hier = hier(
            ffn_out=sample_inputs["ffn_out"],
            zona_acts=sample_inputs["zona_acts"],
            terr_acts=sample_inputs["terr_acts"],
            stream_idx=1,
            nivel_idx=0,
            n_levels=small_cfg.n_levels,
            conf=0.5,
        )
        assert torch.allclose(out_plain, out_hier)


class TestHierarchicalMultimodal:
    def test_acepta_modality_id_no_text(self, small_cfg, sample_inputs):
        mod = HierarchicalModulator(small_cfg)
        out = mod(
            ffn_out=sample_inputs["ffn_out"],
            zona_acts=sample_inputs["zona_acts"],
            terr_acts=sample_inputs["terr_acts"],
            stream_idx=0,
            nivel_idx=0,
            n_levels=small_cfg.n_levels,
            conf=0.5,
            modality_id=ModalityId.IMAGE,
        )
        # Identidad inicial: out == ffn_out
        assert torch.allclose(out, sample_inputs["ffn_out"])

    def test_acepta_loop_idx(self, small_cfg, sample_inputs):
        mod = HierarchicalModulator(small_cfg)
        out = mod(
            ffn_out=sample_inputs["ffn_out"],
            zona_acts=sample_inputs["zona_acts"],
            terr_acts=sample_inputs["terr_acts"],
            stream_idx=0,
            nivel_idx=0,
            n_levels=small_cfg.n_levels,
            conf=0.5,
            loop_idx=3,
            max_loops=8,
        )
        assert torch.allclose(out, sample_inputs["ffn_out"])


class TestBuildContextFreeFunction:
    def test_dimension_71(self, sample_inputs):
        ctx = build_context_v4(
            zona_acts=sample_inputs["zona_acts"],
            terr_acts=sample_inputs["terr_acts"],
            stream_idx=0,
            nivel_idx=0,
            n_levels=3,
            conf=0.5,
        )
        assert ctx.shape[-1] == CONTEXT_DIM == 71

    def test_modality_one_hot_correcto(self, sample_inputs):
        ctx = build_context_v4(
            zona_acts=sample_inputs["zona_acts"],
            terr_acts=sample_inputs["terr_acts"],
            stream_idx=0,
            nivel_idx=0,
            n_levels=3,
            conf=0.5,
            modality_id=ModalityId.AUDIO,  # =2
        )
        # Slot de modalidad: dims [52+4+1+1+1 = 59 .. 59+8 = 67)
        modality_slice = ctx[:, :, 59:67]
        # Debe estar todo en cero excepto el índice 2
        expected = torch.zeros(8)
        expected[int(ModalityId.AUDIO)] = 1.0
        # Promediamos sobre B y L (todos los tokens deben tener el mismo one-hot)
        assert torch.allclose(modality_slice[0, 0], expected)

    def test_stream_one_hot_correcto(self, sample_inputs):
        ctx = build_context_v4(
            zona_acts=sample_inputs["zona_acts"],
            terr_acts=sample_inputs["terr_acts"],
            stream_idx=2,
            nivel_idx=0,
            n_levels=3,
            conf=0.5,
        )
        # Slot de stream: últimas 4 dims
        stream_slice = ctx[:, :, -4:]
        expected = torch.zeros(4)
        expected[2] = 1.0
        assert torch.allclose(stream_slice[0, 0], expected)
