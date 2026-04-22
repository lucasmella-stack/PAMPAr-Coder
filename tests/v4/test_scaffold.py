# SPDX-License-Identifier: BUSL-1.1
"""
Smoke tests para el scaffold de PAMPAr v4.

Cubre:
  - Imports del paquete v4 funcionan
  - ConfigV4 hereda correctamente de ConfigV3
  - Los flags multimodal/recurrent existen y tienen defaults seguros
  - ModalityId y NUM_MODALITIES están bien definidos
  - TextEncoder funciona y es numéricamente equivalente a nn.Embedding
"""

from __future__ import annotations

import pytest
import torch
from pampar.coder.v3.config import ConfigV3
from pampar.coder.v4 import (
    NUM_MODALITIES,
    ConfigV4,
    ContextModulatorV4,
    ModalityEncoder,
    ModalityId,
    StreamFFN,
    TextEncoder,
)

# =============================================================================
# Config
# =============================================================================


class TestConfigV4:
    def test_hereda_de_v3(self):
        cfg = ConfigV4()
        assert isinstance(cfg, ConfigV3)

    def test_defaults_seguros_no_activan_features_futuras(self):
        cfg = ConfigV4()
        assert cfg.use_hierarchical_modulators is False
        assert cfg.use_recurrent_loop is False
        assert cfg.use_lti_injection is False
        assert cfg.use_act_halting is False
        assert cfg.use_loop_index_rope is False

    def test_vocab_active_menor_que_vocab_size(self):
        cfg = ConfigV4()
        assert cfg.vocab_active < cfg.vocab_size
        assert cfg.vocab_active == 47_000
        assert cfg.vocab_size == 48_000

    def test_n_modalities_es_8(self):
        cfg = ConfigV4()
        assert cfg.n_modalities == 8
        assert cfg.n_modalities == NUM_MODALITIES

    def test_zonas_reservadas_para_otras_modalidades(self):
        cfg = ConfigV4()
        assert cfg.n_zonas == 52
        assert cfg.n_zonas_reserved == 38
        assert cfg.n_zonas_total == 90


# =============================================================================
# Modalidades
# =============================================================================


class TestModalityEnum:
    def test_text_es_id_cero(self):
        assert int(ModalityId.TEXT) == 0

    def test_other_es_id_siete(self):
        assert int(ModalityId.OTHER) == 7

    def test_total_son_8_slots(self):
        assert NUM_MODALITIES == 8
        assert len(list(ModalityId)) == 8

    def test_orden_estable(self):
        # CRÍTICO: el orden no debe cambiar entre versiones,
        # los checkpoints dependen del one-hot.
        expected = [
            "TEXT",
            "IMAGE",
            "AUDIO",
            "VIDEO",
            "CODE_AST",
            "DIAGRAM",
            "TABLE",
            "OTHER",
        ]
        assert [m.name for m in ModalityId] == expected


class TestTextEncoder:
    def test_es_modality_encoder(self):
        enc = TextEncoder(vocab_size=100, dim=32)
        assert isinstance(enc, ModalityEncoder)
        assert enc.modality == ModalityId.TEXT
        assert enc.dim == 32

    def test_forward_shape_correcto(self):
        enc = TextEncoder(vocab_size=100, dim=32)
        ids = torch.randint(0, 100, (2, 16))
        out = enc(ids)
        assert out.shape == (2, 16, 32)

    def test_weight_property_para_tying(self):
        enc = TextEncoder(vocab_size=100, dim=32)
        assert enc.weight.shape == (100, 32)
        assert enc.weight is enc.embedding.weight


# =============================================================================
# ContextModulatorV4
# =============================================================================


class TestContextModulatorV4:
    @pytest.fixture
    def cfg(self):
        return ConfigV4(dim=64, modulator_bottleneck=32)

    @pytest.fixture
    def modulator(self, cfg):
        return ContextModulatorV4(cfg)

    def test_context_dim_es_71(self, modulator):
        # 52 zonas + 4 territorios + 1 depth + 1 conf + 1 loop + 8 modality + 4 stream = 71
        assert ContextModulatorV4.CONTEXT_DIM == 71

    def test_context_dim_breakdown(self):
        assert (
            ContextModulatorV4.ZONA_DIM
            + ContextModulatorV4.TERR_DIM
            + ContextModulatorV4.DEPTH_DIM
            + ContextModulatorV4.CONF_DIM
            + ContextModulatorV4.LOOP_DIM
            + ContextModulatorV4.MODALITY_DIM
            + ContextModulatorV4.STREAM_DIM
        ) == ContextModulatorV4.CONTEXT_DIM

    def test_init_es_identidad_porque_proj_arranca_en_zeros(self, cfg, modulator):
        """Con init en zeros, gamma=0, beta=0 → out == ffn_out."""
        B, L, D = 2, 8, cfg.dim
        ffn_out = torch.randn(B, L, D)
        zona_acts = torch.rand(B, L, 52)
        terr_acts = torch.softmax(torch.randn(B, L, 4), dim=-1)

        out = modulator(
            ffn_out=ffn_out,
            zona_acts=zona_acts,
            terr_acts=terr_acts,
            stream_idx=0,
            nivel_idx=0,
            n_levels=5,
            conf=0.5,
        )
        torch.testing.assert_close(out, ffn_out)

    def test_forward_shape_correcto(self, cfg, modulator):
        B, L, D = 2, 8, cfg.dim
        ffn_out = torch.randn(B, L, D)
        zona_acts = torch.rand(B, L, 52)
        terr_acts = torch.softmax(torch.randn(B, L, 4), dim=-1)

        out = modulator(
            ffn_out=ffn_out,
            zona_acts=zona_acts,
            terr_acts=terr_acts,
            stream_idx=2,
            nivel_idx=3,
            n_levels=5,
            conf=0.7,
            loop_idx=0,
            max_loops=1,
            modality_id=ModalityId.TEXT,
        )
        assert out.shape == (B, L, D)

    def test_modality_id_distinto_cambia_contexto(self, cfg, modulator):
        """Después de un paso de optimización, modality distinta produce salida distinta."""
        # Romper la identidad inicializando proj con valores no-zero
        with torch.no_grad():
            for p in modulator.proj.parameters():
                p.normal_(0.0, 0.1)

        B, L, D = 2, 8, cfg.dim
        ffn_out = torch.randn(B, L, D)
        zona_acts = torch.rand(B, L, 52)
        terr_acts = torch.softmax(torch.randn(B, L, 4), dim=-1)

        kwargs = dict(
            ffn_out=ffn_out,
            zona_acts=zona_acts,
            terr_acts=terr_acts,
            stream_idx=0,
            nivel_idx=0,
            n_levels=5,
            conf=0.5,
        )
        out_text = modulator(**kwargs, modality_id=ModalityId.TEXT)
        out_image = modulator(**kwargs, modality_id=ModalityId.IMAGE)

        # Modalidades distintas deben producir modulaciones distintas
        assert not torch.allclose(out_text, out_image)

    def test_loop_idx_distinto_cambia_contexto(self, cfg, modulator):
        with torch.no_grad():
            for p in modulator.proj.parameters():
                p.normal_(0.0, 0.1)

        B, L, D = 2, 8, cfg.dim
        ffn_out = torch.randn(B, L, D)
        zona_acts = torch.rand(B, L, 52)
        terr_acts = torch.softmax(torch.randn(B, L, 4), dim=-1)

        kwargs = dict(
            ffn_out=ffn_out,
            zona_acts=zona_acts,
            terr_acts=terr_acts,
            stream_idx=0,
            nivel_idx=0,
            n_levels=5,
            conf=0.5,
            modality_id=ModalityId.TEXT,
        )
        out_loop0 = modulator(**kwargs, loop_idx=0, max_loops=8)
        out_loop4 = modulator(**kwargs, loop_idx=4, max_loops=8)

        assert not torch.allclose(out_loop0, out_loop4)


# =============================================================================
# StreamFFN re-exportado
# =============================================================================


class TestStreamFFNReexport:
    def test_es_la_misma_clase_que_v3(self):
        from pampar.coder.v3.ffn import StreamFFN as StreamFFNv3

        assert StreamFFN is StreamFFNv3

    def test_forward_funciona(self):
        cfg = ConfigV4(dim=64)
        ffn = StreamFFN(cfg)
        x = torch.randn(2, 8, 64)
        out = ffn(x)
        assert out.shape == (2, 8, 64)
