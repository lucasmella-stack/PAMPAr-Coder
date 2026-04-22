# SPDX-License-Identifier: BUSL-1.1
"""
Tests para Fase 3: ModalityRouter + PamparV4.

Cubre:
  - ModalityRouter: registro, validación dim, error en duplicados/missing
  - encode_text() devuelve embeds correctos + modality_ids = TEXT
  - PamparV4: forward shape, loss, info dict (incluye modality_ids)
  - Weight tying entre TextEncoder y lm_head
  - Equivalencia numérica con PamparV3 cuando se inicializa con misma seed
"""

from __future__ import annotations

import pytest
import torch
from pampar.coder.v3.config import PRESET_V3, ConfigV3
from pampar.coder.v3.modelo import PamparV3
from pampar.coder.v4 import (
    ConfigV4,
    ModalityEncoder,
    ModalityId,
    ModalityRouter,
    PamparV4,
    TextEncoder,
)


@pytest.fixture
def small_cfg_v4():
    """Config v4 chica para tests rápidos en CPU."""
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
    )


# =============================================================================
# ModalityRouter
# =============================================================================


class TestModalityRouterRegistro:
    def test_registra_text_encoder(self):
        router = ModalityRouter(dim=32)
        router.register(TextEncoder(vocab_size=100, dim=32))
        assert router.has(ModalityId.TEXT)
        assert isinstance(router.get(ModalityId.TEXT), TextEncoder)

    def test_dim_mismatch_falla(self):
        router = ModalityRouter(dim=32)
        with pytest.raises(ValueError, match="dim"):
            router.register(TextEncoder(vocab_size=100, dim=64))

    def test_duplicado_falla(self):
        router = ModalityRouter(dim=32)
        router.register(TextEncoder(vocab_size=100, dim=32))
        with pytest.raises(ValueError, match="ya tiene encoder"):
            router.register(TextEncoder(vocab_size=200, dim=32))

    def test_get_modalidad_no_registrada_falla(self):
        router = ModalityRouter(dim=32)
        with pytest.raises(KeyError, match="IMAGE"):
            router.get(ModalityId.IMAGE)

    def test_text_encoder_property_falla_si_no_registrado(self):
        router = ModalityRouter(dim=32)
        with pytest.raises(KeyError):
            _ = router.text_encoder


class TestRouterEncodeText:
    def test_devuelve_embeds_y_modality_ids(self):
        router = ModalityRouter(dim=16)
        router.register(TextEncoder(vocab_size=50, dim=16))
        ids = torch.randint(0, 50, (2, 7))
        embeds, modality_ids = router.encode_text(ids)
        assert embeds.shape == (2, 7, 16)
        assert modality_ids.shape == (2, 7)
        assert (modality_ids == int(ModalityId.TEXT)).all()
        assert modality_ids.dtype == torch.long

    def test_embeds_iguales_a_nn_embedding_directa(self):
        torch.manual_seed(0)
        router = ModalityRouter(dim=16)
        router.register(TextEncoder(vocab_size=50, dim=16))
        text_enc = router.text_encoder

        torch.manual_seed(0)
        ref_emb = torch.nn.Embedding(50, 16)
        # Sincronizar pesos para comparar exacto
        with torch.no_grad():
            ref_emb.weight.copy_(text_enc.weight)

        ids = torch.randint(0, 50, (2, 5))
        embeds, _ = router.encode_text(ids)
        assert torch.allclose(embeds, ref_emb(ids))

    def test_modality_ids_en_mismo_device_que_input(self):
        router = ModalityRouter(dim=8)
        router.register(TextEncoder(vocab_size=20, dim=8))
        ids = torch.randint(0, 20, (1, 4))
        _, modality_ids = router.encode_text(ids)
        assert modality_ids.device == ids.device


# =============================================================================
# PamparV4
# =============================================================================


class TestPamparV4Construccion:
    def test_se_construye_con_config_v4(self, small_cfg_v4):
        model = PamparV4(small_cfg_v4)
        assert model.config is small_cfg_v4
        assert isinstance(model.modality_router, ModalityRouter)
        assert model.modality_router.has(ModalityId.TEXT)

    def test_weight_tying_lm_head_con_text_encoder(self, small_cfg_v4):
        model = PamparV4(small_cfg_v4)
        # El peso debe ser el mismo objeto (no solo iguales)
        assert (
            model.lm_head.weight.data_ptr()
            == model.modality_router.text_encoder.weight.data_ptr()
        )

    def test_tok_emb_property_devuelve_embedding(self, small_cfg_v4):
        """Compat con código v3 que accede a model.tok_emb."""
        model = PamparV4(small_cfg_v4)
        assert isinstance(model.tok_emb, torch.nn.Embedding)
        assert model.tok_emb.num_embeddings == small_cfg_v4.vocab_size
        assert model.tok_emb.embedding_dim == small_cfg_v4.dim


class TestPamparV4Forward:
    def test_forward_shape(self, small_cfg_v4):
        model = PamparV4(small_cfg_v4).eval()
        ids = torch.randint(0, small_cfg_v4.vocab_size, (2, 8))
        logits, loss, info = model(ids)
        assert logits.shape == (2, 8, small_cfg_v4.vocab_size)
        assert loss is None

    def test_forward_con_targets_devuelve_loss(self, small_cfg_v4):
        model = PamparV4(small_cfg_v4).eval()
        ids = torch.randint(0, small_cfg_v4.vocab_size, (2, 8))
        targets = torch.randint(0, small_cfg_v4.vocab_size, (2, 8))
        logits, loss, info = model(ids, targets=targets)
        assert loss is not None
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_info_incluye_modality_ids(self, small_cfg_v4):
        model = PamparV4(small_cfg_v4).eval()
        ids = torch.randint(0, small_cfg_v4.vocab_size, (2, 8))
        _, _, info = model(ids)
        assert "modality_ids" in info
        assert info["modality_ids"].shape == (2, 8)
        assert (info["modality_ids"] == int(ModalityId.TEXT)).all()

    def test_info_incluye_terr_acts_y_exit_nivel(self, small_cfg_v4):
        model = PamparV4(small_cfg_v4).eval()
        ids = torch.randint(0, small_cfg_v4.vocab_size, (1, 4))
        _, _, info = model(ids)
        assert "terr_acts" in info
        assert "exit_nivel" in info
        assert info["exit_nivel"] == small_cfg_v4.n_levels


class TestPamparV4EquivalenciaConV3:
    """
    Equivalencia numérica V3 ↔ V4.

    Importante (post-Fase 6):
        Con `use_mixed_selectivity=True` (default), V4 usa
        `ContextModulatorV4` con contexto de 71 dims (incluyendo
        modality_id one-hot que SIEMPRE activa TEXT=1.0). Por lo tanto,
        V4 NO es bit-equivalent a V3 en Mixed Selectivity, por diseño.

        En `use_mixed_selectivity=False` (legacy), no hay modulators,
        y V4 ES numéricamente equivalente a V3.
    """

    def _make_cfg_v3(self, cfg_v4: ConfigV4) -> ConfigV3:
        """Crea una ConfigV3 con los mismos campos relevantes."""
        return ConfigV3(
            dim=cfg_v4.dim,
            n_streams=cfg_v4.n_streams,
            n_levels=cfg_v4.n_levels,
            n_zonas=cfg_v4.n_zonas,
            vocab_size=cfg_v4.vocab_size,
            n_heads=cfg_v4.n_heads,
            n_kv_heads=cfg_v4.n_kv_heads,
            ffn_mult=cfg_v4.ffn_mult,
            modulator_bottleneck=cfg_v4.modulator_bottleneck,
            max_seq_len=cfg_v4.max_seq_len,
            dropout=cfg_v4.dropout,
            use_checkpoint=cfg_v4.use_checkpoint,
            use_mixed_selectivity=cfg_v4.use_mixed_selectivity,
        )

    @staticmethod
    def _sync_v3_to_v4(model_v3: PamparV3, model_v4: PamparV4) -> None:
        """Copia state_dict de v3 a v4 con renombre tok_emb→text_encoder."""
        sd_v3 = model_v3.state_dict()
        sd_v4 = model_v4.state_dict()
        new_sd: dict = {}
        for k, v in sd_v4.items():
            if k.startswith("modality_router.text_encoder.embedding."):
                v3_key = (
                    "tok_emb." + k.split("modality_router.text_encoder.embedding.")[1]
                )
                new_sd[k] = sd_v3.get(v3_key, v)
            else:
                new_sd[k] = sd_v3.get(k, v)
        model_v4.load_state_dict(new_sd, strict=True)

    def test_legacy_mode_es_equivalente_a_v3(self, small_cfg_v4):
        """En modo legacy (sin modulators), V4 == V3 bit-perfect."""
        cfg_v4_legacy = ConfigV4(
            **{**small_cfg_v4.__dict__, "use_mixed_selectivity": False}
        )
        cfg_v3 = self._make_cfg_v3(cfg_v4_legacy)

        torch.manual_seed(42)
        model_v3 = PamparV3(cfg_v3).eval()
        torch.manual_seed(42)
        model_v4 = PamparV4(cfg_v4_legacy).eval()

        self._sync_v3_to_v4(model_v3, model_v4)

        ids = torch.randint(0, cfg_v4_legacy.vocab_size, (1, 5))
        with torch.no_grad():
            logits_v3, _, _ = model_v3(ids)
            logits_v4, _, _ = model_v4(ids)

        assert torch.allclose(logits_v3, logits_v4, atol=1e-5)

    def test_mixed_mode_diverge_de_v3_pero_es_valido(self, small_cfg_v4):
        """En Mixed Selectivity, V4 diverge intencionalmente de V3."""
        cfg_v3 = self._make_cfg_v3(small_cfg_v4)

        torch.manual_seed(42)
        model_v3 = PamparV3(cfg_v3).eval()
        torch.manual_seed(42)
        model_v4 = PamparV4(small_cfg_v4).eval()

        ids = torch.randint(0, small_cfg_v4.vocab_size, (1, 5))
        with torch.no_grad():
            logits_v3, _, _ = model_v3(ids)
            logits_v4, _, _ = model_v4(ids)

        # Misma forma, ambos finitos, pero NO idénticos.
        assert logits_v3.shape == logits_v4.shape
        assert torch.isfinite(logits_v3).all() and torch.isfinite(logits_v4).all()
        assert not torch.allclose(logits_v3, logits_v4, atol=1e-3)


class TestPamparV4Backward:
    def test_loss_propaga_gradiente_a_text_encoder(self, small_cfg_v4):
        model = PamparV4(small_cfg_v4).train()
        ids = torch.randint(0, small_cfg_v4.vocab_size, (2, 8))
        targets = torch.randint(0, small_cfg_v4.vocab_size, (2, 8))
        _, loss, _ = model(ids, targets=targets)
        loss.backward()
        text_enc = model.modality_router.text_encoder
        assert text_enc.embedding.weight.grad is not None
        assert text_enc.embedding.weight.grad.abs().sum() > 0
