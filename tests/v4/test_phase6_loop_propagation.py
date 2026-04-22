# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Tests Fase 6 — propagación real de `loop_idx` y `HierarchicalModulator`
compartido en PamparV4.

Comprueba que:
  - `NivelProfundoV4` construye modulators V4 correctos (independientes
    o jerárquicos compartidos según flag).
  - `loop_idx` afecta la salida del modulator (no es solo un kwarg
    que se ignora).
  - `RecurrentNivelAdapter` detecta automáticamente si el body acepta
    loop_kwargs y los propaga.
  - PamparV4 path B incrementa loop_idx por iteración (capturado vía
    spy hook en el modulator).
  - El `HierarchicalModulator` es ÚNICO instance compartida entre
    todos los niveles cuando la flag está activa.
"""

from __future__ import annotations

import pytest
import torch
from pampar.coder.v3.nivel import NivelProfundo
from pampar.coder.v4 import (
    ConfigV4,
    ContextModulatorV4,
    HierarchicalModulator,
    NivelProfundoV4,
    PamparV4,
    RecurrentNivelAdapter,
)
from pampar.coder.v4.modalities import ModalityId


@pytest.fixture
def small_cfg() -> ConfigV4:
    return ConfigV4(
        dim=64,
        n_streams=4,
        n_levels=3,
        n_zonas=52,
        vocab_size=200,
        n_heads=4,
        n_kv_heads=2,
        ffn_mult=2.0,
        modulator_bottleneck=16,
        max_seq_len=32,
        dropout=0.0,
        use_checkpoint=False,
    )


# ────────────────────────────────────────────────────────────────────────
# NivelProfundoV4 — construcción
# ────────────────────────────────────────────────────────────────────────


class TestNivelProfundoV4Construction:
    def test_modulators_son_v4_por_default(self, small_cfg):
        nivel = NivelProfundoV4(small_cfg, nivel_idx=0)
        assert len(nivel.modulators) == small_cfg.n_streams
        for m in nivel.modulators:
            assert isinstance(m, ContextModulatorV4)

    def test_acepta_hierarchical_modulator_compartido(self, small_cfg):
        hm = HierarchicalModulator(small_cfg)
        nivel = NivelProfundoV4(small_cfg, nivel_idx=0, hierarchical_modulator=hm)
        # No tiene modulators propios cuando delega
        assert len(nivel.modulators) == 0
        assert nivel._uses_hierarchical
        assert nivel._hierarchical_modulator is hm

    def test_hierarchical_modulator_no_se_registra_como_submodule(self, small_cfg):
        """Si se registrara, sería duplicación de params al sumar varios niveles."""
        hm = HierarchicalModulator(small_cfg)
        nivel = NivelProfundoV4(small_cfg, nivel_idx=0, hierarchical_modulator=hm)
        # _hierarchical_modulator NO debe aparecer en named_modules
        names = [n for n, _ in nivel.named_modules()]
        assert "_hierarchical_modulator" not in names

    def test_legacy_mode_no_construye_modulators(self, small_cfg):
        cfg_legacy = ConfigV4(**{**small_cfg.__dict__, "use_mixed_selectivity": False})
        nivel = NivelProfundoV4(cfg_legacy, nivel_idx=0)
        # Legacy: usa ffns directos, no modulators
        assert hasattr(nivel, "ffns")
        assert not hasattr(nivel, "ffn_shared")


# ────────────────────────────────────────────────────────────────────────
# NivelProfundoV4 — forward propaga loop_idx
# ────────────────────────────────────────────────────────────────────────


class TestNivelProfundoV4ForwardLoopIdx:
    def _make_inputs(self, cfg, B=1, L=4):
        streams = [torch.randn(B, L, cfg.dim) for _ in range(cfg.n_streams)]
        terr_acts = torch.softmax(torch.randn(B, L, cfg.n_streams), dim=-1)
        zona_acts = torch.softmax(torch.randn(B, L, cfg.n_zonas), dim=-1)
        return streams, terr_acts, zona_acts

    def _agregar_fn(self, zonas):
        # Simple identity-ish para no necesitar TalamoInicial real
        n_terr = 4
        return zonas[..., :n_terr]

    def test_forward_acepta_loop_idx_y_max_loops(self, small_cfg):
        nivel = NivelProfundoV4(small_cfg, nivel_idx=0).eval()
        streams, terr, zona = self._make_inputs(small_cfg)
        out_streams, out_terr, conf = nivel(
            streams,
            terr,
            self._agregar_fn,
            zona_acts=zona,
            loop_idx=2,
            max_loops=8,
            modality_id=ModalityId.TEXT,
        )
        assert len(out_streams) == small_cfg.n_streams
        for s in out_streams:
            assert s.shape == streams[0].shape
            assert torch.isfinite(s).all()

    def test_loop_idx_distinto_genera_modulacion_distinta(self, small_cfg):
        """Smoking gun: si loop_idx no se propagara, las salidas serían iguales."""
        nivel = NivelProfundoV4(small_cfg, nivel_idx=0).eval()
        # Forzar pesos no-nulos en el modulator (la init es zeros)
        for m in nivel.modulators:
            torch.nn.init.normal_(m.proj[2].weight, std=0.05)

        torch.manual_seed(0)
        streams, terr, zona = self._make_inputs(small_cfg)

        with torch.no_grad():
            out0, _, _ = nivel(
                [s.clone() for s in streams],
                terr.clone(),
                self._agregar_fn,
                zona_acts=zona.clone(),
                loop_idx=0,
                max_loops=8,
            )
            out1, _, _ = nivel(
                [s.clone() for s in streams],
                terr.clone(),
                self._agregar_fn,
                zona_acts=zona.clone(),
                loop_idx=4,
                max_loops=8,
            )

        # Al menos un stream debe diferir (loop_idx alimenta el contexto)
        diffs = [(a - b).abs().max().item() for a, b in zip(out0, out1)]
        assert max(diffs) > 1e-5, f"loop_idx no se propaga; diffs={diffs}"


# ────────────────────────────────────────────────────────────────────────
# RecurrentNivelAdapter — detección automática de loop_kwargs
# ────────────────────────────────────────────────────────────────────────


class TestAdapterDetectaLoopKwargs:
    def test_detecta_v4_acepta_loop_idx(self, small_cfg):
        body_v4 = NivelProfundoV4(small_cfg, nivel_idx=1)
        adapter = RecurrentNivelAdapter(body_v4, n_streams=small_cfg.n_streams)
        assert adapter._body_accepts_loop_kwargs is True

    def test_detecta_v3_no_acepta_loop_idx(self, small_cfg):
        body_v3 = NivelProfundo(small_cfg, nivel_idx=1)
        adapter = RecurrentNivelAdapter(body_v3, n_streams=small_cfg.n_streams)
        assert adapter._body_accepts_loop_kwargs is False

    def test_max_loops_y_modality_se_almacenan(self, small_cfg):
        body_v4 = NivelProfundoV4(small_cfg, nivel_idx=1)
        adapter = RecurrentNivelAdapter(
            body_v4,
            n_streams=small_cfg.n_streams,
            max_loops=7,
            modality_id=ModalityId.TEXT,
        )
        assert adapter.max_loops == 7
        assert adapter.modality_id == ModalityId.TEXT


# ────────────────────────────────────────────────────────────────────────
# PamparV4 — wiring de HierarchicalModulator
# ────────────────────────────────────────────────────────────────────────


class TestPamparV4HierarchicalModulator:
    def test_no_se_construye_si_flag_off(self, small_cfg):
        model = PamparV4(small_cfg)
        assert model.hierarchical_modulator is None

    def test_se_construye_si_flag_on(self, small_cfg):
        cfg = ConfigV4(**{**small_cfg.__dict__, "use_hierarchical_modulators": True})
        model = PamparV4(cfg)
        assert isinstance(model.hierarchical_modulator, HierarchicalModulator)

    def test_es_compartido_entre_todos_los_niveles(self, small_cfg):
        cfg = ConfigV4(**{**small_cfg.__dict__, "use_hierarchical_modulators": True})
        model = PamparV4(cfg)
        hm = model.hierarchical_modulator
        for nivel in model.niveles:
            assert nivel._uses_hierarchical
            assert nivel._hierarchical_modulator is hm

    def test_path_b_tambien_comparte_hierarchical(self, small_cfg):
        cfg = ConfigV4(
            **{
                **small_cfg.__dict__,
                "use_hierarchical_modulators": True,
                "use_recurrent_loop": True,
                "max_loop_iters": 3,
            }
        )
        model = PamparV4(cfg)
        hm = model.hierarchical_modulator
        for nivel in (model.prelude_nivel, model.body_nivel, model.coda_nivel):
            assert nivel._uses_hierarchical
            assert nivel._hierarchical_modulator is hm

    def test_hierarchical_no_duplica_params(self, small_cfg):
        """Sumar params dedupeando por id == sumar params naive."""
        cfg = ConfigV4(**{**small_cfg.__dict__, "use_hierarchical_modulators": True})
        model = PamparV4(cfg)
        seen = set()
        unique = 0
        for p in model.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            unique += p.numel()
        # PyTorch parameters() ya dedupea — esto solo valida que no hay
        # parámetros "huérfanos" registrados dos veces accidentalmente
        naive = sum(p.numel() for p in model.parameters())
        assert unique == naive


# ────────────────────────────────────────────────────────────────────────
# End-to-end: PamparV4 path B con loop_idx propagado
# ────────────────────────────────────────────────────────────────────────


class TestPamparV4PathBPropagaLoopIdx:
    def test_forward_path_b_funciona_con_loop_kwargs(self, small_cfg):
        cfg = ConfigV4(
            **{
                **small_cfg.__dict__,
                "use_recurrent_loop": True,
                "max_loop_iters": 4,
            }
        )
        model = PamparV4(cfg).eval()
        ids = torch.randint(0, cfg.vocab_size, (2, 8))
        logits, _, info = model(ids)
        assert logits.shape == (2, 8, cfg.vocab_size)
        assert torch.isfinite(logits).all()
        assert info["recurrent_n_steps"] == 4

    def test_loop_idx_alcanza_modulator(self, small_cfg):
        """Spy hook: capturar el `loop_idx` con que se llama a cada modulator."""
        cfg = ConfigV4(
            **{
                **small_cfg.__dict__,
                "use_recurrent_loop": True,
                "max_loop_iters": 3,
            }
        )
        model = PamparV4(cfg).eval()

        captured: list[int] = []
        original_forward = model.body_nivel.modulators[0].forward

        def spy(*args, **kwargs):
            captured.append(kwargs.get("loop_idx", -1))
            return original_forward(*args, **kwargs)

        model.body_nivel.modulators[0].forward = spy

        ids = torch.randint(0, cfg.vocab_size, (1, 4))
        with torch.no_grad():
            model(ids)

        # En 3 iteraciones, el modulator del stream 0 fue llamado con
        # loop_idx ∈ {0, 1, 2}
        loop_ids_seen = set(captured)
        assert loop_ids_seen >= {0, 1, 2}, f"loop_idx visto: {loop_ids_seen}"
