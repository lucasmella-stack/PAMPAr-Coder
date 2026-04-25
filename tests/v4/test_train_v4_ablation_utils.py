"""Tests para los helpers puros de scripts/train_v4_ablation.py.

No tocan disco real ni GPU. Validan:
  - load_config + extends + deep merge
  - cosine_lr (warmup + cosine + bordes)
  - build_model_from_config (filtra fields invalidos)
  - JsonlTokenLoader (tokeniza + train/val split + batches)
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "train_v4_ablation.py"

# Cargamos el script como modulo (no es importable como package)
spec = importlib.util.spec_from_file_location("train_v4_ablation", SCRIPT)
assert spec and spec.loader
train_mod = importlib.util.module_from_spec(spec)
sys.modules["train_v4_ablation"] = train_mod
spec.loader.exec_module(train_mod)


class TestDeepMerge:
    def test_override_gana_en_escalares(self):
        merged = train_mod._deep_merge({"a": 1, "b": 2}, {"b": 99})
        assert merged == {"a": 1, "b": 99}

    def test_recursivo_en_dicts(self):
        merged = train_mod._deep_merge(
            {"model": {"dim": 256, "n_heads": 8}},
            {"model": {"dim": 512}},
        )
        assert merged == {"model": {"dim": 512, "n_heads": 8}}

    def test_no_muta_base(self):
        base = {"a": {"b": 1}}
        train_mod._deep_merge(base, {"a": {"c": 2}})
        assert base == {"a": {"b": 1}}


class TestLoadConfig:
    def test_carga_config_simple_sin_extends(self, tmp_path: Path):
        cfg_path = tmp_path / "c.yaml"
        cfg_path.write_text("model:\n  dim: 128\n", encoding="utf-8")
        out = train_mod.load_config(cfg_path)
        assert out == {"model": {"dim": 128}}

    def test_extends_resuelve_path_relativo(self, tmp_path: Path):
        (tmp_path / "_base.yaml").write_text(
            "model:\n  dim: 256\n  n_heads: 8\n", encoding="utf-8"
        )
        (tmp_path / "child.yaml").write_text(
            "extends: '_base.yaml'\nmodel:\n  dim: 512\n", encoding="utf-8"
        )
        out = train_mod.load_config(tmp_path / "child.yaml")
        assert out == {"model": {"dim": 512, "n_heads": 8}}

    def test_phase6_real_configs_resuelven(self):
        """Sanity sobre los YAML reales del repo."""
        for name in ["A_baseline", "A_hier", "B_full", "B_no_loop", "B_act"]:
            cfg = train_mod.load_config(
                ROOT / "configs" / "phase6_ablation" / f"{name}.yaml"
            )
            assert "model" in cfg and "training" in cfg
            assert cfg["model"]["n_streams"] == 4
            assert cfg["model"]["n_territorios"] == 4


class TestCosineLR:
    def test_warmup_lineal_desde_cero(self):
        # step=0 → ~base*1/warmup, step=warmup-1 → ~base
        assert train_mod.cosine_lr(
            0, warmup=10, total=100, base_lr=1.0
        ) == pytest.approx(0.1)
        assert train_mod.cosine_lr(
            9, warmup=10, total=100, base_lr=1.0
        ) == pytest.approx(1.0)

    def test_post_warmup_decae_a_cero_al_final(self):
        # Al final del entrenamiento debe estar cerca de 0
        end = train_mod.cosine_lr(99, warmup=10, total=100, base_lr=1.0)
        assert 0.0 <= end < 0.01

    def test_warmup_cero_no_divide_por_cero(self):
        # Edge case: warmup=0 no debe explotar
        val = train_mod.cosine_lr(0, warmup=0, total=10, base_lr=1.0)
        assert val == pytest.approx(1.0)


class TestBuildModelFromConfig:
    def test_filtra_fields_invalidos(self):
        cfg = {
            "dim": 64,
            "n_heads": 4,
            "n_kv_heads": 2,
            "n_streams": 4,
            "n_territorios": 4,
            "n_levels": 2,
            "n_zonas": 52,
            "vocab_size": 1000,
            "max_seq_len": 64,
            "field_que_no_existe": 999,  # debe ignorarse sin error
        }
        model, resolved = train_mod.build_model_from_config(cfg)
        assert resolved.dim == 64
        assert resolved.n_levels == 2
        assert isinstance(model, torch.nn.Module)


class TestJsonlTokenLoader:
    @pytest.fixture
    def fake_tokenizer(self):
        """Tokenizer que mapea cada palabra a su hash mod 100, EOS=2."""

        class FakeTokenizer:
            def Encode(self, text: str) -> list[int]:
                return [(hash(w) % 90) + 10 for w in text.split()]

            def PieceToId(self, piece: str) -> int:
                return 2 if piece == "</s>" else -1

        return FakeTokenizer()

    def test_loader_split_y_batch_basico(self, tmp_path: Path, fake_tokenizer):
        # Generamos un JSONL con suficiente texto para ≥4 chunks de seq_len=8
        jsonl = tmp_path / "data.jsonl"
        with jsonl.open("w", encoding="utf-8") as f:
            for i in range(50):
                f.write(
                    json.dumps({"text": " ".join(f"w{i}_{j}" for j in range(20))})
                    + "\n"
                )

        loader = train_mod.JsonlTokenLoader(
            jsonl_path=jsonl,
            tokenizer=fake_tokenizer,
            seq_len=8,
            batch_size=4,
            val_fraction=0.1,
            seed=42,
        )

        device = torch.device("cpu")
        batch = loader.next_train_batch(device)
        assert batch.shape == (4, 9)  # batch_size, seq_len+1
        assert batch.dtype == torch.int64

        val_batches = list(loader.iter_val_batches(device, max_batches=2))
        assert len(val_batches) >= 1
        assert val_batches[0].shape[1] == 9

    def test_loader_falla_si_dataset_muy_chico(self, tmp_path: Path, fake_tokenizer):
        jsonl = tmp_path / "tiny.jsonl"
        jsonl.write_text(json.dumps({"text": "hola"}) + "\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="Pocos chunks"):
            train_mod.JsonlTokenLoader(
                jsonl_path=jsonl,
                tokenizer=fake_tokenizer,
                seq_len=128,
                batch_size=4,
                val_fraction=0.1,
                seed=42,
            )

    def test_split_es_determinista_por_seed(self, tmp_path: Path, fake_tokenizer):
        jsonl = tmp_path / "data.jsonl"
        with jsonl.open("w", encoding="utf-8") as f:
            for i in range(50):
                f.write(
                    json.dumps({"text": " ".join(f"w{i}_{j}" for j in range(20))})
                    + "\n"
                )

        l1 = train_mod.JsonlTokenLoader(jsonl, fake_tokenizer, 8, 4, 0.1, seed=42)
        l2 = train_mod.JsonlTokenLoader(jsonl, fake_tokenizer, 8, 4, 0.1, seed=42)
        # Misma seed → mismas particiones
        np.testing.assert_array_equal(l1._train, l2._train)
        np.testing.assert_array_equal(l1._val, l2._val)
