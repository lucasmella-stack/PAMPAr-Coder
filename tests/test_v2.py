# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""Tests para PAMPAr-Coder v2."""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pampar.coder.v2.config import ConfigV2, PRESET_4GB
from pampar.coder.v2.zonas import Zona, Territorio, ZONA_TERRITORIO
from pampar.coder.v2.llaves import normalizar, clasificar_token, LlavesV2
from pampar.coder.v2.talamo import Talamo
from pampar.coder.v2.bloques import BloqueAttn, BloqueFFN, BloqueTerritorial
from pampar.coder.v2.modelo import PampaRCoderV2, crear_modelo


class TestConfig:
    """Tests de configuración."""
    
    def test_preset_4gb_valid(self):
        cfg = PRESET_4GB
        assert cfg.dim % cfg.n_heads == 0
        assert 0 <= cfg.peso_llaves <= 1
    
    def test_memory_estimate(self):
        cfg = PRESET_4GB
        mem = cfg.memory_estimate_mb()
        assert 50 < mem < 500  # Razonable para 4GB


class TestZonas:
    """Tests del sistema de zonas."""
    
    def test_52_zonas(self):
        assert len(Zona) == 52
    
    def test_4_territorios(self):
        assert len(Territorio) == 4
    
    def test_mapeo_completo(self):
        # Cada zona debe mapear a un territorio
        for z in Zona:
            assert z in ZONA_TERRITORIO
            assert ZONA_TERRITORIO[z] in Territorio
    
    def test_distribucion_zonas(self):
        # SINTAXIS: 15, SEMANTICA: 15, LOGICO: 12, ESTRUCTURAL: 10
        counts = {t: 0 for t in Territorio}
        for z in Zona:
            counts[ZONA_TERRITORIO[z]] += 1
        
        assert counts[Territorio.SINTAXIS] == 15
        assert counts[Territorio.SEMANTICA] == 15
        assert counts[Territorio.LOGICO] == 12
        assert counts[Territorio.ESTRUCTURAL] == 10


class TestLlaves:
    """Tests del sistema LLAVES."""
    
    @pytest.mark.parametrize("token,expected", [
        ("▁def", "def"),
        ("Ġclass", "class"),
        ("##for", "for"),
        ("print", "print"),
    ])
    def test_normalizar(self, token, expected):
        assert normalizar(token) == expected
    
    @pytest.mark.parametrize("token,expected_zona", [
        ("def", Zona.B01_KW_DEF),
        ("class", Zona.B02_KW_CLASS),
        ("if", Zona.B05_KW_CONTROL),
        ("for", Zona.B06_KW_LOOP),
        ("return", Zona.B04_KW_RETURN),
        ("True", Zona.B24_LIT_BOOL),
        ("None", Zona.B25_LIT_NONE),
        ("+", Zona.B31_OP_ARITH),
        ("==", Zona.B32_OP_COMP),
        ("and", Zona.B33_OP_LOGIC),
        ("(", Zona.B11_DELIM_PAREN),
        ("{", Zona.B13_DELIM_BRACE),
    ])
    def test_clasificar_keywords(self, token, expected_zona):
        zona, conf = clasificar_token(token)
        assert zona == expected_zona
        assert conf > 0.5
    
    def test_clasificar_numeros(self):
        zona, _ = clasificar_token("42")
        assert zona == Zona.B21_LIT_INT
        
        zona, _ = clasificar_token("3.14")
        assert zona == Zona.B22_LIT_FLOAT
    
    def test_clasificar_magic(self):
        zona, _ = clasificar_token("__init__")
        assert zona == Zona.B30_MAGIC
    
    def test_llaves_v2_shape(self):
        llaves = LlavesV2(vocab_size=100, n_zonas=52)
        ids = torch.randint(0, 100, (2, 10))
        acts = llaves(ids)
        assert acts.shape == (2, 10, 52)


class TestTalamo:
    """Tests del Tálamo."""
    
    @pytest.fixture
    def talamo(self):
        return Talamo(PRESET_4GB)
    
    def test_forward_shapes(self, talamo):
        B, L, D = 2, 16, PRESET_4GB.dim
        x = torch.randn(B, L, D)
        ids = torch.randint(0, 100, (B, L))
        
        terr, zona = talamo(x, ids)
        
        assert terr.shape == (B, L, 4)
        assert zona.shape == (B, L, 52)
    
    def test_activaciones_normalizadas(self, talamo):
        B, L, D = 1, 8, PRESET_4GB.dim
        x = torch.randn(B, L, D)
        ids = torch.randint(0, 100, (B, L))
        
        terr, zona = talamo(x, ids)
        
        # Activaciones deben estar en [0, 1]
        assert terr.min() >= 0
        assert terr.max() <= 1
        assert zona.min() >= 0
        assert zona.max() <= 1


class TestBloques:
    """Tests de bloques de procesamiento."""
    
    @pytest.fixture
    def config(self):
        return PRESET_4GB
    
    def test_attn_forward(self, config):
        attn = BloqueAttn(config)
        x = torch.randn(2, 16, config.dim)
        out = attn(x)
        assert out.shape == x.shape
    
    def test_ffn_forward(self, config):
        ffn = BloqueFFN(config)
        x = torch.randn(2, 16, config.dim)
        out = ffn(x)
        assert out.shape == x.shape
    
    def test_bloque_territorial_forward(self, config):
        bloque = BloqueTerritorial(config)
        x = torch.randn(2, 16, config.dim)
        terr = torch.rand(2, 16, 4)
        
        out, conf = bloque(x, terr)
        
        assert out.shape == x.shape
        assert 0 <= conf <= 1


class TestModelo:
    """Tests del modelo completo."""
    
    @pytest.fixture
    def modelo(self):
        return crear_modelo(PRESET_4GB)
    
    def test_forward_sin_targets(self, modelo):
        ids = torch.randint(0, 100, (2, 32))
        logits, loss, info = modelo(ids)
        
        assert logits.shape == (2, 32, PRESET_4GB.vocab_size)
        assert loss is None
    
    def test_forward_con_targets(self, modelo):
        ids = torch.randint(0, 100, (2, 32))
        targets = torch.randint(0, 100, (2, 32))
        
        logits, loss, info = modelo(ids, targets)
        
        assert loss is not None
        assert loss.item() > 0
    
    def test_generate(self, modelo):
        prompt = torch.randint(0, 100, (1, 8))
        generated = modelo.generate(prompt, max_tokens=16)
        
        assert generated.shape[0] == 1
        assert generated.shape[1] >= 8
    
    def test_count_params(self, modelo):
        params = modelo.count_params()
        
        assert "total" in params
        assert params["total"] > 0
        assert params["total"] == sum(
            p.numel() for p in modelo.parameters()
        )
    
    def test_early_exit(self, modelo):
        ids = torch.randint(0, 100, (1, 16))
        
        # Con early exit
        _, _, info = modelo(ids, use_early_exit=True)
        
        assert "exit_capa" in info
        assert 1 <= info["exit_capa"] <= PRESET_4GB.n_capas


class TestIntegracion:
    """Tests de integración end-to-end."""
    
    def test_pipeline_completo(self):
        # 1. Crear modelo
        modelo = crear_modelo(PRESET_4GB)
        
        # 2. Forward
        ids = torch.randint(0, 1000, (1, 64))
        targets = torch.randint(0, 1000, (1, 64))
        
        logits, loss, _ = modelo(ids, targets)
        
        # 3. Backward
        loss.backward()
        
        # 4. Verificar gradientes (excluir exit_head que solo se usa en inferencia)
        for name, p in modelo.named_parameters():
            if p.requires_grad and "exit_head" not in name:
                assert p.grad is not None, f"No grad for {name}"
    
    def test_memoria_4gb(self):
        """Verifica que el modelo cabe en 4GB."""
        modelo = crear_modelo(PRESET_4GB)
        params = modelo.count_params()
        
        # FP16: 2 bytes por parámetro
        mem_mb = params["total"] * 2 / (1024 ** 2)
        
        # Modelo debe usar < 500MB para dejar espacio para batch
        assert mem_mb < 500, f"Modelo usa {mem_mb:.1f}MB, muy grande"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
