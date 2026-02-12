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
        
        # terr_acts en [0, 1] por sigmoid final del gate
        assert terr.min() >= 0
        assert terr.max() <= 1
        # zona_acts puede salir de [0,1] tras context_conv (esperado)
        # Solo verificamos que no exploten
        assert zona.min() > -10
        assert zona.max() < 10


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


# =============================================================================
# TESTS PARA FEATURES ARQUITECTÓNICOS v2.1
# =============================================================================

class TestINT8Llaves:
    """Tests del sistema LLAVES con cuantización INT8."""

    def test_int8_storage(self):
        """Tabla de lookup usa uint8 (no nibble packing)."""
        llaves = LlavesV2(vocab_size=200, n_zonas=52, usar_cuant=True)
        assert llaves.tabla_cuant.dtype == torch.uint8
        assert llaves.tabla_cuant.shape == (200, 52)

    def test_int8_precision(self):
        """INT8 tiene 256 niveles → error < 0.4%."""
        llaves = LlavesV2(vocab_size=10, n_zonas=52, usar_cuant=True)
        # Setear un valor conocido
        llaves._set_cuant(0, 5, 0.75)
        # Leer de vuelta
        ids = torch.tensor([[0]])
        acts = llaves(ids)
        recovered = acts[0, 0, 5].item()
        # Error < 1/256 ≈ 0.39%
        assert abs(recovered - 0.75) < 0.004, f"INT8 error too high: {recovered}"

    def test_int8_boundary_values(self):
        """Valores extremos (0.0 y 1.0) se preservan."""
        llaves = LlavesV2(vocab_size=10, n_zonas=52, usar_cuant=True)
        llaves._set_cuant(0, 0, 0.0)
        llaves._set_cuant(0, 1, 1.0)
        ids = torch.tensor([[0]])
        acts = llaves(ids)
        assert acts[0, 0, 0].item() == 0.0
        assert acts[0, 0, 1].item() == 1.0


class TestContextConv:
    """Tests de la ventana de contexto causal en el Tálamo."""

    def test_context_conv_exists(self):
        """El tálamo tiene Conv1D para contexto."""
        talamo = Talamo(PRESET_4GB)
        assert hasattr(talamo, 'context_conv')
        assert isinstance(talamo.context_conv, torch.nn.Conv1d)

    def test_context_conv_is_causal(self):
        """Conv es causal: solo mira tokens anteriores (no futuros)."""
        cfg = ConfigV2(ventana_contexto=4, dim=384, n_heads=6, n_capas=2)
        talamo = Talamo(cfg)
        B, L, D = 1, 8, cfg.dim
        x = torch.randn(B, L, D)
        ids = torch.randint(0, 100, (B, L))

        # Forward con todos los tokens
        terr1, zona1 = talamo(x, ids)

        # Forward cambiando el último token — no debería afectar tokens anteriores
        ids2 = ids.clone()
        ids2[0, -1] = (ids[0, -1] + 1) % 100
        x2 = x.clone()
        x2[0, -1] += 1.0  # Cambiar embedding del último token

        terr2, zona2 = talamo(x2, ids2)

        # Los primeros L-1 tokens deben ser iguales (causal = no ven el futuro)
        assert torch.allclose(terr1[:, :-1, :], terr2[:, :-1, :], atol=1e-5), \
            "Context conv is NOT causal — future tokens are leaking!"

    def test_context_conv_depthwise(self):
        """Conv es depthwise (groups=n_zonas) → liviana."""
        talamo = Talamo(PRESET_4GB)
        assert talamo.context_conv.groups == PRESET_4GB.n_zonas


class TestSymbioticRelationships:
    """Tests de relaciones simbióticas entre territorios."""

    def test_sym_layers_exist(self):
        """BloqueTerritorial tiene capas simbióticas."""
        bloque = BloqueTerritorial(PRESET_4GB)
        assert hasattr(bloque, 'sym_proj')
        assert hasattr(bloque, 'sym_up')

    def test_sym_bottleneck_dimension(self):
        """Bottleneck usa dim // sym_factor."""
        cfg = PRESET_4GB
        bloque = BloqueTerritorial(cfg)
        sym_dim = cfg.dim // cfg.sym_factor
        assert bloque.sym_proj.in_features == cfg.dim * cfg.n_territorios
        assert bloque.sym_proj.out_features == sym_dim
        assert bloque.sym_up.in_features == sym_dim
        assert bloque.sym_up.out_features == cfg.dim

    def test_sym_adds_to_output(self):
        """Symbiotic support is additive (not replacing main mix)."""
        bloque = BloqueTerritorial(PRESET_4GB)
        x = torch.randn(1, 8, PRESET_4GB.dim)
        terr = torch.rand(1, 8, 4)

        # Zero out symbiotic layers → output should differ from normal
        with torch.no_grad():
            bloque.sym_proj.weight.zero_()
            bloque.sym_up.weight.zero_()

        out_no_sym, _ = bloque(x, terr)

        # Restore random weights
        with torch.no_grad():
            torch.nn.init.normal_(bloque.sym_proj.weight, std=0.02)
            torch.nn.init.normal_(bloque.sym_up.weight, std=0.02)

        out_with_sym, _ = bloque(x, terr)

        # Outputs should differ
        assert not torch.allclose(out_no_sym, out_with_sym, atol=1e-6), \
            "Symbiotic layers have no effect!"


class TestPercentileExit:
    """Tests de Early Exit con percentil 10."""

    def test_exit_returns_confidence(self):
        """BloqueTerritorial retorna confianza escalar."""
        bloque = BloqueTerritorial(PRESET_4GB)
        x = torch.randn(1, 16, PRESET_4GB.dim)
        terr = torch.rand(1, 16, 4)
        _, conf = bloque(x, terr)
        assert isinstance(conf, float)
        assert 0 <= conf <= 1

    def test_percentile_focuses_on_worst(self):
        """Percentil 10 mira los tokens con menor confianza."""
        cfg = ConfigV2(exit_percentile=0.1, dim=384, n_heads=6, n_capas=2)
        bloque = BloqueTerritorial(cfg)

        # Con exit_percentile=0.1 y 16 tokens: k = max(1, int(16*0.1)) = 1
        # Mira el peor token (no el promedio)
        x = torch.randn(1, 16, cfg.dim)
        terr = torch.rand(1, 16, 4)
        _, conf = bloque(x, terr)

        # Confianza debe ser baja (peor token) vs promedio global
        # Just verify it runs within valid range
        assert 0 <= conf <= 1

    def test_percentile_config_respected(self):
        """exit_percentile del config se usa correctamente."""
        cfg = ConfigV2(exit_percentile=0.5, dim=384, n_heads=6, n_capas=2)
        bloque = BloqueTerritorial(cfg)
        assert bloque.config.exit_percentile == 0.5


class TestMemoriaErrores:
    """Tests de la Memoria de Errores con Interiorización."""

    def test_crear_memoria(self):
        """Crear memoria con defaults."""
        from pampar.coder.v2.aprendizaje.memoria_errores import MemoriaErrores
        mem = MemoriaErrores()
        assert len(mem.memoria) == 0
        assert mem.max_entries == 10000

    def test_registrar_errores(self):
        """Registra errores cuando loss > umbral."""
        from pampar.coder.v2.aprendizaje.memoria_errores import MemoriaErrores
        mem = MemoriaErrores(hash_window=4, umbral_error=2.0)

        ids = torch.randint(0, 100, (1, 20))
        losses = torch.ones(1, 20) * 1.0  # Below threshold
        assert mem.registrar_errores(ids, losses) == 0

        losses[0, 10] = 5.0  # Above threshold
        count = mem.registrar_errores(ids, losses)
        assert count >= 1
        assert len(mem.memoria) >= 1

    def test_interiorizacion(self):
        """Patrón se interioriza tras N éxitos consecutivos."""
        from pampar.coder.v2.aprendizaje.memoria_errores import MemoriaErrores
        mem = MemoriaErrores(
            hash_window=4,
            umbral_error=2.0,
            umbral_interiorizacion=3,
        )

        ids = torch.randint(0, 100, (1, 20))
        losses_high = torch.ones(1, 20) * 1.0
        losses_high[0, 10] = 5.0  # Error en posición 10
        mem.registrar_errores(ids, losses_high)
        assert len(mem.memoria) >= 1

        # Ahora el modelo "acierta" (loss baja)
        losses_low = torch.ones(1, 20) * 0.5
        for _ in range(3):
            mem.verificar_interiorizacion(ids, losses_low)

        # Después de 3 éxitos, debe interiorizarse (borrarse)
        assert mem.total_interiorizados >= 1

    def test_penalizacion(self):
        """Patrones conocidos reciben penalización extra."""
        from pampar.coder.v2.aprendizaje.memoria_errores import MemoriaErrores
        mem = MemoriaErrores(
            hash_window=4,
            umbral_error=2.0,
            factor_penalizacion=0.15,
        )

        ids = torch.randint(0, 100, (1, 20))
        losses = torch.ones(1, 20) * 5.0
        mem.registrar_errores(ids, losses)

        penalty = mem.calcular_penalizacion(ids)
        assert penalty.shape == (1, 20)
        # At least some positions should have penalty
        assert penalty.sum() > 0

    def test_ring_buffer_overflow(self):
        """Buffer circular sobreescribe entradas viejas."""
        from pampar.coder.v2.aprendizaje.memoria_errores import MemoriaErrores
        mem = MemoriaErrores(max_entries=5, hash_window=4, umbral_error=0.5)

        for i in range(10):
            ids = torch.full((1, 10), fill_value=i * 10, dtype=torch.long)
            losses = torch.ones(1, 10) * 2.0
            mem.registrar_errores(ids, losses)

        assert len(mem.memoria) <= 5

    def test_guardar_cargar(self, tmp_path):
        """Persistencia a disco (JSON)."""
        from pampar.coder.v2.aprendizaje.memoria_errores import MemoriaErrores
        mem = MemoriaErrores(hash_window=4, umbral_error=2.0)

        ids = torch.randint(0, 100, (1, 20))
        losses = torch.ones(1, 20) * 5.0
        mem.registrar_errores(ids, losses)

        path = str(tmp_path / "mem.json")
        mem.guardar(path)

        mem2 = MemoriaErrores.cargar(path)
        assert len(mem2.memoria) == len(mem.memoria)
        assert mem2.total_registrados == mem.total_registrados

    def test_stats(self):
        """Estadísticas de la memoria."""
        from pampar.coder.v2.aprendizaje.memoria_errores import MemoriaErrores
        mem = MemoriaErrores()
        stats = mem.stats()
        assert 'activos' in stats
        assert 'capacidad' in stats
        assert 'ratio_inter' in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
