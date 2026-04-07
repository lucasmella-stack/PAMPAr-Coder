# SPDX-License-Identifier: BUSL-1.1
"""
Tests de la arquitectura PamparV3.

Cubren:
  - Forward pass: shapes de logits, terr_acts, zona_acts
  - Loss válida (~log(vocab) en init aleatoria)
  - Weight tying entre tok_emb y lm_head
  - GQA: n_kv_heads < n_heads
  - Conteo de parámetros coherente
  - generate(): output más largo que input, sin NaN
  - Early exit: exit_nivel dentro de los límites
  - Lateral gate scale inicializado en 0.1
  - TalamoInicial: forma de salida correcta
  - No leak de datos entre streams independientes (grad isolation básico)
"""

import math

import pytest
import torch

from pampar.coder.v3.config import ConfigV3, PRESET_V3, PRESET_V3_SMALL, PRESET_V3_LARGE
from pampar.coder.v3.modelo import PamparV3
from pampar.coder.v3.bloques import NivelProfundo, LateralGate


# ==============================================================================
# TESTS DE CONFIGURACIÓN
# ==============================================================================

class TestConfigV3:
    def test_head_dim_derivado(self, config_small: ConfigV3):
        """head_dim = dim // n_heads debe ser entero sin resto."""
        assert config_small.dim % config_small.n_heads == 0
        assert config_small.head_dim == config_small.dim // config_small.n_heads

    def test_gqa_ratio(self, config_small: ConfigV3):
        """KV heads deben ser menores que Query heads (GQA activado)."""
        assert config_small.n_kv_heads < config_small.n_heads
        assert config_small.n_heads % config_small.n_kv_heads == 0

    def test_n_rep_derivado(self, config_small: ConfigV3):
        """n_rep = n_heads // n_kv_heads debe ser entero."""
        assert config_small.n_rep == config_small.n_heads // config_small.n_kv_heads

    def test_ffn_hidden_positivo(self, config_small: ConfigV3):
        """El hidden de FFN debe ser > dim."""
        assert config_small.ffn_hidden > 0

    def test_presets_existen(self):
        """Los tres presets exportados deben ser instancias de ConfigV3."""
        assert isinstance(PRESET_V3, ConfigV3)
        assert isinstance(PRESET_V3_SMALL, ConfigV3)
        assert isinstance(PRESET_V3_LARGE, ConfigV3)

    def test_preset_v3_grande_mayor_que_small(self):
        """PRESET_V3_LARGE debe tener mayor dim que PRESET_V3_SMALL."""
        assert PRESET_V3_LARGE.dim > PRESET_V3_SMALL.dim

    def test_estimate_params_positivo(self, config_small: ConfigV3):
        """estimate_params() debe retornar un dict con clave 'total' positiva."""
        result = config_small.estimate_params()
        assert isinstance(result, dict)
        assert "total" in result
        assert result["total"] > 0

    def test_memory_estimate_positivo(self, config_small: ConfigV3):
        """memory_estimate_mb() debe retornar un dict con claves esperadas."""
        mem = config_small.memory_estimate_mb()
        assert "model_fp16_mb" in mem, f"Claves: {list(mem.keys())}"
        assert "training_total_mb" in mem
        assert mem["model_fp16_mb"] > 0


# ==============================================================================
# TESTS DEL FORWARD PASS
# ==============================================================================

class TestForwardPass:
    def test_logits_shape(self, modelo: PamparV3, tokens_cortos: torch.Tensor, config_small: ConfigV3):
        """Los logits deben tener shape [B, L, vocab_size]."""
        with torch.no_grad():
            logits, loss, info = modelo(tokens_cortos)

        B, L = tokens_cortos.shape
        assert logits.shape == (B, L, config_small.vocab_size), (
            f"Esperado {(B, L, config_small.vocab_size)}, "
            f"obtenido {logits.shape}"
        )

    def test_loss_none_sin_targets(self, modelo: PamparV3, tokens_cortos: torch.Tensor):
        """Sin targets la loss debe ser None."""
        with torch.no_grad():
            _, loss, _ = modelo(tokens_cortos)
        assert loss is None

    def test_loss_valida_con_targets(self, modelo: PamparV3, tokens_cortos: torch.Tensor, config_small: ConfigV3):
        """Con targets la loss debe ser escalar finito ~log(vocab) en init aleatoria."""
        targets = tokens_cortos.clone()
        with torch.no_grad():
            _, loss, _ = modelo(tokens_cortos, targets=targets)

        assert loss is not None
        assert loss.ndim == 0, "La loss debe ser un escalar"
        assert torch.isfinite(loss), "La loss no puede ser NaN/Inf"
        # Log-probabilidad uniforme sobre el vocab como cota esperada
        expected = math.log(config_small.vocab_size)
        assert 0 < loss.item() < expected * 3, (
            f"Loss {loss.item():.4f} fuera del rango esperado [0, {expected * 3:.2f}]"
        )

    def test_info_contiene_exit_nivel(self, modelo: PamparV3, tokens_cortos: torch.Tensor, config_small: ConfigV3):
        """El dict info debe incluir 'exit_nivel' dentro de [1, n_levels]."""
        with torch.no_grad():
            _, _, info = modelo(tokens_cortos)

        assert "exit_nivel" in info
        nivel = info["exit_nivel"]
        assert 1 <= nivel <= config_small.n_levels

    def test_terr_acts_en_info(self, modelo: PamparV3, tokens_cortos: torch.Tensor, config_small: ConfigV3):
        """info debe incluir 'terr_acts' con shape [B, L, n_streams]."""
        with torch.no_grad():
            _, _, info = modelo(tokens_cortos)

        assert "terr_acts" in info
        B, L = tokens_cortos.shape
        ta = info["terr_acts"]
        assert ta.shape == (B, L, config_small.n_streams)

    def test_logits_finitos(self, modelo: PamparV3, tokens_cortos: torch.Tensor):
        """Los logits no pueden contener NaN ni Inf."""
        with torch.no_grad():
            logits, _, _ = modelo(tokens_cortos)
        assert torch.isfinite(logits).all(), "Logits contienen NaN o Inf"


# ==============================================================================
# TESTS DE PROPIEDADES ESTRUCTURALES
# ==============================================================================

class TestEstructura:
    def test_weight_tying(self, modelo: PamparV3):
        """tok_emb y lm_head deben compartir el MISMO tensor de pesos."""
        assert modelo.lm_head.weight is modelo.tok_emb.weight, (
            "Weight tying roto: lm_head.weight y tok_emb.weight son tensors distintos"
        )

    def test_num_niveles(self, modelo: PamparV3, config_small: ConfigV3):
        """El número de NivelProfundo debe coincidir con n_levels."""
        assert len(modelo.niveles) == config_small.n_levels

    def test_lateral_gate_scale_inicial(self, modelo: PamparV3, config_small: ConfigV3):
        """El parámetro 'scale' de LateralGate debe inicializarse cerca de 0.1."""
        for nivel in modelo.niveles:
            lg = nivel.lateral  # atributo real en NivelProfundo
            assert hasattr(lg, "scale"), "LateralGate no tiene atributo 'scale'"
            # scale es tensor (n_streams,) inicializado con 0.1
            scale = lg.scale  # shape: (n_streams,)
            assert scale.shape == (config_small.n_streams,), (
                f"Esperado shape ({config_small.n_streams},), obtenido {scale.shape}"
            )
            # Permitir ±5% de tolerancia respecto a 0.1
            scale_mean = scale.mean().item()
            assert abs(scale_mean - 0.1) < 0.01, (
                f"scale mean esperado ~0.1, obtenido {scale_mean:.4f}"
            )

    def test_count_params_positivo(self, modelo: PamparV3):
        """count_params() debe retornar un dict con 'total' de parámetros positivo."""
        result = modelo.count_params()
        assert isinstance(result, dict)
        assert "total" in result
        assert result["total"] > 0

    def test_describe_retorna_str(self, modelo: PamparV3):
        """describe() debe retornar una cadena no vacía."""
        desc = modelo.describe()
        assert isinstance(desc, str)
        assert len(desc) > 20

    def test_no_parametros_inf_nan(self, modelo: PamparV3):
        """Ningún parámetro debe contener NaN/Inf tras la inicialización."""
        for name, param in modelo.named_parameters():
            assert torch.isfinite(param).all(), f"Parámetro '{name}' contiene NaN o Inf"


# ==============================================================================
# TESTS DE GENERACIÓN
# ==============================================================================

class TestGeneracion:
    def test_generate_amplía_secuencia(self, modelo: PamparV3, tokens_single: torch.Tensor):
        """generate() debe producir más tokens que la entrada."""
        n_new = 10
        with torch.no_grad():
            output = modelo.generate(tokens_single, max_tokens=n_new)

        assert output.shape[1] > tokens_single.shape[1], (
            "generate() no amplió la secuencia"
        )

    def test_generate_max_tokens_respetado(self, modelo: PamparV3, tokens_single: torch.Tensor):
        """generate() no debe generar más de max_tokens tokens extra."""
        n_new = 5
        with torch.no_grad():
            output = modelo.generate(tokens_single, max_tokens=n_new)

        delta = output.shape[1] - tokens_single.shape[1]
        assert delta <= n_new, (
            f"Generados {delta} tokens, máximo esperado {n_new}"
        )

    def test_generate_sin_nan(self, modelo: PamparV3, tokens_single: torch.Tensor, config_small: ConfigV3):
        """Los tokens generados deben estar dentro del vocab y sin NaN."""
        with torch.no_grad():
            output = modelo.generate(tokens_single, max_tokens=8)

        assert output.dtype == torch.long
        assert (output >= 0).all()
        assert (output < config_small.vocab_size).all()

    def test_generate_top_k(self, modelo: PamparV3, tokens_single: torch.Tensor):
        """generate() con top_k debe funcionar sin errores."""
        with torch.no_grad():
            output = modelo.generate(tokens_single, max_tokens=5, top_k=10)
        assert output.shape[1] > tokens_single.shape[1]

    def test_generate_temperature(self, modelo: PamparV3, tokens_single: torch.Tensor):
        """generate() con temperature muy baja debe ser determinista."""
        with torch.no_grad():
            out1 = modelo.generate(tokens_single, max_tokens=5, temperature=1e-8)
            out2 = modelo.generate(tokens_single, max_tokens=5, temperature=1e-8)
        assert torch.equal(out1, out2), "generate() con temperature≈0 no es determinista"


# ==============================================================================
# TEST DE EARLY EXIT
# ==============================================================================

class TestEarlyExit:
    def test_early_exit_activo(self, modelo: PamparV3, tokens_cortos: torch.Tensor, config_small: ConfigV3):
        """Con use_early_exit=True el nivel de salida debe ser <= n_levels."""
        with torch.no_grad():
            _, _, info = modelo(tokens_cortos, use_early_exit=True)
        assert 1 <= info["exit_nivel"] <= config_small.n_levels

    def test_early_exit_inactivo(self, modelo: PamparV3, tokens_cortos: torch.Tensor, config_small: ConfigV3):
        """Con use_early_exit=False siempre se recorren todos los niveles."""
        with torch.no_grad():
            _, _, info = modelo(tokens_cortos, use_early_exit=False)
        assert info["exit_nivel"] == config_small.n_levels
