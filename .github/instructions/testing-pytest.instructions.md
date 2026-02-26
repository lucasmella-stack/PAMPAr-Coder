```instructions
# Testing Instructions (pytest)

> Para tests en PAMPAr-Coder usando pytest.

## Regla de Oro

**Cada módulo nuevo DEBE tener:**
1. Un test de happy path
2. Un test de error/edge case
3. Un test de shapes (para tensores)

## Estructura de Tests

```

tests/
├── test_modelo.py # Tests del modelo principal
├── test_talamo.py # Tests del tálamo
├── test_llaves.py # Tests de LLAVES
├── test_generation.py # Tests de generación
└── conftest.py # Fixtures compartidos

````

## Fixtures (conftest.py)

```python
import pytest
import torch
from pampar.coder.v2.config import ConfigPampaRCoderV2

@pytest.fixture
def device():
    """Device para tests: CUDA si disponible, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def small_config():
    """Configuración mínima para tests rápidos."""
    return ConfigPampaRCoderV2(
        vocab_size=1000,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        intermediate_size=256,
    )

@pytest.fixture
def batch():
    """Batch de ejemplo para tests."""
    return {
        "input_ids": torch.randint(0, 1000, (2, 16)),
        "attention_mask": torch.ones(2, 16, dtype=torch.long),
        "labels": torch.randint(0, 1000, (2, 16)),
    }
````

## Patrones de Test

### Test de Shapes

```python
class TestModelShapes:
    def test_embedding_output_shape(self, small_config):
        from pampar.coder.v2.modelo import PampaRCoderV2

        model = PampaRCoderV2(small_config)
        input_ids = torch.randint(0, small_config.vocab_size, (2, 16))

        output = model(input_ids)

        assert output.logits.shape == (2, 16, small_config.vocab_size)
        assert output.hidden_states.shape == (2, 16, small_config.hidden_size)

    def test_attention_shape(self, small_config):
        from pampar.coder.v2.talamo import TalamoBrodmann

        talamo = TalamoBrodmann(small_config)
        x = torch.randn(2, 16, small_config.hidden_size)

        out = talamo(x)

        assert out.shape == x.shape
```

### Test de Gradientes

```python
class TestGradientFlow:
    def test_all_parameters_have_gradients(self, small_config):
        model = PampaRCoderV2(small_config)
        input_ids = torch.randint(0, small_config.vocab_size, (1, 8))

        output = model(input_ids, labels=input_ids)
        output.loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No grad: {name}"
                assert not torch.isnan(param.grad).any(), f"NaN grad: {name}"

    def test_gradient_clipping(self, small_config):
        model = PampaRCoderV2(small_config)
        # ... setup con gradientes grandes

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        total_norm = sum(p.grad.norm() ** 2 for p in model.parameters()).sqrt()
        assert total_norm <= 1.0 + 1e-6
```

### Test con Mock

```python
from unittest.mock import MagicMock, patch

class TestTraining:
    @patch("torch.cuda.is_available", return_value=False)
    def test_cpu_fallback(self, mock_cuda):
        """Verifica que funciona sin GPU."""
        from pampar.coder.v2.modelo import PampaRCoderV2

        config = ConfigPampaRCoderV2.from_preset("mini")
        model = PampaRCoderV2(config)

        input_ids = torch.randint(0, config.vocab_size, (1, 8))
        output = model(input_ids)

        assert output.logits is not None
```

### Test Parametrizado

```python
@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [8, 16, 32])
def test_variable_batch_seq(small_config, batch_size, seq_len):
    model = PampaRCoderV2(small_config)
    input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))

    output = model(input_ids)

    assert output.logits.shape == (batch_size, seq_len, small_config.vocab_size)

@pytest.mark.parametrize("preset", ["mini", "1.5B", "3B"])
def test_preset_configs(preset):
    config = ConfigPampaRCoderV2.from_preset(preset)

    assert config.vocab_size == 48000
    assert config.hidden_size > 0
```

### Test de LLAVES

```python
class TestLlaves:
    def test_llaves_are_not_trainable(self):
        from pampar.coder.v2.llaves import LlavesModule

        llaves = LlavesModule()

        for param in llaves.parameters():
            assert not param.requires_grad, "LLAVES no deben ser entrenables"

    def test_llaves_int8_quantization(self):
        from pampar.coder.v2.llaves import LlavesModule

        llaves = LlavesModule()

        assert llaves.lookup_table.dtype == torch.int8

    def test_llaves_pattern_matching(self):
        from pampar.coder.v2.llaves import classify_token

        # Declaración Python
        assert classify_token("def ") in range(1, 16)  # SINTAXIS

        # Operador lógico
        assert classify_token("if ") in range(31, 43)  # LÓGICO
```

## Markers

```python
# En pyproject.toml o pytest.ini:
# [tool.pytest.ini_options]
# markers = [
#     "slow: marks tests as slow",
#     "gpu: marks tests requiring GPU",
# ]

@pytest.mark.slow
def test_full_training_loop():
    """Test lento de training completo."""
    ...

@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_cuda_forward():
    """Test que requiere GPU."""
    ...
```

## Ejecutar Tests

```bash
# Todos los tests
pytest

# Solo tests rápidos
pytest -m "not slow"

# Con coverage
pytest --cov=pampar --cov-report=html

# Verbose con print output
pytest -v -s

# Solo un archivo
pytest tests/test_modelo.py

# Solo un test específico
pytest tests/test_modelo.py::TestModelShapes::test_embedding_output_shape
```

```

```
