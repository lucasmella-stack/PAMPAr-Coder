# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Fixtures compartidas para la suite de tests de PAMPAr-Coder v3.

Patrón:
  - Usar PRESET_V3_SMALL para tests de arquitectura (más rápido, misma API)
  - Usar tmp_path de pytest para aislamiento de I/O
  - No depender de checkpoints pre-entrenados ni GPU
"""

import sys
import tempfile
from pathlib import Path

import pytest
import torch

# Asegurar que el proyecto raíz esté en el PATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from pampar.coder.v3.config import ConfigV3
from pampar.coder.v3.modelo import PamparV3
from pampar.memoria.clasificador import ClasificadorPareto


# =============================================================================
# PRESETS DE CONFIGURACIÓN
# =============================================================================

@pytest.fixture(scope="session")
def config_small() -> ConfigV3:
    """
    Configuración mínima para tests rápidos.

    Misma API que PRESET_V3 pero órdenes de magnitud más pequeña.
    No usa gradient checkpointing (innecesario sin backward en tests).
    """
    return ConfigV3(
        vocab_size=256,
        dim=64,
        n_streams=4,
        n_levels=3,
        n_heads=4,
        n_kv_heads=2,
        ffn_mult=2.0,
        n_zonas=52,            # Hardcodeado en v2 LLAVES — NO cambiar
        n_territorios=4,
        lateral_bottleneck=16,
        ventana_contexto=4,
        max_seq_len=64,
        dropout=0.0,           # Sin dropout en eval
        use_checkpoint=False,  # Sin gradient checkpointing en tests
    )


# =============================================================================
# MODELO
# =============================================================================

@pytest.fixture(scope="session")
def modelo(config_small: ConfigV3) -> PamparV3:
    """
    Instancia de PamparV3 inicializada aleatoriamente.

    Session-scoped: compartida en todos los tests de arquitectura,
    creada una sola vez (evita overhead de init 4× streams × 3 niveles).
    """
    m = PamparV3(config_small)
    m.eval()
    return m


# =============================================================================
# ENTRADAS DE PRUEBA
# =============================================================================

@pytest.fixture
def tokens_cortos(config_small: ConfigV3) -> torch.Tensor:
    """Batch de 2 secuencias de 16 tokens dentro del vocab pequeño."""
    return torch.randint(0, config_small.vocab_size, (2, 16))


@pytest.fixture
def tokens_single(config_small: ConfigV3) -> torch.Tensor:
    """Una sola secuencia de 8 tokens para generate()."""
    return torch.randint(0, config_small.vocab_size, (1, 8))


# =============================================================================
# CLASIFICADOR PARETO
# =============================================================================

@pytest.fixture
def clasificador() -> ClasificadorPareto:
    """ClasificadorPareto sin dependencias externas."""
    return ClasificadorPareto()


# =============================================================================
# CÓDIGO EJEMPLO (fixture reutilizable en test_skills y test_memoria)
# =============================================================================

CODIGO_RICO = '''
from typing import List, Optional
import asyncio
from dataclasses import dataclass

@dataclass
class Resultado:
    """Resultado de la evaluación."""
    valor: float
    exitoso: bool = True
    errores: List[str] = None

    def __post_init__(self):
        if self.errores is None:
            self.errores = []

async def evaluar(items: List[str], max_items: Optional[int] = None) -> Resultado:
    """Evalúa una lista de items de forma asíncrona."""
    try:
        resultado = [x for x in items if x.strip()]
        if max_items:
            resultado = resultado[:max_items]
        return Resultado(valor=len(resultado) / len(items))
    except ZeroDivisionError as e:
        return Resultado(valor=0.0, exitoso=False, errores=[str(e)])
'''.strip()


@pytest.fixture
def codigo_rico() -> str:
    """Fragmento Python denso: dataclass, async, type hints, comprehension, try/except."""
    return CODIGO_RICO
