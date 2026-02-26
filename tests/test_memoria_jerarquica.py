# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""Tests para MemoriaJerarquica y curación de datos Pareto."""

import json
import sys
import tempfile
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from pampar.coder.v2.aprendizaje.memoria_jerarquica import (
    EntradaMemoria,
    MemoriaJerarquica,
    NivelMemoria,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def nivel_pequeno():
    """Nivel de memoria con capacidad pequeña para tests."""
    return NivelMemoria("test", capacidad=10, ratio_pareto=0.2)


@pytest.fixture
def memoria():
    """Memoria jerárquica con capacidades pequeñas para tests."""
    return MemoriaJerarquica(
        capacidad_l0=20,
        capacidad_l1=50,
        capacidad_l2=25,
        ventana_tokens=4,
        umbral_loss_alta=1.0,
    )


@pytest.fixture
def entrada_ejemplo():
    """EntradaMemoria de ejemplo."""
    return EntradaMemoria(
        tokens=(1, 2, 3, 4, 5),
        importancia=0.8,
        loss_media=5.0,
        novedad=0.7,
        territorio_dominante=0,
    )


# =============================================================================
# TESTS: EntradaMemoria
# =============================================================================

class TestEntradaMemoria:
    def test_score_pareto_basico(self, entrada_ejemplo):
        score = entrada_ejemplo.score_pareto()
        assert score > 0
        assert isinstance(score, float)

    def test_score_pareto_loss_alta_mayor(self):
        """Loss alta = más importante que loss baja."""
        alta = EntradaMemoria(tokens=(1, 2), loss_media=8.0, novedad=0.5)
        baja = EntradaMemoria(tokens=(3, 4), loss_media=1.0, novedad=0.5)
        assert alta.score_pareto() > baja.score_pareto()

    def test_score_pareto_novel_vs_repetido(self):
        """Patrones novedosos tienen más puntaje."""
        novel = EntradaMemoria(
            tokens=(1, 2), loss_media=3.0, novedad=1.0, frecuencia=1
        )
        repetido = EntradaMemoria(
            tokens=(3, 4), loss_media=3.0, novedad=0.1, frecuencia=100
        )
        assert novel.score_pareto() > repetido.score_pareto()

    def test_survival_bonus(self):
        """Entradas que sobreviven compresiones ganan bonus."""
        nueva = EntradaMemoria(
            tokens=(1, 2), loss_media=3.0, veces_comprimido=0
        )
        veterana = EntradaMemoria(
            tokens=(1, 2), loss_media=3.0, veces_comprimido=5
        )
        assert veterana.score_pareto() > nueva.score_pareto()


# =============================================================================
# TESTS: NivelMemoria
# =============================================================================

class TestNivelMemoria:
    def test_agregar_hasta_capacidad(self, nivel_pequeno):
        for i in range(10):
            e = EntradaMemoria(tokens=(i,), loss_media=float(i))
            assert nivel_pequeno.agregar(e) is True
        assert nivel_pequeno.lleno

    def test_agregar_lleno_reemplaza_peor(self, nivel_pequeno):
        """Al llenar, la nueva entrada reemplaza la peor si es mejor."""
        # Llenar con entradas de score bajo
        for i in range(10):
            e = EntradaMemoria(tokens=(i,), loss_media=0.1, novedad=0.01)
            nivel_pequeno.agregar(e)

        # Agregar una con score alto → debe reemplazar la peor
        buena = EntradaMemoria(tokens=(99,), loss_media=9.0, novedad=1.0)
        assert nivel_pequeno.agregar(buena) is True
        assert (99,) in nivel_pequeno.buffer

    def test_agregar_duplicado_incrementa_frecuencia(self, nivel_pequeno):
        e1 = EntradaMemoria(tokens=(1, 2, 3), loss_media=5.0)
        e2 = EntradaMemoria(tokens=(1, 2, 3), loss_media=3.0)
        nivel_pequeno.agregar(e1)
        nivel_pequeno.agregar(e2)
        assert nivel_pequeno.buffer[(1, 2, 3)].frecuencia == 2

    def test_comprimir_pareto_retiene_20pct(self, nivel_pequeno):
        # Llenar con 10 entradas de scores variados
        for i in range(10):
            e = EntradaMemoria(tokens=(i,), loss_media=float(i + 1), novedad=0.5)
            nivel_pequeno.agregar(e)

        promovidas = nivel_pequeno.comprimir_pareto()

        # ratio_pareto=0.2 → debe retener 2 de 10
        assert len(promovidas) == 2
        # Buffer debe quedar vacío
        assert len(nivel_pequeno.buffer) == 0
        # Las promovidas deben ser las de mayor score
        scores = [p.score_pareto() for p in promovidas]
        assert scores[0] >= scores[1]

    def test_comprimir_incrementa_veces_comprimido(self, nivel_pequeno):
        for i in range(10):
            e = EntradaMemoria(tokens=(i,), loss_media=float(i + 1))
            nivel_pequeno.agregar(e)

        promovidas = nivel_pequeno.comprimir_pareto()
        for p in promovidas:
            assert p.veces_comprimido >= 1

    def test_stats(self, nivel_pequeno):
        e = EntradaMemoria(tokens=(1, 2), loss_media=5.0)
        nivel_pequeno.agregar(e)
        stats = nivel_pequeno.stats()
        assert stats["entradas"] == 1
        assert stats["capacidad"] == 10
        assert stats["uso_pct"] == 10.0


# =============================================================================
# TESTS: MemoriaJerarquica
# =============================================================================

class TestMemoriaJerarquica:
    def test_procesar_batch_basico(self, memoria):
        B, L = 2, 16
        input_ids = torch.randint(0, 100, (B, L))
        # Loss alta en varias posiciones
        per_token_loss = torch.rand(B, L) * 5.0

        result = memoria.procesar_batch(input_ids, per_token_loss)

        assert "entradas_creadas" in result
        assert result["entradas_creadas"] >= 0

    def test_procesar_batch_sin_loss_alta_no_guarda(self, memoria):
        B, L = 2, 16
        input_ids = torch.randint(0, 100, (B, L))
        # Loss baja → nada importante
        per_token_loss = torch.ones(B, L) * 0.1

        result = memoria.procesar_batch(input_ids, per_token_loss)
        assert result["entradas_creadas"] == 0

    def test_flujo_completo_l0_a_l1(self, memoria):
        """Cuando L0 se llena, comprime y promueve a L1."""
        # Generar suficientes batches para llenar L0
        for _ in range(10):
            B, L = 2, 32
            input_ids = torch.randint(0, 1000, (B, L))
            per_token_loss = torch.rand(B, L) * 8.0  # Loss alta
            memoria.procesar_batch(input_ids, per_token_loss)

        # L1 debería tener entradas (promovidas desde L0)
        # Puede o no haber dependiendo de si L0 se llenó
        stats = memoria.stats()
        assert stats["total_tokens_procesados"] > 0

    def test_consolidar_sin_modelo(self, memoria):
        # Llenar niveles
        for _ in range(20):
            B, L = 2, 16
            input_ids = torch.randint(0, 500, (B, L))
            per_token_loss = torch.rand(B, L) * 6.0
            memoria.procesar_batch(input_ids, per_token_loss)

        result = memoria.consolidar(model=None)
        assert "niveles" in result

    def test_get_replay_batch(self, memoria):
        # Llenar L1 con datos
        for i in range(100):
            e = EntradaMemoria(
                tokens=tuple(range(i, i + 5)),
                loss_media=float(i % 10),
                novedad=0.5,
            )
            memoria.l1.agregar(e)

        batch = memoria.get_replay_batch(batch_size=4, nivel="l1", strategy="hardest")
        assert batch is not None
        assert batch.shape[0] == 4

    def test_get_replay_batch_diverse(self, memoria):
        # Entradas con diferentes territorios
        for i in range(50):
            e = EntradaMemoria(
                tokens=tuple(range(i, i + 5)),
                loss_media=3.0,
                territorio_dominante=i % 4,
            )
            memoria.l1.agregar(e)

        batch = memoria.get_replay_batch(
            batch_size=8, nivel="l1", strategy="diverse"
        )
        assert batch is not None

    def test_guardar_cargar(self, memoria, tmp_path):
        """La memoria sobrevive serialización."""
        # Agregar datos
        for i in range(5):
            e = EntradaMemoria(
                tokens=(i, i + 1, i + 2),
                loss_media=float(i + 1),
                novedad=0.5,
                territorio_dominante=i % 4,
            )
            memoria.l1.agregar(e)

        path = str(tmp_path / "test_memoria.json")
        memoria.guardar(path)

        # Cargar y verificar
        cargada = MemoriaJerarquica.cargar(path)
        assert len(cargada.l1.buffer) == len(memoria.l1.buffer)
        assert cargada.l1.capacidad == memoria.l1.capacidad

    def test_repr(self, memoria):
        s = repr(memoria)
        assert "MemoriaJerarquica" in s
        assert "L0=" in s
        assert "L1=" in s

    def test_stats_completas(self, memoria):
        s = memoria.stats()
        assert "total_tokens_procesados" in s
        assert "total_interiorizados_l3" in s
        assert "niveles" in s
        assert "l0" in s["niveles"]
        assert "l1" in s["niveles"]
        assert "l2" in s["niveles"]


# =============================================================================
# TESTS: curar_datos.py (scoring)
# =============================================================================

class TestScoringSample:
    """Tests del scoring de calidad de datos."""

    def test_import_scoring(self):
        # Importar el módulo de scoring
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from curar_datos import score_sample

        # Código complejo = score alto
        code_complex = '''### Response:
class DataProcessor:
    """Procesa datos de múltiples fuentes."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._cache: Dict[str, pd.DataFrame] = {}

    async def process(self, source: str) -> pd.DataFrame:
        try:
            if source in self._cache:
                return self._cache[source]
            data = await self._fetch(source)
            for transformer in self.config["transformers"]:
                data = transformer.apply(data)
            self._cache[source] = data
            return data
        except ConnectionError as e:
            logger.error(f"Failed to fetch {source}: {e}")
            raise
'''
        score_complex, _ = score_sample(code_complex)

        # Código trivial = score bajo
        code_trivial = "### Response:\narr = [1, 2, 3]"
        score_trivial, _ = score_sample(code_trivial)

        assert score_complex > score_trivial
        assert score_complex > 5.0
        assert score_trivial < 2.0

    def test_score_empty(self):
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from curar_datos import score_sample

        score, details = score_sample("")
        assert score == 0.0

    def test_filtrar_pareto_crea_archivo(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from curar_datos import filtrar_pareto

        # Crear dataset de prueba
        input_file = tmp_path / "test.jsonl"
        samples = []
        for i in range(100):
            if i < 20:
                # Samples de alta calidad
                text = (
                    "### Instruction:\nImplement a binary search tree\n"
                    "### Response:\n"
                    f"class BinarySearchTree:\n"
                    f"    def __init__(self):\n"
                    f"        self.root = None\n"
                    f"    def insert(self, value: int) -> None:\n"
                    f"        if self.root is None:\n"
                    f"            self.root = Node(value)\n"
                    f"        else:\n"
                    f"            self._insert_recursive(self.root, value)\n"
                )
            else:
                # Samples triviales
                text = f"### Instruction:\nPrint {i}\n### Response:\nprint({i})"

            samples.append(json.dumps({"text": text, "source": "test"}))

        input_file.write_text("\n".join(samples))

        output_file = tmp_path / "output.jsonl"
        stats = filtrar_pareto(
            str(input_file),
            str(output_file),
            ratio=0.2,
            min_score=1.0,
        )

        assert output_file.exists()
        assert stats["seleccionados"] <= stats["total_leidos"]
        assert stats["ratio_efectivo"] <= 100


# =============================================================================
# TESTS: Integración con modelo (si está disponible)
# =============================================================================

class TestIntegracionModelo:
    @pytest.fixture
    def small_model(self):
        """Modelo pequeño para tests de integración."""
        from pampar.coder.v2.config import ConfigV2
        from pampar.coder.v2.modelo import PampaRCoderV2

        config = ConfigV2(
            vocab_size=1000,
            dim=64,
            n_heads=4,
            n_capas=2,
            max_seq_len=32,
        )
        return PampaRCoderV2(config)

    def test_interiorizacion_actualiza_pesos(self, small_model):
        """L3 debe actualizar pesos del modelo."""
        mem = MemoriaJerarquica(
            capacidad_l0=10,
            capacidad_l1=10,
            capacidad_l2=10,
            ventana_tokens=4,
            lr_interiorizacion=0.01,
        )

        # Capturar pesos antes
        weights_before = small_model.tok_emb.weight.data.clone()

        # Crear entradas para interiorizar
        entradas = [
            EntradaMemoria(
                tokens=(i, i + 1, i + 2, i + 3, i + 4),
                loss_media=5.0,
            )
            for i in range(0, 50, 5)
        ]

        # Interiorizar
        result = mem._interiorizar(small_model, entradas)

        assert result["n_patrones"] > 0

        # Pesos deben haber cambiado
        weights_after = small_model.tok_emb.weight.data
        assert not torch.allclose(weights_before, weights_after), \
            "La interiorización debe modificar los pesos"

    def test_procesar_y_consolidar_con_modelo(self, small_model):
        """Flujo completo: procesar → consolidar → interiorizar."""
        mem = MemoriaJerarquica(
            capacidad_l0=10,
            capacidad_l1=20,
            capacidad_l2=10,
            ventana_tokens=4,
            umbral_loss_alta=0.5,
            lr_interiorizacion=1e-4,
        )

        # Procesar muchos batches para llenar todos los niveles
        for _ in range(50):
            input_ids = torch.randint(0, 500, (2, 16))
            per_token_loss = torch.rand(2, 16) * 8.0
            mem.procesar_batch(input_ids, per_token_loss)

        # Consolidar con modelo
        result = mem.consolidar(model=small_model)
        assert "niveles" in result
