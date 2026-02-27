# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Tests del sistema de memoria de PAMPAr-Coder v3.

Cubren:
  ClasificadorPareto:
    - Texto vacío → nivel 0
    - Código pobre → nivel 0 o 1 (score bajo)
    - Código denso (dataclass, async, type hints, comprehension) → nivel ≥ 1
    - loss_modelo alta sube importancia
    - actualizar_frecuencia impulsa correctamente
  RAGResidual:
    - agregar entrada nivel 0 → retorna False, no agrega
    - agregar entrada nivel 1 → retorna True, len==1
    - agregar duplicado → retorna False, frecuencia aumenta
    - recuperar → lista ordenada por score
    - eliminar_por_nivel → solo elimina el nivel pedido
    - formatear_contexto → incluye marcador [MEMORIA RELEVANTE]
    - stats → claves esperadas presentes
  ColaFinetune:
    - agregar nivel < 3 → no se agrega
    - agregar nivel 3 → se agrega, len==1
    - agregar duplicado → no se duplica
    - exportar_dataset → JSONL válido con campos instruction/input/output
    - vaciar_post_finetune → vacía la cola
    - proponer_usuario → retorna string indicativo
"""

import json
import tempfile
from pathlib import Path

import pytest

from memoria.clasificador import ClasificadorPareto, EntradaMemoria
from memoria.rag import RAGResidual
from memoria.cola_finetune import ColaFinetune


# ==============================================================================
# HELPERS DE FIXTURES LOCALES
# ==============================================================================

def _entrada(nivel: int = 1, texto: str = "def foo(): pass", tipo: str = "codigo") -> EntradaMemoria:
    """Crea una EntradaMemoria de prueba con importancia acorde al nivel."""
    importancia_por_nivel = {0: 0.1, 1: 0.4, 2: 0.7, 3: 0.9}
    e = EntradaMemoria(
        texto=texto,
        tipo=tipo,
        importancia=importancia_por_nivel.get(nivel, 0.4),
        novedad=0.8,
        densidad=0.5,
        nivel=nivel,
    )
    return e


# ==============================================================================
# TESTS: ClasificadorPareto
# ==============================================================================

class TestClasificadorPareto:
    def test_texto_vacio_es_nivel_cero(self, clasificador: ClasificadorPareto):
        """Un texto vacío siempre debe clasificarse como nivel 0 (ignorar)."""
        resultado = clasificador.clasificar("")
        assert resultado.nivel == 0
        assert resultado.importancia == 0.0

    def test_texto_whitespace_es_nivel_cero(self, clasificador: ClasificadorPareto):
        """Un texto de solo espacios tampoco se guarda."""
        resultado = clasificador.clasificar("   \n\t  ")
        assert resultado.nivel == 0

    def test_codigo_simple_score_bajo(self, clasificador: ClasificadorPareto):
        """Una asignación simple sin patrones avanzados tiene importancia baja."""
        resultado = clasificador.clasificar("x = 1")
        # No debe pasar L2 (importancia >= 0.6)
        assert resultado.importancia < 0.6

    def test_codigo_denso_nivel_minimo_uno(self, clasificador: ClasificadorPareto, codigo_rico: str):
        """Código con dataclass, async, type hints, comprehension → nivel ≥ 1."""
        resultado = clasificador.clasificar(codigo_rico)
        assert resultado.nivel >= 1, (
            f"Código rico debería estar en L1+, obtuvo nivel {resultado.nivel} "
            f"(importancia={resultado.importancia:.4f})"
        )

    def test_importancia_en_rango_valido(self, clasificador: ClasificadorPareto, codigo_rico: str):
        """La importancia siempre debe estar en [0, 1]."""
        resultado = clasificador.clasificar(codigo_rico)
        assert 0.0 <= resultado.importancia <= 1.0

    def test_loss_alta_sube_importancia(self, clasificador: ClasificadorPareto):
        """Un fragmento con loss_modelo alta debe tener mayor importancia."""
        low_loss = clasificador.clasificar("x = 1", loss_modelo=0.1)
        high_loss = clasificador.clasificar("x = 1", loss_modelo=10.0)
        assert high_loss.importancia > low_loss.importancia

    def test_novedad_maxima_sin_existentes(self, clasificador: ClasificadorPareto):
        """Sin fragmentos existentes, novedad debe ser 1.0."""
        resultado = clasificador.clasificar("def f(): pass", fragmentos_existentes=[])
        assert resultado.novedad == 1.0

    def test_novedad_baja_con_texto_identico(self, clasificador: ClasificadorPareto):
        """Si el texto es igual a uno existente, novedad debe bajar drásticamente."""
        fragmento = "def calcular_area(radio): return 3.14 * radio * radio"
        r1 = clasificador.clasificar(fragmento, fragmentos_existentes=[])
        r2 = clasificador.clasificar(fragmento, fragmentos_existentes=[fragmento])
        assert r2.novedad < r1.novedad

    def test_id_generado_automaticamente(self, clasificador: ClasificadorPareto):
        """El id debe ser un hash hex de 16 caracteres."""
        resultado = clasificador.clasificar("def f(): pass")
        assert isinstance(resultado.id, str)
        assert len(resultado.id) == 16
        # Hex characters only
        int(resultado.id, 16)

    def test_territorio_detectado(self, clasificador: ClasificadorPareto):
        """El territorio dominante debe ser uno de los 4 esperados."""
        resultado = clasificador.clasificar("def f(x: int) -> str: return str(x)")
        validos = {"SINTAXIS", "SEMANTICA", "LOGICO", "ESTRUCTURAL", ""}
        assert resultado.territorio_dominante in validos

    def test_tipo_preservado(self, clasificador: ClasificadorPareto):
        """El tipo pasado se debe preservar en la entrada resultante."""
        resultado = clasificador.clasificar("ZeroDivisionError", tipo="error")
        assert resultado.tipo == "error"


# ==============================================================================
# TESTS: RAGResidual
# ==============================================================================

class TestRAGResidual:
    @pytest.fixture
    def rag(self, tmp_path: Path) -> RAGResidual:
        """RAG aislado en directorio temporal."""
        return RAGResidual(directorio=str(tmp_path / "rag"), max_entradas=100, n_resultados=3)

    def test_agregar_nivel_cero_retorna_false(self, rag: RAGResidual):
        """Entradas nivel 0 no deben entrar al RAG."""
        entrada = _entrada(nivel=0)
        resultado = rag.agregar(entrada)
        assert resultado is False
        assert len(rag._entradas) == 0

    def test_agregar_nivel_uno_retorna_true(self, rag: RAGResidual):
        """Entradas nivel 1 se deben guardar en el RAG."""
        entrada = _entrada(nivel=1)
        resultado = rag.agregar(entrada)
        assert resultado is True
        assert len(rag._entradas) == 1

    def test_agregar_nivel_dos_y_tres(self, rag: RAGResidual):
        """Niveles 2 y 3 también se aceptan."""
        assert rag.agregar(_entrada(nivel=2, texto="def a(): pass")) is True
        assert rag.agregar(_entrada(nivel=3, texto="def b(): pass")) is True
        assert len(rag._entradas) == 2

    def test_agregar_duplicado_no_duplica(self, rag: RAGResidual):
        """Agregar el mismo texto dos veces no duplica la entrada."""
        e = _entrada(nivel=1, texto="def foo(): return 42")
        rag.agregar(e)
        resultado = rag.agregar(e)  # Mismo id → duplicado
        assert resultado is False
        assert len(rag._entradas) == 1

    def test_agregar_duplicado_incrementa_frecuencia(self, rag: RAGResidual):
        """El duplicado debe incrementar la frecuencia de la entrada existente."""
        e = _entrada(nivel=1, texto="def foo(): return 42")
        e.frecuencia = 1
        rag.agregar(e)
        rag.agregar(e)
        assert rag._entradas[0].frecuencia == 2

    def test_recuperar_retorna_lista(self, rag: RAGResidual):
        """recuperar() debe retornar una lista (aunque vacía)."""
        result = rag.recuperar("def foo(): pass")
        assert isinstance(result, list)

    def test_recuperar_encuentra_entradas(self, rag: RAGResidual):
        """Tras agregar una entrada, recuperar() debe encontrarla."""
        texto = "def calcular_suma(a: int, b: int) -> int: return a + b"
        rag.agregar(_entrada(nivel=1, texto=texto))
        resultados = rag.recuperar(texto)
        assert len(resultados) >= 1
        entradas = [e.texto for e, _ in resultados]
        assert texto in entradas

    def test_eliminar_por_nivel_correcto(self, rag: RAGResidual):
        """eliminar_por_nivel(2) debe eliminar solo las entradas de nivel 2."""
        rag.agregar(_entrada(nivel=1, texto="def nivel_uno(): pass"))
        rag.agregar(_entrada(nivel=2, texto="class NivelDos: pass"))
        rag.agregar(_entrada(nivel=3, texto="async def nivel_tres(): pass"))

        eliminados = rag.eliminar_por_nivel(2)

        assert eliminados == 1
        niveles_restantes = {e.nivel for e in rag._entradas}
        assert 2 not in niveles_restantes
        assert 1 in niveles_restantes
        assert 3 in niveles_restantes

    def test_formatear_contexto_vacio(self, rag: RAGResidual):
        """Sin resultados, formatear_contexto debe retornar string vacío."""
        ctx = rag.formatear_contexto([])
        assert ctx == ""

    def test_formatear_contexto_incluye_marcador(self, rag: RAGResidual):
        """Con resultados, debe incluir el bloque [MEMORIA RELEVANTE]."""
        e = _entrada(nivel=1, texto="def foo(): pass")
        rag.agregar(e)
        resultados = rag.recuperar("def foo", nivel_minimo=1)
        ctx = rag.formatear_contexto(resultados)
        assert "[MEMORIA RELEVANTE]" in ctx
        assert "[/MEMORIA RELEVANTE]" in ctx

    def test_stats_claves_esperadas(self, rag: RAGResidual):
        """stats() debe incluir las claves esperadas."""
        s = rag.stats()
        for clave in ("total_entradas", "nivel_1_rag", "nivel_2_alta_prio", "nivel_3_finetune", "modo_encoder"):
            assert clave in s, f"Falta clave '{clave}' en stats()"

    def test_stats_contadores_coherentes(self, rag: RAGResidual):
        """Los contadores de stats deben sumar el total."""
        rag.agregar(_entrada(nivel=1, texto="# code 1"))
        rag.agregar(_entrada(nivel=2, texto="# code 2"))
        rag.agregar(_entrada(nivel=3, texto="# code 3"))
        s = rag.stats()
        suma = s["nivel_1_rag"] + s["nivel_2_alta_prio"] + s["nivel_3_finetune"]
        assert suma == s["total_entradas"]


# ==============================================================================
# TESTS: ColaFinetune
# ==============================================================================

class TestColaFinetune:
    @pytest.fixture
    def cola(self, tmp_path: Path) -> ColaFinetune:
        """Cola aislada en directorio temporal con umbral bajo para tests."""
        return ColaFinetune(directorio=str(tmp_path / "cola"), min_ejemplos=5)

    def test_agregar_nivel_menor_tres_no_agrega(self, cola: ColaFinetune):
        """Entradas nivel < 3 no deben entrar en la cola."""
        cola.agregar(_entrada(nivel=2))
        assert len(cola) == 0

    def test_agregar_nivel_tres_agrega(self, cola: ColaFinetune):
        """Entradas nivel 3 deben agregarse."""
        cola.agregar(_entrada(nivel=3))
        assert len(cola) == 1

    def test_agregar_duplicado_no_duplica(self, cola: ColaFinetune):
        """El mismo id no debe aparecer dos veces en la cola."""
        e = _entrada(nivel=3, texto="async def train(): pass")
        cola.agregar(e)
        cola.agregar(e)  # duplicado
        assert len(cola) == 1

    def test_stats_total_correcto(self, cola: ColaFinetune):
        """stats()['total'] debe coincidir con len(cola)."""
        cola.agregar(_entrada(nivel=3, texto="def a(): pass"))
        cola.agregar(_entrada(nivel=3, texto="def b(): pass"))
        assert cola.stats()["total"] == 2

    def test_stats_listos_cuando_supera_umbral(self, cola: ColaFinetune):
        """stats()['listos'] debe ser True cuando len >= min_ejemplos."""
        for i in range(5):
            cola.agregar(_entrada(nivel=3, texto=f"def fn_{i}(): pass"))
        assert cola.stats()["listos"] is True

    def test_exportar_dataset_crea_jsonl(self, cola: ColaFinetune, tmp_path: Path):
        """exportar_dataset() debe crear un archivo JSONL con al menos una línea."""
        cola.agregar(_entrada(nivel=3, texto="class Trainer: pass"))
        ruta_out = str(tmp_path / "ft.jsonl")
        ruta = cola.exportar_dataset(ruta_salida=ruta_out)

        assert ruta.exists()
        lineas = [l for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lineas) >= 1

    def test_exportar_dataset_formato_alpaca(self, cola: ColaFinetune, tmp_path: Path):
        """Cada línea del JSONL debe tener las claves instruction, input, output."""
        cola.agregar(_entrada(nivel=3, texto="def train(model): pass"))
        ruta = cola.exportar_dataset()

        for linea in ruta.read_text(encoding="utf-8").splitlines():
            if linea.strip():
                obj = json.loads(linea)
                assert "instruction" in obj, "Falta 'instruction'"
                assert "input" in obj, "Falta 'input'"
                assert "output" in obj, "Falta 'output'"

    def test_vaciar_post_finetune(self, cola: ColaFinetune):
        """vaciar_post_finetune() debe vaciar la cola y retornar el count previo."""
        cola.agregar(_entrada(nivel=3, texto="def a(): pass"))
        cola.agregar(_entrada(nivel=3, texto="def b(): pass"))
        n = cola.vaciar_post_finetune()
        assert n == 2
        assert len(cola) == 0

    def test_proponer_usuario_retorna_string(self, cola: ColaFinetune):
        """proponer_usuario() siempre debe retornar un string no vacío."""
        msg = cola.proponer_usuario()
        assert isinstance(msg, str)
        assert len(msg) > 10

    def test_persistencia_entre_instancias(self, tmp_path: Path):
        """Los datos persistidos deben cargarse en una nueva instancia."""
        dir_cola = str(tmp_path / "cola_persist")
        c1 = ColaFinetune(directorio=dir_cola, min_ejemplos=50)
        c1.agregar(_entrada(nivel=3, texto="def persistida(): pass"))
        assert len(c1) == 1

        c2 = ColaFinetune(directorio=dir_cola, min_ejemplos=50)
        assert len(c2) == 1
