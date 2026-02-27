# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Tests del Motor de Curiosidad — aprendizaje autónomo guiado por curiosidad.

Cubre:
  - PerfilTema: registro, actualización, detección de dominio
  - MotorCuriosidad: curiosidad zone proximal, selección de tema
  - Persistencia: guardar/cargar estado
  - Integración: flujo completo de varias sesiones
"""

import json
import math
import tempfile
from collections import deque
from pathlib import Path

import pytest

from pampar.coder.v2.aprendizaje.curiosidad import (
    MotorCuriosidad,
    PerfilTema,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def perfil_nuevo() -> PerfilTema:
    return PerfilTema(
        nombre="funciones_basicas",
        categoria="python_basico",
        nivel_dificultad=3,
    )


@pytest.fixture
def motor() -> MotorCuriosidad:
    m = MotorCuriosidad(nivel_actual=1)
    m.registrar_tema("variables", "python_basico", 1)
    m.registrar_tema("listas", "python_basico", 1)
    m.registrar_tema("funciones", "python_basico", 3)
    m.registrar_tema("grafos", "algoritmos", 5)
    return m


INDICE_TEST = {
    "python_basico": [
        {"nombre": "variables",  "nivel": 1, "archivo": "python_basico/variables.jsonl"},
        {"nombre": "listas",     "nivel": 1, "archivo": "python_basico/listas.jsonl"},
        {"nombre": "funciones",  "nivel": 3, "archivo": "python_basico/funciones.jsonl"},
    ],
    "algoritmos": [
        {"nombre": "grafos",     "nivel": 5, "archivo": "algoritmos/grafos.jsonl"},
        {"nombre": "recursion",  "nivel": 4, "archivo": "algoritmos/recursion.jsonl"},
    ],
}


# =============================================================================
# TESTS: PerfilTema
# =============================================================================

class TestPerfilTema:
    def test_valores_iniciales(self, perfil_nuevo):
        """Un perfil nuevo tiene loss alta y no está dominado."""
        assert perfil_nuevo.loss_media == 99.0
        assert perfil_nuevo.dominado is False
        assert perfil_nuevo.n_sesiones == 0
        assert perfil_nuevo.curiosidad == 1.0

    def test_registrar_una_sesion(self, perfil_nuevo):
        """Registrar una sesión actualiza la loss media."""
        perfil_nuevo.registrar_sesion(2.5)
        assert perfil_nuevo.n_sesiones == 1
        assert perfil_nuevo.loss_media == 2.5

    def test_registrar_varias_sesiones_promedia(self, perfil_nuevo):
        """La loss media refleja las últimas 5 sesiones."""
        for l in [3.0, 2.8, 2.6, 2.4, 2.2, 2.0]:
            perfil_nuevo.registrar_sesion(l)
        # Últimas 5: 2.8, 2.6, 2.4, 2.2, 2.0 → media = 2.4
        assert abs(perfil_nuevo.loss_media - 2.4) < 0.01

    def test_detectar_dominio(self, perfil_nuevo):
        """Debe marcarse como dominado tras 5 sesiones consecutivas con loss < 0.8."""
        assert perfil_nuevo.dominado is False
        for _ in range(5):
            perfil_nuevo.registrar_sesion(0.5)
        assert perfil_nuevo.dominado is True

    def test_no_dominio_con_loss_alta(self, perfil_nuevo):
        """No debe marcarse como dominado si alguna sesión tiene loss >= 0.8."""
        for l in [0.5, 0.5, 0.5, 0.5, 1.2]:
            perfil_nuevo.registrar_sesion(l)
        assert perfil_nuevo.dominado is False

    def test_tasa_mejora_positiva(self, perfil_nuevo):
        """La tasa de mejora es positiva cuando la loss baja."""
        for l in [3.0, 2.5, 2.0, 1.5, 1.0]:
            perfil_nuevo.registrar_sesion(l)
        assert perfil_nuevo.tasa_mejora > 0  # Mejorando

    def test_tasa_mejora_negativa(self, perfil_nuevo):
        """La tasa de mejora es negativa cuando la loss sube."""
        for l in [1.0, 1.5, 2.0, 2.5, 3.0]:
            perfil_nuevo.registrar_sesion(l)
        assert perfil_nuevo.tasa_mejora < 0  # Empeorando

    def test_serializacion_round_trip(self, perfil_nuevo):
        """to_dict / from_dict no pierde información."""
        perfil_nuevo.registrar_sesion(2.0)
        perfil_nuevo.registrar_sesion(1.5)
        d = perfil_nuevo.to_dict()
        restaurado = PerfilTema.from_dict(d)

        assert restaurado.nombre == perfil_nuevo.nombre
        assert restaurado.n_sesiones == perfil_nuevo.n_sesiones
        assert abs(restaurado.loss_media - perfil_nuevo.loss_media) < 0.01
        assert restaurado.dominado == perfil_nuevo.dominado


# =============================================================================
# TESTS: Función de curiosidad (zona de desarrollo próximo)
# =============================================================================

class TestCuriosidadZonaProximal:
    def test_tema_dominado_tiene_curiosidad_baja(self, motor):
        """Un tema con loss muy baja (ya dominado) tiene poca curiosidad."""
        tema = motor.temas["variables"]
        for _ in range(5):
            tema.registrar_sesion(0.3)  # Bien dominado

        curiosidad = motor.calcular_curiosidad(tema)
        assert curiosidad < 0.3

    def test_tema_en_zona_optima_tiene_curiosidad_alta(self, motor):
        """Un tema con loss ~1.5 (zona óptima) tiene curiosidad máxima."""
        tema = motor.temas["listas"]
        # 6 sesiones para salir del early exploration path (n_sesiones <= 5)
        for _ in range(6):
            tema.registrar_sesion(1.5)  # Zona exacta

        curiosidad = motor.calcular_curiosidad(tema)
        assert curiosidad > 0.5

    def test_tema_imposible_tiene_curiosidad_media(self, motor):
        """Un tema con loss muy alta (imposible) tiene curiosidad reducida."""
        tema = motor.temas["grafos"]
        tema.registrar_sesion(8.0)  # Demasiado difícil

        curiosidad = motor.calcular_curiosidad(tema)
        assert curiosidad < 0.4

    def test_tema_nunca_visto_tiene_curiosidad_alta(self):
        """Un tema sin sesiones tiene novedad máxima."""
        m = MotorCuriosidad(nivel_actual=3)
        m.registrar_tema("nuevo_tema", "test", 2)
        tema = m.temas["nuevo_tema"]
        curiosidad = m.calcular_curiosidad(tema)
        # Tema nuevo: alta curiosidad por novedad
        assert curiosidad > 0.1

    def test_nivel_superior_reduce_curiosidad(self, motor):
        """Un tema de nivel >> nivel_actual tiene curiosidad penalizada."""
        motor.nivel_actual = 1
        tema_dificil = motor.temas["grafos"]  # nivel 5
        # 6 sesiones para salir del early exploration path (n_sesiones <= 5)
        for _ in range(6):
            tema_dificil.registrar_sesion(2.0)  # Loss óptima, pero nivel 5

        tema_facil = motor.temas["variables"]  # nivel 1
        for _ in range(6):
            tema_facil.registrar_sesion(2.0)  # Misma loss

        c_dificil = motor.calcular_curiosidad(tema_dificil)
        c_facil = motor.calcular_curiosidad(tema_facil)

        assert c_facil > c_dificil  # Nivel apropiado priorizado


# =============================================================================
# TESTS: Selección de temas
# =============================================================================

class TestSeleccionTemas:
    def test_siguiente_tema_devuelve_string(self, motor):
        """siguiente_tema() siempre devuelve un nombre válido."""
        tema = motor.siguiente_tema()
        assert tema in motor.temas

    def test_siguiente_tema_no_repite_inmediatamente(self):
        """siguiente_tema() evita repetir los últimos 5 temas."""
        m = MotorCuriosidad(nivel_actual=1)
        for i in range(10):
            m.registrar_tema(f"tema_{i}", "test", 1)

        vistos = set()
        for _ in range(10):
            t = m.siguiente_tema()
            vistos.add(t)

        # Con 10 temas y cola de 5, debe rotar
        assert len(vistos) > 2

    def test_tops_devuelve_lista_ordenada(self, motor):
        """tops() devuelve los temas con mayor curiosidad en orden."""
        motor.actualizar_todos()
        tops = motor.tops(3)
        assert len(tops) == 3
        nombres = [n for n, _ in tops]
        scores = [s for _, s in tops]

        # Ordenado descendente
        assert scores[0] >= scores[1] >= scores[2]

    def test_sin_temas_devuelve_none(self):
        """Con motor vacío, siguiente_tema() retorna None."""
        m = MotorCuriosidad()
        resultado = m.siguiente_tema()
        assert resultado is None


# =============================================================================
# TESTS: Retroalimentación y avance de nivel
# =============================================================================

class TestRetroalimentacion:
    def test_retroalimentar_actualiza_perfil(self, motor):
        """retroalimentar() actualiza la loss del tema."""
        info = motor.retroalimentar("listas", 2.0)
        assert info["tema"] == "listas"
        assert abs(info["loss_actual"] - 2.0) < 0.01

    def test_tema_inexistente_retorna_dict_vacio(self, motor):
        """No falla si el tema no existe."""
        info = motor.retroalimentar("tema_que_no_existe", 1.0)
        assert info == {}

    def test_detecta_recien_dominado(self, motor):
        """Informa cuando un tema se acaba de dominar."""
        for _ in range(4):
            motor.retroalimentar("variables", 0.5)

        info = motor.retroalimentar("variables", 0.4)
        assert info["recien_dominado"] is True
        assert info["dominado"] is True

    def test_avanza_nivel_con_70_por_ciento(self):
        """Sube de nivel al dominar 70% de los temas del nivel actual."""
        m = MotorCuriosidad(nivel_actual=1)
        m.registrar_tema("tema_a", "test", 1)
        m.registrar_tema("tema_b", "test", 1)
        m.registrar_tema("tema_c", "test", 1)

        # Dominar tema_a y tema_b (67%): no debe subir
        for _ in range(5):
            m.retroalimentar("tema_a", 0.4)
            m.retroalimentar("tema_b", 0.4)

        # Aún en nivel 1 (solo 67%)
        assert m.nivel_actual == 1

        # Dominar tema_c también (100% > 70%): debe subir
        for _ in range(5):
            m.retroalimentar("tema_c", 0.3)

        assert m.nivel_actual == 2


# =============================================================================
# TESTS: Registro desde índice
# =============================================================================

class TestRegistroDesdeIndice:
    def test_carga_todos_los_temas(self):
        """registrar_temas_desde_indice() registra correctamente."""
        m = MotorCuriosidad()
        n = m.registrar_temas_desde_indice(INDICE_TEST)
        assert n == 5
        assert "variables" in m.temas
        assert "grafos" in m.temas
        assert "recursion" in m.temas

    def test_no_duplica_temas(self):
        """Llamar dos veces no duplica."""
        m = MotorCuriosidad()
        m.registrar_temas_desde_indice(INDICE_TEST)
        n2 = m.registrar_temas_desde_indice(INDICE_TEST)
        assert n2 == 0  # No hay nuevos
        assert len(m.temas) == 5

    def test_nivel_correcto(self):
        """El nivel de dificultad se asigna correctamente."""
        m = MotorCuriosidad()
        m.registrar_temas_desde_indice(INDICE_TEST)
        assert m.temas["variables"].nivel_dificultad == 1
        assert m.temas["grafos"].nivel_dificultad == 5


# =============================================================================
# TESTS: Persistencia
# =============================================================================

class TestPersistencia:
    def test_guardar_y_cargar(self, motor):
        """Guardar y cargar preserva el estado completo."""
        motor.retroalimentar("listas", 1.5)
        motor.retroalimentar("variables", 0.5)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            ruta = Path(f.name)

        motor.guardar(ruta)
        assert ruta.exists()

        motor2 = MotorCuriosidad()
        motor2.cargar(ruta)

        assert motor2.nivel_actual == motor.nivel_actual
        assert motor2.sesiones_totales == motor.sesiones_totales
        assert "listas" in motor2.temas
        assert abs(motor2.temas["listas"].loss_media - motor.temas["listas"].loss_media) < 0.01

        ruta.unlink()

    def test_cargar_archivo_inexistente_no_falla(self):
        """Cargar un archivo que no existe es silencioso."""
        m = MotorCuriosidad()
        m.cargar(Path("/tmp/estado_que_no_existe_99999.json"))
        # Sin excepción, motor vacío
        assert len(m.temas) == 0


# =============================================================================
# TESTS: Resumen
# =============================================================================

class TestResumen:
    def test_resumen_con_temas_dominados(self, motor):
        """resumen() reporta correctamente los dominados."""
        for _ in range(5):
            motor.retroalimentar("variables", 0.4)
            motor.retroalimentar("listas", 0.3)

        r = motor.resumen()
        assert r["temas_dominados"] >= 2
        assert r["porcentaje_dominio"] > 0
        assert "tops_curiosidad" in r
        assert r["temas_total"] == 4

    def test_repr(self, motor):
        """__repr__ no falla."""
        s = repr(motor)
        assert "MotorCuriosidad" in s
        assert "nivel=" in s
