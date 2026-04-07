# SPDX-License-Identifier: BUSL-1.1
"""
Tests de las skills de PAMPAr-Coder v3.

Cubren:
  LectorArchivos:
    - Leer archivo existente → exito=True, contenido incluye nombre
    - Leer fuera del workspace → exito=False, "Acceso denegado"
    - Leer archivo inexistente → exito=False
    - Extensión no permitida → exito=False
    - Listar directorio → exito=True, contiene items
    - Rango de líneas funciona
    - buscar_en_workspace encuentra el patrón

  EjecutorCodigo:
    - print simple → exito=True, stdout correcto
    - Código vacío → exito=False
    - Error de sintaxis → exito=False, stderr con traceback
    - Excepción en runtime → exito=False
    - Timeout → exito=False, "TIMEOUT" en contenido
    - Operación bloqueada → exito=False, "no permitida"
    - ejecutar_tests con test que pasa → exito=True
"""

import textwrap
from pathlib import Path

import pytest

from pampar.skills.lector_archivos import LectorArchivos
from pampar.skills.ejecutar_codigo import EjecutorCodigo
from pampar.skills.base import ResultadoSkill


# ==============================================================================
# HELPERS LOCALES
# ==============================================================================

def _crear_archivo(directorio: Path, nombre: str, contenido: str) -> Path:
    """Crea un archivo de texto en el directorio dado y retorna su Path."""
    ruta = directorio / nombre
    ruta.write_text(contenido, encoding="utf-8")
    return ruta


# ==============================================================================
# TESTS: LectorArchivos
# ==============================================================================

class TestLectorArchivos:
    @pytest.fixture
    def workspace(self, tmp_path: Path) -> Path:
        """Directorio workspace temporal con algunos archivos de prueba."""
        (tmp_path / "subdir").mkdir()
        _crear_archivo(tmp_path, "hola.py", "def hola():\n    return 42\n")
        _crear_archivo(tmp_path, "config.json", '{"key": "value"}')
        _crear_archivo(tmp_path / "subdir", "inner.py", "x = 1\ny = 2\nz = 3\n")
        return tmp_path

    @pytest.fixture
    def lector(self, workspace: Path) -> LectorArchivos:
        return LectorArchivos(workspace_root=str(workspace))

    def test_leer_archivo_existente(self, lector: LectorArchivos):
        """Leer un .py existente debe retornar exito=True con su contenido."""
        resultado = lector.execute("hola.py")
        assert resultado.exito is True
        assert "hola.py" in resultado.contenido
        assert "def hola" in resultado.contenido

    def test_leer_archivo_incluye_numeros_linea(self, lector: LectorArchivos):
        """El resultado debe incluir la info de líneas totales."""
        resultado = lector.execute("hola.py")
        assert resultado.exito is True
        assert resultado.datos["total_lineas"] >= 1

    def test_leer_archivo_rango_lineas(self, lector: LectorArchivos):
        """Con linea_inicio y linea_fin solo se leen las líneas del rango."""
        resultado = lector.execute("subdir/inner.py", linea_inicio=1, linea_fin=1)
        assert resultado.exito is True
        assert "x = 1" in resultado.contenido
        assert "y = 2" not in resultado.contenido

    def test_leer_extension_no_permitida(self, workspace: Path):
        """Archivos con extensión no en la whitelist deben rechazarse."""
        _crear_archivo(workspace, "binario.bin", "\x00\x01\x02")
        lector = LectorArchivos(workspace_root=str(workspace))
        resultado = lector.execute("binario.bin")
        assert resultado.exito is False
        assert "no permitida" in resultado.error.lower() or "extensión" in resultado.error.lower()

    def test_leer_archivo_inexistente(self, lector: LectorArchivos):
        """Leer un archivo que no existe debe retornar exito=False."""
        resultado = lector.execute("no_existe.py")
        assert resultado.exito is False
        assert resultado.error != ""

    def test_acceso_fuera_workspace_bloqueado(self, lector: LectorArchivos):
        """Intentar salir del workspace con path traversal debe fallar."""
        resultado = lector.execute("../../etc/passwd")
        assert resultado.exito is False
        assert "Acceso denegado" in resultado.error or resultado.error != ""

    def test_listar_directorio(self, lector: LectorArchivos):
        """Listar un directorio debe retornar exito=True con lista de items."""
        resultado = lector.execute(".")
        assert resultado.exito is True
        assert "[DIRECTORIO" in resultado.contenido

    def test_listar_subdirectorio(self, lector: LectorArchivos):
        """Listar un subdirectorio debe mostrar sus archivos."""
        resultado = lector.execute("subdir")
        assert resultado.exito is True
        assert "inner.py" in resultado.contenido

    def test_buscar_patron_existente(self, lector: LectorArchivos):
        """buscar_en_workspace debe encontrar el patrón en archivos .py."""
        resultado = lector.buscar_en_workspace("def hola", extension=".py")
        assert resultado.exito is True
        assert "hola.py" in resultado.contenido

    def test_buscar_patron_inexistente(self, lector: LectorArchivos):
        """Buscar un patrón que no existe debe retornar exito=True pero sin matches."""
        resultado = lector.buscar_en_workspace("__NO_EXISTE_NUNCA__", extension=".py")
        assert resultado.exito is True
        # El contenido debe indicar que no hay resultados
        assert "No se encontraron" in resultado.contenido or resultado.datos == {}

    def test_leer_json(self, lector: LectorArchivos):
        """Archivos .json también deben ser legibles."""
        resultado = lector.execute("config.json")
        assert resultado.exito is True
        assert "key" in resultado.contenido

    def test_resultado_es_resultadoskill(self, lector: LectorArchivos):
        """El return siempre debe ser una instancia de ResultadoSkill."""
        resultado = lector.execute("hola.py")
        assert isinstance(resultado, ResultadoSkill)


# ==============================================================================
# TESTS: EjecutorCodigo
# ==============================================================================

class TestEjecutorCodigo:
    @pytest.fixture
    def ejecutor(self) -> EjecutorCodigo:
        """Ejecutor con timeout corto para que los tests de timeout sean rápidos."""
        return EjecutorCodigo(timeout=5)

    def test_print_simple(self, ejecutor: EjecutorCodigo):
        """Un print básico debe retornar exito=True y el stdout correcto."""
        resultado = ejecutor.execute("print(42)")
        assert resultado.exito is True
        assert "42" in resultado.contenido
        assert resultado.datos["stdout"] == "42"

    def test_stdout_multilinea(self, ejecutor: EjecutorCodigo):
        """Múltiples prints deben aparecer en el stdout."""
        codigo = "for i in range(3):\n    print(i)"
        resultado = ejecutor.execute(codigo)
        assert resultado.exito is True
        assert "0" in resultado.contenido
        assert "1" in resultado.contenido
        assert "2" in resultado.contenido

    def test_codigo_vacio_falla(self, ejecutor: EjecutorCodigo):
        """Código vacío debe retornar exito=False."""
        resultado = ejecutor.execute("")
        assert resultado.exito is False
        assert resultado.error != ""

    def test_error_sintaxis(self, ejecutor: EjecutorCodigo):
        """Código con error de sintaxis debe dar exito=False con traceback."""
        resultado = ejecutor.execute("def f(\n    pass")
        assert resultado.exito is False
        # El stderr debe contener algún mensaje de error
        assert resultado.error != "" or "STDERR" in resultado.contenido or "Error" in resultado.contenido

    def test_excepcion_en_runtime(self, ejecutor: EjecutorCodigo):
        """Una excepción en runtime debe dar exito=False."""
        resultado = ejecutor.execute("raise ValueError('test error')")
        assert resultado.exito is False
        assert resultado.datos["returncode"] != 0

    def test_returncode_cero_en_exito(self, ejecutor: EjecutorCodigo):
        """Un script exitoso debe tener returncode=0."""
        resultado = ejecutor.execute("x = 1 + 1")
        assert resultado.exito is True
        assert resultado.datos["returncode"] == 0

    def test_timeout_corto(self, ejecutor: EjecutorCodigo):
        """Código que excede el timeout debe retornar exito=False con indicación de TIMEOUT."""
        # Un sleep muy largo para forzar timeout con timeout=1
        resultado = ejecutor.execute("import time; time.sleep(60)", timeout=1)
        assert resultado.exito is False
        assert "TIMEOUT" in resultado.contenido or "timeout" in resultado.error.lower()

    def test_operacion_bloqueada_detectada(self, ejecutor: EjecutorCodigo):
        """Código que usa os.system debe rechazarse antes de ejecutar."""
        resultado = ejecutor.execute("import os; os.system('echo hacked')")
        assert resultado.exito is False
        assert "no permitida" in resultado.error.lower() or "bloqueada" in resultado.error.lower()

    def test_sin_output_reporta_sin_output(self, ejecutor: EjecutorCodigo):
        """Código que no imprime nada debe reportar [Sin output]."""
        resultado = ejecutor.execute("x = 1 + 1  # sin print")
        assert resultado.exito is True
        assert "Sin output" in resultado.contenido

    def test_calculos_correctos(self, ejecutor: EjecutorCodigo):
        """Verificar que el resultado de un cálculo es correcto."""
        resultado = ejecutor.execute("print(2 ** 10)")
        assert resultado.exito is True
        assert "1024" in resultado.contenido

    def test_codigo_con_imports(self, ejecutor: EjecutorCodigo):
        """Código que importa stdlib debe funcionar correctamente."""
        codigo = textwrap.dedent("""
            import math
            print(round(math.pi, 4))
        """)
        resultado = ejecutor.execute(codigo)
        assert resultado.exito is True
        assert "3.1416" in resultado.contenido

    def test_resultado_es_resultadoskill(self, ejecutor: EjecutorCodigo):
        """El return siempre debe ser una instancia de ResultadoSkill."""
        resultado = ejecutor.execute("print('ok')")
        assert isinstance(resultado, ResultadoSkill)

    def test_ejecutar_tests_con_test_simple(self, ejecutor: EjecutorCodigo, tmp_path: Path):
        """ejecutar_tests() con un test que pasa debe retornar exito=True."""
        test_file = tmp_path / "test_simple.py"
        test_file.write_text(
            "def test_suma():\n    assert 1 + 1 == 2\n",
            encoding="utf-8",
        )
        resultado = ejecutor.ejecutar_tests(str(test_file))
        assert resultado.exito is True
        assert "passed" in resultado.contenido.lower()

    def test_ejecutar_tests_con_test_fallido(self, ejecutor: EjecutorCodigo, tmp_path: Path):
        """ejecutar_tests() con un test que falla debe retornar exito=False."""
        test_file = tmp_path / "test_falla.py"
        test_file.write_text(
            "def test_que_falla():\n    assert 1 == 2\n",
            encoding="utf-8",
        )
        resultado = ejecutor.ejecutar_tests(str(test_file))
        assert resultado.exito is False
