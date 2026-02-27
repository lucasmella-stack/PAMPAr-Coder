# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
EjecutorCodigo — Las "manos" del modelo.

Ejecuta código Python en un subproceso aislado con timeout.
El modelo puede ver el output y los errores para razonar sobre ellos
e iterar (detectar bug → ver traceback → corregir → re-ejecutar).

Seguridad:
  - Corre en subproceso separado (no en el mismo proceso del modelo)
  - Timeout configurable (default 10s) — evita loops infinitos
  - stdin cerrado — no puede recibir input
  - Directorio de trabajo aislado configurable

IMPORTANTE: No es un sandbox completo (sin contenedor Docker).
Para producción en entornos multiusuario, envolver con firejail/docker.
Para uso local personal esto es suficiente.
"""

import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Optional

from .base import ResultadoSkill, Skill


class EjecutorCodigo(Skill):
    """
    Ejecuta fragmentos de código Python y retorna el output.

    Args:
        timeout:    Segundos máximos de ejecución (default 10)
        cwd:        Directorio de trabajo para la ejecución
        python_bin: Intérprete Python a usar (default sys.executable)
    """

    name = "ejecutar_codigo"
    description = (
        "Ejecuta un fragmento de código Python y retorna el output y errores. "
        "Úsalo para verificar que el código funciona antes de entregárselo al usuario."
    )

    # Prefijos de módulos bloqueados (seguridad básica)
    _BLOQUEADOS = frozenset({
        "os.system", "subprocess.Popen", "subprocess.run",
        "shutil.rmtree", "__import__('os').remove",
        "open('/etc", "open('/root", "open('/home",
    })

    def __init__(
        self,
        timeout: int = 10,
        cwd: Optional[str] = None,
        python_bin: str = sys.executable,
    ):
        self.timeout = timeout
        self.cwd = cwd
        self.python_bin = python_bin

    def execute(
        self,
        codigo: str,
        timeout: Optional[int] = None,
    ) -> ResultadoSkill:
        """
        Ejecuta el código Python dado y captura stdout/stderr.

        Args:
            codigo:  Código Python a ejecutar
            timeout: Override del timeout (None = usar el default)
        Returns:
            ResultadoSkill con stdout, stderr y código de retorno
        """
        t = timeout or self.timeout
        codigo = textwrap.dedent(codigo).strip()

        if not codigo:
            return ResultadoSkill(
                exito=False, contenido="", error="Código vacío"
            )

        # Chequeo básico de operaciones bloqueadas
        for bloqueado in self._BLOQUEADOS:
            if bloqueado in codigo:
                return ResultadoSkill(
                    exito=False,
                    contenido="",
                    error=f"Operación no permitida detectada: '{bloqueado}'"
                )

        # Escribir código en archivo temporal
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8",
            prefix="pampar_exec_",
        ) as tmp:
            tmp.write(codigo)
            tmp_path = tmp.name

        try:
            result = subprocess.run(
                [self.python_bin, tmp_path],
                capture_output=True,
                text=True,
                timeout=t,
                stdin=subprocess.DEVNULL,  # No input
                cwd=self.cwd,
            )

            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            ret = result.returncode

            # Formatear output para el modelo
            partes = []
            if stdout:
                partes.append(f"[STDOUT]\n{stdout}")
            if stderr:
                label = "[STDERR/TRACEBACK]" if "Traceback" in stderr else "[STDERR]"
                partes.append(f"{label}\n{stderr}")
            if not stdout and not stderr:
                partes.append("[Sin output]")

            contenido = "\n\n".join(partes)
            if ret != 0:
                contenido += f"\n\n[Código de retorno: {ret}]"

            return ResultadoSkill(
                exito=(ret == 0),
                contenido=contenido,
                datos={
                    "returncode": ret,
                    "stdout": stdout,
                    "stderr": stderr,
                    "timeout_usado": t,
                },
                error=stderr if ret != 0 else "",
            )

        except subprocess.TimeoutExpired:
            return ResultadoSkill(
                exito=False,
                contenido=f"[TIMEOUT] El código no terminó en {t} segundos.",
                error=f"Timeout después de {t}s",
                datos={"timeout_segundos": t},
            )
        except Exception as e:
            return ResultadoSkill(
                exito=False,
                contenido="",
                error=f"Error al ejecutar: {e}",
            )
        finally:
            # Siempre limpiar el archivo temporal
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

    def ejecutar_tests(self, ruta_test: str) -> ResultadoSkill:
        """
        Ejecuta tests con pytest en la ruta dada.

        Args:
            ruta_test: Archivo o directorio de tests
        Returns:
            ResultadoSkill con output de pytest
        """
        try:
            result = subprocess.run(
                [self.python_bin, "-m", "pytest", ruta_test, "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=60,  # Tests pueden tardar más
                stdin=subprocess.DEVNULL,
                cwd=self.cwd,
            )

            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            exito = result.returncode == 0

            contenido = stdout
            if stderr:
                contenido += f"\n[STDERR]\n{stderr}"

            return ResultadoSkill(
                exito=exito,
                contenido=contenido,
                datos={"returncode": result.returncode},
                error="" if exito else "Algunos tests fallaron",
            )
        except subprocess.TimeoutExpired:
            return ResultadoSkill(
                exito=False,
                contenido="[TIMEOUT] Tests no terminaron en 60s",
                error="Timeout en tests",
            )
        except Exception as e:
            return ResultadoSkill(exito=False, contenido="", error=str(e))
