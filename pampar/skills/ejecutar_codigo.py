# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
EjecutorCodigo — Las "manos" del modelo.

Ejecuta código en un subproceso aislado con timeout.
Soporta Python, JavaScript (Node.js) y Bash.
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

import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Optional

from .base import ResultadoSkill, Skill

# Lenguajes soportados y sus extensiones
_LANG_CONFIG = {
    "python": {"ext": ".py", "suffix": "pampar_exec_"},
    "javascript": {"ext": ".js", "suffix": "pampar_exec_"},
    "bash": {"ext": ".sh", "suffix": "pampar_exec_"},
}

# Alias de lenguaje → nombre canónico
_LANG_ALIASES = {
    "python": "python",
    "py": "python",
    "python3": "python",
    "javascript": "javascript",
    "js": "javascript",
    "node": "javascript",
    "bash": "bash",
    "sh": "bash",
    "shell": "bash",
}


class EjecutorCodigo(Skill):
    """
    Ejecuta fragmentos de código y retorna el output.

    Soporta Python, JavaScript (Node.js) y Bash.
    Detecta el lenguaje automáticamente o acepta un hint explícito.

    Args:
        timeout:    Segundos máximos de ejecución (default 10)
        cwd:        Directorio de trabajo para la ejecución
        python_bin: Intérprete Python a usar (default sys.executable)
        node_bin:   Intérprete Node.js (auto-detectado si disponible)
        bash_bin:   Intérprete Bash (auto-detectado si disponible)
    """

    name = "ejecutar_codigo"
    description = (
        "Ejecuta un fragmento de código (Python, JavaScript o Bash) y retorna "
        "el output y errores. Úsalo para verificar que el código funciona."
    )

    # Prefijos de operaciones bloqueadas (seguridad básica)
    _BLOQUEADOS = frozenset(
        {
            "os.system",
            "subprocess.Popen",
            "subprocess.run",
            "shutil.rmtree",
            "__import__('os').remove",
            "open('/etc",
            "open('/root",
            "open('/home",
            "child_process.exec",
            "child_process.spawn",  # Node.js
            "require('child_process')",
            "rm -rf /",
            "mkfs.",
            "dd if=",  # Bash
        }
    )

    def __init__(
        self,
        timeout: int = 10,
        cwd: Optional[str] = None,
        python_bin: str = sys.executable,
        node_bin: Optional[str] = None,
        bash_bin: Optional[str] = None,
    ):
        self.timeout = timeout
        self.cwd = cwd
        self.python_bin = python_bin
        self.node_bin = node_bin or shutil.which("node")
        self.bash_bin = bash_bin or shutil.which("bash") or shutil.which("sh")

    def _detect_language(self, codigo: str) -> str:
        """Detecta el lenguaje del código por heurística."""
        first_line = codigo.strip().split("\n")[0].strip()

        # Shebang detection
        if first_line.startswith("#!"):
            if "python" in first_line:
                return "python"
            if "node" in first_line:
                return "javascript"
            if "bash" in first_line or "sh" in first_line:
                return "bash"

        # Keyword heuristics
        js_signals = {
            "const ",
            "let ",
            "var ",
            "function ",
            "=> ",
            "console.log",
            "require(",
            "import {",
            "export ",
        }
        bash_signals = {
            "#!/bin",
            "echo ",
            "fi\n",
            "done\n",
            "esac\n",
            "if [",
            "then\n",
            "$((",
            "${",
        }
        py_signals = {"def ", "import ", "from ", "class ", "print(", "self."}

        js_score = sum(1 for s in js_signals if s in codigo)
        bash_score = sum(1 for s in bash_signals if s in codigo)
        py_score = sum(1 for s in py_signals if s in codigo)

        if js_score > py_score and js_score > bash_score:
            return "javascript"
        if bash_score > py_score and bash_score > js_score:
            return "bash"
        return "python"

    def _get_interpreter(self, lang: str) -> list[str]:
        """Retorna el comando del intérprete para el lenguaje dado."""
        if lang == "python":
            return [self.python_bin]
        if lang == "javascript":
            if not self.node_bin:
                raise RuntimeError(
                    "Node.js no encontrado. Instalar node para ejecutar JavaScript."
                )
            return [self.node_bin]
        if lang == "bash":
            if not self.bash_bin:
                raise RuntimeError(
                    "Bash no encontrado. Instalar bash/sh para ejecutar shell scripts."
                )
            return [self.bash_bin]
        raise ValueError(f"Lenguaje no soportado: {lang}")

    def execute(
        self,
        codigo: str,
        timeout: Optional[int] = None,
        lang: Optional[str] = None,
    ) -> ResultadoSkill:
        """
        Ejecuta el código dado y captura stdout/stderr.

        Args:
            codigo:  Código a ejecutar
            timeout: Override del timeout (None = usar el default)
            lang:    Lenguaje ("python", "javascript", "bash").
                     None = auto-detect.
        Returns:
            ResultadoSkill con stdout, stderr y código de retorno
        """
        t = timeout or self.timeout
        codigo = textwrap.dedent(codigo).strip()

        if not codigo:
            return ResultadoSkill(exito=False, contenido="", error="Código vacío")

        # Resolver lenguaje
        lang_key = _LANG_ALIASES.get(lang, lang) if lang else None
        if not lang_key:
            lang_key = self._detect_language(codigo)
        if lang_key not in _LANG_CONFIG:
            return ResultadoSkill(
                exito=False,
                contenido="",
                error=f"Lenguaje no soportado: {lang_key}. Usar: python, javascript, bash",
            )

        # Verificar intérprete disponible
        try:
            interpreter = self._get_interpreter(lang_key)
        except RuntimeError as e:
            return ResultadoSkill(exito=False, contenido="", error=str(e))

        # Chequeo básico de operaciones bloqueadas
        for bloqueado in self._BLOQUEADOS:
            if bloqueado in codigo:
                return ResultadoSkill(
                    exito=False,
                    contenido="",
                    error=f"Operación no permitida detectada: '{bloqueado}'",
                )

        # Escribir código en archivo temporal
        cfg = _LANG_CONFIG[lang_key]
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=cfg["ext"],
            delete=False,
            encoding="utf-8",
            prefix=cfg["suffix"],
        ) as tmp:
            tmp.write(codigo)
            tmp_path = tmp.name

        try:
            result = subprocess.run(
                [*interpreter, tmp_path],
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
