#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
test_inference_integration.py — Prueba fehaciente del servidor pampar.inference.

Lanza el servidor como subprocess real, carga el checkpoint, y verifica que:
  1. El proceso arranca y emite READY
  2. Responde a una petición de inferencia con código Python real
  3. El código generado al menos tiene sintaxis válida
  4. Responde a una petición de boot con AGENTS.md bien formado
  5. Maneja JSON inválido sin caer

Uso:
  python scripts/test_inference_integration.py
  python scripts/test_inference_integration.py --checkpoint checkpoints/v3_sft_v8.pt
  python scripts/test_inference_integration.py --device cpu --timeout 60
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "v3_sft_v8.pt"
VENV_PYTHON = PROJECT_ROOT.parent / ".venv" / "Scripts" / "python.exe"


def _python_has_torch(python_path: str) -> bool:
    """Verifica rápido si ese intérprete tiene torch disponible."""
    try:
        result = subprocess.run(
            [python_path, "-c", "import torch"],
            capture_output=True, timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def find_python() -> str:
    """Devuelve el primer intérprete Python que tenga torch instalado."""
    candidates = [
        str(VENV_PYTHON) if VENV_PYTHON.exists() else None,
        sys.executable,
    ]
    for candidate in candidates:
        if candidate and _python_has_torch(candidate):
            return candidate
    # Último recurso: sys.executable aunque no tenga torch (el servidor fallará con mensaje claro)
    return sys.executable


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def ok(msg: str) -> None:
    print(f"  ✅  {msg}")

def fail(msg: str) -> None:
    print(f"  ❌  {msg}")

def section(title: str) -> None:
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")


# ---------------------------------------------------------------------------
# Server wrapper
# ---------------------------------------------------------------------------

class InferenceServer:
    """Prozess-wrapper para pampar.inference."""

    def __init__(self, checkpoint: str, device: str, timeout_ready: int = 90):
        python = find_python()
        cmd = [python, "-m", "pampar.inference", "--checkpoint", checkpoint, "--device", device]
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._wait_ready(timeout_ready)

    def _wait_ready(self, timeout: int) -> None:
        print(f"  Esperando READY (timeout={timeout}s)…", end="", flush=True)
        deadline = time.time() + timeout
        stderr_lines: list[str] = []
        while time.time() < deadline:
            line = self.proc.stderr.readline()
            if not line:
                if self.proc.poll() is not None:
                    raise RuntimeError(
                        f"Proceso terminó antes de READY (código {self.proc.returncode}).\n"
                        + "".join(stderr_lines)
                    )
                time.sleep(0.1)
                continue
            stderr_lines.append(line)
            print(".", end="", flush=True)
            if "READY" in line:
                print(" listo.")
                # Leer también la línea {"type":"ready"} de stdout
                self.proc.stdout.readline()
                return
        raise TimeoutError("El servidor no emitió READY a tiempo.\n" + "".join(stderr_lines[-20:]))

    def send(self, msg: dict) -> dict:
        line = json.dumps(msg, ensure_ascii=False) + "\n"
        self.proc.stdin.write(line)
        self.proc.stdin.flush()
        resp_line = self.proc.stdout.readline()
        if not resp_line:
            raise EOFError("El servidor cerró stdout inesperadamente.")
        return json.loads(resp_line)

    def send_raw(self, raw: str) -> str:
        self.proc.stdin.write(raw + "\n")
        self.proc.stdin.flush()
        return self.proc.stdout.readline()

    def close(self) -> None:
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        self.proc.wait(timeout=5)


# ---------------------------------------------------------------------------
# Casos de prueba
# ---------------------------------------------------------------------------

PRUEBA_INFER = {
    "type": "infer",
    "prompt": (
        "### Problem:\n"
        "Write a Python function `suma(a, b)` that returns the sum of two numbers.\n"
        "### Solution:\n"
    ),
    "max_tokens": 80,
    "temperature": 0.1,
}

PRUEBA_INFER_ALGO_REAL = {
    "type": "infer",
    "prompt": (
        "### Problem:\n"
        "Write a Python function `es_par(n)` that returns True if n is even, False otherwise.\n"
        "### Solution:\n"
    ),
    "max_tokens": 60,
    "temperature": 0.1,
}


def run_tests(checkpoint: str, device: str, timeout: int) -> int:
    """Ejecuta todos los tests. Devuelve número de fallos."""
    fallos = 0

    section("Iniciando servidor de inferencia")
    try:
        server = InferenceServer(checkpoint=checkpoint, device=device, timeout_ready=timeout)
    except Exception as exc:
        fail(f"No se pudo iniciar el servidor: {exc}")
        return 1

    # ------------------------------------------------------------------
    # Test 1: responde a inferencia básica
    # ------------------------------------------------------------------
    section("Test 1: Respuesta a petición de inferencia")
    try:
        resp = server.send(PRUEBA_INFER)
        if resp.get("type") == "infer_ok":
            texto = resp.get("text", "").strip()
            if texto:
                ok(f"Texto generado ({len(texto)} chars): {repr(texto[:80])}")
            else:
                fail("infer_ok pero text está vacío")
                fallos += 1
        else:
            fail(f"Respuesta inesperada: {resp}")
            fallos += 1
    except Exception as exc:
        fail(f"Excepción: {exc}")
        fallos += 1

    # ------------------------------------------------------------------
    # Test 2: sintaxis válida en el código generado
    # ------------------------------------------------------------------
    section("Test 2: Código generado tiene sintaxis Python válida")
    try:
        resp = server.send(PRUEBA_INFER_ALGO_REAL)
        texto = resp.get("text", "")
        # Limpiamos markdown si lo hay
        texto_limpio = texto.replace("```python", "").replace("```", "").strip()
        try:
            ast.parse(texto_limpio)
            ok(f"Sintaxis válida: {repr(texto_limpio[:100])}")
        except SyntaxError as se:
            fail(f"SyntaxError en código generado: {se}\nCódigo: {repr(texto_limpio[:200])}")
            fallos += 1
    except Exception as exc:
        fail(f"Excepción: {exc}")
        fallos += 1

    # ------------------------------------------------------------------
    # Test 3: boot genera AGENTS.md
    # ------------------------------------------------------------------
    section("Test 3: Boot genera AGENTS.md")
    try:
        resp = server.send({"type": "boot", "workspace": str(PROJECT_ROOT)})
        if resp.get("type") == "boot_ok":
            md = resp.get("agents_md", "")
            checks = [
                ("## Quick Reference" in md, "Sección Quick Reference"),
                ("## Boot protocol"    in md, "Sección Boot protocol"),
                (len(md) > 200,              "Contenido suficiente (>200 chars)"),
            ]
            for passed, label in checks:
                if passed:
                    ok(label)
                else:
                    fail(label)
                    fallos += 1
        else:
            fail(f"Respuesta inesperada: {resp}")
            fallos += 1
    except Exception as exc:
        fail(f"Excepción: {exc}")
        fallos += 1

    # ------------------------------------------------------------------
    # Test 4: prompt vacío → error (no crash)
    # ------------------------------------------------------------------
    section("Test 4: Prompt vacío → error controlado")
    try:
        resp = server.send({"type": "infer", "prompt": ""})
        if resp.get("type") == "error":
            ok(f"Error controlado: {resp.get('message', '')[:60]}")
        else:
            fail(f"Se esperaba error, se recibió: {resp}")
            fallos += 1
    except Exception as exc:
        fail(f"Excepción: {exc}")
        fallos += 1

    # ------------------------------------------------------------------
    # Test 5: JSON inválido no baja el servidor
    # ------------------------------------------------------------------
    section("Test 5: JSON inválido no mata el servidor")
    try:
        server.send_raw("esto no es json {{{{")
        # Leer error report
        raw = server.proc.stdout.readline()
        if raw.strip():
            resp = json.loads(raw)
            if resp.get("type") == "error":
                ok("El servidor respondió error y sigue vivo")
            else:
                ok(f"El servidor respondió (tipo={resp.get('type')}) y sigue vivo")
        else:
            fail("No hubo respuesta al JSON inválido")
            fallos += 1
        # Verificar que el servidor sigue respondiendo
        resp2 = server.send({"type": "infer", "prompt": "### Problem:\nhi\n### Solution:\n", "max_tokens": 20})
        if resp2.get("type") == "infer_ok":
            ok("Servidor sigue respondiendo después del JSON inválido")
        else:
            fail(f"El servidor dejó de responder: {resp2}")
            fallos += 1
    except Exception as exc:
        fail(f"Excepción: {exc}")
        fallos += 1

    # ------------------------------------------------------------------
    # Resumen
    # ------------------------------------------------------------------
    server.close()
    section("RESUMEN")
    if fallos == 0:
        print(f"  ✅  TODOS LOS TESTS PASARON (5/5)")
    else:
        print(f"  ❌  {fallos} test(s) fallaron")

    return fallos


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Test de integración del servidor pampar.inference")
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help=f"Ruta al checkpoint .pt (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"],
        help="Dispositivo de inferencia"
    )
    parser.add_argument(
        "--timeout", type=int, default=90,
        help="Segundos máximos para esperar READY del servidor"
    )
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"ERROR: Checkpoint no encontrado: {args.checkpoint}")
        print(f"       Checkpoints disponibles:")
        for pt in sorted((PROJECT_ROOT / "checkpoints").glob("*.pt")):
            print(f"         {pt.name}")
        sys.exit(1)

    fallos = run_tests(args.checkpoint, args.device, args.timeout)
    sys.exit(0 if fallos == 0 else 1)


if __name__ == "__main__":
    main()
