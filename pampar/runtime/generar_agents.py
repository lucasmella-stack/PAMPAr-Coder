# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
generar_agents.py — Generador determinista del AGENTS.md contextual.

Forma parte del protocolo de boot (Paso 3): dado el ResultadoScan del entorno,
genera el AGENTS.md actualizado con el contexto real del despliegue.

Uso desde el BootProtocol:
    from pampar.runtime.generar_agents import generar_agents_md
    md = generar_agents_md(scan_resultado, proyecto="mi-app")
    Path("AGENTS.md").write_text(md, encoding="utf-8")
"""

from __future__ import annotations

import datetime
from typing import Optional

from .scanner import ResultadoScan


def generar_agents_md(
    scan: ResultadoScan,
    proyecto: Optional[str] = None,
    descripcion: Optional[str] = None,
    agente_nombre: str = "PAMPAr",
) -> str:
    """
    Genera un AGENTS.md contextual desde el resultado del Scanner.

    Args:
        scan: Resultado del Scanner con info del entorno.
        proyecto: Nombre del proyecto (auto-detectado si es None).
        descripcion: Breve descripción del proyecto.
        agente_nombre: Nombre del agente (default: PAMPAr).

    Returns:
        Contenido del AGENTS.md como string.
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    sis = scan.sistema
    proyecto = proyecto or _detectar_proyecto(scan)
    descripcion = descripcion or _inferir_descripcion(scan)

    # ── Sección de cabecera ─────────────────────────────────────────────────
    lineas: list[str] = [
        f"# {agente_nombre} — Protocolo de Despliegue",
        "",
        f"> Generado automáticamente por el Scanner al boot — {now}",
        f"> Para la identidad invariante del modelo, ver `CONCIENCIA.md`.",
        "",
        "---",
        "",
    ]

    # ── Quick Reference ────────────────────────────────────────────────────
    gpu_row = sis.gpu if sis.gpu else "CPU only"
    voz_row = ", ".join(scan.voz) if scan.voz else "no disponible"

    lineas += [
        "## Quick Reference",
        "",
        "| Área            | Valor detectado                       |",
        "| --------------- | ------------------------------------- |",
        f"| Proyecto        | `{proyecto}`                          |",
        f"| OS              | {sis.os}                              |",
        f"| Python          | {sis.python_version}                  |",
        f"| GPU             | {gpu_row}                             |",
        f"| RAM             | {f'{sis.ram_gb:.1f} GB' if sis.ram_gb else 'desconocida'}                   |",
        f"| Voz             | {voz_row}                             |",
        "",
    ]

    # ── Sistema detectado ──────────────────────────────────────────────────
    lineas += [
        "## Sistema detectado",
        "",
        f"- **OS**: {sis.os}",
        f"- **Python**: {sis.python_version}",
    ]

    if sis.gpu:
        vram = f"{sis.vram_mb / 1024:.1f} GB" if sis.vram_mb else "desconocida"
        lineas.append(f"- **GPU**: {sis.gpu} \u2014 {vram} VRAM")
    else:
        lineas.append("- **GPU**: no disponible (solo CPU)")

    if sis.ram_gb:
        lineas += [
            f"- **RAM**: {sis.ram_gb:.1f} GB",
            "",
        ]
    else:
        lineas.append("")

    # ── Workspace ─────────────────────────────────────────────────────────
    if scan.archivos:
        # Agrupar por extensión
        ext_count: dict[str, int] = {}
        for a in scan.archivos:
            ext = a.ruta.rsplit(".", 1)[-1].lower() if "." in a.ruta else "sin ext"
            ext_count[ext] = ext_count.get(ext, 0) + 1

        total_files = len(scan.archivos)
        ext_str = ", ".join(
            f"{ext.upper()}: {n}"
            for ext, n in sorted(ext_count.items(), key=lambda x: -x[1])[:5]
        )

        total_funcs = sum(len(a.funciones) for a in scan.archivos)
        total_classes = sum(len(a.clases) for a in scan.archivos)

        lineas += [
            "## Workspace",
            "",
            f"- **Archivos**: {total_files} ({ext_str})",
            f"- **Funciones**: {total_funcs}",
            f"- **Clases**: {total_classes}",
            "",
        ]

    # ── Paquetes clave ────────────────────────────────────────────────────
    _PAQUETES_CLAVE = {
        "torch", "tensorflow", "keras",
        "fastapi", "flask", "django", "uvicorn", "starlette",
        "pandas", "numpy", "scipy", "matplotlib", "plotly", "seaborn",
        "transformers", "diffusers", "peft", "trl", "bitsandbytes",
        "sqlalchemy", "alembic", "psycopg2", "pymongo",
        "celery", "redis", "pika",
        "pytest", "vitest", "playwright",
        "sentencepiece", "tokenizers",
        "langchain", "openai", "anthropic",
        "docker", "kubernetes",
        "click", "typer", "rich",
        "pydantic", "fastapi",
        "httpx", "requests", "aiohttp",
    }

    paquetes_relevantes = {
        k: v for k, v in scan.paquetes.items()
        if any(clave in k.lower() for clave in _PAQUETES_CLAVE)
    }

    if paquetes_relevantes:
        lineas += ["## Paquetes clave", ""]
        for pkg, ver in sorted(paquetes_relevantes.items()):
            lineas.append(f"- `{pkg}=={ver}`")
        lineas.append("")

    # ── Servicios detectados ──────────────────────────────────────────────
    servicios_activos = [k for k, v in scan.servicios.items() if v]
    servicios_inactivos = [k for k, v in scan.servicios.items() if not v]

    lineas += ["## Servicios", ""]
    if servicios_activos:
        lineas.append(f"- **Activos**: {', '.join(servicios_activos)}")
    if servicios_inactivos:
        lineas.append(f"- **Inactivos**: {', '.join(servicios_inactivos)}")
    if not scan.servicios:
        lineas.append("- No se detectaron servicios de red activos")
    lineas.append("")

    # ── Boot protocol ─────────────────────────────────────────────────────
    lineas += [
        "## Boot protocol",
        "",
        "El agente ejecuta la siguiente secuencia al iniciar:",
        "",
        "```",
        "1. CONCIENCIA.md  →  identidad invariante  → RAG L3",
        "2. Scanner        →  inspección del entorno",
        "3. AGENTS.md      →  contexto del despliegue  → RAG L2",
        "4. Workspace      →  archivos y funciones      → RAG L1",
        "```",
        "",
    ]

    return "\n".join(lineas)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers privados
# ─────────────────────────────────────────────────────────────────────────────

def _detectar_proyecto(scan: ResultadoScan) -> str:
    """Intenta detectar el nombre del proyecto desde el workspace."""
    # Buscar pyproject.toml, setup.py, package.json
    nombres_clave = {
        "pyproject.toml": r'name\s*=\s*["\']([^"\']+)',
        "package.json": r'"name"\s*:\s*"([^"]+)"',
    }
    for archivo in scan.archivos:
        base = archivo.ruta.split("/")[-1].split("\\")[-1]
        if base in nombres_clave:
            return base  # fallback — nombre del archivo de config
    return "workspace"


def _inferir_descripcion(scan: ResultadoScan) -> str:
    """Infiere una descripción breve del proyecto desde los paquetes detectados."""
    pkgs = set(scan.paquetes.keys())

    if any(p in pkgs for p in ("torch", "transformers", "peft", "trl")):
        return "pipeline de ML / fine-tuning de LLMs"
    if any(p in pkgs for p in ("fastapi", "uvicorn", "starlette")):
        return "API REST con FastAPI"
    if "django" in pkgs:
        return "aplicación web Django"
    if "flask" in pkgs:
        return "aplicación web Flask"
    if any(p in pkgs for p in ("pandas", "numpy", "scikit-learn")):
        return "proyecto de Data Science / análisis de datos"
    if any(p in pkgs for p in ("click", "typer", "rich")):
        return "herramienta CLI"
    if any(p in pkgs for p in ("celery", "redis")):
        return "sistema de colas y procesamiento asíncrono"
    return "proyecto Python"
