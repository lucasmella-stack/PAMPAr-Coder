#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
generar_dataset_agentes.py — Dataset sintético para Milestone 3.

Genera pares (resumen_scan → AGENTS.md) variando el entorno sintéticamente.
El modelo aprende a producir un AGENTS.md coherente dado el output del Scanner.

Formato de cada ejemplo:
  {"text": "### Scan:\\n{resumen_scan}\\n### Protocolo:\\n{agents_md}"}

Uso:
  python -X utf8 scripts/generar_dataset_agentes.py
  python -X utf8 scripts/generar_dataset_agentes.py --n 500 --out data/agents_sft.jsonl
"""

import argparse
import json
import random
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Vocabulario de variación sintética
# ─────────────────────────────────────────────────────────────────────────────

_OS_VARIANTS = [
    ("Windows", "10.0.26200"),
    ("Windows", "11.0.22621"),
    ("Linux", "Ubuntu 22.04"),
    ("Linux", "Debian 12"),
    ("macOS", "14.0 Sonoma"),
]

_PY_VERSIONS = ["3.10.14", "3.11.9", "3.12.4", "3.13.3"]

_GPU_VARIANTS = [
    None,
    ("NVIDIA GeForce GTX 1650", 4095),
    ("NVIDIA GeForce RTX 3080", 10240),
    ("NVIDIA GeForce RTX 4090", 24576),
    ("AMD Radeon RX 6800 XT", 16384),
]

_RAM_VARIANTS = [8.0, 16.0, 32.0, 64.0]

_FRAMEWORK_STACKS = [
    {
        "nombre": "ML / PyTorch",
        "paquetes": ["torch==2.5.1", "torchvision==0.20.1", "sentencepiece==0.2.0",
                     "transformers==4.47.1", "accelerate==1.2.0", "peft==0.13.2"],
        "lenguajes": {"Python": 45, "Markdown": 6, "JSON": 4},
        "servicios": {},
        "descripcion": "repositorio de modelos de lenguaje con PyTorch",
    },
    {
        "nombre": "Web / FastAPI",
        "paquetes": ["fastapi==0.115.0", "uvicorn==0.32.0", "pydantic==2.10.0",
                     "sqlalchemy==2.0.36", "alembic==1.14.0", "httpx==0.28.0"],
        "lenguajes": {"Python": 38, "TypeScript": 12, "JSON": 8, "Markdown": 5},
        "servicios": {"PostgreSQL": True, "Redis": False, "HTTP-8000": True},
        "descripcion": "API REST con FastAPI, PostgreSQL y autenticación JWT",
    },
    {
        "nombre": "Data / Pandas",
        "paquetes": ["pandas==2.2.3", "numpy==2.1.3", "matplotlib==3.9.3",
                     "scikit-learn==1.5.2", "jupyter==1.1.1", "plotly==5.24.1"],
        "lenguajes": {"Python": 22, "JSON": 15, "Markdown": 8},
        "servicios": {"PostgreSQL": True, "HTTP-8000": False},
        "descripcion": "análisis de datos con Pandas, visualización y ML clásico",
    },
    {
        "nombre": "Next.js / Full-stack",
        "paquetes": ["requests==2.32.3", "python-dotenv==1.0.1",
                     "black==25.11.0", "pytest==9.0.1"],
        "lenguajes": {"TypeScript": 65, "Python": 8, "JSON": 20, "Markdown": 7, "YAML": 4},
        "servicios": {"PostgreSQL": True, "Redis": True, "HTTP-3000": True},
        "descripcion": "aplicación full-stack Next.js con API Python como BFF",
    },
    {
        "nombre": "CLI / Script",
        "paquetes": ["click==8.1.8", "rich==13.9.4", "typer==0.15.1",
                     "httpx==0.28.0", "python-dotenv==1.0.1"],
        "lenguajes": {"Python": 18, "Shell": 5, "Markdown": 4},
        "servicios": {},
        "descripcion": "herramienta de línea de comandos con interfaz rica",
    },
    {
        "nombre": "Django / Web",
        "paquetes": ["django==5.1.4", "djangorestframework==3.15.2",
                     "celery==5.4.0", "redis==5.2.1", "pillow==11.0.0"],
        "lenguajes": {"Python": 55, "HTML": 20, "CSS": 10, "JavaScript": 8, "Markdown": 5},
        "servicios": {"PostgreSQL": True, "Redis": True, "HTTP-8000": True},
        "descripcion": "aplicación Django con Celery para tareas asíncronas",
    },
    {
        "nombre": "Embedded / IoT",
        "paquetes": ["pyserial==3.5", "RPi.GPIO==0.7.1", "paho-mqtt==2.1.0",
                     "numpy==2.1.3", "python-dotenv==1.0.1"],
        "lenguajes": {"Python": 25, "C": 8, "Markdown": 4},
        "servicios": {"HTTP-8000": False},
        "descripcion": "firmware de control con MQTT y lectura de sensores",
    },
    {
        "nombre": "LLM / Fine-tuning",
        "paquetes": ["torch==2.5.1", "transformers==4.47.1", "peft==0.13.2",
                     "datasets==3.1.0", "evaluate==0.4.3", "bitsandbytes==0.45.0",
                     "trl==0.12.2"],
        "lenguajes": {"Python": 40, "Markdown": 8, "JSON": 12},
        "servicios": {},
        "descripcion": "pipeline de fine-tuning con LoRA/QLoRA sobre LLMs",
    },
]

_VOZ_VARIANTS = [
    [],
    ["SAPI"],
    ["espeak"],
    ["say"],
    ["SAPI", "espeak"],
]

_PROYECTO_NOMBRES = [
    "mi-proyecto", "analysis-tool", "backend-api", "data-pipeline",
    "ml-training", "web-scraper", "chatbot", "sensor-monitor",
    "code-reviewer", "deploy-helper", "report-generator", "text-classifier",
]

_TESTS_VARIANTS = [
    ("pytest", "pytest==9.0.1", "0 tests"),
    ("pytest", "pytest==9.0.1", "12 tests pasando"),
    ("pytest", "pytest==9.0.1", "47 tests pasando"),
    ("pytest", "pytest==9.0.1", "109 tests pasando"),
    ("unittest", None, "sin tests integrados aún"),
]

_CHECKPOINTS_VARIANTS = [
    None,
    "checkpoints/v3_sft_v8.pt  (16/16 eval)",
    "checkpoints/model_best.pt  (fine-tuned)",
    "checkpoints/backbone.pt",
]


# ─────────────────────────────────────────────────────────────────────────────
# Generadores
# ─────────────────────────────────────────────────────────────────────────────

def _resumen_scan(cfg: dict) -> str:
    """Genera el resumen del Scanner para un entorno sintético."""
    os_name, os_ver = cfg["os"]
    gpu = cfg["gpu"]
    ram = cfg["ram"]
    py = cfg["python"]
    stack = cfg["stack"]
    voz = cfg["voz"]
    langs = stack["lenguajes"]

    partes: list[str] = [
        "El Scanner del sistema detectó el siguiente entorno. "
        "Genera el archivo AGENTS.md contextual para este despliegue.\n",
        "## Entorno detectado\n",
        f"- **OS**: {os_name} {os_ver}",
        f"- **Python**: {py}",
    ]

    if gpu:
        gpu_name, vram = gpu
        partes.append(f"- **GPU**: {gpu_name} ({vram} MB)")
    else:
        partes.append("- **GPU**: no disponible (solo CPU)")

    partes.append(f"- **RAM**: {ram:.1f} GB")

    langs_str = ", ".join(f"{k}: {v}" for k, v in sorted(langs.items(), key=lambda x: -x[1]))
    partes.append(f"- **Archivos**: {langs_str}")

    pkgs_str = ", ".join(stack["paquetes"])
    partes.append(f"- **Paquetes** ({len(stack['paquetes'])} relevantes): {pkgs_str}")

    servicios = stack.get("servicios", {})
    if servicios:
        activos = [k for k, v in servicios.items() if v]
        inactivos = [k for k, v in servicios.items() if not v]
        if activos:
            partes.append(f"- **Servicios activos**: {', '.join(activos)}")
        if inactivos:
            partes.append(f"- **Servicios inactivos**: {', '.join(inactivos)}")

    if voz:
        partes.append(f"- **Voz**: {', '.join(voz)}")
    else:
        partes.append("- **Voz**: no detectada")

    partes.append(f"\n**Proyecto**: {cfg['proyecto']} — {stack['descripcion']}")

    return "\n".join(partes)


def _agents_md(cfg: dict) -> str:
    """Genera el AGENTS.md para el entorno sintético dado."""
    os_name, os_ver = cfg["os"]
    gpu = cfg["gpu"]
    ram = cfg["ram"]
    py = cfg["python"]
    stack = cfg["stack"]
    voz = cfg["voz"]
    proyecto = cfg["proyecto"]
    test_fw, test_pkg, test_estado = cfg["tests"]
    ckpt = cfg["checkpoint"]

    fecha = "Mar 2026"

    # Detectar tipo de proyecto
    es_ml = any(p in stack["paquetes"][0] for p in ["torch", "tensorflow", "sklearn"]) or "ML" in stack["nombre"]
    es_web = any("fastapi" in p or "django" in p or "flask" in p for p in stack["paquetes"])
    es_data = any("pandas" in p for p in stack["paquetes"])

    # Quick reference según stack
    lang_principal = max(stack["lenguajes"].items(), key=lambda x: x[1])[0]
    frameworks = []
    for p in stack["paquetes"]:
        nombre = p.split("==")[0]
        if nombre in ("torch", "transformers", "fastapi", "django", "pandas",
                      "numpy", "click", "typer", "celery", "peft", "trl"):
            frameworks.append(nombre)
    frameworks_str = ", ".join(frameworks[:4]) if frameworks else lang_principal

    servicios = stack.get("servicios", {})
    activos = [k for k, v in servicios.items() if v]

    # Construir AGENTS.md
    lineas: list[str] = [
        f"# {proyecto} — Protocolo de Despliegue",
        "",
        f"> **PAMPAr** activo en: `{proyecto}`",
        f"> Generado automáticamente por el Scanner al boot — {fecha}.",
        f"> Para identidad invariante del modelo ver `CONCIENCIA.md`.",
        "",
        "---",
        "",
        "## Visión",
        "",
        f"PAMPAr se desplegó en un entorno de **{stack['descripcion']}**.",
        "El modelo trae el razonamiento computacional en sus pesos.",
        "Este archivo describe el laboratorio donde aterrizó.",
        "",
        "---",
        "",
        "## Quick Reference",
        "",
        "| Area | Valor |",
        "| ---- | ----- |",
        f"| Lenguaje principal | {lang_principal} |",
        f"| Stack | {frameworks_str} |",
        f"| Testing | {test_fw} — {test_estado} |",
        f"| OS | {os_name} {os_ver} |",
        f"| Python | {py} |",
    ]

    if gpu:
        lineas.append(f"| GPU | {gpu[0]} ({gpu[1]} MB VRAM) |")
    else:
        lineas.append("| GPU | No disponible — solo CPU |")

    if ckpt:
        lineas.append(f"| Checkpoint | `{ckpt}` |")

    if activos:
        lineas.append(f"| Servicios | {', '.join(activos)} |")

    if voz:
        lineas.append(f"| Voz | {', '.join(voz)} |")

    lineas += [
        "",
        "---",
        "",
        "## Sistema detectado",
        "",
        f"- **OS**: {os_name} {os_ver}",
        f"- **Python**: {py}",
        f"- **RAM**: {ram:.1f} GB",
    ]

    if gpu:
        lineas.append(f"- **GPU**: {gpu[0]}, {gpu[1]} MB VRAM")
    else:
        lineas.append("- **GPU**: no disponible")

    if voz:
        lineas.append(f"- **Síntesis de voz**: {', '.join(voz)} disponible")

    lineas += [
        "",
        "---",
        "",
        "## Paquetes clave",
        "",
    ]
    for pkg in stack["paquetes"]:
        nombre, *ver = pkg.split("==")
        ver_str = f" `{ver[0]}`" if ver else ""
        lineas.append(f"- `{nombre}`{ver_str}")

    if servicios:
        lineas += ["", "---", "", "## Servicios", ""]
        for svc, activo in servicios.items():
            estado = "✅ activo" if activo else "❌ no detectado"
            lineas.append(f"- **{svc}**: {estado}")

    # Reglas específicas del dominio
    lineas += ["", "---", "", "## Reglas de este entorno", ""]

    if es_ml:
        lineas += [
            "- Checkpoints en `checkpoints/` — nunca sobrescribir el mejor sin backup",
            "- Datasets en `data/` — no modificar archivos curados",
            "- Training siempre con `torch.no_grad()` en eval",
            "- Eval antes y después de cualquier cambio de arquitectura",
        ]
    if es_web:
        lineas += [
            "- API routes validadas con Pydantic antes de llegar a la DB",
            "- Variables de entorno en `.env` — nunca en código",
            "- Migraciones con Alembic — nunca modificar DB en producción directamente",
        ]
    if es_data:
        lineas += [
            "- Datos de entrada en `data/raw/` — inmutables",
            "- Resultados procesados en `data/processed/`",
            "- Notebooks en `notebooks/` — no en el root",
        ]
    if not es_ml and not es_web and not es_data:
        lineas += [
            "- Tests deben pasar antes de cualquier commit",
            "- Conventional commits: feat/fix/docs/test/refactor/chore",
            "- No secrets en código — usar variables de entorno",
        ]

    lineas += ["", "---", "", "## Boot protocol", "",
               "```",
               "1. CONCIENCIA.md → RAG L3 (identidad invariante)",
               "2. Scanner → workspace + paquetes + servicios + sistema",
               "3. AGENTS.md (este) → RAG L2 (contexto del entorno)",
               "4. Listo — identidad + contexto + acciones disponibles",
               "```",
               ""]

    return "\n".join(lineas)


# ─────────────────────────────────────────────────────────────────────────────
# Generación del dataset
# ─────────────────────────────────────────────────────────────────────────────

def generar_configuracion() -> dict:
    """Genera una configuración aleatoria de entorno sintético."""
    rng = random
    return {
        "os": rng.choice(_OS_VARIANTS),
        "python": rng.choice(_PY_VERSIONS),
        "gpu": rng.choice(_GPU_VARIANTS),
        "ram": rng.choice(_RAM_VARIANTS),
        "stack": rng.choice(_FRAMEWORK_STACKS),
        "voz": rng.choice(_VOZ_VARIANTS),
        "proyecto": rng.choice(_PROYECTO_NOMBRES),
        "tests": rng.choice(_TESTS_VARIANTS),
        "checkpoint": rng.choice(_CHECKPOINTS_VARIANTS),
    }


def generar_ejemplo(cfg: dict) -> dict:
    """Genera un ejemplo SFT completo (problema + solución)."""
    problema = _resumen_scan(cfg)
    solucion = _agents_md(cfg)
    texto = f"### Scan:\n{problema}\n### Protocolo:\n{solucion}"
    return {"text": texto}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300, help="Número de ejemplos a generar")
    parser.add_argument("--out", default="data/agents_sft.jsonl", help="Archivo de salida")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ejemplos: list[dict] = []
    vistos: set[str] = set()

    intentos = 0
    while len(ejemplos) < args.n and intentos < args.n * 10:
        intentos += 1
        cfg = generar_configuracion()
        ej = generar_ejemplo(cfg)
        # Deduplicar por stack+OS+GPU (no queremos ejemplos idénticos)
        clave = f"{cfg['stack']['nombre']}|{cfg['os']}|{cfg['gpu']}|{cfg['proyecto']}"
        if clave not in vistos:
            vistos.add(clave)
            ejemplos.append(ej)

    out_path.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in ejemplos),
        encoding="utf-8",
    )

    # Estadísticas
    lens = [len(e["text"]) for e in ejemplos]
    print(f"✅ Generados: {len(ejemplos)} ejemplos → {out_path}")
    print(f"   Longitud promedio: {sum(lens) // len(lens)} chars")
    print(f"   Longitud mín/máx: {min(lens)} / {max(lens)} chars")
    print(f"   Tamaño archivo: {out_path.stat().st_size // 1024} KB")

    # Muestra 1 ejemplo
    print(f"\n--- Ejemplo (primeros 600 chars) ---")
    print(ejemplos[0]["text"][:600])
    print("...")


if __name__ == "__main__":
    main()
