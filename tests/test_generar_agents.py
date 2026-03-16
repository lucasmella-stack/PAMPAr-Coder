# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Tests de pampar.runtime.generar_agents — Milestone 3: generador determinista
del AGENTS.md contextual.

Cubren:
  - generar_agents_md produce string no vacío
  - Secciones principales presentes (Quick Reference, Sistema, Boot protocol)
  - Info de GPU se incluye cuando está presente
  - Rama CPU-only funciona
  - Paquetes clave filtrados correctamente
  - Servicios activos/inactivos listados
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pampar.runtime.generar_agents import generar_agents_md
from pampar.runtime.scanner import InfoArchivo, InfoSistema, ResultadoScan


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def scan_con_gpu() -> ResultadoScan:
    return ResultadoScan(
        workspace_root="/workspace",
        archivos=[
            InfoArchivo(ruta="scripts/train.py", funciones=["main", "entrenar"], clases=[], lineas=150),
            InfoArchivo(ruta="pampar/__init__.py", funciones=[], clases=[], lineas=10),
        ],
        lenguajes={"Python": 2},
        paquetes={
            "torch": "2.5.1",
            "transformers": "4.47.1",
            "peft": "0.13.2",
            "sentencepiece": "0.2.0",
            "colorama": "0.4.6",  # no es clave → no debe aparecer
        },
        servicios={"PostgreSQL": True, "Redis": False, "HTTP-8000": True},
        sistema=InfoSistema(
            os="Linux Ubuntu 22.04",
            os_version="22.04",
            python_version="3.11.9",
            arquitectura="x86_64",
            gpu="NVIDIA GeForce RTX 4090",
            vram_mb=24576,
            ram_gb=64.0,
        ),
        voz=["espeak"],
    )


@pytest.fixture
def scan_sin_gpu() -> ResultadoScan:
    return ResultadoScan(
        workspace_root="/workspace",
        archivos=[],
        lenguajes={},
        paquetes={
            "fastapi": "0.115.0",
            "uvicorn": "0.32.0",
            "pydantic": "2.10.0",
        },
        servicios={"PostgreSQL": False, "Redis": False},
        sistema=InfoSistema(
            os="macOS 14.0 Sonoma",
            os_version="14.0",
            python_version="3.12.4",
            arquitectura="arm64",
            gpu=None,
            vram_mb=None,
            ram_gb=16.0,
        ),
        voz=[],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestGenerarAgentsMd:
    def test_retorna_string_no_vacio(self, scan_con_gpu):
        md = generar_agents_md(scan_con_gpu)
        assert isinstance(md, str)
        assert len(md) > 100

    def test_tiene_seccion_quick_reference(self, scan_con_gpu):
        md = generar_agents_md(scan_con_gpu)
        assert "## Quick Reference" in md

    def test_tiene_seccion_sistema_detectado(self, scan_con_gpu):
        md = generar_agents_md(scan_con_gpu)
        assert "## Sistema detectado" in md

    def test_tiene_seccion_boot_protocol(self, scan_con_gpu):
        md = generar_agents_md(scan_con_gpu)
        assert "## Boot protocol" in md

    def test_incluye_os_en_quick_reference(self, scan_con_gpu):
        md = generar_agents_md(scan_con_gpu)
        assert "Linux Ubuntu 22.04" in md

    def test_incluye_python_version(self, scan_con_gpu):
        md = generar_agents_md(scan_con_gpu)
        assert "3.11.9" in md

    def test_incluye_gpu_cuando_disponible(self, scan_con_gpu):
        md = generar_agents_md(scan_con_gpu)
        assert "RTX 4090" in md

    def test_incluye_vram(self, scan_con_gpu):
        md = generar_agents_md(scan_con_gpu)
        # 24576 MB = 24.0 GB
        assert "24.0 GB" in md

    def test_cpu_only_cuando_no_gpu(self, scan_sin_gpu):
        md = generar_agents_md(scan_sin_gpu)
        assert "CPU only" in md or "solo CPU" in md

    def test_incluye_paquetes_clave(self, scan_con_gpu):
        md = generar_agents_md(scan_con_gpu)
        assert "## Paquetes clave" in md
        assert "torch" in md
        assert "transformers" in md

    def test_excluye_paquetes_no_clave(self, scan_con_gpu):
        # 'colorama' no está en _PAQUETES_CLAVE
        md = generar_agents_md(scan_con_gpu)
        idx_paquetes = md.find("## Paquetes clave")
        assert idx_paquetes >= 0
        idx_siguiente = md.find("\n## ", idx_paquetes + 1)
        if idx_siguiente < 0:
            idx_siguiente = len(md)
        seccion_paquetes = md[idx_paquetes:idx_siguiente]
        assert "colorama" not in seccion_paquetes

    def test_servicios_activos_e_inactivos(self, scan_con_gpu):
        md = generar_agents_md(scan_con_gpu)
        assert "Activos" in md
        assert "PostgreSQL" in md
        assert "HTTP-8000" in md

    def test_servicios_inactivos(self, scan_con_gpu):
        md = generar_agents_md(scan_con_gpu)
        assert "Inactivos" in md
        assert "Redis" in md

    def test_voz_incluida(self, scan_con_gpu):
        md = generar_agents_md(scan_con_gpu)
        assert "espeak" in md

    def test_voz_no_disponible_cuando_vacia(self, scan_sin_gpu):
        md = generar_agents_md(scan_sin_gpu)
        assert "no disponible" in md

    def test_ram_incluida(self, scan_con_gpu):
        md = generar_agents_md(scan_con_gpu)
        assert "64.0 GB" in md

    def test_proyecto_custom(self, scan_con_gpu):
        md = generar_agents_md(scan_con_gpu, proyecto="mi-proyecto-custom")
        assert "mi-proyecto-custom" in md

    def test_nombre_agente_en_titulo(self, scan_con_gpu):
        md = generar_agents_md(scan_con_gpu, agente_nombre="PAMPAr")
        assert "PAMPAr" in md

    def test_son_headers_markdown_validos(self, scan_con_gpu):
        md = generar_agents_md(scan_con_gpu)
        headers = [l for l in md.splitlines() if l.startswith("## ")]
        assert len(headers) >= 3, f"Esperaba al menos 3 headers ##, encontró: {headers}"

    def test_tiene_tabla_quick_reference(self, scan_con_gpu):
        md = generar_agents_md(scan_con_gpu)
        # quick reference debe tener líneas con |
        idx = md.find("## Quick Reference")
        assert idx >= 0
        seccion = md[idx:idx + 500]
        assert "|" in seccion

    def test_boot_protocol_tiene_pasos(self, scan_con_gpu):
        md = generar_agents_md(scan_con_gpu)
        assert "RAG L3" in md
        assert "RAG L2" in md

    def test_fastapi_en_paquetes_clave(self, scan_sin_gpu):
        md = generar_agents_md(scan_sin_gpu)
        assert "fastapi" in md

    def test_sin_paquetes_relevantes_no_hay_seccion(self):
        scan = ResultadoScan(
            workspace_root="/test",
            archivos=[],
            paquetes={"colorama": "0.4.6", "certifi": "2025.1.31"},  # ninguno clave
            servicios={},
            sistema=InfoSistema(os="Linux", python_version="3.11"),
        )
        md = generar_agents_md(scan)
        assert "## Paquetes clave" not in md
