# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Scanner — Inspección del entorno al boot.

Escanea el workspace, paquetes instalados, servicios disponibles y
capacidades del sistema. El resultado se usa para generar el AGENTS.md
contextual de cada despliegue.

Principios:
  - Solo lectura: ast.parse, no exec/eval
  - Sin dependencias externas: usa stdlib pura
  - Timeout en detección de servicios
  - Resultado como dataclass serializable
"""

import ast
import importlib.metadata
import platform
import shutil
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class InfoArchivo:
    """Metadata de un archivo Python del workspace."""

    ruta: str
    funciones: List[str] = field(default_factory=list)
    clases: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    lineas: int = 0


@dataclass
class InfoSistema:
    """Información del sistema operativo y hardware."""

    os: str = ""
    os_version: str = ""
    python_version: str = ""
    arquitectura: str = ""
    gpu: Optional[str] = None
    vram_mb: Optional[int] = None
    ram_gb: Optional[float] = None


@dataclass
class ResultadoScan:
    """Resultado completo del scan del entorno."""

    workspace_root: str = ""
    archivos: List[InfoArchivo] = field(default_factory=list)
    lenguajes: Dict[str, int] = field(default_factory=dict)
    paquetes: Dict[str, str] = field(default_factory=dict)
    servicios: Dict[str, bool] = field(default_factory=dict)
    sistema: InfoSistema = field(default_factory=InfoSistema)
    voz: List[str] = field(default_factory=list)

    @property
    def resumen(self) -> str:
        """Genera resumen legible del scan para inyectar en contexto."""
        partes: list[str] = []

        partes.append(f"## Entorno detectado\n")
        partes.append(f"- **OS**: {self.sistema.os} {self.sistema.os_version}")
        partes.append(f"- **Python**: {self.sistema.python_version}")
        if self.sistema.gpu:
            partes.append(f"- **GPU**: {self.sistema.gpu} ({self.sistema.vram_mb} MB)")
        if self.sistema.ram_gb:
            partes.append(f"- **RAM**: {self.sistema.ram_gb:.1f} GB")

        if self.lenguajes:
            langs = ", ".join(f"{k}: {v}" for k, v in sorted(
                self.lenguajes.items(), key=lambda x: -x[1],
            ))
            partes.append(f"- **Archivos**: {langs}")

        if self.paquetes:
            top = list(self.paquetes.items())[:20]
            pkgs = ", ".join(f"{n}=={v}" for n, v in top)
            partes.append(f"- **Paquetes** ({len(self.paquetes)} total): {pkgs}")

        if self.servicios:
            activos = [k for k, v in self.servicios.items() if v]
            if activos:
                partes.append(f"- **Servicios activos**: {', '.join(activos)}")

        if self.voz:
            partes.append(f"- **Voz**: {', '.join(self.voz)}")

        return "\n".join(partes)


class Scanner:
    """
    Inspecciona el entorno donde PAMPAr se despliega.

    Solo lectura — usa ast.parse (no exec), importlib.metadata,
    socket con timeout, platform, shutil.which.

    Args:
        workspace_root: Directorio raíz del workspace a inspeccionar.
        scan_depth:     Profundidad máxima de directorios a escanear.
        service_timeout: Timeout en segundos para detectar servicios.
    """

    EXTENSIONES = {
        ".py": "Python",
        ".js": "JavaScript",
        ".ts": "TypeScript",
        ".json": "JSON",
        ".md": "Markdown",
        ".yaml": "YAML",
        ".yml": "YAML",
        ".toml": "TOML",
        ".sh": "Shell",
        ".sql": "SQL",
    }

    SERVICIOS = {
        "PostgreSQL": ("127.0.0.1", 5432),
        "Redis": ("127.0.0.1", 6379),
        "HTTP-8000": ("127.0.0.1", 8000),
        "HTTP-3000": ("127.0.0.1", 3000),
    }

    MOTORES_VOZ = ["espeak", "espeak-ng", "say", "festival"]

    # Directorios a ignorar siempre
    _IGNORAR = {
        "__pycache__", ".git", ".venv", "venv", "node_modules",
        ".mypy_cache", ".pytest_cache", ".tox", "dist", "build",
        "egg-info",
    }

    def __init__(
        self,
        workspace_root: str = ".",
        scan_depth: int = 5,
        service_timeout: float = 0.3,
    ):
        self.root = Path(workspace_root).resolve()
        self.scan_depth = scan_depth
        self.service_timeout = service_timeout

    def scan(self) -> ResultadoScan:
        """
        Ejecuta un scan completo del entorno.

        Returns:
            ResultadoScan con toda la información detectada.
        """
        resultado = ResultadoScan(workspace_root=str(self.root))
        resultado.sistema = self._scan_sistema()
        resultado.archivos, resultado.lenguajes = self._scan_workspace()
        resultado.paquetes = self._scan_paquetes()
        resultado.servicios = self._scan_servicios()
        resultado.voz = self._scan_voz()
        return resultado

    def _scan_sistema(self) -> InfoSistema:
        """Detecta OS, Python, GPU, RAM."""
        info = InfoSistema(
            os=platform.system(),
            os_version=platform.version(),
            python_version=platform.python_version(),
            arquitectura=platform.machine(),
        )

        # GPU detection via torch (si está disponible)
        try:
            import torch
            if torch.cuda.is_available():
                info.gpu = torch.cuda.get_device_name(0)
                info.vram_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
        except ImportError:
            pass

        # RAM detection
        try:
            import psutil
            info.ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        except ImportError:
            pass

        return info

    def _scan_workspace(self) -> tuple[list[InfoArchivo], dict[str, int]]:
        """Inspecciona archivos del workspace con ast.parse para .py."""
        archivos: list[InfoArchivo] = []
        lenguajes: dict[str, int] = {}

        if not self.root.is_dir():
            return archivos, lenguajes

        for path in self._iter_files():
            ext = path.suffix.lower()
            lang = self.EXTENSIONES.get(ext)
            if lang:
                lenguajes[lang] = lenguajes.get(lang, 0) + 1

            if ext == ".py":
                info = self._analizar_python(path)
                if info:
                    archivos.append(info)

        return archivos, lenguajes

    def _iter_files(self):
        """Itera archivos del workspace respetando profundidad y exclusiones."""
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue

            # Verificar profundidad
            try:
                rel = path.relative_to(self.root)
            except ValueError:
                continue
            if len(rel.parts) > self.scan_depth:
                continue

            # Ignorar directorios excluidos
            if any(
                p in self._IGNORAR or p.endswith(".egg-info")
                for p in rel.parts
            ):
                continue

            yield path

    def _analizar_python(self, path: Path) -> Optional[InfoArchivo]:
        """Analiza un archivo .py con ast.parse (no exec)."""
        try:
            source = path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source, filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            return None

        rel_path = str(path.relative_to(self.root)).replace("\\", "/")
        info = InfoArchivo(ruta=rel_path, lineas=len(source.splitlines()))

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                info.funciones.append(node.name)
            elif isinstance(node, ast.ClassDef):
                info.clases.append(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    info.imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    info.imports.append(node.module)

        return info

    def _scan_paquetes(self) -> dict[str, str]:
        """Detecta paquetes instalados via importlib.metadata."""
        paquetes: dict[str, str] = {}
        try:
            for dist in importlib.metadata.distributions():
                name = dist.metadata.get("Name", "")
                version = dist.metadata.get("Version", "")
                if name:
                    paquetes[name] = version
        except Exception:
            pass
        return paquetes

    def _scan_servicios(self) -> dict[str, bool]:
        """Detecta servicios activos con socket connect + timeout."""
        servicios: dict[str, bool] = {}
        for nombre, (host, port) in self.SERVICIOS.items():
            servicios[nombre] = self._puerto_abierto(host, port)
        return servicios

    def _puerto_abierto(self, host: str, port: int) -> bool:
        """Verifica si un puerto está abierto (solo localhost)."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(self.service_timeout)
                return s.connect_ex((host, port)) == 0
        except OSError:
            return False

    def _scan_voz(self) -> list[str]:
        """Detecta motores de voz disponibles en el sistema."""
        disponibles: list[str] = []
        for motor in self.MOTORES_VOZ:
            if shutil.which(motor):
                disponibles.append(motor)

        # Windows SAPI check
        if platform.system() == "Windows":
            try:
                import winreg
                key = winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"SOFTWARE\Microsoft\Speech\Voices",
                )
                winreg.CloseKey(key)
                disponibles.append("SAPI")
            except (ImportError, OSError):
                pass

        return disponibles
