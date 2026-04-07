# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Boot Protocol — Secuencia de arranque de PAMPAr.

Implementa el protocolo de 3 archivos:
  1. CONCIENCIA.md → identidad invariante → RAG L3
  2. Scanner → inspección del entorno
  3. AGENTS.md contextual → resumen del entorno → RAG L2

El boot se ejecuta una vez al iniciar el Agente.
El resultado es un RAGResidual pre-cargado con identidad + contexto.
"""

from pathlib import Path
from typing import Optional

from pampar.memoria.clasificador import EntradaMemoria
from pampar.memoria.rag import RAGResidual
from .scanner import ResultadoScan, Scanner
from .generar_agents import generar_agents_md


def _fragmentar_markdown(texto: str) -> list[str]:
    """
    Divide un archivo Markdown en fragmentos semánticos por secciones.

    Cada fragmento es una sección (## header + contenido).
    Los fragmentos cortos (< 20 chars) se descartan.
    """
    fragmentos: list[str] = []
    actual: list[str] = []

    for linea in texto.splitlines():
        if linea.startswith("## ") and actual:
            bloque = "\n".join(actual).strip()
            if len(bloque) >= 20:
                fragmentos.append(bloque)
            actual = [linea]
        else:
            actual.append(linea)

    if actual:
        bloque = "\n".join(actual).strip()
        if len(bloque) >= 20:
            fragmentos.append(bloque)

    return fragmentos


class BootProtocol:
    """
    Ejecuta la secuencia de arranque de PAMPAr.

    Carga CONCIENCIA.md como identidad (L3, nunca se purga),
    ejecuta el Scanner para inspeccionar el entorno, y
    vectoriza el resultado como contexto del entorno (L2).

    Args:
        workspace_root: Directorio raíz del workspace.
        conciencia_path: Ruta al archivo CONCIENCIA.md.
        scan_depth: Profundidad de escaneo para el Scanner.
    """

    def __init__(
        self,
        workspace_root: str = ".",
        conciencia_path: Optional[str] = None,
        scan_depth: int = 5,
    ):
        self.workspace_root = Path(workspace_root).resolve()

        if conciencia_path:
            self.conciencia_path = Path(conciencia_path)
        else:
            self.conciencia_path = self._buscar_conciencia()

        self.scanner = Scanner(
            workspace_root=str(self.workspace_root),
            scan_depth=scan_depth,
        )
        self._scan_resultado: Optional[ResultadoScan] = None

    def ejecutar(self, rag: RAGResidual) -> ResultadoScan:
        """
        Ejecuta el boot completo: identidad + scan + vectorización.

        Args:
            rag: RAGResidual donde inyectar identidad y contexto.
        Returns:
            ResultadoScan con la información del entorno detectado.
        """
        # 1. Cargar e inyectar CONCIENCIA (identidad L3)
        self._cargar_conciencia(rag)

        # 2. Escanear el entorno
        self._scan_resultado = self.scanner.scan()

        # 3. Vectorizar el resumen del scan como contexto L2
        self._inyectar_contexto(rag, self._scan_resultado)

        return self._scan_resultado

    def _buscar_conciencia(self) -> Path:
        """Busca CONCIENCIA.md en ubicaciones conocidas."""
        # Primero buscar relativo al paquete pampar/ (donde pertenece)
        paquete_dir = Path(__file__).resolve().parent.parent
        candidatos = [
            paquete_dir / "CONCIENCIA.md",
            self.workspace_root / "pampar" / "CONCIENCIA.md",
            self.workspace_root / "CONCIENCIA.md",
            Path.home() / ".pampar" / "CONCIENCIA.md",
        ]
        for path in candidatos:
            if path.is_file():
                return path
        return candidatos[0]  # default al paquete

    def _cargar_conciencia(self, rag: RAGResidual) -> None:
        """Carga CONCIENCIA.md y lo fragmenta en entradas L3."""
        if not self.conciencia_path.is_file():
            return

        texto = self.conciencia_path.read_text(encoding="utf-8")
        fragmentos = _fragmentar_markdown(texto)

        for i, fragmento in enumerate(fragmentos):
            entrada = EntradaMemoria(
                texto=fragmento,
                tipo="identidad",
                nivel=3,
                importancia=1.0,
                frecuencia=1,
            )
            rag.agregar(entrada)

    def _inyectar_contexto(self, rag: RAGResidual, scan: ResultadoScan) -> None:
        """Genera el AGENTS.md contextual y lo inyecta como entradas L2 en el RAG."""
        # Generar el AGENTS.md determinista desde el scan
        agents_md = generar_agents_md(scan)

        # Fragmentar por secciones ## y agregar cada sección como entrada L2
        fragmentos = _fragmentar_markdown(agents_md)
        for fragmento in fragmentos:
            entrada = EntradaMemoria(
                texto=fragmento,
                tipo="entorno",
                nivel=2,
                importancia=0.8,
                frecuencia=1,
            )
            rag.agregar(entrada)

        # Archivos Python del workspace como entradas individuales L1
        for archivo in scan.archivos[:50]:  # Cap a 50 archivos más relevantes
            partes: list[str] = [f"Archivo: {archivo.ruta} ({archivo.lineas} líneas)"]
            if archivo.clases:
                partes.append(f"  Clases: {', '.join(archivo.clases)}")
            if archivo.funciones:
                partes.append(f"  Funciones: {', '.join(archivo.funciones[:10])}")
            if archivo.imports:
                partes.append(f"  Imports: {', '.join(archivo.imports[:10])}")

            entrada_archivo = EntradaMemoria(
                texto="\n".join(partes),
                tipo="workspace",
                nivel=1,
                importancia=0.5,
                frecuencia=1,
            )
            rag.agregar(entrada_archivo)

    @property
    def scan_resultado(self) -> Optional[ResultadoScan]:
        """Resultado del último scan (None si no se ha ejecutado boot)."""
        return self._scan_resultado

    def generar_system_prompt(self) -> str:
        """
        Genera el system prompt dinámico basado en CONCIENCIA + scan.

        Este prompt se usa como fallback cuando el RAG no ha sido inicializado
        o como prompt base mínimo.
        """
        partes: list[str] = []

        # Identidad mínima (siempre presente)
        partes.append(
            "Sos PAMPAr, un asistente de programación local y offline "
            "especializado en Python.\n"
            "Tenés acceso a la memoria de interacciones previas y "
            "podés ejecutar código cuando sea necesario."
        )

        # Acciones disponibles
        partes.append(
            "Para leer un archivo: [LEER: ruta/al/archivo.py]\n"
            "Para ejecutar código: [EJECUTAR:\ncodigo_python_aqui\n]\n"
            "Para ejecutar tests: [TESTS: ruta/tests/]"
        )

        # Contexto del entorno (si hay scan)
        if self._scan_resultado:
            resumen = self._scan_resultado.resumen
            if resumen:
                partes.append(resumen)

        partes.append("Respondé siempre en español. El código va siempre en inglés.")

        return "\n\n".join(partes)
