# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
LectorArchivos — Los "ojos" del modelo.

Permite a PAMPAr leer archivos y directorios del sistema local.
El modelo puede ver el código del usuario, su estructura de proyecto,
archivos de configuración, errores en logs, etc.

Seguridad:
  - Raíz configurable: solo puede leer dentro del workspace_root
  - Límite de tamaño: archivos > max_bytes se truncan con advertencia
  - Extensiones permitidas: solo texto plano y código
"""

from pathlib import Path
from typing import List, Optional

from .base import ResultadoSkill, Skill


# Extensiones permitidas para lectura
EXTENSIONES_TEXTO = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs",
    ".java", ".c", ".cpp", ".h", ".hpp", ".cs", ".go", ".rs",
    ".html", ".css", ".scss", ".less",
    ".json", ".jsonl", ".yaml", ".yml", ".toml", ".env",
    ".md", ".txt", ".rst", ".csv",
    ".sh", ".bash", ".zsh", ".ps1",
    ".sql", ".graphql",
    ".ipynb",
}


class LectorArchivos(Skill):
    """
    Lee archivos y directorios del workspace del usuario.

    Args:
        workspace_root: Directorio raíz donde puede leer (sandboxing)
        max_bytes:      Límite de bytes por archivo (default 50KB)
    """

    name = "lector_archivos"
    description = (
        "Lee el contenido de archivos o lista un directorio del proyecto. "
        "Úsalo cuando necesites ver código existente antes de modificarlo."
    )

    def __init__(
        self,
        workspace_root: str = ".",
        max_bytes: int = 50 * 1024,  # 50KB
    ):
        self.root = Path(workspace_root).resolve()
        self.max_bytes = max_bytes

    def execute(
        self,
        ruta: str,
        linea_inicio: int = 1,
        linea_fin: Optional[int] = None,
    ) -> ResultadoSkill:
        """
        Lee un archivo o lista un directorio.

        Args:
            ruta:         Path relativo al workspace_root
            linea_inicio: Primera línea a leer (1-based, default 1)
            linea_fin:    Última línea a leer (None = hasta el final)
        Returns:
            ResultadoSkill con el contenido del archivo o listado de dir
        """
        try:
            ruta_abs = (self.root / ruta).resolve()
        except Exception as e:
            return ResultadoSkill(exito=False, contenido="", error=f"Ruta inválida: {e}")

        # Verificar sandboxing
        if not str(ruta_abs).startswith(str(self.root)):
            return ResultadoSkill(
                exito=False, contenido="",
                error=f"Acceso denegado: fuera del workspace ({self.root})"
            )

        if not ruta_abs.exists():
            return ResultadoSkill(
                exito=False, contenido="",
                error=f"No existe: {ruta}"
            )

        if ruta_abs.is_dir():
            return self._listar_directorio(ruta_abs, ruta)

        return self._leer_archivo(ruta_abs, ruta, linea_inicio, linea_fin)

    def _leer_archivo(
        self,
        ruta_abs: Path,
        ruta_rel: str,
        linea_inicio: int,
        linea_fin: Optional[int],
    ) -> ResultadoSkill:
        """Lee el contenido de un archivo de texto."""
        sufijo = ruta_abs.suffix.lower()
        if sufijo not in EXTENSIONES_TEXTO:
            return ResultadoSkill(
                exito=False, contenido="",
                error=f"Extensión no permitida: {sufijo}. Solo archivos de texto/código."
            )

        size = ruta_abs.stat().st_size
        truncado = size > self.max_bytes

        try:
            contenido = ruta_abs.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return ResultadoSkill(exito=False, contenido="", error=str(e))

        lineas = contenido.splitlines()
        total_lineas = len(lineas)

        # Aplicar rango de líneas
        li = max(0, linea_inicio - 1)
        lf = linea_fin if linea_fin else total_lineas
        lineas_sel = lineas[li:lf]
        contenido_sel = "\n".join(lineas_sel)

        # Truncar si excede max_bytes
        if len(contenido_sel.encode()) > self.max_bytes:
            contenido_sel = contenido_sel.encode()[:self.max_bytes].decode(errors="replace")
            truncado = True

        aviso = f"\n[TRUNCADO: archivo de {size // 1024}KB, mostrando hasta {self.max_bytes // 1024}KB]" if truncado else ""

        contenido_final = (
            f"[ARCHIVO: {ruta_rel}] ({total_lineas} líneas)\n"
            f"```{sufijo.lstrip('.')}\n"
            f"{contenido_sel}"
            f"\n```{aviso}"
        )
        return ResultadoSkill(
            exito=True,
            contenido=contenido_final,
            datos={
                "ruta": ruta_rel,
                "total_lineas": total_lineas,
                "lineas_leidas": len(lineas_sel),
                "truncado": truncado,
            }
        )

    def _listar_directorio(self, ruta_abs: Path, ruta_rel: str) -> ResultadoSkill:
        """Lista el contenido de un directorio."""
        try:
            items = sorted(ruta_abs.iterdir(), key=lambda p: (p.is_file(), p.name))
        except PermissionError:
            return ResultadoSkill(
                exito=False, contenido="",
                error=f"Sin permiso para leer: {ruta_rel}"
            )

        lineas = [f"[DIRECTORIO: {ruta_rel}]"]
        for item in items[:100]:  # Limitar a 100 items
            prefijo = "📄" if item.is_file() else "📁"
            size = f" ({item.stat().st_size // 1024}KB)" if item.is_file() else ""
            lineas.append(f"  {prefijo} {item.name}{size}")

        if len(list(ruta_abs.iterdir())) > 100:
            lineas.append("  ... (más de 100 items, mostrando primeros 100)")

        return ResultadoSkill(
            exito=True,
            contenido="\n".join(lineas),
            datos={"ruta": ruta_rel, "n_items": len(items)}
        )

    def buscar_en_workspace(
        self,
        patron: str,
        extension: str = ".py",
        max_resultados: int = 10,
    ) -> ResultadoSkill:
        """
        Busca texto en archivos del workspace.

        Args:
            patron:        String a buscar
            extension:     Extensión de archivo a buscar (default .py)
            max_resultados: Máximo de resultados
        Returns:
            ResultadoSkill con matches encontrados
        """
        matches: List[str] = []
        for archivo in self.root.rglob(f"*{extension}"):
            if "__pycache__" in str(archivo) or ".git" in str(archivo):
                continue
            try:
                contenido = archivo.read_text(encoding="utf-8", errors="replace")
                for n, linea in enumerate(contenido.splitlines(), 1):
                    if patron.lower() in linea.lower():
                        rel = str(archivo.relative_to(self.root))
                        matches.append(f"{rel}:{n}: {linea.strip()}")
                        if len(matches) >= max_resultados:
                            break
            except Exception:
                continue
            if len(matches) >= max_resultados:
                break

        if not matches:
            return ResultadoSkill(
                exito=True,
                contenido=f"No se encontraron resultados para '{patron}' en archivos {extension}",
            )

        contenido = f"[BÚSQUEDA: '{patron}' en *{extension}] — {len(matches)} resultado(s)\n"
        contenido += "\n".join(matches)
        return ResultadoSkill(exito=True, contenido=contenido, datos={"matches": matches})
