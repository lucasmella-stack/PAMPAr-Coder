# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
LectorBiblioteca — carga y tokeniza datos de biblioteca/ para entrenamiento.

Soporta:
  - JSONL con claves "text", "content", "instruction"+"output", o texto plano
  - División automática en chunks de max_seq_len tokens
  - Cache en memoria por archivo (no releer en cada iteración)
  - Batch aleatorio listo para enviar al modelo
"""

import json
import logging
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class LectorBiblioteca:
    """
    Lee y tokeniza archivos JSONL de la biblioteca de conocimiento.

    Flujo:
      1. cargar_archivo(ruta_relativa) → lista de chunks tokenizados (cacheada)
      2. obtener_batch(ruta_relativa, device) → Tensor [B, L+1]

    Args:
        raiz:        Ruta a biblioteca/ (carpeta con los .jsonl)
        tokenizer:   SentencePieceProcessor ya cargado
        max_seq_len: Máxima longitud de secuencia (excl. el token objetivo)
        batch_size:  Chunks por batch
    """

    def __init__(
        self,
        raiz: Path,
        tokenizer,
        max_seq_len: int = 512,
        batch_size: int = 4,
    ) -> None:
        self.raiz = Path(raiz)
        self.tok = tokenizer
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self._cache: dict[str, list[list[int]]] = {}

    # ── Parseo ────────────────────────────────────────────────────────────────

    def _extraer_texto(self, linea: str) -> str:
        """
        Extrae texto de una línea JSONL.

        Formatos soportados:
          {"text": "..."}
          {"content": "..."}
          {"instruction": "...", "output": "..."}  — formato Alpaca
          Texto plano (fallback)
        """
        try:
            obj = json.loads(linea)
            if "text" in obj:
                return obj["text"]
            if "content" in obj:
                return obj["content"]
            if "instruction" in obj and "output" in obj:
                return f"{obj['instruction']}\n{obj.get('input', '')}\n{obj['output']}"
            # Último recurso: concatenar todos los valores string
            return " ".join(str(v) for v in obj.values() if isinstance(v, str))
        except json.JSONDecodeError:
            return linea.strip()

    # ── Carga ─────────────────────────────────────────────────────────────────

    def cargar_archivo(self, ruta_relativa: str) -> list[list[int]]:
        """
        Carga y tokeniza todos los chunks de un archivo JSONL.

        Returns:
            Lista de listas de token IDs. Vacía si el archivo no existe.
        """
        if ruta_relativa in self._cache:
            return self._cache[ruta_relativa]

        ruta = self.raiz / ruta_relativa
        if not ruta.exists():
            return []

        chunks: list[list[int]] = []
        try:
            for linea in ruta.read_text(encoding="utf-8").splitlines():
                if not linea.strip():
                    continue
                texto = self._extraer_texto(linea)
                ids = self.tok.Encode(texto)
                # Dividir en chunks solapados — +1 para el token target
                for i in range(0, max(1, len(ids) - self.max_seq_len), self.max_seq_len // 2):
                    chunk = ids[i : i + self.max_seq_len + 1]
                    if len(chunk) >= 8:
                        chunks.append(chunk)
        except Exception as exc:
            logger.warning("Error leyendo %s: %s", ruta, exc)
            return []

        self._cache[ruta_relativa] = chunks
        return chunks

    def tiene_datos(self, ruta_relativa: str) -> bool:
        """True si el archivo existe y tiene al menos un chunk válido."""
        return len(self.cargar_archivo(ruta_relativa)) > 0

    def n_chunks(self, ruta_relativa: str) -> int:
        """Número de chunks disponibles para un archivo."""
        return len(self.cargar_archivo(ruta_relativa))

    # ── Batch ─────────────────────────────────────────────────────────────────

    def obtener_batch(
        self,
        ruta_relativa: str,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """
        Devuelve un batch aleatorio de tokens listos para el modelo.

        Returns:
            Tensor [B, L] o None si no hay datos para este archivo.
        """
        chunks = self.cargar_archivo(ruta_relativa)
        if not chunks:
            return None

        indices = torch.randint(0, len(chunks), (self.batch_size,))
        seleccionados = [chunks[i] for i in indices]

        max_len = min(max(len(c) for c in seleccionados), self.max_seq_len + 1)

        padded = []
        for chunk in seleccionados:
            trunc = chunk[:max_len]
            pad = [0] * (max_len - len(trunc))
            padded.append(trunc + pad)

        return torch.tensor(padded, dtype=torch.long, device=device)

    def invalidar_cache(self) -> None:
        """Limpia el cache en memoria (útil si los archivos cambian en disco)."""
        self._cache.clear()
