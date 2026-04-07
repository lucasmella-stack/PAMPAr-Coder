# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
RAG Residual — Almacenamiento y Recuperación de Memoria Local.

Vector store local con FAISS (sin necesidad de servidor externo).
Almacena fragmentos clasificados por el ClasificadorPareto con nivel >= 1.

El RAG Residual actúa como el hipocampo:
  - Guarda todo lo que clasificó como importante (L1+)
  - El modelo puede recuperar los N más similares para usarlos como contexto
  - Los fragmentos L3 se promueven a la ColaFinetune y se pueden eliminar del RAG

Dependencias:
    pip install faiss-cpu sentence-transformers

Si no están instaladas, el RAG funciona en modo degradado (BM25 sobre texto plano).
"""

import json
import math
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

from .clasificador import EntradaMemoria


# =============================================================================
# ENCODER DE TEXTO (MODO LIGERO)
# =============================================================================

class EncoderTexto:
    """
    Encoder minimalista para vectorizar fragmentos de código.

    Modo 1 (completo): usa sentence-transformers/all-MiniLM-L6-v2
        → 384D embeddings, ~22MB modelo, offline total
    Modo 2 (degradado): BM25-like TF-IDF sobre vocabulario de código
        → Sin GPU, sin dependencias externas, funciona siempre

    Intenta cargar modo 1. Si falla, cae a modo 2 silenciosamente.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._encoder = None
        self._dim = 384
        self._vocab: dict[str, int] = {}

        try:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(model_name)
            self._dim = self._encoder.get_sentence_embedding_dimension()
        except ImportError:
            pass  # Modo degradado

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def modo_completo(self) -> bool:
        return self._encoder is not None

    def encode(self, textos: List[str]) -> "list[list[float]]":
        """
        Codifica textos a vectores float.

        Args:
            textos: Lista de fragmentos a codificar
        Returns:
            Lista de vectores (listas de float)
        """
        if self._encoder is not None:
            vecs = self._encoder.encode(textos, normalize_embeddings=True)
            return vecs.tolist()
        else:
            return [self._tfidf_encode(t) for t in textos]

    def _tfidf_encode(self, texto: str) -> list[float]:
        """
        TF-IDF sobre vocabulario de tokens Python.

        Vector disperso normalizado — suficiente para recuperación BM25-like.
        """
        tokens = re.findall(r"\w+", texto.lower())
        if not tokens:
            return [0.0] * self._dim

        # Actualizar vocabulario
        for t in tokens:
            if t not in self._vocab:
                if len(self._vocab) < self._dim:
                    self._vocab[t] = len(self._vocab)

        # Vector TF
        vec = [0.0] * self._dim
        for t in tokens:
            if t in self._vocab:
                vec[self._vocab[t]] += 1 / len(tokens)

        # Normalizar
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


# =============================================================================
# RAG RESIDUAL
# =============================================================================

class RAGResidual:
    """
    Almacén de memoria local con recuperación semántica.

    Persistido en disco como JSON (entradas) + npy (vectores).
    No requiere servidor — funciona completamente offline.

    Args:
        directorio: Ruta donde guardar el índice y las entradas
        max_entradas: Máximo de entradas en el RAG (FIFO por importancia)
        n_resultados: Cuántos fragmentos recuperar por consulta
    """

    def __init__(
        self,
        directorio: str = "memoria/data",
        max_entradas: int = 5_000,
        n_resultados: int = 5,
    ):
        self.dir = Path(directorio)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_entradas = max_entradas
        self.n_resultados = n_resultados

        self.encoder = EncoderTexto()
        self._entradas: list[EntradaMemoria] = []
        self._vectores: list[list[float]] = []
        self._indice = None  # índice FAISS si disponible

        self._cargar()
        self._construir_indice()

    # ── Persistencia ──────────────────────────────────────────────────────────

    def _cargar(self) -> None:
        """Carga entradas desde disco al iniciar."""
        ruta_json = self.dir / "entradas.json"
        if not ruta_json.exists():
            return

        try:
            data = json.loads(ruta_json.read_text(encoding="utf-8"))
            for item in data:
                e = EntradaMemoria(**{
                    k: v for k, v in item.items()
                    if k in EntradaMemoria.__dataclass_fields__
                })
                self._entradas.append(e)
        except Exception:
            pass  # Archivo corrupto → empezar de cero

        ruta_vecs = self.dir / "vectores.json"
        if ruta_vecs.exists():
            try:
                self._vectores = json.loads(ruta_vecs.read_text(encoding="utf-8"))
            except Exception:
                self._vectores = []

    def _guardar(self) -> None:
        """Persiste el estado actual en disco."""
        try:
            entradas_data = [
                {k: getattr(e, k) for k in e.__dataclass_fields__}
                for e in self._entradas
            ]
            (self.dir / "entradas.json").write_text(
                json.dumps(entradas_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (self.dir / "vectores.json").write_text(
                json.dumps(self._vectores),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _construir_indice(self) -> None:
        """Construye índice FAISS si está disponible."""
        if not self._vectores:
            return
        try:
            import faiss
            import numpy as np
            mat = np.array(self._vectores, dtype=np.float32)
            self._indice = faiss.IndexFlatIP(self.encoder.dim)  # Inner product = coseno si normalizado
            self._indice.add(mat)
        except ImportError:
            pass  # Sin FAISS → búsqueda lineal en _buscar_lineal

    # ── Operaciones ───────────────────────────────────────────────────────────

    def agregar(self, entrada: EntradaMemoria) -> bool:
        """
        Agrega una entrada al RAG si su nivel Pareto es >= 1.

        Returns:
            True si fue agregada, False si fue descartada (nivel 0)
        """
        if entrada.nivel < 1:
            return False

        # Verificar duplicados por ID
        ids_existentes = {e.id for e in self._entradas}
        if entrada.id in ids_existentes:
            # Actualizar frecuencia de la existente
            for e in self._entradas:
                if e.id == entrada.id:
                    e.frecuencia += 1
                    e.importancia = min(1.0, e.importancia + 0.05)
            self._guardar()
            return False

        # Vectorizar el texto
        vec = self.encoder.encode([entrada.texto])[0]

        # Agregar al almacén
        self._entradas.append(entrada)
        self._vectores.append(vec)

        # Reconstruir índice FAISS con la nueva entrada
        self._agregar_al_indice(vec)

        # Cap: si excede max_entradas, eliminar las de menor importancia
        if len(self._entradas) > self.max_entradas:
            self._podar()

        self._guardar()
        return True

    def _agregar_al_indice(self, vec: list[float]) -> None:
        """Agrega un vector al índice FAISS o lo resetea."""
        try:
            import faiss
            import numpy as np
            if self._indice is None:
                self._construir_indice()
                return
            self._indice.add(np.array([vec], dtype=np.float32))
        except ImportError:
            pass

    def recuperar(
        self,
        query: str,
        nivel_minimo: int = 1,
    ) -> List[Tuple[EntradaMemoria, float]]:
        """
        Recupera los N fragmentos más similares a la consulta.

        Args:
            query:        Texto de consulta (código o pregunta del usuario)
            nivel_minimo: Solo recuperar entradas con nivel >= este valor
        Returns:
            Lista de (EntradaMemoria, score_similitud) ordenada por relevancia
        """
        if not self._entradas:
            return []

        vec_query = self.encoder.encode([query])[0]

        # Filtrar por nivel mínimo
        indices_validos = [
            i for i, e in enumerate(self._entradas)
            if e.nivel >= nivel_minimo
        ]
        if not indices_validos:
            return []

        try:
            import faiss
            import numpy as np
            if self._indice is not None:
                q = np.array([vec_query], dtype=np.float32)
                k = min(self.n_resultados * 2, len(indices_validos))
                scores, idxs = self._indice.search(q, k)

                resultados = []
                for idx, score in zip(idxs[0], scores[0]):
                    if idx in indices_validos:
                        resultados.append((self._entradas[idx], float(score)))
                    if len(resultados) >= self.n_resultados:
                        break
                return resultados
        except (ImportError, Exception):
            pass

        # Fallback: búsqueda lineal por coseno
        return self._buscar_lineal(vec_query, indices_validos)

    def _buscar_lineal(
        self,
        vec_query: list[float],
        indices_validos: list[int],
    ) -> List[Tuple[EntradaMemoria, float]]:
        """Búsqueda por coseno lineal — O(n) pero siempre funciona."""
        def coseno(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            norma = math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(x*x for x in b))
            return dot / norma if norma else 0.0

        scored = [
            (self._entradas[i], coseno(vec_query, self._vectores[i]))
            for i in indices_validos
            if i < len(self._vectores)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:self.n_resultados]

    def _podar(self) -> None:
        """Elimina las entradas de menor importancia cuando el RAG está lleno."""
        # Ordenar por importancia × log(frecuencia) desc
        scored = sorted(
            enumerate(self._entradas),
            key=lambda x: x[1].importancia * math.log(x[1].frecuencia + 1),
            reverse=True,
        )
        keep_indices = sorted([i for i, _ in scored[:self.max_entradas]])
        self._entradas = [self._entradas[i] for i in keep_indices]
        self._vectores = [self._vectores[i] for i in keep_indices if i < len(self._vectores)]
        self._construir_indice()

    def eliminar_por_nivel(self, nivel: int) -> int:
        """
        Elimina del RAG todas las entradas de un nivel dado.

        Se usa cuando L3 entra en fine-tune y ya no necesita estar en el RAG.
        Returns: cantidad de entradas eliminadas
        """
        antes = len(self._entradas)
        indices_mantener = [i for i, e in enumerate(self._entradas) if e.nivel != nivel]
        self._entradas = [self._entradas[i] for i in indices_mantener]
        self._vectores = [self._vectores[i] for i in indices_mantener if i < len(self._vectores)]
        self._construir_indice()
        self._guardar()
        return antes - len(self._entradas)

    def stats(self) -> dict:
        """Estadísticas del estado actual del RAG."""
        niveles = {1: 0, 2: 0, 3: 0}
        for e in self._entradas:
            niveles[e.nivel] = niveles.get(e.nivel, 0) + 1
        return {
            "total_entradas": len(self._entradas),
            "nivel_1_rag": niveles.get(1, 0),
            "nivel_2_alta_prio": niveles.get(2, 0),
            "nivel_3_finetune": niveles.get(3, 0),
            "modo_encoder": "sentence-transformers" if self.encoder.modo_completo else "tfidf-fallback",
            "indice_faiss": self._indice is not None,
        }

    def formatear_contexto(
        self, resultados: List[Tuple[EntradaMemoria, float]]
    ) -> str:
        """
        Convierte resultados del RAG en texto de contexto para el prompt.

        Args:
            resultados: Output de recuperar()
        Returns:
            String para insertar antes del prompt del usuario
        """
        if not resultados:
            return ""

        partes = ["[MEMORIA RELEVANTE]"]
        for entrada, score in resultados:
            partes.append(
                f"--- (similitud: {score:.2f}, importancia: {entrada.importancia:.2f}) ---\n"
                f"{entrada.texto.strip()}"
            )
        partes.append("[/MEMORIA RELEVANTE]")
        return "\n".join(partes)
