# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Sistema de Memoria Residual — RAG Pareto + Cola de Fine-tune.

Pipeline:
    Usuario envía código / texto
        → Clasificador Pareto: ¿qué es importante?
        → RAGResidual: guarda con score en vector DB local (FAISS)
        → Modelo usa RAGResidual como contexto externo (retrieval)

    Cuando L3 tiene suficiente data de alta calidad:
        → ColaFinetune propone un fine-tune
        → Usuario acepta → se lanza training → L3 se vacía
        → El conocimiento queda en los pesos (como aprender a caminar)
"""

from .clasificador import ClasificadorPareto, EntradaMemoria
from .rag import RAGResidual
from .cola_finetune import ColaFinetune

__all__ = [
    "ClasificadorPareto",
    "EntradaMemoria",
    "RAGResidual",
    "ColaFinetune",
]
