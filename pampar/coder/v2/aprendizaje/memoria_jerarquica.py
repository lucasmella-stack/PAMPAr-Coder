# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Memoria Jerárquica con Compresión Pareto Recursiva.

Principio de Pareto aplicado a la memoria del modelo:
  De 100,000 tokens procesados, solo ~20% es realmente importante.
  De ese 20%, solo ~20% es esencial.
  Y así recursivamente hasta que el conocimiento se INTERIORIZA
  en los pesos del modelo — como aprender a caminar.

Niveles de memoria (inspirado en neurociencia):

  L0 — Memoria de Trabajo (Working Memory)
       Últimos N tokens, resolución completa.
       Capacidad: ~4K tokens. Se vacía constantemente.
       Análogo: Hipocampo, cortex prefrontal.

  L1 — Memoria de Corto Plazo (Short-term)
       20% más importante de L0, comprimido.
       Capacidad: ~10K representaciones comprimidas.
       Criterio: tokens con alta loss, patrones novedosos.
       Análogo: Hipocampo → cortex temporal.

  L2 — Memoria de Largo Plazo (Long-term)
       20% más importante de L1, muy comprimido.
       Capacidad: ~5K embeddings promediados.
       Criterio: patrones que se repiten Y causan errores.
       Análogo: Neocórtex (memorias semánticas consolidadas).

  L3 — Conocimiento Interiorizado (Procedural)
       20% de L2 → actualiza pesos del modelo directamente.
       Capacidad: infinita (son los pesos).
       Análogo: Cerebelo, ganglios basales.
       "No recuerdo cuándo aprendí a caminar, pero camino."

Flujo:
  Datos → L0 (todo) → L1 (20% importante) → L2 (4% esencial)
  → L3 (0.8% que se convierte en conocimiento procedimental)

  Cada nivel comprime con una función de scoring distinta:
  - L0→L1: Importancia = f(loss, novedad, diversidad territorial)
  - L1→L2: Importancia = f(frecuencia, consistencia, impacto en loss)
  - L2→L3: Importancia = f(estabilidad, generalización)

Innovación PAMPAr: Los territorios de Brodmann permiten scoring
más inteligente que en un transformer estándar. Sabemos QUÉ tipo
de token es cada uno, así que podemos priorizar mejor.
"""

import json
import math
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# ENTRADA DE MEMORIA (compartida entre niveles)
# =============================================================================

@dataclass
class EntradaMemoria:
    """Una unidad de memoria en cualquier nivel."""

    # Contenido
    tokens: Tuple[int, ...]         # Ventana de token IDs
    embedding: Optional[torch.Tensor] = None  # Representación comprimida

    # Scoring
    importancia: float = 0.0        # Score de importancia [0, 1]
    loss_media: float = 0.0         # Loss promedio en este patrón
    novedad: float = 0.0            # Qué tan novedoso es (vs lo visto antes)
    frecuencia: int = 1             # Cuántas veces se ha visto
    territorio_dominante: int = -1  # Territorio más activo (0-3)

    # Metadata
    timestamp: float = field(default_factory=time.time)
    nivel_origen: int = 0           # De qué nivel vino
    veces_comprimido: int = 0       # Cuántas compresiones ha sobrevivido

    def score_pareto(self) -> float:
        """
        Score compuesto para decidir si esta entrada sobrevive
        la próxima compresión Pareto.

        Prioriza:
        1. Alta loss (el modelo no sabe esto → necesita aprenderlo)
        2. Novedad (patrones nuevos > patrones repetidos)
        3. Diversidad territorial (patrones multi-territorio > mono)
        4. Frecuencia moderada (muy raro = ruido, muy común = ya sabido)
        """
        # Loss alta = más importante (el modelo necesita esto)
        loss_score = min(self.loss_media / 10.0, 1.0)

        # Novedad decae con frecuencia (pero no a 0)
        freq_factor = 1.0 / (1.0 + math.log1p(self.frecuencia))

        # Bonus por sobrevivir compresiones previas
        survival_bonus = 1.0 + 0.1 * self.veces_comprimido

        return (
            0.5 * loss_score
            + 0.3 * self.novedad * freq_factor
            + 0.2 * self.importancia
        ) * survival_bonus


# =============================================================================
# NIVEL DE MEMORIA
# =============================================================================

class NivelMemoria:
    """
    Un nivel en la jerarquía de memoria.

    Implementa un buffer con capacidad fija. Cuando se llena,
    aplica compresión Pareto: retiene el top 20% y descarta el resto.
    El 20% retenido se promueve al siguiente nivel.
    """

    def __init__(
        self,
        nombre: str,
        capacidad: int,
        ratio_pareto: float = 0.2,
    ):
        self.nombre = nombre
        self.capacidad = capacidad
        self.ratio_pareto = ratio_pareto

        # Buffer ordenado por timestamp
        self.buffer: OrderedDict[Tuple[int, ...], EntradaMemoria] = OrderedDict()

        # Estadísticas
        self.total_recibidos = 0
        self.total_promovidos = 0
        self.total_descartados = 0
        self.compresiones_realizadas = 0

    @property
    def uso(self) -> float:
        """Porcentaje de uso del buffer."""
        return len(self.buffer) / self.capacidad if self.capacidad > 0 else 0.0

    @property
    def lleno(self) -> bool:
        """¿Está lleno?"""
        return len(self.buffer) >= self.capacidad

    def agregar(self, entrada: EntradaMemoria) -> bool:
        """
        Agrega una entrada al buffer.

        Returns:
            True si se agregó, False si fue descartada.
        """
        key = entrada.tokens
        self.total_recibidos += 1

        # Si ya existe, actualizar frecuencia y promediar loss
        if key in self.buffer:
            existente = self.buffer[key]
            existente.frecuencia += 1
            existente.loss_media = (
                existente.loss_media * 0.7 + entrada.loss_media * 0.3
            )
            # Mover al final (más reciente)
            self.buffer.move_to_end(key)
            return True

        # Si hay espacio, agregar directamente
        if not self.lleno:
            self.buffer[key] = entrada
            return True

        # Buffer lleno → la entrada nueva compite con la peor existente
        peor_key = min(self.buffer, key=lambda k: self.buffer[k].score_pareto())
        if entrada.score_pareto() > self.buffer[peor_key].score_pareto():
            del self.buffer[peor_key]
            self.buffer[key] = entrada
            self.total_descartados += 1
            return True

        self.total_descartados += 1
        return False

    def comprimir_pareto(self) -> List[EntradaMemoria]:
        """
        Aplica compresión Pareto: retiene top 20%, descarta el resto.

        Returns:
            Lista de entradas promovidas (top 20%) para el siguiente nivel.
        """
        if not self.buffer:
            return []

        n_total = len(self.buffer)
        n_retener = max(1, int(n_total * self.ratio_pareto))

        # Rankear por score Pareto
        ranking = sorted(
            self.buffer.values(),
            key=lambda e: e.score_pareto(),
            reverse=True,
        )

        # Top 20% → promovidas al siguiente nivel
        promovidas = ranking[:n_retener]
        for p in promovidas:
            p.veces_comprimido += 1

        # Vaciar buffer
        descartadas = n_total - n_retener
        self.buffer.clear()

        # Estadísticas
        self.total_promovidos += n_retener
        self.total_descartados += descartadas
        self.compresiones_realizadas += 1

        return promovidas

    def stats(self) -> Dict:
        """Estadísticas del nivel."""
        scores = [e.score_pareto() for e in self.buffer.values()]
        return {
            "nombre": self.nombre,
            "entradas": len(self.buffer),
            "capacidad": self.capacidad,
            "uso_pct": round(self.uso * 100, 1),
            "total_recibidos": self.total_recibidos,
            "total_promovidos": self.total_promovidos,
            "total_descartados": self.total_descartados,
            "compresiones": self.compresiones_realizadas,
            "score_medio": round(sum(scores) / max(len(scores), 1), 4),
            "score_max": round(max(scores, default=0), 4),
        }


# =============================================================================
# MEMORIA JERÁRQUICA COMPLETA
# =============================================================================

class MemoriaJerarquica:
    """
    Sistema de memoria jerárquica con compresión Pareto recursiva.

    4 niveles que imitan la consolidación de memoria humana:
    - L0: Memoria de trabajo (todo, efímero)
    - L1: Corto plazo (20% importante)
    - L2: Largo plazo (4% esencial)
    - L3: Interiorizado (0.8% → pesos del modelo)

    Uso en training loop:
        memoria = MemoriaJerarquica()

        for batch in dataloader:
            logits, loss, info = model(input_ids, targets)

            # Registrar lo que el modelo acaba de ver
            memoria.procesar_batch(
                input_ids=input_ids,
                per_token_loss=per_token_loss,
                terr_acts=terr_acts,
                hidden_states=hidden_states,
            )

            # Periódicamente: consolidar e interiorizar
            if step % consolidation_interval == 0:
                updates = memoria.consolidar(model)

    La interiorización (L3) actualiza directamente los
    pesos del modelo — es el equivalente a "ya no recuerdo
    cuándo aprendí esto, pero lo sé hacer".
    """

    def __init__(
        self,
        capacidad_l0: int = 4096,
        capacidad_l1: int = 10000,
        capacidad_l2: int = 5000,
        ventana_tokens: int = 16,
        ratio_pareto: float = 0.2,
        umbral_loss_alta: float = 3.0,
        lr_interiorizacion: float = 1e-5,
    ):
        """
        Args:
            capacidad_l0: Tokens en memoria de trabajo.
            capacidad_l1: Entradas en corto plazo.
            capacidad_l2: Entradas en largo plazo.
            ventana_tokens: Tamaño de ventana para crear entradas.
            ratio_pareto: Fracción a retener (0.2 = Pareto 80/20).
            umbral_loss_alta: Loss mínima para considerar "importante".
            lr_interiorizacion: Learning rate para actualizar pesos (L3).
        """
        self.ventana = ventana_tokens
        self.ratio_pareto = ratio_pareto
        self.umbral_loss_alta = umbral_loss_alta
        self.lr_interiorizacion = lr_interiorizacion

        # 4 niveles de memoria
        self.l0 = NivelMemoria("L0_trabajo", capacidad_l0, ratio_pareto)
        self.l1 = NivelMemoria("L1_corto_plazo", capacidad_l1, ratio_pareto)
        self.l2 = NivelMemoria("L2_largo_plazo", capacidad_l2, ratio_pareto)

        # L3 no es un buffer — son actualizaciones acumuladas a los pesos
        self.l3_updates: List[Dict[str, torch.Tensor]] = []
        self.total_interiorizados = 0

        # Tracking de novedad: qué patrones ya hemos visto
        self._patrones_vistos: Dict[Tuple[int, ...], int] = defaultdict(int)
        self._total_procesados = 0

    def procesar_batch(
        self,
        input_ids: torch.Tensor,           # [B, L]
        per_token_loss: torch.Tensor,       # [B, L]
        terr_acts: Optional[torch.Tensor] = None,  # [B, L, 4]
        hidden_states: Optional[torch.Tensor] = None,  # [B, L, D]
    ) -> Dict:
        """
        Procesa un batch de entrenamiento, extrayendo los patrones
        importantes para la memoria.

        Returns:
            Dict con estadísticas del procesamiento.
        """
        B, L = input_ids.shape
        entradas_creadas = 0
        self._total_procesados += B * L

        for b in range(B):
            # Recorrer secuencia con ventana deslizante
            for pos in range(self.ventana, L):
                token_loss = per_token_loss[b, pos].item()

                # Solo recordar lo importante (alta loss = no lo sabe)
                if token_loss < self.umbral_loss_alta:
                    continue

                # Extraer ventana de tokens
                start = pos - self.ventana
                tokens = tuple(input_ids[b, start:pos + 1].cpu().tolist())

                # Calcular novedad
                veces = self._patrones_vistos[tokens]
                self._patrones_vistos[tokens] += 1
                novedad = 1.0 / (1.0 + veces)

                # Territorio dominante
                terr_dom = -1
                if terr_acts is not None:
                    # terr_acts puede tener L' < L (computado sobre input_ids[:, :-1])
                    terr_pos = min(pos, terr_acts.shape[1] - 1)
                    terr_dom = int(terr_acts[b, terr_pos].argmax().item())

                # Embedding comprimido (promedio de la ventana)
                emb = None
                if hidden_states is not None:
                    emb = hidden_states[b, start:pos + 1].mean(dim=0).detach().cpu()

                entrada = EntradaMemoria(
                    tokens=tokens,
                    embedding=emb,
                    importancia=min(token_loss / 10.0, 1.0),
                    loss_media=token_loss,
                    novedad=novedad,
                    territorio_dominante=terr_dom,
                    nivel_origen=0,
                )

                self.l0.agregar(entrada)
                entradas_creadas += 1

        # Si L0 está lleno, comprimir automáticamente
        promovidas_a_l1 = 0
        if self.l0.lleno:
            promovidas = self.l0.comprimir_pareto()
            for p in promovidas:
                p.nivel_origen = 1
                self.l1.agregar(p)
                promovidas_a_l1 += 1

        return {
            "entradas_creadas": entradas_creadas,
            "promovidas_l0_l1": promovidas_a_l1,
            "l0_uso": self.l0.uso,
            "l1_uso": self.l1.uso,
        }

    def consolidar(
        self,
        model: Optional[nn.Module] = None,
    ) -> Dict:
        """
        Consolidación periódica: comprime niveles y opcionalmente
        interioriza conocimiento en los pesos del modelo.

        Analogía: El "sueño" del modelo.

        Args:
            model: Si se proporciona, aplica interiorización L3.

        Returns:
            Dict con métricas de consolidación.
        """
        metricas: Dict = {
            "compresiones": {},
            "interiorizacion": None,
        }

        # L1 → L2: Comprimir corto plazo a largo plazo
        if self.l1.lleno:
            promovidas = self.l1.comprimir_pareto()
            for p in promovidas:
                p.nivel_origen = 2
                self.l2.agregar(p)
            metricas["compresiones"]["l1_a_l2"] = len(promovidas)

        # L2 → L3: Interiorizar en pesos del modelo
        if self.l2.lleno and model is not None:
            promovidas = self.l2.comprimir_pareto()
            resultado = self._interiorizar(model, promovidas)
            metricas["interiorizacion"] = resultado

        # Estadísticas globales
        metricas["niveles"] = {
            "l0": self.l0.stats(),
            "l1": self.l1.stats(),
            "l2": self.l2.stats(),
            "l3_interiorizados": self.total_interiorizados,
        }

        return metricas

    def _interiorizar(
        self,
        model: nn.Module,
        entradas: List[EntradaMemoria],
    ) -> Dict:
        """
        Interiorización: convierte memorias L2 en actualizaciones
        de pesos del modelo.

        Similar a cómo el cerebro convierte memorias episódicas
        (hipocampo) en conocimiento procedimental (cerebelo).

        Método: Micro-gradient step con los patrones más importantes.
        Se crea un mini-batch con esos patrones y se hace un paso
        de optimización con learning rate muy bajo.

        Args:
            model: PampaRCoderV2
            entradas: Patrones a interiorizar.

        Returns:
            Dict con métricas.
        """
        if not entradas:
            return {"n_patrones": 0, "skip": True}

        # Filtrar entradas sin tokens válidos
        validas = [e for e in entradas if len(e.tokens) >= 2]
        if not validas:
            return {"n_patrones": 0, "skip": True}

        device = next(model.parameters()).device

        # Crear mini-batch con los patrones a interiorizar
        max_len = max(len(e.tokens) for e in validas)
        batch_size = min(len(validas), 32)  # Cap para memoria

        input_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        targets = torch.full(
            (batch_size, max_len), -100, dtype=torch.long, device=device
        )

        for i, entrada in enumerate(validas[:batch_size]):
            toks = torch.tensor(entrada.tokens, dtype=torch.long)
            input_ids[i, : len(toks)] = toks
            # Target = shifted: predecir siguiente token
            targets[i, : len(toks) - 1] = toks[1:]

        # Micro-step con LR muy bajo (no queremos olvidar lo anterior)
        model.train()
        was_training = model.training

        # Forward
        logits, loss, _ = model(input_ids, targets)

        if loss is not None:
            # Escalar loss por LR de interiorización
            scaled_loss = loss * self.lr_interiorizacion
            scaled_loss.backward()

            # Aplicar gradientes directamente (sin optimizer)
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        param.data -= self.lr_interiorizacion * param.grad
                        param.grad.zero_()

        self.total_interiorizados += len(validas[:batch_size])

        if not was_training:
            model.eval()

        return {
            "n_patrones": len(validas[:batch_size]),
            "loss_media": loss.item() if loss is not None else 0.0,
            "lr": self.lr_interiorizacion,
        }

    # =========================================================================
    # SCORING INTELIGENTE POR TERRITORIO
    # =========================================================================

    def get_replay_batch(
        self,
        batch_size: int = 16,
        nivel: str = "l1",
        strategy: str = "hardest",
    ) -> Optional[torch.Tensor]:
        """
        Genera un mini-batch de replay desde la memoria.

        Estrategias:
        - "hardest": Los patrones con mayor loss media.
        - "diverse": Mezcla de territorios para balance.
        - "novel": Los patrones más novedosos.
        - "pareto": Top Pareto scoring.

        Returns:
            [batch_size, ventana+1] tensor de token IDs, o None.
        """
        nivel_obj = {"l0": self.l0, "l1": self.l1, "l2": self.l2}[nivel]

        if len(nivel_obj.buffer) < batch_size:
            return None

        entradas = list(nivel_obj.buffer.values())

        if strategy == "hardest":
            entradas.sort(key=lambda e: e.loss_media, reverse=True)
        elif strategy == "novel":
            entradas.sort(key=lambda e: e.novedad, reverse=True)
        elif strategy == "diverse":
            # Seleccionar uniformemente entre territorios
            by_terr: Dict[int, List[EntradaMemoria]] = defaultdict(list)
            for e in entradas:
                by_terr[e.territorio_dominante].append(e)
            entradas = []
            per_terr = max(1, batch_size // max(len(by_terr), 1))
            for terr_entries in by_terr.values():
                terr_entries.sort(key=lambda e: e.score_pareto(), reverse=True)
                entradas.extend(terr_entries[:per_terr])
        else:  # pareto
            entradas.sort(key=lambda e: e.score_pareto(), reverse=True)

        selected = entradas[:batch_size]
        max_len = max(len(e.tokens) for e in selected)

        batch = torch.zeros(len(selected), max_len, dtype=torch.long)
        for i, e in enumerate(selected):
            toks = torch.tensor(e.tokens, dtype=torch.long)
            batch[i, : len(toks)] = toks

        return batch

    # =========================================================================
    # PERSISTENCIA
    # =========================================================================

    def guardar(self, path: str) -> None:
        """Guarda toda la jerarquía a disco."""
        data = {
            "config": {
                "ventana": self.ventana,
                "ratio_pareto": self.ratio_pareto,
                "umbral_loss_alta": self.umbral_loss_alta,
                "lr_interiorizacion": self.lr_interiorizacion,
                "capacidades": {
                    "l0": self.l0.capacidad,
                    "l1": self.l1.capacidad,
                    "l2": self.l2.capacidad,
                },
            },
            "stats": {
                "total_procesados": self._total_procesados,
                "total_interiorizados": self.total_interiorizados,
            },
            "niveles": {
                "l0": self._serializar_nivel(self.l0),
                "l1": self._serializar_nivel(self.l1),
                "l2": self._serializar_nivel(self.l2),
            },
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def cargar(cls, path: str) -> "MemoriaJerarquica":
        """Carga la jerarquía desde disco."""
        data = json.loads(Path(path).read_text())
        cfg = data["config"]
        caps = cfg["capacidades"]

        mem = cls(
            capacidad_l0=caps["l0"],
            capacidad_l1=caps["l1"],
            capacidad_l2=caps["l2"],
            ventana_tokens=cfg["ventana"],
            ratio_pareto=cfg["ratio_pareto"],
            umbral_loss_alta=cfg["umbral_loss_alta"],
            lr_interiorizacion=cfg["lr_interiorizacion"],
        )

        mem._total_procesados = data["stats"]["total_procesados"]
        mem.total_interiorizados = data["stats"]["total_interiorizados"]

        # Restaurar buffers
        for nivel_str, nivel_obj in [
            ("l0", mem.l0),
            ("l1", mem.l1),
            ("l2", mem.l2),
        ]:
            for entry in data["niveles"][nivel_str]["entries"]:
                tokens = tuple(entry["tokens"])
                nivel_obj.buffer[tokens] = EntradaMemoria(
                    tokens=tokens,
                    importancia=entry["importancia"],
                    loss_media=entry["loss_media"],
                    novedad=entry["novedad"],
                    frecuencia=entry["frecuencia"],
                    territorio_dominante=entry["territorio_dominante"],
                    timestamp=entry["timestamp"],
                    nivel_origen=entry["nivel_origen"],
                    veces_comprimido=entry["veces_comprimido"],
                )

        return mem

    def _serializar_nivel(self, nivel: NivelMemoria) -> Dict:
        """Serializa un nivel a dict."""
        return {
            "stats": nivel.stats(),
            "entries": [
                {
                    "tokens": list(e.tokens),
                    "importancia": e.importancia,
                    "loss_media": e.loss_media,
                    "novedad": e.novedad,
                    "frecuencia": e.frecuencia,
                    "territorio_dominante": e.territorio_dominante,
                    "timestamp": e.timestamp,
                    "nivel_origen": e.nivel_origen,
                    "veces_comprimido": e.veces_comprimido,
                }
                for e in nivel.buffer.values()
            ],
        }

    # =========================================================================
    # ESTADÍSTICAS
    # =========================================================================

    def stats(self) -> Dict:
        """Estadísticas completas del sistema de memoria."""
        return {
            "total_tokens_procesados": self._total_procesados,
            "total_interiorizados_l3": self.total_interiorizados,
            "niveles": {
                "l0": self.l0.stats(),
                "l1": self.l1.stats(),
                "l2": self.l2.stats(),
            },
            "ratio_compresion_efectiva": (
                round(
                    self.total_interiorizados
                    / max(self._total_procesados, 1)
                    * 100,
                    4,
                )
            ),
        }

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"MemoriaJerarquica("
            f"procesados={s['total_tokens_procesados']:,}, "
            f"L0={s['niveles']['l0']['entradas']}, "
            f"L1={s['niveles']['l1']['entradas']}, "
            f"L2={s['niveles']['l2']['entradas']}, "
            f"L3_interiorizado={s['total_interiorizados_l3']}, "
            f"compresión={s['ratio_compresion_efectiva']}%)"
        )
