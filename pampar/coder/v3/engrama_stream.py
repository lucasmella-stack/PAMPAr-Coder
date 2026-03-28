# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
EngramaStream — Memoria de activaciones para inferencia adaptativa.

Captura activaciones de forward passes exitosos, indexadas por routing
territorial (Tálamo), y las re-inyecta como residual en futuros forward
passes cuando el modelo encuentra patrones similares.

El modelo mejora con cada uso exitoso SIN retraining.

Componentes:
  BancoEngrama   — almacén de activaciones indexado por (territorio, zona)
  EngramaCapture — lógica de captura post-inferencia
  EngramaInject  — lógica de inyección durante forward pass

Flujo:
  1. El modelo genera tokens (forward normal)
  2. Se evalúa la calidad (AST parse, loss, etc.)
  3. Si calidad > umbral → EngramaCapture graba activaciones por nivel
  4. En futuros forward passes → EngramaInject busca en el banco
     e inyecta residuales antes de los FFNs

La clave de búsqueda es (territorio_dominante, zona_top) que el Tálamo
ya calcula — no necesita compute adicional.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F


@dataclass
class Engrama:
    """Una activación capturada de un forward pass exitoso."""

    nivel: int                          # nivel de profundidad (0-4)
    activacion: torch.Tensor            # [D] vector de activación promedio
    territorio: int                     # territorio dominante (0-3)
    zona: int                           # zona top (0-51)
    score: float = 0.0                  # calidad del forward pass fuente
    timestamp: float = field(default_factory=time.time)
    n_usos: int = 0                     # veces inyectado
    decaimiento: float = 1.0            # factor temporal (se reduce con el tiempo)


class BancoEngrama:
    """
    Almacén de activaciones indexado por (territorio, zona, nivel).

    Cada entrada guarda un vector de dim=640 que representa el patrón
    de activación exitoso para ese tipo de token en ese nivel.

    Lookup: O(1) por clave, O(K) para promediar K engramas.
    Memoria: ~640 floats × 52 zonas × 5 niveles × K engramas
           = 640 × 260 × K × 4 bytes. Con K=10: ~6.7 MB.
    """

    def __init__(
        self,
        dim: int = 640,
        max_engramas_por_clave: int = 10,
        score_minimo: float = 0.5,
        decaimiento_por_hora: float = 0.99,
    ):
        self.dim = dim
        self.max_k = max_engramas_por_clave
        self.score_minimo = score_minimo
        self.decaimiento_hora = decaimiento_por_hora

        # banco[nivel][(territorio, zona)] = List[Engrama]
        self._banco: Dict[int, Dict[Tuple[int, int], List[Engrama]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._total_engramas = 0
        self._total_inyecciones = 0

    @property
    def total_engramas(self) -> int:
        return self._total_engramas

    @property
    def total_inyecciones(self) -> int:
        return self._total_inyecciones

    def agregar(
        self,
        nivel: int,
        territorio: int,
        zona: int,
        activacion: torch.Tensor,
        score: float,
    ) -> None:
        """
        Agrega un engrama al banco.

        Si la clave ya tiene max_k engramas, reemplaza el de menor score.

        Args:
            nivel:      nivel de profundidad (0-4)
            territorio: territorio dominante (0-3)
            zona:       zona principal (0-51)
            activacion: [D] vector de activación
            score:      calidad del forward pass fuente (0-1)
        """
        if score < self.score_minimo:
            return

        clave = (territorio, zona)
        lista = self._banco[nivel][clave]

        eng = Engrama(
            nivel=nivel,
            activacion=activacion.detach().cpu(),
            territorio=territorio,
            zona=zona,
            score=score,
        )

        if len(lista) < self.max_k:
            lista.append(eng)
            self._total_engramas += 1
        else:
            # Reemplazar el de menor score
            min_idx = min(range(len(lista)), key=lambda i: lista[i].score)
            if lista[min_idx].score < score:
                lista[min_idx] = eng

    def buscar(
        self,
        nivel: int,
        territorio: int,
        zona: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """
        Busca engramas para una clave y devuelve el promedio ponderado.

        Ponderación por score × decaimiento temporal.

        Args:
            nivel:      nivel de profundidad
            territorio: territorio dominante del token actual
            zona:       zona principal del token actual
            device:     device del modelo

        Returns:
            [D] vector promedio ponderado, o None si no hay engramas
        """
        clave = (territorio, zona)
        lista = self._banco[nivel].get(clave)
        if not lista:
            return None

        # Aplicar decaimiento temporal
        ahora = time.time()
        for eng in lista:
            horas = (ahora - eng.timestamp) / 3600
            eng.decaimiento = self.decaimiento_hora ** horas

        # Promedio ponderado por score * decaimiento
        pesos = torch.tensor(
            [e.score * e.decaimiento for e in lista],
            dtype=torch.float32,
        )
        peso_total = pesos.sum()
        if peso_total < 1e-8:
            return None

        pesos = pesos / peso_total

        resultado = torch.zeros(self.dim, dtype=torch.float32)
        for i, eng in enumerate(lista):
            resultado += pesos[i] * eng.activacion
            eng.n_usos += 1

        self._total_inyecciones += 1
        return resultado.to(device)

    def buscar_batch(
        self,
        nivel: int,
        territorios: torch.Tensor,
        zonas: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Búsqueda batch para una secuencia completa.

        Args:
            nivel:       nivel de profundidad
            territorios: [L] territorio dominante por token
            zonas:       [L] zona principal por token
            device:      device del modelo

        Returns:
            engramas: [L, D] vectores de engrama (cero si no hay match)
            mask:     [L] bool, True si hay engrama disponible
        """
        L = territorios.shape[0]
        engramas = torch.zeros(L, self.dim, dtype=torch.float32)
        mask = torch.zeros(L, dtype=torch.bool)

        for pos in range(L):
            t = territorios[pos].item()
            z = zonas[pos].item()
            eng = self.buscar(nivel, t, z, torch.device("cpu"))
            if eng is not None:
                engramas[pos] = eng
                mask[pos] = True

        return engramas.to(device), mask.to(device)

    def stats(self) -> Dict[str, object]:
        """Estadísticas del banco."""
        por_nivel: Dict[int, int] = {}
        por_territorio: Dict[int, int] = defaultdict(int)
        por_zona: Dict[int, int] = defaultdict(int)
        scores: List[float] = []

        for nivel, claves in self._banco.items():
            count = sum(len(v) for v in claves.values())
            por_nivel[nivel] = count
            for (t, z), engs in claves.items():
                por_territorio[t] += len(engs)
                por_zona[z] += len(engs)
                scores.extend(e.score for e in engs)

        return {
            "total": self._total_engramas,
            "inyecciones": self._total_inyecciones,
            "por_nivel": dict(por_nivel),
            "por_territorio": dict(por_territorio),
            "score_mean": sum(scores) / len(scores) if scores else 0.0,
            "claves_unicas": sum(
                len(claves) for claves in self._banco.values()
            ),
        }

    def guardar(self, ruta: Path) -> None:
        """Persiste el banco a disco."""
        ruta.parent.mkdir(parents=True, exist_ok=True)
        data: List[Dict] = []
        for nivel, claves in self._banco.items():
            for (t, z), engs in claves.items():
                for eng in engs:
                    data.append({
                        "nivel": nivel,
                        "territorio": t,
                        "zona": z,
                        "activacion": eng.activacion.tolist(),
                        "score": eng.score,
                        "timestamp": eng.timestamp,
                        "n_usos": eng.n_usos,
                    })
        ruta.write_text(
            json.dumps(data, ensure_ascii=False),
            encoding="utf-8",
        )

    def cargar(self, ruta: Path) -> int:
        """Carga el banco desde disco."""
        if not ruta.exists():
            return 0
        data = json.loads(ruta.read_text(encoding="utf-8"))
        count = 0
        for item in data:
            eng = Engrama(
                nivel=item["nivel"],
                activacion=torch.tensor(item["activacion"], dtype=torch.float32),
                territorio=item["territorio"],
                zona=item["zona"],
                score=item["score"],
                timestamp=item.get("timestamp", time.time()),
                n_usos=item.get("n_usos", 0),
            )
            clave = (eng.territorio, eng.zona)
            self._banco[eng.nivel][clave].append(eng)
            count += 1
        self._total_engramas = count
        return count


class EngramaCapture:
    """
    Captura activaciones de forward passes exitosos.

    Se invoca DESPUÉS de que el modelo genera y se evalúa la calidad.
    Usa el GhidraProbe para acceder a las activaciones capturadas.
    """

    def __init__(self, banco: BancoEngrama, score_minimo: float = 0.6):
        self.banco = banco
        self.score_minimo = score_minimo

    @torch.no_grad()
    def capturar_desde_probe(
        self,
        probe_raw: Dict[str, Any],
        score: float,
        n_levels: int = 5,
    ) -> int:
        """
        Extrae activaciones del GhidraProbe y las almacena en el banco.

        Args:
            probe_raw:  dict de tensores raw del GhidraProbe
            score:      calidad evaluada del forward pass (0-1)
            n_levels:   número de niveles del modelo

        Returns:
            Número de engramas almacenados
        """
        if score < self.score_minimo:
            return 0

        count = 0

        # Para cada nivel, capturar x_attn (salida de atención) por (territorio, zona)
        # Usamos x_attn en vez de streams_out para que la escala coincida
        # con el punto de inyección (paso 2.5 en NivelProfundo)
        terr_n0 = probe_raw.get("terr_acts_n0")  # [B, L, 4]
        zona_n0 = probe_raw.get("zona_acts_n0")   # [B, L, 52]

        if terr_n0 is None or zona_n0 is None:
            return 0

        # Territorio y zona dominantes por token
        terr_dom = terr_n0[0].argmax(dim=-1)  # [L]
        zona_dom = zona_n0[0].argmax(dim=-1)  # [L]
        L = terr_dom.shape[0]

        for nivel_idx in range(n_levels):
            # Preferir x_attn (escala correcta para inyección)
            # Fallback a streams_out si x_attn no está disponible
            x_attn_key = f"x_attn_n{nivel_idx}"
            x_attn = probe_raw.get(x_attn_key)  # [B, L, D]

            if x_attn is not None:
                x_out = x_attn[0]  # [L, D] — ya es la salida de atención
            else:
                # Fallback: usar streams_out combinados (escala diferente)
                streams_key = f"streams_out_n{nivel_idx}"
                streams = probe_raw.get(streams_key)
                terr_key = f"terr_acts_n{nivel_idx}"
                terr_acts = probe_raw.get(terr_key)
                if streams is None or terr_acts is None:
                    continue
                n_streams = len(streams)
                x_out = sum(
                    streams[t][0] * terr_acts[0, :, t:t + 1]
                    for t in range(n_streams)
                )  # [L, D]

            # Agrupar tokens por (territorio, zona) y promediar
            seen: Dict[Tuple[int, int], List[torch.Tensor]] = defaultdict(list)
            for pos in range(L):
                t = terr_dom[pos].item()
                z = zona_dom[pos].item()
                seen[(t, z)].append(x_out[pos])

            for (t, z), vecs in seen.items():
                avg = torch.stack(vecs).mean(dim=0)  # [D]
                self.banco.agregar(nivel_idx, t, z, avg, score)
                count += 1

        return count

    @torch.no_grad()
    def capturar_directo(
        self,
        model: object,
        input_ids: torch.Tensor,
        score: float,
    ) -> int:
        """
        Captura directa sin GhidraProbe (forward pass adicional).

        Más simple pero requiere un forward pass extra.
        Usar cuando no hay probe activo.
        """
        if score < self.score_minimo:
            return 0

        from .modelo import PamparV3
        from .talamo import TalamoInicial

        assert isinstance(model, PamparV3)

        model.eval()
        config = model.config
        count = 0

        x = model.emb_drop(model.tok_emb(input_ids))
        terr_acts, zona_acts = model.talamo(x, input_ids)
        streams = [x.clone() for _ in range(config.n_streams)]

        terr_dom = terr_acts[0].argmax(dim=-1)  # [L]
        zona_dom = zona_acts[0].argmax(dim=-1)   # [L]
        L = terr_dom.shape[0]

        for nivel_idx, nivel in enumerate(model.niveles):
            streams, terr_acts, _ = nivel(
                streams, terr_acts, TalamoInicial.agregar_fn
            )

            x_out = sum(
                streams[t][0] * terr_acts[0, :, t:t + 1]
                for t in range(config.n_streams)
            )  # [L, D]

            seen: Dict[Tuple[int, int], List[torch.Tensor]] = defaultdict(list)
            for pos in range(L):
                t = terr_dom[pos].item()
                z = zona_dom[pos].item()
                seen[(t, z)].append(x_out[pos])

            for (t, z), vecs in seen.items():
                avg = torch.stack(vecs).mean(dim=0)
                self.banco.agregar(nivel_idx, t, z, avg, score)
                count += 1

        return count
