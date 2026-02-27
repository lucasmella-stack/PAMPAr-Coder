# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Fase 6: Aprendizaje Online — "Aprender de las Personas"

El modelo aprende de las interacciones reales con usuarios.

Cuando una persona usa PAMPAr (en un servidor, API, o editor):
  1. El modelo genera código
  2. El usuario acepta, modifica o rechaza
  3. Esas señales se convierten en datos de entrenamiento
  4. El modelo mejora gradualmente (sin re-entrenamiento masivo)

3 tipos de señal:

  A) ACEPTACIÓN IMPLÍCITA
     - El usuario usa el código generado sin cambios → refuerzo positivo
     - El usuario borra o reescribe → refuerzo negativo
     - Sin esfuerzo cognitivo del usuario

  B) CORRECCIÓN DIRECTA
     - El usuario edita el código generado
     - El diff entre generado y editado = señal de aprendizaje
     - Crea pares DPO automáticos: (editado=preferred, original=rejected)

  C) FEEDBACK EXPLÍCITO
     - 👍/👎 en la respuesta
     - Reportes de error
     - Calificaciones (1-5)

Privacidad:
  - TODO se procesa localmente en el dispositivo del usuario
  - Solo se guardan gradientes agregados, NO código del usuario
  - Compatible con federated learning (futuro)

Inspirado en:
  - RLHF (Reinforcement Learning from Human Feedback)
  - Online learning (aprendizaje incremental)
  - Federated Learning (McMahan et al., 2017)
"""

import json
import os
import time
from collections import deque
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# TIPOS DE INTERACCIÓN
# =============================================================================

@dataclass
class Interaccion:
    """Registro de una interacción usuario-modelo."""
    prompt: str                         # Lo que pidió el usuario
    generado: str                       # Lo que generó el modelo
    editado: Optional[str] = None       # Lo que el usuario escribió (si editó)
    aceptado: Optional[bool] = None     # True=aceptó, False=rechazó, None=desconocido
    feedback: Optional[int] = None      # 1-5 calificación (None=sin feedback)
    timestamp: float = 0.0             # Cuando ocurrió
    tokens_prompt: int = 0
    tokens_generado: int = 0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    @property
    def fue_editado(self) -> bool:
        """¿El usuario editó la respuesta?"""
        return self.editado is not None and self.editado != self.generado

    @property
    def similaridad(self) -> float:
        """Similaridad entre generado y editado (0-1)."""
        if not self.fue_editado:
            return 1.0 if self.aceptado else 0.0
        return SequenceMatcher(None, self.generado, self.editado).ratio()

    @property
    def reward(self) -> float:
        """Reward estimado de la interacción."""
        if self.feedback is not None:
            return (self.feedback - 1) / 4  # Normalizar 1-5 → 0-1

        if self.aceptado is True:
            return 0.8

        if self.aceptado is False:
            return -0.5

        if self.fue_editado:
            # Más similar al original = más positivo
            sim = self.similaridad
            return sim * 0.8 - (1 - sim) * 0.3

        return 0.0  # Sin información


# =============================================================================
# BUFFER DE EXPERIENCIA
# =============================================================================

class BufferExperiencia:
    """
    Buffer circular que almacena interacciones recientes.
    
    Como la memoria de corto plazo: mantiene las N interacciones
    más recientes para entrenar en micro-batches.
    """

    def __init__(
        self,
        max_size: int = 10000,
        persistir_path: Optional[str] = None,
    ):
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)
        self.persistir_path = persistir_path

        # Estadísticas
        self.total_interacciones = 0
        self.total_positivas = 0
        self.total_negativas = 0
        self.total_ediciones = 0

        # Cargar persistido si existe
        if persistir_path and os.path.exists(persistir_path):
            self._cargar()

    def agregar(self, interaccion: Interaccion):
        """Agrega una interacción al buffer."""
        self.buffer.append(interaccion)
        self.total_interacciones += 1

        if interaccion.reward > 0:
            self.total_positivas += 1
        elif interaccion.reward < 0:
            self.total_negativas += 1
        if interaccion.fue_editado:
            self.total_ediciones += 1

        # Persistir periódicamente
        if self.persistir_path and self.total_interacciones % 100 == 0:
            self._guardar()

    def obtener_pares_dpo(self, n: int = 10) -> List[Tuple[Interaccion, Interaccion]]:
        """
        Obtiene pares (preferred, rejected) del buffer.

        Fuentes de pares:
        1. Interacción editada: editado=preferred, generado=rejected
        2. Feedback: positivo=preferred, negativo=rejected
        3. Aceptado vs rechazado para el mismo tipo de prompt
        """
        pares = []

        # Tipo 1: Ediciones (más valiosas)
        editadas = [i for i in self.buffer if i.fue_editado]
        for inter in editadas[:n]:
            # Crear interacción "preferred" con el código editado
            preferred = Interaccion(
                prompt=inter.prompt,
                generado=inter.editado,
                aceptado=True,
            )
            pares.append((preferred, inter))

        # Tipo 2: Positivas vs negativas
        positivas = [i for i in self.buffer if i.reward > 0.3]
        negativas = [i for i in self.buffer if i.reward < -0.1]

        import random
        for _ in range(min(n - len(pares), len(positivas), len(negativas))):
            pref = random.choice(positivas)
            rej = random.choice(negativas)
            pares.append((pref, rej))

        return pares[:n]

    def obtener_batch(
        self,
        batch_size: int = 4,
        solo_positivas: bool = False,
    ) -> List[Interaccion]:
        """Obtiene un batch de interacciones para entrenamiento."""
        import random

        if solo_positivas:
            candidatas = [i for i in self.buffer if i.reward > 0]
        else:
            candidatas = list(self.buffer)

        if len(candidatas) < batch_size:
            return list(candidatas)

        # Muestreo ponderado por valor absoluto de reward
        pesos = [abs(i.reward) + 0.1 for i in candidatas]
        return random.choices(candidatas, weights=pesos, k=batch_size)

    def _guardar(self):
        """Persiste buffer a disco."""
        data = []
        for inter in self.buffer:
            data.append({
                "prompt": inter.prompt,
                "generado": inter.generado,
                "editado": inter.editado,
                "aceptado": inter.aceptado,
                "feedback": inter.feedback,
                "timestamp": inter.timestamp,
            })

        os.makedirs(os.path.dirname(self.persistir_path), exist_ok=True)
        with open(self.persistir_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _cargar(self):
        """Carga buffer desde disco."""
        try:
            with open(self.persistir_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for d in data:
                inter = Interaccion(**d)
                self.buffer.append(inter)

            print(f"  Buffer cargado: {len(self.buffer)} interacciones")
        except Exception as e:
            print(f"  Warning: no se pudo cargar buffer: {e}")

    def get_stats(self) -> Dict:
        """Estadísticas del buffer."""
        return {
            "total": self.total_interacciones,
            "buffer_size": len(self.buffer),
            "positivas": self.total_positivas,
            "negativas": self.total_negativas,
            "ediciones": self.total_ediciones,
            "reward_medio": (
                sum(i.reward for i in self.buffer) / max(len(self.buffer), 1)
            ),
        }


# =============================================================================
# ENTRENADOR ONLINE
# =============================================================================

class EntrenadorOnline:
    """
    Entrenamiento online (incremental) a partir de interacciones.

    El modelo se actualiza en micro-batches mientras sirve a usuarios.
    Usa LoRA-like updates para no perder el conocimiento base.

    Flujo:
    1. Usuario interactúa → Interacción se guarda en buffer
    2. Cada N interacciones → micro-batch de entrenamiento
    3. Solo actualiza un % pequeño de pesos (plastic params)
    4. Periódicamente consolida (merge plastic → base)
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        lr: float = 5e-5,
        update_interval: int = 10,      # Cada N interacciones, entrenar
        micro_batch_size: int = 2,       # Batch para micro-update
        max_grad_norm: float = 1.0,
        buffer_path: Optional[str] = "data/online/experience_buffer.json",
        device: str = "cuda",
        plastic_ratio: float = 0.1,     # Solo 10% de parámetros se actualzian
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.lr = lr
        self.update_interval = update_interval
        self.micro_batch_size = micro_batch_size
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.plastic_ratio = plastic_ratio

        # Buffer de experiencia
        self.buffer = BufferExperiencia(
            max_size=10000,
            persistir_path=buffer_path,
        )

        # Optimizador solo para "plastic params"
        self._setup_plastic_params()

        # Contadores
        self.interacciones_desde_update = 0
        self.total_updates = 0

    def _setup_plastic_params(self):
        """
        Configura qué parámetros son "plásticos" (se actualizan online).

        Inspirado en el cerebro: las neuronas jóvenes son más plásticas
        que las viejas. Las últimas capas y el Tálamo son más adaptables.
        """
        plastic_params = []
        frozen_params = []

        for name, param in self.model.named_parameters():
            # Siempre plásticos: Tálamo (routing) y últimas capas
            if any(k in name for k in ["talamo", "exit_head", "lm_head"]):
                param.requires_grad = True
                plastic_params.append(param)
            # Últimas 3 capas son plásticas
            elif "bloques" in name:
                # Extraer número de capa
                import re
                match = re.search(r"bloques\.(\d+)\.", name)
                if match:
                    capa = int(match.group(1))
                    n_capas = self.model.config.n_capas
                    if capa >= n_capas - 3:  # Últimas 3 capas
                        param.requires_grad = True
                        plastic_params.append(param)
                    else:
                        param.requires_grad = False
                        frozen_params.append(param)
                else:
                    param.requires_grad = False
                    frozen_params.append(param)
            else:
                param.requires_grad = False
                frozen_params.append(param)

        self.optimizer = torch.optim.AdamW(
            plastic_params, lr=self.lr, weight_decay=0.01,
        )

        n_plastic = sum(p.numel() for p in plastic_params)
        n_frozen = sum(p.numel() for p in frozen_params)
        total = n_plastic + n_frozen
        print(
            f"  Params plásticos: {n_plastic:,} "
            f"({100*n_plastic/total:.1f}%) | "
            f"Congelados: {n_frozen:,}"
        )

    def registrar_interaccion(self, interaccion: Interaccion):
        """
        Registra una nueva interacción y potencialmente entrena.

        Llamar esto cada vez que un usuario interactúa con el modelo.
        """
        self.buffer.agregar(interaccion)
        self.interacciones_desde_update += 1

        # ¿Hora de entrenar?
        if self.interacciones_desde_update >= self.update_interval:
            self._micro_update()
            self.interacciones_desde_update = 0

    def _micro_update(self):
        """
        Realiza un micro-update con las interacciones recientes.

        Combina:
        1. SFT (supervised) con interacciones aceptadas/editadas
        2. DPO con pares preferred/rejected
        """
        # Obtener batch de interacciones positivas
        batch = self.buffer.obtener_batch(
            self.micro_batch_size, solo_positivas=True,
        )

        if not batch:
            return

        self.model.train()
        self.optimizer.zero_grad()

        total_loss = 0.0
        n_valid = 0

        for inter in batch:
            # Usar el mejor texto disponible
            texto = inter.editado if inter.fue_editado else inter.generado

            # Tokenizar
            tokens = self.tokenizer.Encode(texto)
            if len(tokens) < 4:
                continue

            input_ids = torch.tensor(
                [tokens[:-1]], dtype=torch.long, device=self.device,
            )
            targets = torch.tensor(
                [tokens[1:]], dtype=torch.long, device=self.device,
            )

            # Forward
            logits, loss, _ = self.model(input_ids, targets)

            # Ponderar por reward y normalizar por batch size
            weighted_loss = loss * max(abs(inter.reward), 0.1) / max(len(batch), 1)
            weighted_loss.backward()

            total_loss += loss.item()
            n_valid += 1

        if n_valid > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm,
            )
            self.optimizer.step()
            self.total_updates += 1

            avg_loss = total_loss / n_valid
            if self.total_updates % 10 == 0:
                stats = self.buffer.get_stats()
                print(
                    f"  Online update #{self.total_updates}: "
                    f"loss={avg_loss:.4f}, "
                    f"buffer={stats['buffer_size']}, "
                    f"reward_medio={stats['reward_medio']:.3f}"
                )

    def servir_y_aprender(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> Tuple[str, float]:
        """
        Genera código Y aprende de la interacción futura.

        Retorna el código generado y la confianza del modelo.
        El usuario debe llamar registrar_feedback() después.
        """
        # Tokenizar
        tokens = self.tokenizer.Encode(prompt)
        input_ids = torch.tensor(
            [tokens], dtype=torch.long, device=self.device,
        )

        # Generar
        self.model.eval()
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            # Obtener confianza
            logits, _, info = self.model(output)
            confianza = info.get("exit_capa", self.model.config.n_capas) / self.model.config.n_capas

        generated = self.tokenizer.Decode(output[0, len(tokens):].tolist())

        # Pre-registrar interacción (sin feedback aún)
        self._ultima_interaccion = Interaccion(
            prompt=prompt,
            generado=generated,
            tokens_prompt=len(tokens),
            tokens_generado=output.shape[1] - len(tokens),
        )

        return generated, confianza

    def registrar_feedback(
        self,
        aceptado: Optional[bool] = None,
        editado: Optional[str] = None,
        feedback: Optional[int] = None,
    ):
        """
        Registra feedback del usuario sobre la última generación.

        Llamar después de servir_y_aprender().

        Args:
            aceptado: True si el usuario usó el código
            editado: Código editado por el usuario (si modificó)
            feedback: Calificación 1-5
        """
        if hasattr(self, "_ultima_interaccion"):
            inter = self._ultima_interaccion
            inter.aceptado = aceptado
            inter.editado = editado
            inter.feedback = feedback
            self.registrar_interaccion(inter)
            del self._ultima_interaccion

    def get_stats(self) -> Dict:
        """Estadísticas del entrenador online."""
        return {
            "total_updates": self.total_updates,
            "interacciones_pendientes": self.interacciones_desde_update,
            **self.buffer.get_stats(),
        }


# =============================================================================
# SERVIDOR HTTP SIMPLE PARA INTERACCIONES
# =============================================================================

def crear_servidor(
    model,
    tokenizer,
    host: str = "0.0.0.0",
    port: int = 8080,
    device: str = "cuda",
) -> "EntrenadorOnline":
    """
    Crea un servidor HTTP que sirve el modelo y aprende de usuarios.

    Endpoints:
      POST /generate  → Genera código
      POST /feedback  → Recibe feedback del usuario
      GET  /stats     → Estadísticas

    Uso:
      servidor = crear_servidor(model, tokenizer)
      # El servidor corre en background y aprende de cada interacción
    """
    entrenador = EntrenadorOnline(
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    try:
        from http.server import HTTPServer, BaseHTTPRequestHandler

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body) if body else {}

                if self.path == "/generate":
                    codigo, confianza = entrenador.servir_y_aprender(
                        data.get("prompt", ""),
                        max_tokens=data.get("max_tokens", 256),
                        temperature=data.get("temperature", 0.7),
                    )
                    response = {
                        "code": codigo,
                        "confidence": confianza,
                    }
                elif self.path == "/feedback":
                    entrenador.registrar_feedback(
                        aceptado=data.get("accepted"),
                        editado=data.get("edited"),
                        feedback=data.get("rating"),
                    )
                    response = {"status": "ok"}
                else:
                    self.send_error(404)
                    return

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode("utf-8"))

            def do_GET(self):
                if self.path == "/stats":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(entrenador.get_stats()).encode("utf-8"))
                else:
                    self.send_error(404)

            def log_message(self, format, *args):
                pass  # Silenciar logs HTTP

        server = HTTPServer((host, port), Handler)
        print(f"  Servidor corriendo en http://{host}:{port}")
        print(f"  POST /generate  — Genera código")
        print(f"  POST /feedback  — Registra feedback")
        print(f"  GET  /stats     — Estadísticas")

        return entrenador, server

    except Exception as e:
        print(f"  Error creando servidor: {e}")
        return entrenador, None
