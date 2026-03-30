#!/usr/bin/env python3
"""
bio_mechanisms.py — Mecanismos bio-inspirados para el Classroom de PamparV3.

5 mecanismos basados en neurociencia real:
  1. Neuromodulación  — dopamina/norepinefrina ajustan LR dinámicamente
  2. LTP              — fortalece LateralGate.scale de streams consistentes
  3. Sleep Replay     — consolidación periódica (REM aleatorio + SWS ordenado)
  4. Neurogenesis     — inyecta LoRA adapters en StreamFFN para conocimiento nuevo
  5. Synaptic Pruning — poda conexiones laterales débiles

Uso:
  from bio_mechanisms import BioOrchestrator

  bio = BioOrchestrator(model, config, optimizer, replay_buffer)
  # Después de cada lección:
  bio.after_lesson(lesson_result, terr_acts_history)
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# 1. NEUROMODULACIÓN — Dopamina + Norepinefrina
# =============================================================================


class Neuromodulator:
    """
    Modula el learning rate según el resultado de la lección.

    - Dopamina (recompensa):  sube tras éxito → consolida aprendizaje
    - Norepinefrina (alerta): sube tras error/novedad → aumenta plasticidad

    El LR efectivo se escala: lr_effective = lr_base × modulation_factor
    """

    def __init__(
        self, baseline_lr: float, min_mult: float = 0.3, max_mult: float = 3.0
    ):
        self.baseline_lr = baseline_lr
        self.min_mult = min_mult
        self.max_mult = max_mult

        # Estado interno (decaimiento exponencial)
        self.dopamine: float = 1.0  # Recompensa acumulada
        self.norepinephrine: float = 1.0  # Alerta/novedad

        # Historial para detectar tendencias
        self._recent_correct: deque[bool] = deque(maxlen=10)
        self._recent_losses: deque[float] = deque(maxlen=10)

    def update(self, correct: bool, loss: float, level: int) -> float:
        """
        Actualiza neuromoduladores y retorna el factor de modulación del LR.

        Returns:
            factor multiplicativo para el LR (ej: 1.5 = 50% más LR)
        """
        self._recent_correct.append(correct)
        self._recent_losses.append(loss)

        # Decaimiento natural (tau ~5 lecciones)
        decay = 0.8
        self.dopamine *= decay
        self.norepinephrine *= decay

        if correct:
            # Éxito → dopamina sube (más en niveles altos)
            self.dopamine += 0.3 * (1.0 + level * 0.1)
        else:
            # Error → norepinefrina sube (más plasticidad)
            self.norepinephrine += 0.4

        # Detectar novedad: si el loss es mucho mayor que el promedio reciente
        if len(self._recent_losses) > 3:
            avg_loss = sum(self._recent_losses) / len(self._recent_losses)
            if loss > avg_loss * 1.5:
                self.norepinephrine += 0.2  # Material nuevo/difícil

        # Factor de modulación combinado
        # Dopamina alta + Norepinefrina baja = consolidar (LR moderado)
        # Dopamina baja + Norepinefrina alta = explorar (LR alto)
        factor = 0.5 * self.dopamine + 0.7 * self.norepinephrine

        # Clampear a rango seguro
        factor = max(self.min_mult, min(self.max_mult, factor))

        return factor

    def apply_to_optimizer(
        self, optimizer: torch.optim.Optimizer, factor: float
    ) -> None:
        """Aplica el factor de modulación a todos los param groups."""
        for group in optimizer.param_groups:
            # Cada grupo tiene su propio baseline (definido por lr_base × mult)
            if "baseline_lr" in group:
                group["lr"] = group["baseline_lr"] * factor


# =============================================================================
# 2. LTP — Long-Term Potentiation (Fortalecimiento sináptico)
# =============================================================================


class LTPManager:
    """
    Fortalece las conexiones laterales (LateralGate.scale) de streams
    que se activan consistentemente juntos.

    Regla de Hebb: "Neurons that fire together wire together."
    Si un stream tiene alta activación territorial repetidamente,
    su scale en LateralGate crece → más comunicación lateral.
    """

    def __init__(self, n_streams: int = 4, n_levels: int = 5):
        self.n_streams = n_streams
        self.n_levels = n_levels

        # Acumulador de activaciones por stream por nivel
        self._activation_accum: list[torch.Tensor] = [
            torch.zeros(n_streams) for _ in range(n_levels)
        ]
        self._count: int = 0
        self._apply_every: int = 5  # Aplicar LTP cada 5 lecciones

    def accumulate(self, terr_acts_per_level: list[torch.Tensor]) -> None:
        """
        Acumula activaciones territoriales de la última lección.

        Args:
            terr_acts_per_level: lista de [B, L, 4] por nivel, o un solo [B, L, 4]
        """
        self._count += 1

        for lvl_idx, terr_acts in enumerate(terr_acts_per_level):
            if lvl_idx >= self.n_levels:
                break
            # Promedio espacial: [4] — activación media de cada stream
            mean_act = terr_acts.detach().float().mean(dim=(0, 1))  # [4]
            self._activation_accum[lvl_idx] += mean_act.cpu()

    def should_apply(self) -> bool:
        """Retorna True si es momento de aplicar LTP."""
        return self._count > 0 and self._count % self._apply_every == 0

    @torch.no_grad()
    def apply(self, model: nn.Module, strength: float = 0.02) -> dict[str, float]:
        """
        Fortalece LateralGate.scale según activaciones acumuladas.

        Returns:
            dict con cambios aplicados por nivel
        """
        if self._count == 0:
            return {}

        changes: dict[str, float] = {}

        for name, module in model.named_modules():
            if not hasattr(module, "scale") or "lateral" not in name.lower():
                continue

            # Extraer índice de nivel del nombre del módulo
            lvl_idx = self._extract_level_index(name)
            if lvl_idx is None or lvl_idx >= self.n_levels:
                continue

            # Activación promedio normalizada
            avg_act = self._activation_accum[lvl_idx] / self._count
            avg_act = avg_act / (avg_act.max() + 1e-8)  # Normalizar a [0, 1]

            # LTP: streams con alta activación → su scale crece
            # La delta es proporcional a la activación y strength
            delta = strength * avg_act.to(module.scale.device)
            module.scale.data += delta

            # Clampear scale a rango razonable [0.01, 0.5]
            module.scale.data.clamp_(0.01, 0.5)

            changes[name] = delta.mean().item()

        # Reset acumuladores
        self._activation_accum = [
            torch.zeros(self.n_streams) for _ in range(self.n_levels)
        ]
        self._count = 0

        return changes

    def _extract_level_index(self, name: str) -> Optional[int]:
        """Extrae el índice del nivel desde el nombre del módulo."""
        # Buscar patrones como 'niveles.0.lateral', 'niveles.3.lateral'
        parts = name.split(".")
        for i, part in enumerate(parts):
            if part == "niveles" and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    pass
        return None


# =============================================================================
# 3. SLEEP CONSOLIDATION — Replay durante "sueño"
# =============================================================================


class SleepConsolidator:
    """
    Consolidación periódica que simula las fases del sueño:

    - REM: replay aleatorio de experiencias recientes (creatividad/generalización)
    - SWS (Slow-Wave Sleep): replay ordenado por importancia (consolidación fuerte)

    Se ejecuta cada N lecciones, hace un mini-entrenamiento con replay puro.
    """

    def __init__(self, every_n: int = 15, rem_ratio: float = 0.6):
        self.every_n = every_n
        self.rem_ratio = rem_ratio  # 60% REM, 40% SWS
        self._lesson_count = 0

    def should_sleep(self) -> bool:
        """¿Es hora de dormir?"""
        self._lesson_count += 1
        return self._lesson_count % self.every_n == 0

    def consolidate(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        replay_buffer: object,
        device: torch.device,
        n_steps: int = 3,
    ) -> float:
        """
        Ejecuta consolidación de sueño.

        Args:
            model: PamparV3
            optimizer: el optimizador con LR diferencial
            replay_buffer: ReplayBuffer con .buffer y .sample()
            device: dispositivo
            n_steps: pasos de consolidación

        Returns:
            loss promedio durante consolidación
        """
        buffer = getattr(replay_buffer, "buffer", [])
        if len(buffer) < 4:
            return 0.0

        model.train()
        total_loss = 0.0

        # Reducir LR durante sueño (como en sueño real, actividad reducida)
        sleep_lr_factor = 0.3
        original_lrs: list[float] = []
        for group in optimizer.param_groups:
            original_lrs.append(group["lr"])
            group["lr"] = group["lr"] * sleep_lr_factor

        for step in range(n_steps):
            # Fase REM: replay aleatorio (generalización)
            n_rem = max(1, int(len(buffer) * self.rem_ratio))
            rem_samples = random.sample(list(buffer), min(n_rem, len(buffer)))

            # Fase SWS: replay ordenado por nivel (lo más difícil primero)
            sws_samples = sorted(
                list(buffer),
                key=lambda x: x.get("level", 1),
                reverse=True,
            )
            n_sws = max(1, len(buffer) - n_rem)
            sws_samples = sws_samples[:n_sws]

            # Combinar
            all_samples = rem_samples + sws_samples

            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=device)
            n = 0

            for sample in all_samples:
                tokens = sample["tokens"].to(device)
                if tokens.dim() == 1:
                    tokens = tokens.unsqueeze(0)
                if tokens.shape[-1] < 3:
                    continue

                input_ids = tokens[:, :-1]
                targets = tokens[:, 1:]
                logits, _, _ = model(input_ids, targets=targets)

                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=0,
                )
                batch_loss = batch_loss + loss
                n += 1

            if n > 0:
                batch_loss = batch_loss / n
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += batch_loss.item()

        # Restaurar LR originales
        for group, orig_lr in zip(optimizer.param_groups, original_lrs):
            group["lr"] = orig_lr

        return total_loss / max(1, n_steps)


# =============================================================================
# 4. NEUROGENESIS — LoRA adapters para conocimiento nuevo
# =============================================================================


class StreamLoRA(nn.Module):
    """
    Adapter LoRA minimalista para StreamFFN.

    Inyecta una rama paralela de bajo rango que captura conocimiento nuevo
    sin modificar los pesos originales (como neuronas nuevas en el hipocampo).

    Original: y = FFN(x)
    Con LoRA: y = FFN(x) + scale * B(A(x))

    Params: dim × rank + rank × dim ≈ 640×8×2 = 10K por adapter
    """

    def __init__(self, dim: int, rank: int = 8):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        self.scale = nn.Parameter(torch.tensor(0.01))

        # Inicialización: down normal, up zeros (empiezan como identidad)
        nn.init.kaiming_normal_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna solo el delta LoRA (se suma al output original)."""
        return self.scale * self.up(F.silu(self.down(x)))


class NeurogenesisManager:
    """
    Gestiona la creación y activación de LoRA adapters en StreamFFN.

    Solo crea adapters cuando detecta que un stream necesita aprender
    algo genuinamente nuevo (alta pérdida + baja activación territorial).
    """

    def __init__(self, dim: int = 640, rank: int = 8, max_adapters: int = 8):
        self.dim = dim
        self.rank = rank
        self.max_adapters = max_adapters
        self._adapters: dict[str, StreamLoRA] = {}
        self._hooked: bool = False

    @property
    def adapter_count(self) -> int:
        return len(self._adapters)

    def should_grow(self, loss: float, threshold: float = 4.0) -> bool:
        """Determina si necesitamos crear neuronas nuevas."""
        return loss > threshold and self.adapter_count < self.max_adapters

    def create_adapter(
        self, model: nn.Module, level_idx: int, stream_idx: int, device: torch.device
    ) -> Optional[str]:
        """
        Crea un LoRA adapter para un StreamFFN específico.

        Returns:
            nombre del adapter creado, o None si ya existe/límite alcanzado
        """
        key = f"lora_L{level_idx}_S{stream_idx}"
        if key in self._adapters or self.adapter_count >= self.max_adapters:
            return None

        adapter = StreamLoRA(self.dim, self.rank).to(device)
        self._adapters[key] = adapter

        # Registrar como submodule del modelo para que el optimizer lo vea
        if not hasattr(model, "_bio_lora_adapters"):
            model._bio_lora_adapters = nn.ModuleDict()
        model._bio_lora_adapters[key] = adapter

        return key

    def get_adapter(self, level_idx: int, stream_idx: int) -> Optional[StreamLoRA]:
        """Retorna el adapter para un nivel/stream, si existe."""
        key = f"lora_L{level_idx}_S{stream_idx}"
        return self._adapters.get(key)

    def add_adapters_to_optimizer(
        self, optimizer: torch.optim.Optimizer, lr: float
    ) -> None:
        """Añade los parámetros de los nuevos adapters al optimizador."""
        existing_params = set()
        for group in optimizer.param_groups:
            for p in group["params"]:
                existing_params.add(id(p))

        new_params = []
        for adapter in self._adapters.values():
            for p in adapter.parameters():
                if id(p) not in existing_params:
                    new_params.append(p)

        if new_params:
            optimizer.add_param_group(
                {
                    "params": new_params,
                    "lr": lr,
                    "label": "neurogenesis_lora",
                }
            )


# =============================================================================
# 5. SYNAPTIC PRUNING — Poda de conexiones débiles
# =============================================================================


class SynapticPruner:
    """
    Poda conexiones laterales débiles (LateralGate.scale bajo).

    En el cerebro, ~50% de las sinapsis se eliminan durante el desarrollo.
    Aquí, si un LateralGate.scale cae por debajo del umbral durante
    varias lecciones consecutivas, lo reducimos agresivamente.

    Esto libera "capacidad" y evita ruido de conexiones irrelevantes.
    """

    def __init__(self, every_n: int = 30, threshold: float = 0.03, decay: float = 0.5):
        self.every_n = every_n
        self.threshold = threshold
        self.decay = decay
        self._lesson_count = 0

    def should_prune(self) -> bool:
        """¿Es momento de podar?"""
        self._lesson_count += 1
        return self._lesson_count % self.every_n == 0

    @torch.no_grad()
    def prune(self, model: nn.Module) -> dict[str, list[int]]:
        """
        Poda conexiones laterales débiles.

        Returns:
            dict con streams podados por nivel
        """
        pruned: dict[str, list[int]] = {}

        for name, module in model.named_modules():
            if not hasattr(module, "scale") or "lateral" not in name.lower():
                continue

            scale = module.scale.data  # [n_streams]
            weak_mask = scale < self.threshold

            if weak_mask.any():
                # Reducir, no eliminar completamente (permitir recuperación)
                module.scale.data[weak_mask] *= self.decay

                # Registrar qué streams se podaron
                pruned_streams = weak_mask.nonzero(as_tuple=True)[0].tolist()
                pruned[name] = pruned_streams

        return pruned


# =============================================================================
# ORCHESTRATOR — Coordina todos los mecanismos
# =============================================================================


@dataclass
class BioState:
    """Estado observable de los mecanismos bio para logging/UI."""

    dopamine: float = 1.0
    norepinephrine: float = 1.0
    lr_factor: float = 1.0
    ltp_applied: bool = False
    ltp_changes: dict = field(default_factory=dict)
    sleep_triggered: bool = False
    sleep_loss: float = 0.0
    adapters_created: int = 0
    adapters_total: int = 0
    pruned_streams: dict = field(default_factory=dict)


class BioOrchestrator:
    """
    Coordina los 5 mecanismos bio-inspirados.

    Se llama una vez después de cada lección con el resultado y las
    activaciones territoriales. Él decide qué mecanismos activar.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        replay_buffer: object,
        device: torch.device,
        baseline_lr: float = 5e-6,
        dim: int = 640,
        n_streams: int = 4,
        n_levels: int = 5,
        sleep_every: int = 15,
        prune_every: int = 30,
    ):
        self.model = model
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer
        self.device = device

        # Inicializar los 5 mecanismos
        self.neuromod = Neuromodulator(baseline_lr)
        self.ltp = LTPManager(n_streams, n_levels)
        self.sleep = SleepConsolidator(every_n=sleep_every)
        self.neurogenesis = NeurogenesisManager(dim=dim, rank=8, max_adapters=8)
        self.pruner = SynapticPruner(every_n=prune_every)

        # Guardar baseline LRs en el optimizer para modulación
        for group in optimizer.param_groups:
            group["baseline_lr"] = group["lr"]

    def after_lesson(
        self,
        correct: bool,
        loss: float,
        level: int,
        terr_acts_per_level: Optional[list[torch.Tensor]] = None,
    ) -> BioState:
        """
        Hook principal — se llama después de cada lección.

        Args:
            correct: si el alumno acertó
            loss: loss CE de la lección
            level: nivel del curriculum
            terr_acts_per_level: activaciones territoriales por nivel (opcional)

        Returns:
            BioState con el estado de todos los mecanismos
        """
        state = BioState()

        # 1. NEUROMODULACIÓN — ajustar LR
        factor = self.neuromod.update(correct, loss, level)
        self.neuromod.apply_to_optimizer(self.optimizer, factor)
        state.dopamine = self.neuromod.dopamine
        state.norepinephrine = self.neuromod.norepinephrine
        state.lr_factor = factor

        # 2. LTP — acumular y potenciar si toca
        if terr_acts_per_level is not None:
            self.ltp.accumulate(terr_acts_per_level)
        if self.ltp.should_apply():
            changes = self.ltp.apply(self.model)
            state.ltp_applied = True
            state.ltp_changes = changes

        # 3. SLEEP — consolidar si toca
        if self.sleep.should_sleep():
            sleep_loss = self.sleep.consolidate(
                self.model, self.optimizer, self.replay_buffer, self.device
            )
            state.sleep_triggered = True
            state.sleep_loss = sleep_loss

        # 4. NEUROGENESIS — crear adapters si el loss es alto
        if self.neurogenesis.should_grow(loss):
            # Encontrar el stream menos activo (más necesitado)
            if terr_acts_per_level:
                last_terr = terr_acts_per_level[-1]  # Último nivel
                mean_act = last_terr.detach().float().mean(dim=(0, 1))
                weakest_stream = mean_act.argmin().item()
                # Crear en el nivel más profundo
                deepest_level = len(terr_acts_per_level) - 1
                key = self.neurogenesis.create_adapter(
                    self.model, deepest_level, weakest_stream, self.device
                )
                if key:
                    state.adapters_created = 1
                    # Añadir al optimizer
                    self.neurogenesis.add_adapters_to_optimizer(
                        self.optimizer, lr=self.neuromod.baseline_lr * factor
                    )
        state.adapters_total = self.neurogenesis.adapter_count

        # 5. PRUNING — podar conexiones débiles periódicamente
        if self.pruner.should_prune():
            pruned = self.pruner.prune(self.model)
            state.pruned_streams = pruned

        return state
