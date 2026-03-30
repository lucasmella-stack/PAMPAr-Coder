"""classroom_memory.py — EWC y Replay Buffer para protección de memoria."""

from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


class EWC:
    """
    Elastic Weight Consolidation (Kirkpatrick et al., 2017).

    Simula LTP biológica: identifica pesos importantes (alta Fisher info)
    y penaliza moverlos durante entrenamiento nuevo.

    L_total = L_task + (λ/2) * Σ F_i * (θ_i - θ*_i)²
    """

    def __init__(self, model: nn.Module, lam: float = 500.0):
        self.lam = lam
        self.params_star: dict[str, torch.Tensor] = {}
        self.fisher: dict[str, torch.Tensor] = {}

    def compute_fisher(
        self,
        model: nn.Module,
        data_loader: list[torch.Tensor],
        device: torch.device,
        n_samples: int = 200,
    ) -> None:
        """Calcula la Diagonal Fisher Information Matrix sobre datos existentes."""
        model.eval()

        self.params_star = {
            n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad
        }

        self.fisher = {
            n: torch.zeros_like(p.data)
            for n, p in model.named_parameters()
            if p.requires_grad
        }

        n = min(n_samples, len(data_loader))
        samples = random.sample(data_loader, n) if len(data_loader) > n else data_loader

        for tokens in samples:
            model.zero_grad()
            tokens = tokens.to(device)
            if tokens.dim() == 1:
                tokens = tokens.unsqueeze(0)

            input_ids = tokens[:, :-1]
            targets = tokens[:, 1:]
            logits, _, _ = model(input_ids, targets=targets)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=0,
            )
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher[name] += param.grad.data.pow(2) / n

        model.zero_grad()

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Calcula la penalización EWC: (λ/2) * Σ F_i * (θ_i - θ*_i)²"""
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if name in self.fisher:
                loss += (
                    self.fisher[name] * (param - self.params_star[name]).pow(2)
                ).sum()
        return (self.lam / 2.0) * loss


class ReplayBuffer:
    """
    Buffer circular de ejemplos exitosos.

    Mezcla ejemplos nuevos con viejos para evitar olvido catastrófico.
    Como el replay neuronal durante el sueño: reactiva memorias viejas
    mientras integra las nuevas.
    """

    def __init__(self, maxsize: int = 100):
        self.buffer: deque[dict] = deque(maxlen=maxsize)

    def add(
        self, problem: str, solution: str, tokens: torch.Tensor, level: int
    ) -> None:
        self.buffer.append(
            {
                "problem": problem,
                "solution": solution,
                "tokens": tokens.cpu(),
                "level": level,
                "timestamp": time.time(),
            }
        )

    def sample(self, n: int) -> list[dict]:
        if len(self.buffer) == 0:
            return []
        n = min(n, len(self.buffer))
        return random.sample(list(self.buffer), n)

    def __len__(self) -> int:
        return len(self.buffer)


@dataclass
class LessonResult:
    """Resultado de una lección."""

    lesson_id: int
    level: int
    problem: str
    student_answer: str
    teacher_solution: str
    correct: bool
    feedback: str
    loss: float
    ewc_penalty: float
    brain_score: float
    timestamp: float = field(default_factory=time.time)
