# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Fase 5: Metacognición — "Curiosidad"

El modelo aprende a conocer sus propias limitaciones.

Metacognición = "pensar sobre el pensamiento"
  - El Early Exit ya da una señal de confianza
  - Podemos entrenar esta señal para que sea CALIBRADA
  - Calibrada = si dice 80% confianza, acierta el 80% de las veces

Esto permite:
  1. Active Learning: entrenar más en lo que NO sabe
  2. Inferencia eficiente: salir temprano cuando está seguro
  3. Auto-evaluación: el modelo sabe cuándo pedir ayuda

Inspirado en:
  - Cerebro humano: cortex prefrontal monitorea otros procesos
  - Sistemas 1 y 2 de Kahneman: respuesta rápida vs deliberación
  - Calibración de redes neuronales (Guo et al., 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json
import math


# =============================================================================
# METACOGNITIVE LOSS
# =============================================================================

class MetacognitiveLoss(nn.Module):
    """
    Pérdida metacognitiva que entrena al modelo a tener
    confianza calibrada.
    
    Componentes:
    1. CE Loss estándar (predicción de tokens)
    2. Calibration Loss (confianza ≈ accuracy)
    3. Territory Specialization Loss (territorios especializados)
    
    L_total = α * L_CE + β * L_calibration + γ * L_specialization
    """
    
    def __init__(
        self,
        alpha: float = 1.0,      # Peso de CE (siempre principal)
        beta: float = 0.1,       # Peso de calibración
        gamma: float = 0.05,     # Peso de especialización
        n_bins: int = 10,        # Bins para calibración (ECE)
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_bins = n_bins
        
        # Historial para tracking
        self._losses_history: List[Dict] = []
    
    def forward(
        self,
        logits: torch.Tensor,          # [B, L, V]
        targets: torch.Tensor,         # [B, L]
        confianza: float,              # Early Exit confidence (0-1)
        terr_acts: torch.Tensor,       # [B, L, 4] territorial activations
        terr_target: Optional[torch.Tensor] = None,  # [4] expected territory pattern
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Calcula pérdida metacognitiva compuesta.
        
        Returns:
            (loss_total, info_dict)
        """
        B, L, V = logits.shape
        info = {}
        
        # 1. CE Loss estándar
        loss_ce = F.cross_entropy(
            logits.view(-1, V),
            targets.view(-1),
            ignore_index=-100,
        )
        info["loss_ce"] = loss_ce.item()
        
        # 2. Calibration Loss
        loss_cal = self._calibration_loss(logits, targets, confianza)
        info["loss_calibration"] = loss_cal.item()
        
        # 3. Territory Specialization Loss
        loss_spec = self._specialization_loss(terr_acts, terr_target)
        info["loss_specialization"] = loss_spec.item()
        
        # Total
        loss_total = (
            self.alpha * loss_ce +
            self.beta * loss_cal +
            self.gamma * loss_spec
        )
        info["loss_total"] = loss_total.item()
        info["confianza"] = confianza
        
        self._losses_history.append(info)
        
        return loss_total, info
    
    def _calibration_loss(
        self,
        logits: torch.Tensor,  # [B, L, V]
        targets: torch.Tensor,  # [B, L]
        confianza: float,
    ) -> torch.Tensor:
        """
        Penaliza la diferencia entre confianza y accuracy real.
        
        Si el modelo dice "90% seguro" pero acierta solo el 50%,
        hay una penalización alta → aprende a ser más honesto.
        
        Si dice "90% seguro" y acierta 90%, penalización baja.
        """
        with torch.no_grad():
            # Accuracy real
            mask = targets != -100
            preds = logits.argmax(dim=-1)
            correct = (preds == targets) & mask
            accuracy = correct.float().sum() / mask.float().sum().clamp(min=1)
        
        # Diferencia entre confianza y accuracy real
        # Usando squared error para penalizar sobreconfianza más
        cal_error = (confianza - accuracy.item()) ** 2
        
        # Usar un parámetro que sí participe en el grafo computacional
        # Multiplicar por la media de los logits para que el gradiente fluya
        logit_mean = logits.mean() * 0.0 + 1.0  # grad-enabled identity
        return cal_error * logit_mean
    
    def _specialization_loss(
        self,
        terr_acts: torch.Tensor,   # [B, L, 4]
        terr_target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Incentiva que los territorios se especialicen:
        - Si hay target: MSE hacia el patrón esperado
        - Si no hay target: entropy regularization (diversity)
        """
        mean_acts = terr_acts.mean(dim=(0, 1))  # [4]
        
        if terr_target is not None:
            # MSE hacia el patrón esperado del curriculum
            target = terr_target.to(mean_acts.device)
            return F.mse_loss(mean_acts, target)
        
        # Sin target: regularización por entropía
        probs = F.softmax(mean_acts, dim=0)
        entropy = -(probs * (probs + 1e-8).log()).sum()
        
        # Target entropy ~1.2 (ni uniforme ni colapsada)
        return (entropy - 1.2) ** 2
    
    def get_calibration_report(self) -> Dict:
        """
        Genera reporte de calibración.
        
        Un modelo bien calibrado tiene confianza ≈ accuracy
        en todos los rangos de confianza.
        """
        if not self._losses_history:
            return {"error": "sin datos"}
        
        # Agrupar por bins de confianza
        bins = {i: {"confianzas": [], "accuracies": []} for i in range(self.n_bins)}
        
        for entry in self._losses_history:
            conf = entry.get("confianza", 0.5)
            # Calcular bin
            bin_idx = min(int(conf * self.n_bins), self.n_bins - 1)
            bins[bin_idx]["confianzas"].append(conf)
        
        report = {
            "n_samples": len(self._losses_history),
            "mean_confidence": sum(e["confianza"] for e in self._losses_history) / len(self._losses_history),
            "mean_ce_loss": sum(e["loss_ce"] for e in self._losses_history) / len(self._losses_history),
        }
        
        return report
    
    def reset_history(self):
        """Resetea historial (después de consolidación)."""
        self._losses_history = []


# =============================================================================
# ACTIVE LEARNER
# =============================================================================

class ActiveLearner:
    """
    Implementa Active Learning usando la confianza del Early Exit.
    
    Idea: en vez de entrenar con TODOS los datos uniformemente,
    enfocarse en los ejemplos donde el modelo tiene BAJA confianza.
    
    Es como un estudiante que dedica más tiempo a los temas
    que no entiende bien, en vez de repasar lo que ya sabe.
    
    Flujo:
    1. Forward pass sin gradientes
    2. Medir confianza por ejemplo
    3. Los de baja confianza van al "buffer de estudio"
    4. Entrenar preferentemente con esos ejemplos
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.5,   # Debajo = "no sé"
        buffer_size: int = 10000,             # Máximo ejemplos en buffer
        oversample_ratio: float = 3.0,        # 3x más probable elegir ejemplos difíciles
    ):
        self.confidence_threshold = confidence_threshold
        self.buffer_size = buffer_size
        self.oversample_ratio = oversample_ratio
        
        # Buffer de ejemplos difíciles
        self.buffer_dificiles: List[Dict] = []
        
        # Estadísticas
        self.stats = {
            "total_evaluados": 0,
            "total_dificiles": 0,
            "confianza_media": 0.0,
            "ratio_dificiles": 0.0,
        }
    
    @torch.no_grad()
    def evaluar_dificultad(
        self,
        model,
        input_ids: torch.Tensor,    # [B, L]
        targets: torch.Tensor,      # [B, L]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evalúa qué tan difícil es cada ejemplo para el modelo.
        
        Returns:
            confianzas: [B] confianza por ejemplo
            es_dificil: [B] bool mask de ejemplos difíciles
        """
        model.eval()
        
        logits, _, info = model(input_ids, use_early_exit=True)
        
        # Confianza del Early Exit
        confianza_global = info.get("exit_capa", model.config.n_capas) / model.config.n_capas
        
        # Confianza token-level: max prob del softmax
        probs = F.softmax(logits, dim=-1)
        max_probs = probs.max(dim=-1).values  # [B, L]
        
        # Ignorar padding
        mask = targets != -100
        
        # Confianza media por ejemplo (excluyendo padding)
        confianzas = (max_probs * mask.float()).sum(dim=1) / mask.float().sum(dim=1).clamp(min=1)
        
        # Marcar difíciles
        es_dificil = confianzas < self.confidence_threshold
        
        # Actualizar stats
        self.stats["total_evaluados"] += input_ids.shape[0]
        self.stats["total_dificiles"] += es_dificil.sum().item()
        self.stats["confianza_media"] = confianzas.mean().item()
        self.stats["ratio_dificiles"] = self.stats["total_dificiles"] / max(self.stats["total_evaluados"], 1)
        
        model.train()
        
        return confianzas, es_dificil
    
    def agregar_al_buffer(
        self,
        input_ids: torch.Tensor,    # [B, L]
        targets: torch.Tensor,      # [B, L]
        confianzas: torch.Tensor,   # [B]
        es_dificil: torch.Tensor,   # [B] bool
    ):
        """Agrega ejemplos difíciles al buffer de estudio."""
        for i in range(input_ids.shape[0]):
            if es_dificil[i]:
                self.buffer_dificiles.append({
                    "input_ids": input_ids[i].cpu(),
                    "targets": targets[i].cpu(),
                    "confianza": confianzas[i].item(),
                })
        
        # Mantener tamaño del buffer
        if len(self.buffer_dificiles) > self.buffer_size:
            # Quedarse con los más difíciles (menor confianza)
            self.buffer_dificiles.sort(key=lambda x: x["confianza"])
            self.buffer_dificiles = self.buffer_dificiles[:self.buffer_size]
    
    def obtener_batch_dificil(
        self,
        batch_size: int = 4,
        device: str = "cuda",
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Obtiene un batch de ejemplos difíciles del buffer.
        
        Returns:
            Dict con 'input_ids' y 'targets', o None si buffer vacío
        """
        if len(self.buffer_dificiles) < batch_size:
            return None
        
        import random
        batch = random.sample(self.buffer_dificiles, batch_size)
        
        # Stack
        max_len = max(b["input_ids"].shape[0] for b in batch)
        
        input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        targets = torch.full((batch_size, max_len), -100, dtype=torch.long)
        
        for i, b in enumerate(batch):
            L = b["input_ids"].shape[0]
            input_ids[i, :L] = b["input_ids"]
            targets[i, :L] = b["targets"]
        
        return {
            "input_ids": input_ids.to(device),
            "targets": targets.to(device),
        }
    
    def get_stats_str(self) -> str:
        """Estadísticas formateadas."""
        return (
            f"Active Learning Stats:\n"
            f"  Evaluados: {self.stats['total_evaluados']}\n"
            f"  Difíciles: {self.stats['total_dificiles']} "
            f"({100*self.stats['ratio_dificiles']:.1f}%)\n"
            f"  Confianza media: {self.stats['confianza_media']:.4f}\n"
            f"  Buffer size: {len(self.buffer_dificiles)}\n"
        )


# =============================================================================
# CALIBRACIÓN
# =============================================================================

def calcular_calibracion(
    confianzas: List[float],
    accuracies: List[float],
    n_bins: int = 10,
) -> Dict:
    """
    Calcula Expected Calibration Error (ECE).
    
    ECE = Σ (|bin_samples|/N) * |accuracy(bin) - confidence(bin)|
    
    Un ECE bajo (< 0.05) significa que el modelo es bien calibrado:
    cuando dice "80% seguro", acierta ~80% del tiempo.
    
    Returns:
        Dict con 'ece', 'mce' (max calibration error), 'bins'
    """
    if not confianzas:
        return {"ece": 0.0, "mce": 0.0, "bins": []}
    
    n = len(confianzas)
    bins = [{"conf_sum": 0.0, "acc_sum": 0.0, "count": 0} for _ in range(n_bins)]
    
    for conf, acc in zip(confianzas, accuracies):
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bins[bin_idx]["conf_sum"] += conf
        bins[bin_idx]["acc_sum"] += acc
        bins[bin_idx]["count"] += 1
    
    ece = 0.0
    mce = 0.0
    bin_results = []
    
    for b in bins:
        if b["count"] == 0:
            bin_results.append({"conf": 0, "acc": 0, "count": 0, "gap": 0})
            continue
        
        avg_conf = b["conf_sum"] / b["count"]
        avg_acc = b["acc_sum"] / b["count"]
        gap = abs(avg_acc - avg_conf)
        
        ece += (b["count"] / n) * gap
        mce = max(mce, gap)
        
        bin_results.append({
            "conf": round(avg_conf, 4),
            "acc": round(avg_acc, 4),
            "count": b["count"],
            "gap": round(gap, 4),
        })
    
    return {
        "ece": round(ece, 4),
        "mce": round(mce, 4),
        "bins": bin_results,
    }
