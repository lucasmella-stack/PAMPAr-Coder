# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Fase 4: Neuroplasticidad — "Sueño"

Implementa mecanismos de aprendizaje Hebbiano y consolidación
inspirados en cómo el cerebro fortalece conexiones durante el sueño.

Principios:
  1. "Neuronas que disparan juntas, se conectan juntas" (Hebb, 1949)
     → Si dos territorios se activan juntos en predicciones exitosas,
       fortalecer su conexión (gate en el Tálamo)
  
  2. Poda sináptica (eliminación de conexiones débiles)
     → Territorios que raramente se activan para ciertos tipos de tokens
       reducen sus pesos → el modelo se vuelve más eficiente
  
  3. Consolidación por replay
     → Repetir patrones exitosos (sin datos nuevos)
     → Refuerza memorias "importantes" como en el sueño REM

  4. Equilibrio homeostático
     → Evitar que un territorio domine toda la red
     → Mantener diversidad funcional

Innovación PAMPAr: Esto es posible porque tenemos una arquitectura
con fronteras explícitas entre territorios y gates en el Tálamo.
Un transformer estándar NO puede hacer esto porque no tiene
estructura modular explícita.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json
import math


# =============================================================================
# ESTADÍSTICAS DE CO-ACTIVACIÓN
# =============================================================================

@dataclass
class EstadisticasTerritoriales:
    """
    Rastreo de patrones de activación territorial.
    
    Registra qué territorios se activan juntos y cuándo las predicciones
    son correctas/incorrectas. Esta información guía el ajuste Hebbiano.
    """
    
    # Matriz de co-activación: [4, 4] cuántas veces los territorios i,j
    # se activan juntos en predicciones EXITOSAS
    coactivacion_exito: torch.Tensor = None      # [4, 4]
    
    # Misma idea pero para predicciones FALLIDAS
    coactivacion_fallo: torch.Tensor = None       # [4, 4]
    
    # Activación media por territorio (para homeostasis)
    activacion_media: torch.Tensor = None         # [4]
    
    # Contadores
    n_exitos: int = 0
    n_fallos: int = 0
    
    def __post_init__(self):
        if self.coactivacion_exito is None:
            self.coactivacion_exito = torch.zeros(4, 4)
        if self.coactivacion_fallo is None:
            self.coactivacion_fallo = torch.zeros(4, 4)
        if self.activacion_media is None:
            self.activacion_media = torch.zeros(4)
    
    def registrar(
        self,
        terr_acts: torch.Tensor,   # [B, L, 4]
        prediccion_correcta: torch.Tensor,  # [B] bool
    ):
        """
        Registra patrones de activación con su resultado.
        
        Args:
            terr_acts: Activaciones territoriales [B, L, 4]
            prediccion_correcta: Máscara de predicciones correctas [B]
        """
        # Promediar sobre secuencia
        mean_acts = terr_acts.mean(dim=1).detach().cpu()  # [B, 4]
        
        for b in range(mean_acts.shape[0]):
            acts = mean_acts[b]  # [4]
            
            # Matriz de co-activación: outer product de activaciones
            coact = torch.outer(acts, acts)  # [4, 4]
            
            if prediccion_correcta[b]:
                self.coactivacion_exito += coact
                self.n_exitos += 1
            else:
                self.coactivacion_fallo += coact
                self.n_fallos += 1
            
            # Actualizar media exponencial
            alpha = 0.01
            self.activacion_media = (
                (1 - alpha) * self.activacion_media + alpha * acts
            )
    
    def get_hebbian_signal(self) -> torch.Tensor:
        """
        Calcula la señal Hebbiana: diferencia entre co-activación
        en éxitos vs fallos.
        
        Returns:
            [4, 4] tensor: positivo = fortalecer, negativo = debilitar
        """
        if self.n_exitos == 0 and self.n_fallos == 0:
            return torch.zeros(4, 4)
        
        # Normalizar por número de muestras
        exito_norm = self.coactivacion_exito / max(self.n_exitos, 1)
        fallo_norm = self.coactivacion_fallo / max(self.n_fallos, 1)
        
        # Señal Hebbiana: lo que funciona - lo que no
        signal = exito_norm - fallo_norm
        
        return signal
    
    def reset(self):
        """Resetea estadísticas (después de consolidación)."""
        self.coactivacion_exito.zero_()
        self.coactivacion_fallo.zero_()
        self.n_exitos = 0
        self.n_fallos = 0


# =============================================================================
# CONSOLIDACIÓN HEBBIANA
# =============================================================================

class ConsolidacionHebbiana:
    """
    Aplica ajustes Hebbianos al modelo basándose en las estadísticas
    de co-activación territorial.
    
    Es como el proceso de consolidación durante el sueño:
    - Fortalecer conexiones exitosas
    - Debilitar conexiones que causan errores
    - Mantener equilibrio entre territorios
    
    Se aplica periódicamente (cada N steps o al final de una fase).
    """
    
    def __init__(
        self,
        learning_rate: float = 0.001,      # Tasa de ajuste Hebbiano
        homeostasis_target: float = 0.25,   # Cada territorio ~25% activo
        homeostasis_weight: float = 0.01,   # Fuerza de homeostasis
        poda_threshold: float = 0.01,       # Umbral para podar pesos
        max_adjustment: float = 0.05,       # Máximo ajuste por paso
    ):
        self.lr = learning_rate
        self.homeostasis_target = homeostasis_target
        self.homeostasis_weight = homeostasis_weight
        self.poda_threshold = poda_threshold
        self.max_adjustment = max_adjustment
        
        self.stats = EstadisticasTerritoriales()
        self.historial: List[Dict] = []
    
    def registrar_paso(
        self,
        terr_acts: torch.Tensor,          # [B, L, 4]
        loss: float,
        logits: torch.Tensor,             # [B, L, V]
        targets: torch.Tensor,            # [B, L]
    ):
        """Registra un paso de entrenamiento para estadísticas Hebbianas."""
        with torch.no_grad():
            # Determinar predicciones correctas por ejemplo en el batch
            preds = logits.argmax(dim=-1)  # [B, L]
            
            # Comparar con targets (ignorar -100)
            mask = targets != -100
            correct_per_token = (preds == targets) & mask   # [B, L]
            correct_ratio = correct_per_token.float().sum(dim=1) / mask.float().sum(dim=1).clamp(min=1)  # [B]
            
            # Umbral: >50% correcto = "predicción exitosa"
            prediccion_correcta = correct_ratio > 0.5
            
            self.stats.registrar(terr_acts, prediccion_correcta)
    
    def consolidar(self, model) -> Dict:
        """
        Aplica consolidación Hebbiana al modelo.
        
        Modifica los pesos del gate territorial en el Tálamo
        basándose en los patrones de co-activación observados.
        
        Args:
            model: PampaRCoderV2
            
        Returns:
            Dict con métricas de la consolidación
        """
        signal = self.stats.get_hebbian_signal()  # [4, 4]
        
        metricas = {
            "signal_mean": signal.mean().item(),
            "signal_max": signal.max().item(),
            "signal_min": signal.min().item(),
            "n_exitos": self.stats.n_exitos,
            "n_fallos": self.stats.n_fallos,
            "activacion_media": self.stats.activacion_media.tolist(),
            "ajustes_aplicados": 0,
        }
        
        if self.stats.n_exitos + self.stats.n_fallos < 100:
            metricas["razon_skip"] = "insuficientes datos (<100 muestras)"
            return metricas
        
        # 1. Ajuste Hebbiano del gate territorial en el Tálamo
        with torch.no_grad():
            gate_weight = model.talamo.terr_gate.weight  # [4, 4]
            
            # Limitar ajuste
            adjustment = signal.to(gate_weight.device) * self.lr
            adjustment = adjustment.clamp(-self.max_adjustment, self.max_adjustment)
            
            gate_weight.add_(adjustment)
            metricas["ajustes_aplicados"] += 1
        
        # 2. Homeostasis: evitar que un territorio domine
        with torch.no_grad():
            act_media = self.stats.activacion_media.to(gate_weight.device)
            
            # Si un territorio está demasiado activo, reducir su bias
            # Si está poco activo, aumentarlo
            desviacion = act_media - self.homeostasis_target
            
            if hasattr(model.talamo.terr_gate, 'bias') and model.talamo.terr_gate.bias is not None:
                bias_adjustment = -desviacion * self.homeostasis_weight
                model.talamo.terr_gate.bias.add_(bias_adjustment)
                metricas["homeostasis_ajuste"] = desviacion.tolist()
        
        # 3. Poda de FFN territories con baja activación
        pesos_podados = podar_pesos(model, self.stats.activacion_media, self.poda_threshold)
        metricas["pesos_podados"] = pesos_podados
        
        # Guardar historial
        self.historial.append(metricas)
        
        # Reset stats para siguiente ronda
        self.stats.reset()
        
        return metricas
    
    def get_estado(self) -> Dict:
        """Estado para checkpoint."""
        return {
            "stats": {
                "coactivacion_exito": self.stats.coactivacion_exito.tolist(),
                "coactivacion_fallo": self.stats.coactivacion_fallo.tolist(),
                "activacion_media": self.stats.activacion_media.tolist(),
                "n_exitos": self.stats.n_exitos,
                "n_fallos": self.stats.n_fallos,
            },
            "historial": self.historial,
        }
    
    def cargar_estado(self, estado: Dict):
        """Restaura estado desde checkpoint."""
        s = estado["stats"]
        self.stats.coactivacion_exito = torch.tensor(s["coactivacion_exito"])
        self.stats.coactivacion_fallo = torch.tensor(s["coactivacion_fallo"])
        self.stats.activacion_media = torch.tensor(s["activacion_media"])
        self.stats.n_exitos = s["n_exitos"]
        self.stats.n_fallos = s["n_fallos"]
        self.historial = estado.get("historial", [])


# =============================================================================
# PODA DE PESOS (Synaptic Pruning)
# =============================================================================

def podar_pesos(
    model,
    activacion_media: torch.Tensor,  # [4]
    threshold: float = 0.01,
) -> int:
    """
    Poda de pesos inspirada en la poda sináptica del cerebro.
    
    Territorios con muy baja activación media tienen sus FFN
    con pesos cercanos a 0 empujados a exactamente 0.
    Esto ahorra memoria y cómputo (sparsity).
    
    Args:
        model: PampaRCoderV2
        activacion_media: [4] activación media por territorio
        threshold: Umbral de activación bajo el cual podar
    
    Returns:
        Número de pesos podados
    """
    total_podados = 0
    
    for bloque in model.bloques:
        for t, ffn in enumerate(bloque.ffns):
            if activacion_media[t] < threshold:
                # Este territorio está casi muerto → podar pesos pequeños
                with torch.no_grad():
                    for param in ffn.parameters():
                        mask = param.abs() < 0.001  # Pesos muy pequeños
                        param.data[mask] = 0.0
                        total_podados += mask.sum().item()
    
    return total_podados


# =============================================================================
# AJUSTE DE FRONTERAS (Frontier Learning)
# =============================================================================

def ajustar_fronteras(
    model,
    signal: torch.Tensor,     # [4, 4] Hebbian signal
    learning_rate: float = 0.001,
) -> int:
    """
    Ajusta las conexiones entre territorios basándose en la señal Hebbiana.
    
    En la arquitectura PAMPAr, las "fronteras" entre territorios
    se implementan via el mix layer de cada BloqueTerritorial.
    
    Si territorios i,j se co-activan exitosamente, fortalecer
    las filas/columnas correspondientes en mix.weight.
    
    Args:
        model: PampaRCoderV2
        signal: [4, 4] señal Hebbiana
        learning_rate: Tasa de ajuste
    
    Returns:
        Número de bloques ajustados
    """
    n_ajustados = 0
    dim = model.config.dim
    
    for bloque in model.bloques:
        with torch.no_grad():
            # mix.weight: [dim, dim*4] — combina 4 FFN outputs
            # Cada bloque de dim columnas corresponde a un territorio
            weight = bloque.mix.weight  # [dim, dim*4]
            
            for i in range(4):
                for j in range(4):
                    if i == j:
                        continue
                    
                    sig = signal[i, j].item()
                    if abs(sig) < 0.001:
                        continue
                    
                    # Ajustar las columnas del territorio j
                    # que afectan la salida (filas) correspondiente 
                    # a patrones del territorio i
                    col_start = j * dim
                    col_end = (j + 1) * dim
                    
                    # Escalar las conexiones entre territorios
                    adjustment = sig * learning_rate
                    weight[:, col_start:col_end] *= (1 + adjustment)
                    
                    n_ajustados += 1
    
    return n_ajustados


# =============================================================================
# TERRITORY ENTROPY REGULARIZATION
# =============================================================================

def territory_entropy_loss(
    terr_acts: torch.Tensor,  # [B, L, 4]
    target_entropy: float = 1.2,  # ~uniform pero no totalmente
    weight: float = 0.01,
) -> torch.Tensor:
    """
    Regularización de entropía territorial.
    
    Evita dos extremos:
    1. Colapso: todos los territorios activos igual → no hay especialización
    2. Muerte: solo 1 territorio activo → pierde capacidad
    
    Target entropy ~1.2 (entre máximo log(4)=1.386 y mínimo 0)
    permite especialización pero mantiene diversidad.
    
    Args:
        terr_acts: [B, L, 4] activaciones territoriales
        target_entropy: Entropía objetivo
        weight: Peso de la regularización
    
    Returns:
        Scalar loss
    """
    # Normalizar a distribución
    probs = F.softmax(terr_acts, dim=-1)  # [B, L, 4]
    
    # Entropía por posición
    entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1)  # [B, L]
    
    # Mean sobre batch y secuencia
    mean_entropy = entropy.mean()
    
    # Penalizar desviación del target
    loss = (mean_entropy - target_entropy) ** 2
    
    return weight * loss
