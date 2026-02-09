# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
🧠 Entrenamiento Cerebral — Pipeline completo de 6 fases

Entrena PAMPAr-Coder 1.5B como aprende un cerebro humano:

  Fase 1: INFANCIA     — Curriculum Learning (simple → complejo)
  Fase 2: EXPERIMENTAR — Self-Play con ejecución de código
  Fase 3: FILOSOFAR    — Knowledge Distillation (profesor → alumno)
  Fase 4: SUEÑO        — Consolidación Hebbiana
  Fase 5: CURIOSIDAD   — Active Learning metacognitivo
  Fase 6: SOCIAL       — Online Learning de interacciones con usuarios

Uso:
  # Entrenamiento completo (todas las fases)
  python scripts/train_cerebral.py

  # Solo una fase específica
  python scripts/train_cerebral.py --fase 1

  # Resumir desde checkpoint
  python scripts/train_cerebral.py --resume

  # Modo local (tu PC, sin cloud)
  python scripts/train_cerebral.py --local --preset 8gb
  
  # Solo self-play en tu PC (después de pre-training en cloud)
  python scripts/train_cerebral.py --fase 2 --checkpoint checkpoints/fase1_best.pt

  # Destilación con profesor local gratis
  python scripts/train_cerebral.py --fase 3 --profesor qwen-7b --local

  # Servidor online (aprende de usuarios)
  python scripts/train_cerebral.py --fase 6 --checkpoint checkpoints/cerebral_final.pt

Requisitos:
  - Fase 1: GPU cloud (A40 recomendado) o local 24GB+
  - Fase 2, 4, 5: Tu PC con 8GB+ VRAM
  - Fase 3: GPU + profesor (API o Ollama local)
  - Fase 6: Tu PC, aprende mientras sirve usuarios
"""

import argparse
import copy
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Ajustar path
script_dir = Path(__file__).parent
project_dir = script_dir.parent
sys.path.insert(0, str(project_dir))

from pampar.coder.v2.config import PRESET_1_5B, PRESET_8GB, PRESET_4GB, ConfigV2
from pampar.coder.v2.modelo import PampaRCoderV2, crear_modelo
from pampar.coder.v2.aprendizaje.curriculum import (
    NivelDificultad,
    CurriculumManager,
    CurriculumDataset,
    territory_alignment_loss,
)
from pampar.coder.v2.aprendizaje.self_play import (
    SelfPlayEngine,
    dpo_loss,
)
from pampar.coder.v2.aprendizaje.neuroplasticidad import (
    ConsolidacionHebbiana,
    territory_entropy_loss,
)
from pampar.coder.v2.aprendizaje.metacognicion import (
    MetacognitiveLoss,
    ActiveLearner,
)
from pampar.coder.v2.aprendizaje.destilacion import (
    PROFESORES,
    ClienteProfesor,
    GeneradorDestilacion,
    territory_aware_distillation,
)
from pampar.coder.v2.aprendizaje.aprendizaje_online import (
    EntrenadorOnline,
    Interaccion,
    crear_servidor,
)


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

CONFIG = {
    # Archivos de datos
    "data_dir": "data",
    "tokenizer_path": "data/tokenizer/pampar_48k.model",
    "checkpoint_dir": "checkpoints/cerebral",

    # Hiperparámetros compartidos
    "max_seq_len": 2048,       # Context window para training
    "batch_size": 4,
    "grad_accum": 8,           # Effective batch = 32
    "max_grad_norm": 1.0,
    "weight_decay": 0.1,
    "warmup_steps": 1000,

    # Fase 1: Curriculum
    "fase1": {
        "lr": 3e-4,
        "min_lr": 3e-5,
        "steps_por_nivel": 5000,
        "revision_ratio": 0.1,
        "criterio_avance": 2.0,
    },

    # Fase 2: Self-Play
    "fase2": {
        "lr": 1e-4,
        "n_rondas": 1000,
        "intentos_por_ronda": 4,
        "dpo_beta": 0.1,
        "dpo_batch": 4,
        "max_tokens_gen": 256,
        "temperature": 0.8,
    },

    # Fase 4: Consolidación
    "fase4": {
        "hebbian_lr": 0.001,
        "homeostasis_weight": 0.01,
        "poda_threshold": 0.01,
        "consolidation_interval": 500,
    },

    # Fase 3: Destilación
    "fase3": {
        "lr": 1e-4,
        "profesor": "qwen-7b",   # Default: local gratis
        "modo": "cot",           # Chain-of-thought
        "n_ejemplos": 500,
        "batch_size": 4,
        "grad_accum": 8,
        "temperature_kd": 4.0,   # Temperature para soft targets
        "alpha_kd": 0.7,         # 70% soft + 30% hard
    },

    # Fase 5: Active Learning  
    "fase5": {
        "lr": 5e-5,
        "confidence_threshold": 0.5,
        "oversample_ratio": 3.0,
        "steps": 2000,
    },

    # Fase 6: Online Learning
    "fase6": {
        "lr": 5e-5,
        "update_interval": 10,
        "host": "0.0.0.0",
        "port": 8080,
        "plastic_ratio": 0.1,
    },
}


# =============================================================================
# UTILIDADES
# =============================================================================

def cargar_tokenizer(path: str):
    """Carga el tokenizer SentencePiece."""
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(path)
    return sp


def crear_optimizer(model, lr, weight_decay):
    """Crea AdamW con decay separado."""
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "norm" in name or "bias" in name or "emb" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return torch.optim.AdamW([
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=lr, betas=(0.9, 0.95))


def get_lr_schedule(step, warmup_steps, total_steps, max_lr, min_lr):
    """Cosine schedule con warmup."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def guardar_checkpoint(model, optimizer, fase, paso, loss, path, extra=None):
    """Guarda checkpoint con metadatos."""
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "fase": fase,
        "paso": paso,
        "loss": loss,
        "config": {
            k: getattr(model.config, k)
            for k in model.config.__dataclass_fields__
        },
        "timestamp": datetime.now().isoformat(),
    }
    if extra:
        ckpt.update(extra)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)
    print(f"  💾 Checkpoint guardado: {path} (loss={loss:.4f})")


def log_metricas(fase, paso, metricas, log_file="checkpoints/cerebral/training_log.jsonl"):
    """Logea métricas a archivo JSONL."""
    metricas["fase"] = fase
    metricas["paso"] = paso
    metricas["timestamp"] = datetime.now().isoformat()

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(metricas, ensure_ascii=False) + "\n")


def collate_pad(batch):
    """Pad variable-length sequences to same length in batch."""
    max_len = max(b["input_ids"].size(0) for b in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    targets = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        L = b["input_ids"].size(0)
        input_ids[i, :L] = b["input_ids"]
        targets[i, :L] = b["targets"]
    terr_target = torch.stack([b["terr_target"] for b in batch])
    return {"input_ids": input_ids, "targets": targets, "terr_target": terr_target}


# =============================================================================
# FASE 1: INFANCIA — Curriculum Learning
# =============================================================================

def fase_1_infancia(model, tokenizer, device, config):
    """
    Pre-training con curriculum: del nivel BASICO al PATRONES.
    
    Como un niño que aprende primero a hablar, luego a leer,
    luego a escribir ensayos, luego a programar...
    """
    print("\n" + "=" * 60)
    print("🧒 FASE 1: INFANCIA — Curriculum Learning")
    print("=" * 60)

    cfg = config["fase1"]
    optimizer = crear_optimizer(model, cfg["lr"], config["weight_decay"])
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # Curriculum manager
    curriculum = CurriculumManager(
        criterio_avance=cfg["criterio_avance"],
        min_epochs_nivel=1,
        max_epochs_nivel=5,
    )

    # Archivos de datos
    data_dir = Path(config["data_dir"])
    archivos_datos = sorted(
        list(data_dir.glob("**/*.jsonl")),
    )
    
    if not archivos_datos:
        print(f"  ⚠️  No se encontraron archivos JSONL en {data_dir}")
        print("  Ejecuta: python scripts/generar_curriculum.py primero")
        return model

    print(f"  Archivos de datos: {len(archivos_datos)}")
    for a in archivos_datos[:5]:
        print(f"    - {a.name}")

    # Entrenamiento por niveles
    consolidacion = ConsolidacionHebbiana(
        learning_rate=config["fase4"]["hebbian_lr"],
    )
    
    paso_global = 0
    mejor_loss = float("inf")

    for nivel in NivelDificultad:
        print(f"\n  📚 Nivel {nivel.value}: {nivel.name}")
        print(f"  {'─' * 40}")

        # Dataset para este nivel
        dataset = CurriculumDataset(
            archivos=archivos_datos,
            tokenizer=tokenizer,
            nivel_actual=nivel,
            max_seq_len=config["max_seq_len"],
            revision_ratio=cfg["revision_ratio"],
        )

        dataloader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            num_workers=0,  # IterableDataset
            pin_memory=True,
            collate_fn=collate_pad,
        )

        optimizer.zero_grad()
        loss_accum = 0.0
        n_accum = 0
        epoch_losses = []

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)
            terr_target = batch["terr_target"][0].to(device)  # Same for all in batch

            # Forward con AMP
            with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                logits, loss_ce, info = model(input_ids, targets)

                # Metacognitive loss
                confianza = info.get("exit_capa", model.config.n_capas) / model.config.n_capas

                # Territory alignment loss
                terr_acts, _ = model.talamo(
                    model.emb_drop(model.tok_emb(input_ids)), input_ids
                )
                loss_terr = territory_alignment_loss(terr_acts, terr_target, weight=0.05)

                # Territory entropy
                loss_entropy = territory_entropy_loss(terr_acts, weight=0.01)

                # Total loss
                loss = loss_ce + loss_terr + loss_entropy
                loss = loss / config["grad_accum"]

            # Backward
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            loss_accum += loss.item() * config["grad_accum"]
            n_accum += 1

            # Registrar para Hebbian
            with torch.no_grad():
                consolidacion.registrar_paso(terr_acts.detach(), loss_ce.item(), logits.detach(), targets)

            # Gradient accumulation step
            if n_accum >= config["grad_accum"]:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                    optimizer.step()
                optimizer.zero_grad()

                paso_global += 1
                avg_loss = loss_accum / n_accum
                epoch_losses.append(avg_loss)

                # LR schedule
                lr = get_lr_schedule(
                    paso_global, config["warmup_steps"],
                    cfg["steps_por_nivel"] * 6,  # Total estimado
                    cfg["lr"], cfg["min_lr"],
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                # Log
                if paso_global % 50 == 0:
                    print(
                        f"    Paso {paso_global:6d} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {lr:.2e} | "
                        f"Conf: {confianza:.3f} | "
                        f"Nivel: {nivel.name}"
                    )
                    log_metricas(1, paso_global, {
                        "loss": avg_loss,
                        "lr": lr,
                        "confianza": confianza,
                        "nivel": nivel.name,
                        "loss_terr": loss_terr.item(),
                        "loss_entropy": loss_entropy.item(),
                    })

                # Checkpoint
                if paso_global % 500 == 0:
                    ckpt_path = f"{config['checkpoint_dir']}/fase1_paso_{paso_global}.pt"
                    if avg_loss < mejor_loss:
                        mejor_loss = avg_loss
                        guardar_checkpoint(
                            model, optimizer, 1, paso_global, avg_loss,
                            f"{config['checkpoint_dir']}/fase1_best.pt",
                            extra={"curriculum": curriculum.get_estado()},
                        )
                    guardar_checkpoint(
                        model, optimizer, 1, paso_global, avg_loss, ckpt_path,
                        extra={"curriculum": curriculum.get_estado()},
                    )

                # Consolidación periódica (Hebbian durante training)
                if paso_global % config["fase4"]["consolidation_interval"] == 0:
                    resultado = consolidacion.consolidar(model)
                    if resultado.get("ajustes_aplicados", 0) > 0:
                        print(f"    🧠 Consolidación Hebbiana: signal={resultado['signal_mean']:.4f}")

                loss_accum = 0.0
                n_accum = 0

                # ¿Suficientes pasos en este nivel?
                if paso_global % cfg["steps_por_nivel"] == 0:
                    break

        # Fin del nivel — reportar al curriculum
        avg_epoch_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        resultado = curriculum.reportar_epoch(avg_epoch_loss)
        print(f"\n    📊 Nivel {nivel.name}: loss={avg_epoch_loss:.4f}, acción={resultado['accion']}")
        print(f"      Razón: {resultado['razon']}")

    # Guardar checkpoint final de Fase 1
    guardar_checkpoint(
        model, optimizer, 1, paso_global, mejor_loss,
        f"{config['checkpoint_dir']}/fase1_final.pt",
        extra={"curriculum": curriculum.get_estado()},
    )

    print(f"\n  ✅ Fase 1 completada: {paso_global} pasos, mejor loss={mejor_loss:.4f}")
    return model


# =============================================================================
# FASE 2: EXPERIMENTACIÓN — Self-Play
# =============================================================================

def fase_2_experimentar(model, tokenizer, device, config):
    """
    Self-play: el modelo genera código, lo ejecuta y aprende.
    
    No necesita datasets masivos — se entrena consigo mismo.
    PUEDE CORRER EN TU PC.
    """
    print("\n" + "=" * 60)
    print("🔬 FASE 2: EXPERIMENTACIÓN — Self-Play")
    print("=" * 60)

    cfg = config["fase2"]
    optimizer = crear_optimizer(model, cfg["lr"], config["weight_decay"])

    # Motor de self-play
    engine = SelfPlayEngine(
        model=model,
        tokenizer=tokenizer,
        nivel=1,
        n_intentos=cfg["intentos_por_ronda"],
        max_tokens=cfg["max_tokens_gen"],
        temperature=cfg["temperature"],
        device=str(device),
    )

    mejor_ratio = 0.0
    
    for ronda in range(cfg["n_rondas"]):
        # Ajustar nivel basado en rendimiento
        if engine.stats["total_generados"] > 0:
            ratio_exito = (
                engine.stats["correctos"] + engine.stats["ejecutan"]
            ) / engine.stats["total_generados"]
            
            if ratio_exito > 0.7 and engine.nivel < 6:
                engine.nivel += 1
                print(f"\n  📈 Subiendo nivel a {engine.nivel}")
            
            if ratio_exito > mejor_ratio:
                mejor_ratio = ratio_exito

        # Jugar ronda
        model.eval()
        resultados = engine.jugar_ronda()
        model.train()

        # Entrenar con DPO si hay pares
        batch_dpo = engine.obtener_lote_dpo(cfg["dpo_batch"])
        if batch_dpo is not None:
            optimizer.zero_grad()

            loss = dpo_loss(
                model,
                batch_dpo["preferred_ids"],
                batch_dpo["rejected_ids"],
                batch_dpo["prompt_lens"],
                beta=cfg["dpo_beta"],
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            optimizer.step()

            if ronda % 20 == 0:
                print(
                    f"  Ronda {ronda:4d}/{cfg['n_rondas']} | "
                    f"DPO Loss: {loss.item():.4f} | "
                    f"Nivel: {engine.nivel} | "
                    f"Exito: {100*mejor_ratio:.1f}%"
                )
                log_metricas(2, ronda, {
                    "dpo_loss": loss.item(),
                    "nivel": engine.nivel,
                    "ratio_exito": mejor_ratio,
                    **engine.stats,
                })

        # Checkpoint
        if (ronda + 1) % 200 == 0:
            guardar_checkpoint(
                model, optimizer, 2, ronda, mejor_ratio,
                f"{config['checkpoint_dir']}/fase2_ronda_{ronda}.pt",
            )

    # Stats finales
    print(f"\n{engine.get_stats_str()}")

    guardar_checkpoint(
        model, optimizer, 2, cfg["n_rondas"], mejor_ratio,
        f"{config['checkpoint_dir']}/fase2_final.pt",
    )

    print(f"  ✅ Fase 2 completada: {cfg['n_rondas']} rondas, ratio éxito={100*mejor_ratio:.1f}%")
    return model


# =============================================================================
# FASE 3: FILOSOFAR — Knowledge Distillation
# =============================================================================

def fase_3_filosofar(model, tokenizer, device, config):
    """
    Knowledge Distillation: aprender de un profesor.

    Como un estudiante que aprende de un maestro experimentado:
    1. El profesor resuelve problemas
    2. PAMPAr aprende la RESPUESTA y el RAZONAMIENTO
    3. Soft targets enseñan la distribución de probabilidad
    """
    print("\n" + "=" * 60)
    print("🎓 FASE 3: FILOSOFAR — Knowledge Distillation")
    print("=" * 60)

    cfg = config["fase3"]

    # Seleccionar profesor
    profesor_key = cfg["profesor"]
    if profesor_key in PROFESORES:
        config_prof = PROFESORES[profesor_key]
    else:
        print(f"  ❌ Profesor '{profesor_key}' no encontrado")
        print(f"  Disponibles: {', '.join(PROFESORES.keys())}")
        return model

    print(f"  Profesor: {config_prof.nombre} ({config_prof.modelo})")
    print(f"  Modo: {cfg['modo']}")
    print(f"  Ejemplos: {cfg['n_ejemplos']}")

    # Crear cliente y generador
    cliente = ClienteProfesor(config_prof)
    generador = GeneradorDestilacion(cliente, modo=cfg["modo"])

    # Optimizer
    optimizer = crear_optimizer(model, cfg["lr"], config["weight_decay"])
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # Generar y entrenar incrementalmente
    model.train()
    paso = 0
    mejor_loss = float("inf")
    max_seq_len = config["max_seq_len"]

    for nivel in range(1, 7):
        n_por_nivel = cfg["n_ejemplos"] // 6
        print(f"\n  📚 Nivel {nivel}: {n_por_nivel} ejemplos")

        optimizer.zero_grad()
        loss_accum = 0.0
        n_accum = 0

        for i in range(n_por_nivel):
            try:
                # Generar ejemplo con profesor
                ejemplo = generador.generar(nivel=nivel)
                if not ejemplo:
                    continue

                # Tokenizar
                texto = ejemplo.get("input", "") + "\n" + ejemplo.get("output", "")
                tokens = tokenizer.Encode(texto)[:max_seq_len]
                if len(tokens) < 4:
                    continue

                input_ids = torch.tensor(
                    [tokens[:-1]], dtype=torch.long, device=device,
                )
                targets = torch.tensor(
                    [tokens[1:]], dtype=torch.long, device=device,
                )

                # Forward
                with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                    logits, loss_ce, info = model(input_ids, targets)

                    # Territory-aware distillation
                    loss_terr = territory_aware_distillation(
                        model, tokenizer,
                        ejemplo["output"], nivel,
                    )

                    loss = (1 - cfg["alpha_kd"]) * loss_ce + cfg["alpha_kd"] * loss_terr
                    loss = loss / cfg["grad_accum"]

                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                loss_accum += loss.item() * cfg["grad_accum"]
                n_accum += 1

                if n_accum >= cfg["grad_accum"]:
                    if scaler:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                        optimizer.step()
                    optimizer.zero_grad()

                    paso += 1
                    avg_loss = loss_accum / n_accum

                    if avg_loss < mejor_loss:
                        mejor_loss = avg_loss

                    if paso % 20 == 0:
                        print(
                            f"    Paso {paso:5d} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"Nivel: {nivel} | "
                            f"CE: {loss_ce.item():.4f}"
                        )
                        log_metricas(3, paso, {
                            "loss": avg_loss,
                            "loss_ce": loss_ce.item(),
                            "nivel": nivel,
                            "profesor": profesor_key,
                        })

                    loss_accum = 0.0
                    n_accum = 0

            except Exception as e:
                if i < 3:  # Solo mostrar primeros errores
                    print(f"    ⚠️  Error: {e}")

    # Guardar
    guardar_checkpoint(
        model, optimizer, 3, paso, mejor_loss,
        f"{config['checkpoint_dir']}/fase3_final.pt",
        extra={"profesor": profesor_key, "modo": cfg["modo"]},
    )

    print(f"\n  ✅ Fase 3 completada: {paso} pasos, mejor loss={mejor_loss:.4f}")
    return model


# =============================================================================
# FASE 4: SUEÑO — Consolidación
# =============================================================================

def fase_4_sueno(model, tokenizer, device, config):
    """
    Consolidación sin datos nuevos — replay y fortalecimiento.
    
    Como cuando dormimos y el cerebro consolida memorias.
    GRATIS en tu PC (no necesita datos).
    """
    print("\n" + "=" * 60)
    print("😴 FASE 4: SUEÑO — Consolidación Hebbiana")
    print("=" * 60)

    cfg = config["fase4"]
    consolidacion = ConsolidacionHebbiana(
        learning_rate=cfg["hebbian_lr"],
        homeostasis_weight=cfg["homeostasis_weight"],
        poda_threshold=cfg["poda_threshold"],
    )

    # Para consolidar necesitamos correr algunos forward passes
    # y recopilar estadísticas de activación
    data_dir = Path(config["data_dir"])
    archivos_datos = sorted(list(data_dir.glob("**/*.jsonl")))

    if not archivos_datos:
        print("  ⚠️  Sin datos para consolidación, aplicando solo poda")
        # Poda básica sin estadísticas
        from pampar.coder.v2.aprendizaje.neuroplasticidad import podar_pesos
        act_media = torch.tensor([0.25, 0.25, 0.25, 0.25])
        n_podados = podar_pesos(model, act_media, cfg["poda_threshold"])
        print(f"  🔪 Pesos podados: {n_podados}")
        return model

    dataset = CurriculumDataset(
        archivos=archivos_datos,
        tokenizer=tokenizer,
        nivel_actual=NivelDificultad.PATRONES,  # Todos los niveles
        max_seq_len=config["max_seq_len"],
        revision_ratio=1.0,  # Todo incluido
    )

    dataloader = DataLoader(
        dataset, batch_size=config["batch_size"], num_workers=0,
        collate_fn=collate_pad,
    )

    model.eval()
    n_batches = 0
    max_batches = 500  # Solo necesitamos suficientes para estadísticas

    print("  Recopilando estadísticas de activación...")
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)

            logits, loss, info = model(input_ids, targets)
            
            terr_acts, _ = model.talamo(
                model.emb_drop(model.tok_emb(input_ids)), input_ids
            )
            
            consolidacion.registrar_paso(terr_acts, loss.item(), logits, targets)

            n_batches += 1
            if n_batches >= max_batches:
                break
            if n_batches % 100 == 0:
                print(f"    {n_batches}/{max_batches} batches procesados")

    # Aplicar consolidación
    print("\n  Aplicando consolidación Hebbiana...")
    resultado = consolidacion.consolidar(model)

    print(f"    Signal Hebbian: mean={resultado['signal_mean']:.4f}, "
          f"max={resultado['signal_max']:.4f}")
    print(f"    Activación media: {resultado['activacion_media']}")
    print(f"    Pesos podados: {resultado['pesos_podados']}")
    print(f"    Éxitos/Fallos: {resultado['n_exitos']}/{resultado['n_fallos']}")

    log_metricas(4, 0, resultado)

    guardar_checkpoint(
        model, torch.optim.AdamW(model.parameters(), lr=1e-5), 
        4, 0, 0.0,
        f"{config['checkpoint_dir']}/fase4_consolidado.pt",
        extra={"consolidacion": consolidacion.get_estado()},
    )

    print("  ✅ Fase 4 completada: modelo consolidado")
    return model


# =============================================================================
# FASE 5: CURIOSIDAD — Active Learning
# =============================================================================

def fase_5_curiosidad(model, tokenizer, device, config):
    """
    Active Learning: entrena más en lo que NO sabe.
    
    El modelo identifica sus débilidades usando Early Exit
    y dedica más tiempo a esos ejemplos.
    
    PUEDE CORRER EN TU PC.
    """
    print("\n" + "=" * 60)
    print("🔍 FASE 5: CURIOSIDAD — Active Learning")
    print("=" * 60)

    cfg = config["fase5"]
    optimizer = crear_optimizer(model, cfg["lr"], config["weight_decay"])
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # Active learner
    learner = ActiveLearner(
        confidence_threshold=cfg["confidence_threshold"],
        oversample_ratio=cfg["oversample_ratio"],
    )

    # Dataset completo
    data_dir = Path(config["data_dir"])
    archivos_datos = sorted(list(data_dir.glob("**/*.jsonl")))

    if not archivos_datos:
        print("  ⚠️  Sin datos para active learning")
        return model

    dataset = CurriculumDataset(
        archivos=archivos_datos,
        tokenizer=tokenizer,
        nivel_actual=NivelDificultad.PATRONES,
        max_seq_len=config["max_seq_len"],
        revision_ratio=1.0,
    )

    dataloader = DataLoader(
        dataset, batch_size=config["batch_size"], num_workers=0,
        collate_fn=collate_pad,
    )

    # Fase 1: Evaluar todo y encontrar puntos débiles
    print("  Evaluando dificultad de ejemplos...")
    model.eval()
    n_eval = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)

        confianzas, es_dificil = learner.evaluar_dificultad(model, input_ids, targets)
        learner.agregar_al_buffer(input_ids, targets, confianzas, es_dificil)

        n_eval += input_ids.shape[0]
        if n_eval >= 5000:  # Evaluar hasta 5000 ejemplos
            break
        if n_eval % 1000 == 0:
            print(f"    Evaluados: {n_eval}")

    print(f"\n{learner.get_stats_str()}")

    # Fase 2: Entrenar con ejemplos difíciles
    print("  Entrenando con ejemplos difíciles...")
    model.train()
    mejor_loss = float("inf")
    
    for paso in range(cfg["steps"]):
        batch_dificil = learner.obtener_batch_dificil(
            config["batch_size"], str(device)
        )
        
        if batch_dificil is None:
            print("  ⚠️  Buffer de difíciles agotado")
            break

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            logits, loss, info = model(
                batch_dificil["input_ids"],
                batch_dificil["targets"],
            )

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            optimizer.step()

        if loss.item() < mejor_loss:
            mejor_loss = loss.item()

        if paso % 100 == 0:
            print(f"    Paso {paso}/{cfg['steps']} | Loss: {loss.item():.4f}")
            log_metricas(5, paso, {"loss": loss.item()})

    guardar_checkpoint(
        model, optimizer, 5, cfg["steps"], mejor_loss,
        f"{config['checkpoint_dir']}/fase5_final.pt",
    )

    print(f"  ✅ Fase 5 completada")
    return model


# =============================================================================
# FASE 6: SOCIAL — Online Learning
# =============================================================================

def fase_6_social(model, tokenizer, device, config):
    """
    Servidor que aprende de las personas que lo usan.

    Lanza un servidor HTTP donde:
    1. Usuarios envían prompts → modelo genera código
    2. Usuarios dan feedback (aceptar/editar/rechazar)
    3. El modelo se actualiza gradualmente

    Solo actualiza parámetros "plásticos" (Tálamo + últimas capas).
    PUEDE CORRER EN TU PC.
    """
    print("\n" + "=" * 60)
    print("👥 FASE 6: SOCIAL — Online Learning")
    print("=" * 60)

    cfg = config["fase6"]

    print(f"  Host: {cfg['host']}:{cfg['port']}")
    print(f"  LR: {cfg['lr']}")
    print(f"  Update interval: cada {cfg['update_interval']} interacciones")
    print(f"  Plastic ratio: {cfg['plastic_ratio'] * 100:.0f}% de params")

    # Crear entrenador online
    entrenador, servidor = crear_servidor(
        model=model,
        tokenizer=tokenizer,
        host=cfg["host"],
        port=cfg["port"],
        device=str(device),
    )

    if servidor is None:
        print("  ❌ No se pudo crear el servidor")
        return model

    print(f"\n  🌐 Servidor listo. Endpoints:")
    print(f"    POST http://localhost:{cfg['port']}/generate")
    print(f"      Body: {{\"prompt\": \"def fibonacci(n):\", \"max_tokens\": 256}}")
    print(f"    POST http://localhost:{cfg['port']}/feedback")
    print(f"      Body: {{\"accepted\": true}} o {{\"edited\": \"código corregido\"}}")
    print(f"    GET  http://localhost:{cfg['port']}/stats")
    print(f"\n  Presiona Ctrl+C para detener y guardar.")

    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\n  Deteniendo servidor...")
        servidor.shutdown()

    # Guardar modelo actualizado
    stats = entrenador.get_stats()
    print(f"\n  📊 Estadísticas finales:")
    print(f"    Updates online: {stats['total_updates']}")
    print(f"    Interacciones totales: {stats['total']}")
    print(f"    Reward medio: {stats['reward_medio']:.3f}")

    guardar_checkpoint(
        model,
        entrenador.optimizer,
        6, stats["total_updates"], stats["reward_medio"],
        f"{config['checkpoint_dir']}/fase6_online.pt",
        extra={"online_stats": stats},
    )

    print(f"  ✅ Fase 6 completada: {stats['total_updates']} updates online")
    return model


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="🧠 Entrenamiento Cerebral PAMPAr-Coder"
    )
    parser.add_argument(
        "--fase", type=int, default=0,
        help="Fase específica (1-6). 0=todas (excluye 6/online)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resumir desde último checkpoint"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Checkpoint específico para cargar"
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Modo local (tu PC)"
    )
    parser.add_argument(
        "--preset", type=str, default="1.5b",
        choices=["4gb", "8gb", "1.5b"],
        help="Preset del modelo"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Directorio de datos"
    )
    parser.add_argument(
        "--profesor", type=str, default=None,
        help=f"Profesor para Fase 3: {', '.join(PROFESORES.keys())}"
    )
    parser.add_argument(
        "--modo-kd", type=str, default=None,
        choices=["codigo", "cot", "correccion"],
        help="Modo de destilación para Fase 3"
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="Puerto para Fase 6 (online learning server)"
    )
    args = parser.parse_args()

    # Seleccionar preset
    presets = {"4gb": PRESET_4GB, "8gb": PRESET_8GB, "1.5b": PRESET_1_5B}
    model_config = presets[args.preset]

    # Ajustar config para modo local
    config = copy.deepcopy(CONFIG)
    if args.local:
        config["batch_size"] = 2
        config["grad_accum"] = 16
        config["max_seq_len"] = 1024
        print("🖥️  Modo local activado (batch=2, seq_len=1024)")

    if args.batch_size:
        config["batch_size"] = args.batch_size
    
    if args.data_dir:
        config["data_dir"] = args.data_dir

    if args.profesor:
        config["fase3"]["profesor"] = args.profesor
    if args.modo_kd:
        config["fase3"]["modo"] = args.modo_kd
    if args.port:
        config["fase6"]["port"] = args.port

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🧠 PAMPAr-Coder — Entrenamiento Cerebral")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  Preset: {args.preset}")
    print(f"  Params: {model_config.estimate_params():,}")

    # Crear modelo
    print("\n  Creando modelo...")
    model = crear_modelo(model_config).to(device)
    print(f"  Params reales: {sum(p.numel() for p in model.parameters()):,}")

    # Cargar tokenizer
    tokenizer_path = str(project_dir / config["tokenizer_path"])
    if not os.path.exists(tokenizer_path):
        print(f"  ⚠️  Tokenizer no encontrado: {tokenizer_path}")
        print(f"  Buscando en ubicaciones alternativas...")
        # Buscar en el directorio del proyecto
        for alt in [
            project_dir / "data" / "tokenizer" / "pampar_48k.model",
            Path("data/tokenizer/pampar_48k.model"),
        ]:
            if alt.exists():
                tokenizer_path = str(alt)
                break
        else:
            print("  ❌ Tokenizer no encontrado. Ejecuta prepare_tokenizer.py primero.")
            return

    print(f"  Cargando tokenizer: {tokenizer_path}")
    tokenizer = cargar_tokenizer(tokenizer_path)
    n_registered = model.registrar_tokenizer(tokenizer)
    print(f"  Tokens registrados en LLAVES: {n_registered}")

    # Cargar checkpoint si existe
    if args.checkpoint:
        print(f"\n  Cargando checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"  Fase: {ckpt.get('fase', '?')}, Paso: {ckpt.get('paso', '?')}")

    # Ejecutar fases
    print(f"\n{'═' * 60}")
    print(f"  INICIANDO ENTRENAMIENTO CEREBRAL")
    print(f"{'═' * 60}")

    t_start = time.time()

    fases = {
        1: ("INFANCIA", fase_1_infancia),
        2: ("EXPERIMENTAR", fase_2_experimentar),
        3: ("FILOSOFAR", fase_3_filosofar),
        4: ("SUEÑO", fase_4_sueno),
        5: ("CURIOSIDAD", fase_5_curiosidad),
        6: ("SOCIAL", fase_6_social),
    }

    if args.fase > 0:
        # Solo una fase
        if args.fase in fases:
            nombre, func = fases[args.fase]
            model = func(model, tokenizer, device, config)
        else:
            print(f"  ❌ Fase {args.fase} no implementada")
    else:
        # Todas las fases (excluye 6=online, que necesita servidor)
        for fase_num, (nombre, func) in sorted(fases.items()):
            if fase_num == 6:
                print("\n  ℹ️  Fase 6 (Online) se omite en modo automático.")
                print("     Ejecuta: python scripts/train_cerebral.py --fase 6")
                continue
            model = func(model, tokenizer, device, config)

    t_total = time.time() - t_start
    print(f"\n{'═' * 60}")
    print(f"  ENTRENAMIENTO COMPLETADO en {t_total/3600:.1f} horas")
    print(f"{'═' * 60}")

    # Guardar modelo final
    guardar_checkpoint(
        model,
        torch.optim.AdamW(model.parameters(), lr=1e-5),
        99, 0, 0.0,
        f"{config['checkpoint_dir']}/cerebral_final.pt",
    )


if __name__ == "__main__":
    main()
