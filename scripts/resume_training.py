# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
🔄 Resume Training — Retoma entrenamiento desde checkpoint

Script para cuando el pod se para y necesitás retomar.

Uso:
  # En el pod: resume la última fase
  python scripts/resume_training.py

  # Resume fase específica
  python scripts/resume_training.py --fase 2

  # Resume con checkpoint específico
  python scripts/resume_training.py --checkpoint checkpoints/cerebral/fase1_final.pt --fase 2

  # Solo evaluar con HumanEval-Mini (rápido)
  python scripts/resume_training.py --eval-only --checkpoint checkpoints/cerebral/fase1_final.pt

Este script automatiza:
  1. Detectar último checkpoint
  2. Cargar modelo + estado
  3. Continuar con la siguiente fase (o la misma)
  4. Instalar watchdog auto-stop para el pod
"""

import argparse
import json
import os
import sys
import glob
import subprocess
from pathlib import Path
from datetime import datetime

import torch

# Ajustar path
script_dir = Path(__file__).parent
project_dir = script_dir.parent
sys.path.insert(0, str(project_dir))


def find_latest_checkpoint(checkpoint_dir: str = "checkpoints/cerebral") -> dict:
    """Encuentra el checkpoint más reciente y su fase."""
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return {"path": None, "fase": 0, "paso": 0}
    
    # Buscar todos los checkpoints
    ckpts = list(ckpt_dir.glob("*.pt"))
    if not ckpts:
        return {"path": None, "fase": 0, "paso": 0}
    
    # Ordenar por fecha de modificación
    ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    latest = ckpts[0]
    
    # Extraer fase del nombre
    name = latest.stem
    fase = 0
    paso = 0
    
    if "fase1" in name:
        fase = 1
    elif "fase2" in name:
        fase = 2
    elif "fase3" in name:
        fase = 3
    elif "fase4" in name:
        fase = 4
    elif "fase5" in name:
        fase = 5
    elif "fase6" in name:
        fase = 6
    elif "cerebral_final" in name:
        fase = 99
    
    # Cargar metadata si existe
    try:
        ckpt = torch.load(str(latest), map_location="cpu", weights_only=False)
        fase = ckpt.get("fase", fase)
        paso = ckpt.get("paso", paso)
        loss = ckpt.get("loss", "?")
        print(f"  📦 Checkpoint: {latest.name}")
        print(f"     Fase: {fase}, Paso: {paso}, Loss: {loss}")
        print(f"     Fecha: {ckpt.get('timestamp', 'desconocida')}")
    except Exception as e:
        print(f"  ⚠️  Error leyendo metadata: {e}")
    
    return {"path": str(latest), "fase": fase, "paso": paso}


def detect_next_fase(current_fase: int) -> int:
    """Determina la siguiente fase a ejecutar."""
    FINAL_PHASES = {
        0: 1,   # Sin checkpoint → empezar desde cero
        1: 2,   # Terminó Fase 1 → hacer Fase 2
        2: 3,   # Terminó Fase 2 → hacer Fase 3
        3: 4,   # ...
        4: 5,
        5: 6,
        99: 0,  # Ya completó todo
    }
    
    if "final" not in str(current_fase):
        # Si el checkpoint no es "final", repetir la misma fase
        return current_fase
    
    return FINAL_PHASES.get(current_fase, current_fase + 1)


def setup_watchdog(api_key: str = None, pod_id: str = None):
    """Configura watchdog para parar el pod cuando termine."""
    if not api_key or not pod_id:
        # Intentar leer de env
        api_key = api_key or os.environ.get("RUNPOD_API_KEY", "")
        pod_id = pod_id or os.environ.get("RUNPOD_POD_ID", "")
    
    if not api_key or not pod_id:
        print("  ⚠️  Sin RUNPOD_API_KEY o RUNPOD_POD_ID — watchdog no configurado")
        return
    
    watchdog_script = f"""#!/bin/bash
TRAIN_PID=$$TRAIN_PID$$
POD_ID="{pod_id}"
API_KEY="{api_key}"

while kill -0 $TRAIN_PID 2>/dev/null; do
    sleep 30
done

echo "[watchdog] Training finished at $(date)" >> /workspace/watchdog.log
curl -s -X POST https://api.runpod.io/graphql \\
  -H 'Content-Type: application/json' \\
  -H "Authorization: Bearer $API_KEY" \\
  -d '{{"query": "mutation {{ podStop(input: {{podId: \\\\"'$POD_ID'\\\\"}}) {{ id }} }}"}}' \\
  >> /workspace/watchdog.log 2>&1
"""
    
    with open("/workspace/watchdog_resume.sh", "w") as f:
        f.write(watchdog_script)
    os.chmod("/workspace/watchdog_resume.sh", 0o755)
    print("  🐕 Watchdog script preparado")


def main():
    parser = argparse.ArgumentParser(description="🔄 Resume Training")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--fase", type=int, default=None,
                       help="Fase a ejecutar (override auto-detection)")
    parser.add_argument("--preset", type=str, default="1.5b",
                       choices=["4gb", "8gb", "1.5b"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-only", action="store_true",
                       help="Solo evaluar, no entrenar")
    parser.add_argument("--eval-mini", action="store_true",
                       help="Evaluar con HumanEval-Mini (rápido)")
    args = parser.parse_args()
    
    print(f"\n🔄 PAMPAr-Coder — Resume Training")
    print(f"{'═' * 50}")
    
    # 1. Encontrar checkpoint
    if args.checkpoint:
        ckpt_info = {"path": args.checkpoint, "fase": 0, "paso": 0}
        # Cargar info
        try:
            ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            ckpt_info["fase"] = ckpt.get("fase", 0)
            ckpt_info["paso"] = ckpt.get("paso", 0)
            print(f"  📦 Checkpoint: {args.checkpoint}")
            print(f"     Fase: {ckpt_info['fase']}, Paso: {ckpt_info['paso']}")
        except Exception as e:
            print(f"  ⚠️  Error: {e}")
    else:
        print("  Buscando último checkpoint...")
        ckpt_info = find_latest_checkpoint()
    
    if ckpt_info["path"] is None:
        print("  ❌ No se encontró checkpoint")
        print("  Ejecuta: python scripts/train_cerebral.py --fase 1 --preset 1.5b")
        return
    
    # 2. Determinar siguiente fase
    if args.fase is not None:
        next_fase = args.fase
    else:
        # Si checkpoint es "final", avanzar; si no, repetir
        if "final" in Path(ckpt_info["path"]).stem:
            next_fase = ckpt_info["fase"] + 1
        else:
            next_fase = ckpt_info["fase"]
    
    if next_fase > 6:
        print("  ✅ ¡Entrenamiento completo! Todas las fases terminadas.")
        print("  Ejecuta evaluación: python scripts/evaluate_v2.py --checkpoint", ckpt_info["path"])
        return
    
    # 3. Eval-only?
    if args.eval_only or args.eval_mini:
        benchmark = "mini" if args.eval_mini else "humaneval"
        cmd = [
            sys.executable, str(project_dir / "scripts" / "evaluate_v2.py"),
            "--checkpoint", ckpt_info["path"],
            "--preset", args.preset,
            "--benchmark", benchmark,
            "--save-samples",
        ]
        print(f"\n  Ejecutando evaluación ({benchmark})...")
        os.execvp(cmd[0], cmd)
        return
    
    # 4. Lanzar entrenamiento
    print(f"\n  ▶️  Lanzando Fase {next_fase} desde checkpoint {Path(ckpt_info['path']).name}")
    
    cmd = [
        sys.executable, str(project_dir / "scripts" / "train_cerebral.py"),
        "--fase", str(next_fase),
        "--preset", args.preset,
        "--batch-size", str(args.batch_size),
        "--checkpoint", ckpt_info["path"],
    ]
    
    print(f"  Comando: {' '.join(cmd)}")
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
