# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Monitor de entrenamiento en tiempo real.

Muestra el progreso del entrenamiento background sin interferir.
Ejecutar en otra terminal mientras entrena.
"""

import json
import time
import argparse
from pathlib import Path
from datetime import datetime


def format_time(hours: float) -> str:
    """Formatea horas a string legible."""
    if hours < 1:
        return f"{hours * 60:.0f} min"
    elif hours < 24:
        return f"{hours:.1f} h"
    else:
        return f"{hours / 24:.1f} dias"


def monitor(checkpoint_dir: Path, refresh: float = 2.0):
    """Monitorea el entrenamiento."""
    status_file = checkpoint_dir / "training_status.json"
    
    print("\n" + "="*50)
    print(" PAMPAr-Coder - Monitor de Entrenamiento")
    print("="*50)
    print(f" Status file: {status_file}")
    print(" Presiona Ctrl+C para salir")
    print("="*50 + "\n")
    
    last_step = 0
    
    while True:
        try:
            if status_file.exists():
                with open(status_file, 'r') as f:
                    status = json.load(f)
                
                running = "Entrenando" if status.get('running', False) else "Pausado"
                step = status.get('global_step', 0)
                loss = status.get('loss', 0)
                speed = status.get('speed', 0)
                elapsed = status.get('elapsed_hours', 0)
                epoch = status.get('epoch', 0)
                
                # Calcular steps/min
                steps_diff = step - last_step
                steps_per_min = steps_diff * (60 / refresh) if steps_diff > 0 else 0
                last_step = step
                
                # Limpiar línea y mostrar
                print(f"\r Step: {step:,} | Epoch: {epoch} | Loss: {loss:.4f} | "
                      f"Speed: {speed:.1f} s/s | Time: {format_time(elapsed)} | "
                      f"Status: {running}      ", end="", flush=True)
            else:
                print(f"\r Esperando inicio del entrenamiento...    ", end="", flush=True)
            
            time.sleep(refresh)
            
        except KeyboardInterrupt:
            print("\n\n Monitor detenido.")
            break
        except Exception as e:
            print(f"\r Error leyendo status: {e}    ", end="", flush=True)
            time.sleep(refresh)


def main():
    parser = argparse.ArgumentParser(description='Monitor de entrenamiento')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directorio de checkpoints')
    parser.add_argument('--refresh', type=float, default=2.0,
                        help='Intervalo de refresh en segundos')
    args = parser.parse_args()
    
    monitor(Path(args.checkpoint_dir), args.refresh)


if __name__ == "__main__":
    main()
