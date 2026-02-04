# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Script launcher que ejecuta el entrenamiento en un proceso separado,
resistente a interrupciones del terminal/VS Code.
"""

import subprocess
import sys
import os

# Cambiar al directorio del proyecto
os.chdir(r"C:\Users\lucas\Documents\Be Web\Lunux-AI\PAMPAr-Coder")

# Comando de entrenamiento
cmd = [
    sys.executable,
    "scripts/train_stable.py",
    "--epochs", "5",
    "--batch-size", "4", 
    "--max-samples", "5000",
    "--lr", "1e-4"
]

# Crear archivo de log
log_file = open("training_log.txt", "w", encoding="utf-8")

print("=" * 60)
print("  Iniciando entrenamiento en proceso separado...")
print("  Log: training_log.txt")
print("  Para ver progreso: Get-Content training_log.txt -Wait")
print("=" * 60)

# Ejecutar en proceso separado con creationflags para Windows
CREATE_NEW_PROCESS_GROUP = 0x00000200
DETACHED_PROCESS = 0x00000008

try:
    process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        creationflags=CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS
    )
    print(f"\n  PID: {process.pid}")
    print("  El proceso continuará incluso si cierras esta terminal.")
    print("\n  Comandos útiles:")
    print(f"    - Ver log: Get-Content training_log.txt -Tail 50")
    print(f"    - Seguir log: Get-Content training_log.txt -Wait")
    print(f"    - Matar proceso: Stop-Process -Id {process.pid}")
except Exception as e:
    print(f"Error: {e}")
    log_file.close()
