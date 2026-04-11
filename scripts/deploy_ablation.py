#!/usr/bin/env python3
# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
deploy_ablation.py — Despliega pod RunPod y lanza ablación.

Uso:
  python scripts/deploy_ablation.py --create
  python scripts/deploy_ablation.py --status
  python scripts/deploy_ablation.py --terminate
"""

import argparse
import json
import os
import sys
from pathlib import Path

import runpod

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Config ───────────────────────────────────────────────────────────────────

POD_CONFIG = {
    "name": "pampar-ablation",
    "image_name": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
    "gpu_type_id": "NVIDIA GeForce RTX 3090",
    "cloud_type": "COMMUNITY",
    "gpu_count": 1,
    "volume_in_gb": 50,
    "container_disk_in_gb": 40,
    "ports": "8888/http,22/tcp",
    "volume_mount_path": "/workspace",
    "docker_args": "",
    "env": {
        "JUPYTER_PASSWORD": "pampar2026",
        "PUBLIC_KEY": "",
    },
}

# GPUs de fallback (orden de preferencia por precio/rendimiento)
GPU_FALLBACKS = [
    "NVIDIA GeForce RTX 3090",
    "NVIDIA RTX A5000",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX A6000",
]

SETUP_SCRIPT = r"""#!/bin/bash
set -e

echo "=== PAMPAr Ablation Setup ==="

# Instalar dependencias
pip install sentencepiece

# Clonar repo si no existe
if [ ! -d /workspace/PAMPAr-Coder ]; then
    echo "Clonando repo..."
    cd /workspace
    git clone https://github.com/lucasmella-stack/PAMPAr-Coder.git
    cd PAMPAr-Coder
else
    echo "Repo ya existe, actualizando..."
    cd /workspace/PAMPAr-Coder
    git pull
fi

# Crear directorio de resultados
mkdir -p /workspace/PAMPAr-Coder/ablation_results

# Verificar GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GiB')"

echo "=== Setup completo ==="
echo ""
echo "Para lanzar la ablación:"
echo "  cd /workspace/PAMPAr-Coder"
echo "  python scripts/ablation_train.py --all --batch-size 16 --max-pasos 30000"
echo ""
echo "Para Jupyter monitoring:"
echo "  jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser"
"""


def _load_api_key() -> str:
    """Carga la API key desde .env o variable de entorno."""
    key = os.environ.get("RUNPOD_API_KEY", "")
    if not key:
        env_path = ROOT / ".env"
        if env_path.exists():
            with env_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("RUNPOD_API_KEY="):
                        key = line.split("=", 1)[1].strip()
                        break
    if not key:
        print("ERROR: RUNPOD_API_KEY no encontrada")
        sys.exit(1)
    return key


def _load_ssh_pubkey() -> str:
    """Carga la SSH public key si existe."""
    paths = [
        Path.home() / ".ssh" / "runpod_key.pub",
        Path.home() / "ssh-runpod.pub",
        Path.home() / ".ssh" / "id_rsa.pub",
        Path.home() / ".ssh" / "id_ed25519.pub",
    ]
    for p in paths:
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    return ""


def create_pod() -> None:
    """Crea un pod en RunPod."""
    api_key = _load_api_key()
    runpod.api_key = api_key

    # SSH key
    ssh_key = _load_ssh_pubkey()
    if ssh_key:
        POD_CONFIG["env"]["PUBLIC_KEY"] = ssh_key
        print(f"SSH key encontrada: {ssh_key[:40]}...")

    # Intentar con cada GPU
    pod = None
    for gpu in GPU_FALLBACKS:
        print(f"Intentando con {gpu}...")
        try:
            pod = runpod.create_pod(
                name=POD_CONFIG["name"],
                image_name=POD_CONFIG["image_name"],
                gpu_type_id=gpu,
                cloud_type=POD_CONFIG["cloud_type"],
                gpu_count=POD_CONFIG["gpu_count"],
                volume_in_gb=POD_CONFIG["volume_in_gb"],
                container_disk_in_gb=POD_CONFIG["container_disk_in_gb"],
                ports=POD_CONFIG["ports"],
                volume_mount_path=POD_CONFIG["volume_mount_path"],
                docker_args=POD_CONFIG["docker_args"],
                env=POD_CONFIG["env"],
            )
            if pod and pod.get("id"):
                print(f"\n✓ Pod creado: {pod['id']}")
                print(f"  GPU: {gpu}")
                print(f"  Status: {pod.get('desiredStatus', 'RUNNING')}")
                break
        except Exception as e:
            print(f"  No disponible: {e}")
            continue

    if not pod or not pod.get("id"):
        print("\n✗ No se pudo crear el pod con ninguna GPU disponible")
        sys.exit(1)

    # Guardar pod info
    info_path = ROOT / "ablation_results" / "pod_info.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    with info_path.open("w", encoding="utf-8") as f:
        json.dump(pod, f, indent=2)
    print(f"\nInfo guardada en: {info_path}")

    print("\n── Próximos pasos ──")
    print("1. Espera ~2min a que el pod inicie")
    print("2. python scripts/deploy_ablation.py --status")
    print("3. Conéctate por SSH o Jupyter y ejecuta el setup:")
    print(f"   ssh root@<IP> -i ~/.ssh/runpod_key")
    print("4. Copia y pega el setup script, o clona y ejecuta manualmente")


def get_status() -> None:
    """Muestra el estado de los pods."""
    api_key = _load_api_key()
    runpod.api_key = api_key

    pods = runpod.get_pods()
    if not pods:
        print("No hay pods activos")
        return

    for pod in pods:
        gpu = pod.get("machine", {}).get("gpu", "?")
        status = pod.get("desiredStatus", "?")
        runtime = pod.get("runtime", {})
        uptime = runtime.get("uptimeInSeconds", 0) if runtime else 0
        cost = (
            runtime.get("gpus", [{}])[0].get("cost", 0)
            if runtime and runtime.get("gpus")
            else 0
        )

        print(f"\n── Pod: {pod['id']} ──")
        print(f"  Nombre: {pod.get('name', '?')}")
        print(f"  GPU: {gpu}")
        print(f"  Status: {status}")
        print(f"  Uptime: {uptime // 3600}h {(uptime % 3600) // 60}m")
        if cost:
            print(f"  Costo/hr: ${cost:.3f}")
            print(f"  Costo total: ${(cost * uptime / 3600):.2f}")


def terminate_pod() -> None:
    """Termina todos los pods de ablación."""
    api_key = _load_api_key()
    runpod.api_key = api_key

    pods = runpod.get_pods()
    ablation_pods = [p for p in pods if "ablation" in p.get("name", "").lower()]

    if not ablation_pods:
        print("No hay pods de ablación activos")
        return

    for pod in ablation_pods:
        print(f"Terminando pod {pod['id']} ({pod.get('name', '?')})...")
        runpod.terminate_pod(pod["id"])
        print(f"  ✓ Terminado")


def main() -> None:
    p = argparse.ArgumentParser(description="Gestión de pod RunPod para ablación")
    p.add_argument("--create", action="store_true", help="Crear pod")
    p.add_argument("--status", action="store_true", help="Ver estado")
    p.add_argument("--terminate", action="store_true", help="Terminar pods")
    p.add_argument("--setup-script", action="store_true", help="Imprimir setup script")
    args = p.parse_args()

    if args.setup_script:
        print(SETUP_SCRIPT)
    elif args.create:
        create_pod()
    elif args.status:
        get_status()
    elif args.terminate:
        terminate_pod()
    else:
        p.print_help()


if __name__ == "__main__":
    main()
