#!/usr/bin/env python3
# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
download_ablation.py — Descarga resultados de ablación + termina el pod.

Monitorea el pod cada N minutos; cuando los 4 experimentos terminan,
descarga ablation_results/ vía scp y opcionalmente termina el pod.

Uso:
  python scripts/download_ablation.py --host 194.14.47.19 --port 22935
  python scripts/download_ablation.py --host 194.14.47.19 --port 22935 --interval 5
  python scripts/download_ablation.py --host 194.14.47.19 --port 22935 --download-now
  python scripts/download_ablation.py --host 194.14.47.19 --port 22935 --no-terminate

Opciones:
  --host          IP del pod RunPod
  --port          Puerto SSH del pod
  --key           Path a la clave SSH privada (default: ~/.ssh/runpod_key)
  --interval      Minutos entre checks (default: 5)
  --download-now  Descargar inmediatamente sin esperar a que termine
  --no-terminate  No terminar el pod después de descargar
  --out-dir       Carpeta local destino (default: ./ablation_results)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EXPECTED_EXPERIMENTS = ["pampar_v3", "no_llaves", "single_stream", "vanilla_gpt"]
MAX_STEPS = 30_000
REMOTE_LOG = "/workspace/PAMPAr-Coder/ablation.log"
REMOTE_RESULTS = "/workspace/PAMPAr-Coder/ablation_results"


# ---------------------------------------------------------------------------
# SSH helpers
# ---------------------------------------------------------------------------


def _ssh_base(host: str, port: int, key: Path) -> list[str]:
    """Construye el prefijo base para comandos SSH."""
    return [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "BatchMode=yes",
        "-i",
        str(key),
        f"root@{host}",
        "-p",
        str(port),
    ]


def run_remote(host: str, port: int, key: Path, command: str) -> tuple[int, str]:
    """Ejecuta un comando remoto, devuelve (returncode, stdout+stderr)."""
    result = subprocess.run(
        _ssh_base(host, port, key) + [command],
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = (result.stdout + result.stderr).strip()
    return result.returncode, output


def scp_download(
    host: str, port: int, key: Path, remote_path: str, local_path: Path
) -> bool:
    """Descarga un directorio remoto vía scp."""
    local_path.mkdir(parents=True, exist_ok=True)
    cmd = [
        "scp",
        "-r",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=30",
        "-i",
        str(key),
        "-P",
        str(port),
        f"root@{host}:{remote_path}",
        str(local_path),
    ]
    print(f"  scp: {remote_path} → {local_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ERROR] scp falló:\n{result.stderr}", file=sys.stderr)
        return False
    return True


# ---------------------------------------------------------------------------
# Estado del entrenamiento
# ---------------------------------------------------------------------------


def check_training_status(host: str, port: int, key: Path) -> dict:
    """
    Consulta el estado del entrenamiento en el pod remoto.

    Returns:
        dict con claves:
          - reachable: bool
          - process_alive: bool
          - last_log_lines: str
          - finished_experiments: list[str]
          - gpu_util: str
          - gpu_mem: str
    """
    status: dict = {
        "reachable": False,
        "process_alive": False,
        "last_log_lines": "",
        "finished_experiments": [],
        "gpu_util": "?",
        "gpu_mem": "?",
    }

    # Verificar conectividad
    rc, _ = run_remote(host, port, key, "echo OK")
    if rc != 0:
        return status
    status["reachable"] = True

    # Proceso vivo
    rc, out = run_remote(
        host, port, key, "ps aux | grep ablation_train | grep -v grep | wc -l"
    )
    status["process_alive"] = out.strip() == "1"

    # Últimas líneas del log
    rc, out = run_remote(
        host, port, key, f"tail -12 {REMOTE_LOG} 2>/dev/null || echo NO_LOG"
    )
    status["last_log_lines"] = out

    # GPU
    rc, out = run_remote(
        host,
        port,
        key,
        "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || echo ?,?",
    )
    parts = out.strip().split(",")
    if len(parts) == 2:
        status["gpu_util"] = parts[0].strip()
        status["gpu_mem"] = parts[1].strip()

    # Experimentos completados (tienen checkpoint.pt al paso 30K)
    finished: list[str] = []
    for exp in EXPECTED_EXPERIMENTS:
        rc, out = run_remote(
            host,
            port,
            key,
            f"grep -c '\"step\": {MAX_STEPS}' {REMOTE_RESULTS}/{exp}/metrics.jsonl 2>/dev/null || echo 0",
        )
        if out.strip() not in ("0", ""):
            finished.append(exp)
    status["finished_experiments"] = finished

    return status


def all_done(status: dict) -> bool:
    """True si los 4 experimentos terminaron."""
    return set(EXPECTED_EXPERIMENTS) == set(status["finished_experiments"])


# ---------------------------------------------------------------------------
# Descarga
# ---------------------------------------------------------------------------


def download_results(host: str, port: int, key: Path, out_dir: Path) -> bool:
    """Descarga ablation_results/ del pod a out_dir."""
    print(f"\n{'=' * 60}")
    print(f"  DESCARGANDO RESULTADOS")
    print(f"{'=' * 60}")
    print(f"  Origen: root@{host}:{REMOTE_RESULTS}")
    print(f"  Destino: {out_dir}")
    print()

    success = scp_download(host, port, key, REMOTE_RESULTS + "/", out_dir)

    if success:
        print(f"\n  Descarga completa → {out_dir}")
        # Verificar qué llegó
        for exp in EXPECTED_EXPERIMENTS:
            exp_dir = out_dir / exp
            if exp_dir.exists():
                files = list(exp_dir.iterdir())
                print(f"    {exp:>16}: {len(files)} archivos")
            else:
                print(f"    {exp:>16}: [no encontrado]")
    else:
        print("\n  [ERROR] Descarga falló", file=sys.stderr)

    return success


# ---------------------------------------------------------------------------
# Terminar pod
# ---------------------------------------------------------------------------


def terminate_pod_via_api() -> bool:
    """Termina el pod de ablación via RunPod API."""
    try:
        import runpod
    except ImportError:
        print(
            "  [INFO] runpod no instalado, saltando terminación via API",
            file=sys.stderr,
        )
        return False

    # Cargar API key
    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not api_key:
        env_path = ROOT / ".env"
        if env_path.exists():
            with env_path.open(encoding="utf-8") as f:
                for line in f:
                    if line.startswith("RUNPOD_API_KEY="):
                        api_key = line.split("=", 1)[1].strip()
                        break

    if not api_key:
        print(
            "  [WARNING] RUNPOD_API_KEY no encontrada, no se puede terminar el pod",
            file=sys.stderr,
        )
        return False

    runpod.api_key = api_key

    pods = runpod.get_pods()
    ablation_pods = [p for p in pods if "ablation" in p.get("name", "").lower()]

    if not ablation_pods:
        print("  [INFO] No se encontraron pods de ablación activos")
        return True

    for pod in ablation_pods:
        pod_id = pod["id"]
        print(f"  Terminando pod {pod_id} ({pod.get('name', '?')})...")
        runpod.terminate_pod(pod_id)
        print(f"  ✓ Pod {pod_id} terminado")

    return True


# ---------------------------------------------------------------------------
# Loop de monitoreo
# ---------------------------------------------------------------------------


def monitor_loop(
    host: str,
    port: int,
    key: Path,
    interval_min: int,
    out_dir: Path,
    no_terminate: bool,
) -> None:
    """Monitorea el pod periódicamente hasta completar o falla."""
    print(f"\nMonitoreando pod {host}:{port} cada {interval_min} min")
    print(f"Ctrl+C para salir sin descargar\n")

    check_count = 0

    while True:
        check_count += 1
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] Check #{check_count}...")

        try:
            status = check_training_status(host, port, key)
        except subprocess.TimeoutExpired:
            print("  [WARNING] Timeout al conectar al pod")
            _sleep(interval_min)
            continue
        except Exception as e:
            print(f"  [WARNING] Error: {e}")
            _sleep(interval_min)
            continue

        if not status["reachable"]:
            print("  Pod no alcanzable (¿ya terminó o fue detenido?)")
            _sleep(interval_min)
            continue

        proc = "VIVO" if status["process_alive"] else "MUERTO"
        finished = status["finished_experiments"]
        n_done = len(finished)

        print(
            f"  Proceso: {proc} | GPU: {status['gpu_util']} | Mem: {status['gpu_mem']}"
        )
        print(f"  Experimentos finalizados: {n_done}/4 {finished}")

        if status["last_log_lines"]:
            last = status["last_log_lines"].splitlines()[-1]
            print(f"  Último log: {last}")

        if all_done(status):
            print("\n  TODOS LOS EXPERIMENTOS COMPLETADOS")
            break

        if not status["process_alive"] and n_done < 4:
            print("\n  [WARNING] Proceso muerto antes de completar los 4 experimentos.")
            print("  Descargando lo que hay...")
            break

        _sleep(interval_min)

    # Descargar resultados
    download_results(host, port, key, out_dir)

    # Terminar pod
    if not no_terminate:
        print(f"\n{'=' * 60}")
        print("  TERMINANDO POD")
        print(f"{'=' * 60}")
        terminated = terminate_pod_via_api()
        if not terminated:
            print(
                f"\n  Para terminar manualmente:\n"
                f"  python scripts/deploy_ablation.py --terminate"
            )

    print("\nDone.")


def _sleep(interval_min: int) -> None:
    """Duerme interval_min minutos con progreso visual."""
    total_sec = interval_min * 60
    print(f"  Esperando {interval_min} min...", end="", flush=True)
    time.sleep(total_sec)
    print(" listo")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Descargar resultados de ablación y terminar pod"
    )
    parser.add_argument("--host", required=True, help="IP del pod RunPod")
    parser.add_argument("--port", type=int, default=22935, help="Puerto SSH")
    parser.add_argument(
        "--key",
        type=Path,
        default=Path.home() / ".ssh" / "runpod_key",
        help="Path a la clave SSH privada",
    )
    parser.add_argument("--interval", type=int, default=5, help="Minutos entre checks")
    parser.add_argument(
        "--download-now",
        action="store_true",
        help="Descargar inmediatamente sin esperar",
    )
    parser.add_argument(
        "--no-terminate",
        action="store_true",
        help="No terminar el pod después de descargar",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "ablation_results",
        help="Carpeta local destino para los resultados",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Solo mostrar estado y salir (sin descargar)",
    )
    args = parser.parse_args()

    # Verificar clave SSH
    if not args.key.exists():
        print(f"[ERROR] Clave SSH no encontrada: {args.key}", file=sys.stderr)
        print("  Especifica --key /ruta/a/tu/clave", file=sys.stderr)
        sys.exit(1)

    # Solo status
    if args.status:
        print(f"Consultando estado de {args.host}:{args.port}...")
        status = check_training_status(args.host, args.port, args.key)
        if not status["reachable"]:
            print("Pod no alcanzable")
            sys.exit(1)
        print(f"Proceso vivo: {status['process_alive']}")
        print(f"GPU: {status['gpu_util']} | Mem: {status['gpu_mem']}")
        print(f"Experimentos terminados: {status['finished_experiments']}")
        print(f"\nÚltimas líneas del log:\n{status['last_log_lines']}")
        sys.exit(0)

    # Descarga inmediata
    if args.download_now:
        ok = download_results(args.host, args.port, args.key, args.out_dir)
        if ok and not args.no_terminate:
            terminate_pod_via_api()
        sys.exit(0 if ok else 1)

    # Loop de monitoreo
    monitor_loop(
        host=args.host,
        port=args.port,
        key=args.key,
        interval_min=args.interval,
        out_dir=args.out_dir,
        no_terminate=args.no_terminate,
    )


if __name__ == "__main__":
    main()
