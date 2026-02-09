# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Lanzador cloud: sube archivos y arranca entrenamiento 1.5B en RunPod.

Uso:
    python scripts/launch_1_5B.py              # Lanzar entrenamiento
    python scripts/launch_1_5B.py --resume     # Resumir desde último checkpoint
    python scripts/launch_1_5B.py --status     # Ver estado
    python scripts/launch_1_5B.py --stop       # Parar pod
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

# Agregar raíz del proyecto
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# RunPod config (use env vars for secrets)
API_KEY = os.environ.get("RUNPOD_API_KEY", "")
POD_ID = os.environ.get("RUNPOD_POD_ID", "")
SSH_HOST = None  # Se detecta al arrancar
SSH_PORT = None
SSH_KEY = os.path.expanduser("~/.ssh/runpod_key")

WORKSPACE = "/workspace/PAMPAr-Coder"

# Archivos a subir (relativos a PAMPAr-Coder/)
UPLOAD_FILES = [
    # Modelo
    "pampar/coder/v2/__init__.py",
    "pampar/coder/v2/config.py",
    "pampar/coder/v2/modelo.py",
    "pampar/coder/v2/bloques.py",
    "pampar/coder/v2/talamo.py",
    "pampar/coder/v2/zonas.py",
    "pampar/coder/v2/llaves.py",
    "pampar/__init__.py",
    "pampar/coder/__init__.py",
    
    # Script de entrenamiento
    "scripts/train_1_5B.py",
    
    # Tokenizer
    "data/tokenizer/pampar_48k.model",
    "data/tokenizer/pampar_48k.vocab",
]

# Datos - estos son más grandes, se verifican si ya existen en el pod
DATA_FILES = [
    "data/code/github_code.jsonl",
    "data/code/train.jsonl",
    "data/code/train_massive.jsonl",
    "data/distillation/codealpaca_20k.jsonl",
    "data/distillation/codeexercises_python.jsonl",
    "data/distillation/code_feedback.jsonl",
    "data/distillation/distillation_data.jsonl",
    "data/distillation/evol_instruct_code_80k.jsonl",
]


def get_pod_info():
    """Obtiene info del pod via API."""
    import requests
    
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    query = f'''{{
        pod(input: {{ podId: "{POD_ID}" }}) {{
            id
            name
            desiredStatus
            runtime {{
                uptimeInSeconds
                ports {{ ip isIpPublic privatePort publicPort type }}
                gpus {{ id gpuUtilPercent memoryUtilPercent }}
            }}
        }}
    }}'''
    
    r = requests.post("https://api.runpod.io/graphql", json={"query": query}, headers=headers)
    data = r.json()
    
    if "errors" in data:
        print(f"Error API: {data['errors']}")
        return None
    
    return data.get("data", {}).get("pod")


def start_pod():
    """Arranca el pod."""
    import requests
    
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    mutation = f'''mutation {{
        podResume(input: {{ podId: "{POD_ID}", gpuCount: 1 }}) {{
            id desiredStatus
        }}
    }}'''
    
    r = requests.post("https://api.runpod.io/graphql", json={"query": mutation}, headers=headers)
    data = r.json()
    print(f"Start pod: {data}")
    return data


def stop_pod():
    """Para el pod."""
    import requests
    
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    mutation = f'''mutation {{
        podStop(input: {{ podId: "{POD_ID}" }}) {{
            id desiredStatus
        }}
    }}'''
    
    r = requests.post("https://api.runpod.io/graphql", json={"query": mutation}, headers=headers)
    print(f"Stop: {r.json()}")


def get_ssh_info(pod_info):
    """Extrae SSH host/port del pod info."""
    if not pod_info or not pod_info.get("runtime"):
        return None, None
    
    ports = pod_info["runtime"].get("ports", [])
    for p in ports:
        if p.get("privatePort") == 22 and p.get("isIpPublic"):
            return p["ip"], p["publicPort"]
    
    return None, None


def wait_for_pod(timeout=120):
    """Espera a que el pod esté listo con SSH."""
    print("Esperando a que el pod arranque...")
    
    for i in range(timeout // 5):
        pod = get_pod_info()
        if pod and pod.get("desiredStatus") == "RUNNING":
            host, port = get_ssh_info(pod)
            if host and port:
                # Verificar SSH
                time.sleep(10)  # Dar tiempo al SSH server
                print(f"  Pod listo! SSH: {host}:{port}")
                return host, port
        
        status = pod.get("desiredStatus", "?") if pod else "?"
        print(f"  [{i*5}s] Status: {status}")
        time.sleep(5)
    
    print("Timeout esperando pod!")
    return None, None


def ssh_exec(host, port, cmd, timeout=30):
    """Ejecuta comando via SSH."""
    import paramiko
    
    key = paramiko.Ed25519Key.from_private_key_file(SSH_KEY)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        client.connect(host, port=port, username="root", pkey=key, timeout=10)
        stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        return out, err
    finally:
        client.close()


def scp_upload(host, port, local_path, remote_path):
    """Sube archivo via SCP."""
    import paramiko
    
    key = paramiko.Ed25519Key.from_private_key_file(SSH_KEY)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        client.connect(host, port=port, username="root", pkey=key, timeout=10)
        sftp = client.open_sftp()
        
        # Crear directorio remoto si no existe
        remote_dir = str(Path(remote_path).parent)
        try:
            sftp.stat(remote_dir)
        except FileNotFoundError:
            # Crear recursivamente
            parts = remote_dir.split("/")
            current = ""
            for part in parts:
                if not part:
                    continue
                current += f"/{part}"
                try:
                    sftp.stat(current)
                except FileNotFoundError:
                    sftp.mkdir(current)
        
        sftp.put(str(local_path), remote_path)
        sftp.close()
    finally:
        client.close()


def upload_files(host, port, project_root, files, force=False):
    """Sube archivos al pod."""
    for rel_path in files:
        local = project_root / rel_path
        remote = f"{WORKSPACE}/{rel_path}"
        
        if not local.exists():
            print(f"  ⚠️  No existe: {local}")
            continue
        
        size_mb = local.stat().st_size / 1024**2
        
        if not force:
            # Verificar si ya está en el pod
            out, _ = ssh_exec(host, port, f"stat --printf='%s' {remote} 2>/dev/null || echo 'NOFILE'")
            if out.strip() != "NOFILE":
                remote_size = int(out.strip())
                if abs(remote_size - local.stat().st_size) < 100:
                    print(f"  ✓ Ya existe: {rel_path} ({size_mb:.1f}MB)")
                    continue
        
        print(f"  ↑ Subiendo: {rel_path} ({size_mb:.1f}MB)...", end="", flush=True)
        scp_upload(host, port, local, remote)
        print(" ✓")


def launch_training(host, port, resume=False):
    """Lanza el entrenamiento en el pod."""
    # Instalar dependencias si faltan
    print("\nVerificando dependencias...")
    ssh_exec(host, port, "pip install -q sentencepiece 2>/dev/null", timeout=60)
    
    # Comando de entrenamiento
    cmd = f"cd {WORKSPACE} && nohup python scripts/train_1_5B.py"
    if resume:
        # Buscar último checkpoint
        out, _ = ssh_exec(host, port, f"ls -t {WORKSPACE}/checkpoints_1_5B/step_*.pt 2>/dev/null | head -1")
        last_ckpt = out.strip()
        if last_ckpt:
            cmd += f" --resume {last_ckpt}"
            print(f"  Resumiendo desde: {last_ckpt}")
        else:
            print("  No se encontró checkpoint, entrenando desde cero")
    
    cmd += f" > {WORKSPACE}/training_1_5B.log 2>&1 &"
    
    print(f"\n🚀 Lanzando entrenamiento...")
    ssh_exec(host, port, cmd, timeout=10)
    
    # Verificar que arrancó
    time.sleep(5)
    out, _ = ssh_exec(host, port, "pgrep -f train_1_5B.py")
    if out.strip():
        pid = out.strip().split()[0]
        print(f"  ✅ Entrenamiento corriendo (PID: {pid})")
        
        # Mostrar primeras líneas del log
        time.sleep(5)
        out, _ = ssh_exec(host, port, f"tail -20 {WORKSPACE}/training_1_5B.log 2>/dev/null")
        if out:
            print("\n--- Log (últimas 20 líneas) ---")
            print(out)
    else:
        print("  ❌ El proceso no arrancó!")
        out, _ = ssh_exec(host, port, f"tail -30 {WORKSPACE}/training_1_5B.log 2>/dev/null")
        print("--- Error log ---")
        print(out)


def show_status(host, port):
    """Muestra el estado actual del entrenamiento."""
    # Proceso corriendo?
    out, _ = ssh_exec(host, port, "pgrep -af train_1_5B.py")
    running = bool(out.strip())
    print(f"Proceso: {'✅ CORRIENDO' if running else '❌ DETENIDO'}")
    if out.strip():
        print(f"  {out.strip()}")
    
    # GPU
    out, _ = ssh_exec(host, port, "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader")
    if out.strip():
        print(f"GPU: {out.strip()}")
    
    # Últimas líneas del log
    out, _ = ssh_exec(host, port, f"tail -15 {WORKSPACE}/training_1_5B.log 2>/dev/null")
    if out:
        print(f"\n--- Log reciente ---")
        print(out)
    
    # Checkpoints
    out, _ = ssh_exec(host, port, f"ls -lh {WORKSPACE}/checkpoints_1_5B/*.pt 2>/dev/null | tail -10")
    if out:
        print(f"\n--- Checkpoints ---")
        print(out)


def main():
    parser = argparse.ArgumentParser(description="PAMPAr-Coder 1.5B Cloud Launcher")
    parser.add_argument("--status", action="store_true", help="Ver estado")
    parser.add_argument("--stop", action="store_true", help="Parar pod")
    parser.add_argument("--resume", action="store_true", help="Resumir entrenamiento")
    parser.add_argument("--skip-data", action="store_true", help="No subir datos (ya están)")
    parser.add_argument("--force-upload", action="store_true", help="Forzar re-upload")
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent  # PAMPAr-Coder/
    
    if args.stop:
        stop_pod()
        return
    
    # Verificar pod
    pod = get_pod_info()
    if not pod:
        print("No se puede conectar al pod")
        return
    
    status = pod.get("desiredStatus", "UNKNOWN")
    print(f"Pod: {pod.get('name', POD_ID)} | Status: {status}")
    
    # Si está parado, arrancarlo
    if status != "RUNNING":
        print("Arrancando pod...")
        start_pod()
        host, port = wait_for_pod()
    else:
        host, port = get_ssh_info(pod)
    
    if not host or not port:
        print("No se pudo obtener SSH info")
        return
    
    print(f"SSH: {host}:{port}")
    
    # Solo status?
    if args.status:
        show_status(host, port)
        return
    
    # Subir código y tokenizer
    print("\n📦 Subiendo código y tokenizer...")
    upload_files(host, port, project_root, UPLOAD_FILES, force=args.force_upload)
    
    # Subir datos (si no skip)
    if not args.skip_data:
        print("\n📦 Verificando datos de entrenamiento...")
        upload_files(host, port, project_root, DATA_FILES)
    
    # Lanzar entrenamiento
    launch_training(host, port, resume=args.resume)
    
    print("\n" + "=" * 60)
    print("🧠 Entrenamiento lanzado!")
    print("=" * 60)
    print(f"  Monitor: python scripts/launch_1_5B.py --status")
    print(f"  SSH:     ssh -i {SSH_KEY} -p {port} root@{host}")
    print(f"  Log:     tail -f {WORKSPACE}/training_1_5B.log")
    print(f"  Parar:   python scripts/launch_1_5B.py --stop")


if __name__ == "__main__":
    main()
