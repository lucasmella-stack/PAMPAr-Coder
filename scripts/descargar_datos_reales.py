#!/usr/bin/env python3
# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Descarga código Python real de HuggingFace y lo añade a biblioteca/.

Fuentes (100% gratuitas, sin auth):
  1. flytech/python-codes-25k          — 25K scripts Python reales
  2. iamtarun/python_code_instructions — 18K pares instrucción+código Python
  3. bigcode/humanevalpack             — 164 problemas con tests verificados

Uso:
  python scripts/descargar_datos_reales.py
  python scripts/descargar_datos_reales.py --max-por-fuente 5000
  python scripts/descargar_datos_reales.py --solo flytech
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BIBLIOTECA = ROOT / "biblioteca"
INDICE_PATH = BIBLIOTECA / "indice.json"


# ─────────────────────────────────────────────────────────────────────────────
# Filtros de calidad
# ─────────────────────────────────────────────────────────────────────────────

def es_python_valido(texto: str, min_lineas: int = 5, max_lineas: int = 500) -> bool:
    """Heurísticas básicas de calidad para filtrar basura."""
    if not texto or not texto.strip():
        return False
    lineas = [l for l in texto.splitlines() if l.strip()]
    n = len(lineas)
    if n < min_lineas or n > max_lineas:
        return False
    # Debe tener algo reconociblemente Python
    tiene_python = any(kw in texto for kw in ("def ", "class ", "import ", "for ", "if ", "return "))
    if not tiene_python:
        return False
    # Rechazar archivos que parezcan datos o configuración pura
    if texto.count("=") > n * 3:
        return False
    return True


def limpiar_texto(texto: str) -> str:
    """Limpieza básica: quitar BOM, trailing spaces excesivos."""
    return texto.strip().replace("\r\n", "\n").replace("\r", "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Descargadores por fuente
# ─────────────────────────────────────────────────────────────────────────────

def descargar_flytech(max_muestras: int, destino: Path) -> int:
    """flytech/python-codes-25k — scripts Python reales."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  ✗ datasets no instalado. Ejecutar: pip install datasets")
        return 0

    print("  Conectando a flytech/python-codes-25k ...", flush=True)
    try:
        ds = load_dataset("flytech/python-codes-25k", split="train", streaming=True)
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return 0

    destino.mkdir(parents=True, exist_ok=True)
    salida = destino / "flytech_python.jsonl"
    guardados = 0

    with salida.open("w", encoding="utf-8") as f:
        for ejemplo in ds:
            if guardados >= max_muestras:
                break
            # El dataset tiene columna 'output' con el código
            texto = ejemplo.get("output") or ejemplo.get("text") or ejemplo.get("content") or ""
            texto = limpiar_texto(texto)
            if es_python_valido(texto):
                f.write(json.dumps({"text": texto}, ensure_ascii=False) + "\n")
                guardados += 1
                if guardados % 1000 == 0:
                    print(f"    → {guardados} muestras...", flush=True)

    print(f"  ✓ flytech: {guardados} muestras → {salida.name}")
    return guardados


def descargar_instrucciones(max_muestras: int, destino: Path) -> int:
    """iamtarun/python_code_instructions_18k_alpaca — pares instrucción+código."""
    try:
        from datasets import load_dataset
    except ImportError:
        return 0

    print("  Conectando a iamtarun/python_code_instructions_18k_alpaca ...", flush=True)
    try:
        ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train", streaming=True)
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return 0

    destino.mkdir(parents=True, exist_ok=True)
    salida = destino / "python_instrucciones.jsonl"
    guardados = 0

    with salida.open("w", encoding="utf-8") as f:
        for ejemplo in ds:
            if guardados >= max_muestras:
                break
            instruccion = ejemplo.get("instruction", "")
            entrada = ejemplo.get("input", "")
            output = ejemplo.get("output", "")
            if not output:
                continue
            # Formato: instrucción + código = contexto completo de aprendizaje
            if entrada:
                texto = f"# {instruccion}\n# Input: {entrada}\n{output}"
            else:
                texto = f"# {instruccion}\n{output}"
            texto = limpiar_texto(texto)
            if es_python_valido(texto, min_lineas=3):
                f.write(json.dumps({"text": texto}, ensure_ascii=False) + "\n")
                guardados += 1
                if guardados % 1000 == 0:
                    print(f"    → {guardados} muestras...", flush=True)

    print(f"  ✓ instrucciones: {guardados} muestras → {salida.name}")
    return guardados


def descargar_humaneval(destino: Path) -> int:
    """bigcode/humanevalpack Python — 164 problemas con tests verificados."""
    try:
        from datasets import load_dataset
    except ImportError:
        return 0

    print("  Conectando a bigcode/humanevalpack (python) ...", flush=True)
    try:
        ds = load_dataset("bigcode/humanevalpack", "python", split="test", streaming=False)
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return 0

    destino.mkdir(parents=True, exist_ok=True)
    salida = destino / "humaneval_python.jsonl"
    guardados = 0

    with salida.open("w", encoding="utf-8") as f:
        for ejemplo in ds:
            # prompt + canonical_solution = función completa
            prompt = ejemplo.get("prompt", "")
            solucion = ejemplo.get("canonical_solution", "")
            tests = ejemplo.get("test", "")
            texto = limpiar_texto(f"{prompt}{solucion}\n\n{tests}")
            if es_python_valido(texto, min_lineas=3):
                f.write(json.dumps({"text": texto}, ensure_ascii=False) + "\n")
                guardados += 1

    print(f"  ✓ humaneval: {guardados} muestras → {salida.name}")
    return guardados


# ─────────────────────────────────────────────────────────────────────────────
# Actualizar índice
# ─────────────────────────────────────────────────────────────────────────────

def actualizar_indice(archivos_nuevos: list[dict]) -> None:
    """Añade las nuevas fuentes a biblioteca/indice.json."""
    if INDICE_PATH.exists():
        indice = json.loads(INDICE_PATH.read_text(encoding="utf-8"))
    else:
        indice = {"version": "1.0", "descripcion": "Biblioteca de conocimiento PamparV3"}

    # Añadir categoría python_real si no existe
    if "python_real" not in indice:
        indice["python_real"] = []

    nombres_existentes = {t["nombre"] for t in indice.get("python_real", [])}

    for archivo in archivos_nuevos:
        nombre = archivo["nombre"]
        if nombre not in nombres_existentes:
            indice["python_real"].append(archivo)
            nombres_existentes.add(nombre)
            print(f"  + índice: {nombre}")

    INDICE_PATH.write_text(json.dumps(indice, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✓ indice.json actualizado — {len(indice.get('python_real', []))} fuentes reales")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Descarga datos Python reales para PamparV3")
    p.add_argument("--max-por-fuente", type=int, default=8000,
                   help="Máximo de muestras por dataset (default: 8000)")
    p.add_argument("--solo", choices=["flytech", "instrucciones", "humaneval"],
                   help="Descargar solo una fuente específica")
    args = p.parse_args()

    destino = BIBLIOTECA / "python_real"
    archivos_nuevos = []
    total = 0

    print("\n── Descargando datos Python reales ─────────────────────────")

    if not args.solo or args.solo == "flytech":
        print("\n[1/3] flytech/python-codes-25k")
        n = descargar_flytech(args.max_por_fuente, destino)
        if n > 0:
            total += n
            archivos_nuevos.append({
                "nombre": "flytech_python",
                "nivel": 2,
                "archivo": "python_real/flytech_python.jsonl",
            })

    if not args.solo or args.solo == "instrucciones":
        print("\n[2/3] python_code_instructions_18k")
        n = descargar_instrucciones(args.max_por_fuente, destino)
        if n > 0:
            total += n
            archivos_nuevos.append({
                "nombre": "python_instrucciones",
                "nivel": 2,
                "archivo": "python_real/python_instrucciones.jsonl",
            })

    if not args.solo or args.solo == "humaneval":
        print("\n[3/3] humanevalpack Python")
        n = descargar_humaneval(destino)
        if n > 0:
            total += n
            archivos_nuevos.append({
                "nombre": "humaneval_python",
                "nivel": 3,
                "archivo": "python_real/humaneval_python.jsonl",
            })

    if archivos_nuevos:
        print(f"\n── Actualizando índice ──────────────────────────────────────")
        actualizar_indice(archivos_nuevos)

    print(f"\n✅ Total descargado: {total:,} muestras")
    print(f"   Destino: {destino}")
    if total > 0:
        print("\nPróximo paso:")
        print("  & 'C:\\Users\\lucas\\AppData\\Local\\Programs\\Python\\Python313\\python.exe' \\")
        print("    scripts/train_v3.py --checkpoint checkpoints/v3_train.pt --lr 3e-5 --batch-size 2 --seq-len 256")


if __name__ == "__main__":
    main()
