# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Descarga los mejores datasets de código para entrenamiento.

Datasets:
1. CodeSearchNet (GitHub) - 6M funciones en 6 lenguajes
2. The Stack Smol - Subset curado de The Stack
3. CodeAlpaca - 20K instrucciones de código
4. Evol-Instruct-Code - 80K instrucciones evolucionadas
5. CodeFeedback - Código con feedback
"""

import os
import json
import gzip
import requests
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

DATA_DIR = Path(__file__).parent.parent / "data" / "code"

DATASETS = {
    "codesearchnet": {
        "url": "https://huggingface.co/datasets/code_search_net/resolve/main/python/train.jsonl.gz",
        "description": "CodeSearchNet Python (500K funciones)",
        "format": "jsonl.gz",
        "fields": ["code", "docstring"],
    },
    "codealpaca": {
        "url": "https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k/resolve/main/code_alpaca_20k.json",
        "description": "CodeAlpaca 20K instrucciones",
        "format": "json",
        "fields": ["instruction", "output"],
    },
    "evol_instruct": {
        "url": "https://huggingface.co/datasets/nickrosh/Evol-Instruct-Code-80k-v1/resolve/main/EvolInstruct-Code-80k.json",
        "description": "Evol-Instruct-Code 80K",
        "format": "json",
        "fields": ["instruction", "output"],
    },
    "commitpack": {
        "url": "https://huggingface.co/datasets/bigcode/commitpack/resolve/main/data/python/train-00000-of-00020.parquet",
        "description": "CommitPack Python (commits reales)",
        "format": "parquet",
        "fields": ["new_contents", "message"],
    },
}


# =============================================================================
# FUNCIONES DE DESCARGA
# =============================================================================

def download_file(url: str, dest: Path, chunk_size: int = 8192) -> bool:
    """
    Descarga archivo con barra de progreso.
    
    Args:
        url: URL a descargar
        dest: Ruta destino
        chunk_size: Tamaño de chunk
        
    Returns:
        True si exitoso
    """
    try:
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        
        total = int(resp.headers.get("content-length", 0))
        
        with open(dest, "wb") as f:
            with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        return True
    except Exception as e:
        print(f"❌ Error descargando {url}: {e}")
        return False


def process_jsonl_gz(path: Path, output: Path, fields: List[str], max_samples: int = None):
    """Procesa archivo JSONL.GZ."""
    count = 0
    
    with gzip.open(path, "rt", encoding="utf-8") as f_in:
        with open(output, "w", encoding="utf-8") as f_out:
            for line in tqdm(f_in, desc="Processing"):
                if max_samples and count >= max_samples:
                    break
                
                try:
                    data = json.loads(line)
                    
                    # Extraer campos relevantes
                    record = {}
                    for field in fields:
                        if field in data:
                            record[field] = data[field]
                    
                    if record:
                        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        count += 1
                except:
                    continue
    
    return count


def process_json(path: Path, output: Path, fields: List[str], max_samples: int = None):
    """Procesa archivo JSON (lista de objetos)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    count = 0
    with open(output, "w", encoding="utf-8") as f_out:
        for item in tqdm(data, desc="Processing"):
            if max_samples and count >= max_samples:
                break
            
            record = {}
            for field in fields:
                if field in item:
                    record[field] = item[field]
            
            # Normalizar a formato instruction/output
            if "instruction" in record and "output" in record:
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    
    return count


def process_parquet(path: Path, output: Path, fields: List[str], max_samples: int = None):
    """Procesa archivo Parquet."""
    try:
        import pandas as pd
        df = pd.read_parquet(path)
        
        if max_samples:
            df = df.head(max_samples)
        
        count = 0
        with open(output, "w", encoding="utf-8") as f:
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
                record = {}
                for field in fields:
                    if field in row:
                        record[field] = str(row[field])
                
                if record:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
        
        return count
    except ImportError:
        print("⚠️ pandas y pyarrow requeridos para parquet")
        return 0


# =============================================================================
# MAIN
# =============================================================================

def download_datasets(
    datasets: List[str] = None,
    max_samples_per_dataset: int = 50000,
):
    """
    Descarga y procesa datasets.
    
    Args:
        datasets: Lista de nombres (None = todos)
        max_samples_per_dataset: Máximo por dataset
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if datasets is None:
        datasets = list(DATASETS.keys())
    
    print("=" * 60)
    print("📦 PAMPAr-Coder: Descarga de Datasets de Código")
    print("=" * 60)
    
    stats = {}
    
    for name in datasets:
        if name not in DATASETS:
            print(f"⚠️ Dataset desconocido: {name}")
            continue
        
        info = DATASETS[name]
        print(f"\n📥 {info['description']}")
        
        # Descargar
        ext = info["format"].split(".")[-1]
        raw_path = DATA_DIR / f"{name}_raw.{info['format']}"
        output_path = DATA_DIR / f"{name}.jsonl"
        
        # Skip si ya existe
        if output_path.exists():
            print(f"   ✅ Ya existe: {output_path}")
            with open(output_path) as f:
                stats[name] = sum(1 for _ in f)
            continue
        
        # Descargar
        if not raw_path.exists():
            if not download_file(info["url"], raw_path):
                continue
        
        # Procesar
        print(f"   🔄 Procesando...")
        
        if info["format"] == "jsonl.gz":
            count = process_jsonl_gz(raw_path, output_path, info["fields"], max_samples_per_dataset)
        elif info["format"] == "json":
            count = process_json(raw_path, output_path, info["fields"], max_samples_per_dataset)
        elif info["format"] == "parquet":
            count = process_parquet(raw_path, output_path, info["fields"], max_samples_per_dataset)
        else:
            print(f"   ❌ Formato no soportado: {info['format']}")
            continue
        
        stats[name] = count
        print(f"   ✅ {count:,} muestras guardadas")
        
        # Limpiar raw
        if raw_path.exists():
            raw_path.unlink()
    
    # Resumen
    print("\n" + "=" * 60)
    print("📊 Resumen de Datasets:")
    print("=" * 60)
    
    total = 0
    for name, count in stats.items():
        print(f"   {name:20}: {count:>10,} muestras")
        total += count
    
    print("-" * 60)
    print(f"   {'TOTAL':20}: {total:>10,} muestras")
    
    return stats


def merge_datasets(output: str = "train_all.jsonl"):
    """
    Combina todos los datasets en uno solo.
    """
    output_path = DATA_DIR / output
    count = 0
    
    print(f"\n🔄 Combinando datasets en {output}...")
    
    with open(output_path, "w", encoding="utf-8") as f_out:
        for jsonl in DATA_DIR.glob("*.jsonl"):
            if jsonl.name == output:
                continue
            
            with open(jsonl, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    f_out.write(line)
                    count += 1
    
    print(f"✅ {count:,} muestras totales en {output_path}")
    return count


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Descarga datasets de código")
    parser.add_argument("--datasets", nargs="+", help="Datasets específicos")
    parser.add_argument("--max-samples", type=int, default=50000,
                        help="Máximo muestras por dataset")
    parser.add_argument("--merge", action="store_true",
                        help="Combinar datasets al final")
    
    args = parser.parse_args()
    
    # Descargar
    download_datasets(args.datasets, args.max_samples)
    
    # Combinar si se pide
    if args.merge:
        merge_datasets()
