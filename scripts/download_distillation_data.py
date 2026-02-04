# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Descarga datasets gratuitos de código destilado para PAMPAr-Coder.

Datasets incluidos:
- CodeAlpaca-20k (GPT-4 destilado)
- Code-Feedback (instrucciones de código)  
- Evol-Instruct-Code (WizardCoder style)
- Self-instruct código

Total estimado: ~100-200M tokens GRATIS
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict
import time

# Intentar importar datasets
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Instalando datasets...")
    os.system("pip install datasets")
    from datasets import load_dataset


# Datasets gratuitos de código destilado
DATASETS_CONFIG = [
    {
        "name": "CodeAlpaca-20k",
        "hf_id": "sahil2801/CodeAlpaca-20k",
        "split": "train",
        "text_field": ["instruction", "input", "output"],
        "format": "instruction",
        "estimated_tokens": "10M",
    },
    {
        "name": "Code-Feedback", 
        "hf_id": "m-a-p/Code-Feedback",
        "split": "train",
        "text_field": ["query", "response"],
        "format": "qa",
        "estimated_tokens": "50M",
    },
    {
        "name": "Evol-Instruct-Code-80k",
        "hf_id": "nickrosh/Evol-Instruct-Code-80k-v1",
        "split": "train", 
        "text_field": ["instruction", "output"],
        "format": "instruction",
        "estimated_tokens": "40M",
    },
    {
        "name": "CodeExercises-Python",
        "hf_id": "jinaai/code_exercises",
        "split": "train",
        "text_field": ["problem", "solution"],
        "format": "qa",
        "estimated_tokens": "20M",
    },
]


def format_instruction(item: Dict, fields: List[str]) -> str:
    """Formatea un ejemplo de instrucción."""
    parts = []
    
    if "instruction" in fields and item.get("instruction"):
        parts.append(f"### Instruction:\n{item['instruction']}")
    
    if "input" in fields and item.get("input"):
        parts.append(f"### Input:\n{item['input']}")
    
    if "query" in fields and item.get("query"):
        parts.append(f"### Query:\n{item['query']}")
    
    if "problem" in fields and item.get("problem"):
        parts.append(f"### Problem:\n{item['problem']}")
        
    if "output" in fields and item.get("output"):
        parts.append(f"### Response:\n{item['output']}")
    
    if "response" in fields and item.get("response"):
        parts.append(f"### Response:\n{item['response']}")
    
    if "solution" in fields and item.get("solution"):
        parts.append(f"### Solution:\n{item['solution']}")
    
    return "\n\n".join(parts)


def download_dataset(config: Dict, output_dir: Path, max_samples: int = None) -> int:
    """Descarga un dataset y lo guarda en formato JSONL."""
    name = config["name"]
    hf_id = config["hf_id"]
    
    print(f"\n{'='*60}")
    print(f"Descargando: {name}")
    print(f"HuggingFace ID: {hf_id}")
    print(f"Tokens estimados: {config['estimated_tokens']}")
    print(f"{'='*60}")
    
    try:
        # Cargar dataset
        ds = load_dataset(hf_id, split=config["split"], trust_remote_code=True)
        
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        
        # Preparar archivo de salida
        output_file = output_dir / f"{name.lower().replace('-', '_')}.jsonl"
        
        count = 0
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in ds:
                try:
                    text = format_instruction(item, config["text_field"])
                    if text and len(text) > 50:  # Filtrar ejemplos muy cortos
                        json.dump({"text": text, "source": name}, f, ensure_ascii=False)
                        f.write('\n')
                        count += 1
                except Exception as e:
                    continue
        
        print(f"✅ {name}: {count:,} ejemplos guardados en {output_file.name}")
        return count
        
    except Exception as e:
        print(f"❌ Error descargando {name}: {e}")
        return 0


def merge_datasets(output_dir: Path) -> Path:
    """Combina todos los datasets en uno solo."""
    merged_file = output_dir / "distillation_data.jsonl"
    
    total = 0
    with open(merged_file, 'w', encoding='utf-8') as out:
        for jsonl_file in output_dir.glob("*.jsonl"):
            if jsonl_file.name == "distillation_data.jsonl":
                continue
            
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    out.write(line)
                    total += 1
    
    print(f"\n{'='*60}")
    print(f"📦 Dataset combinado: {merged_file.name}")
    print(f"   Total ejemplos: {total:,}")
    print(f"   Tamaño: {merged_file.stat().st_size / (1024*1024):.1f} MB")
    print(f"{'='*60}")
    
    return merged_file


def main():
    parser = argparse.ArgumentParser(description='Descarga datasets de destilación')
    parser.add_argument('--output-dir', type=str, default='data/distillation',
                        help='Directorio de salida')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Máximo de samples por dataset (para testing)')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='Datasets específicos a descargar')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("🚀 PAMPAr-Coder - Descarga de Datasets de Destilación")
    print("="*60)
    print(f"\nDatasets disponibles:")
    for cfg in DATASETS_CONFIG:
        print(f"  • {cfg['name']}: ~{cfg['estimated_tokens']} tokens")
    
    total_samples = 0
    start_time = time.time()
    
    for config in DATASETS_CONFIG:
        if args.datasets and config["name"] not in args.datasets:
            continue
        
        count = download_dataset(config, output_dir, args.max_samples)
        total_samples += count
    
    # Combinar todos los datasets
    merged = merge_datasets(output_dir)
    
    elapsed = time.time() - start_time
    print(f"\n✅ Descarga completada en {elapsed/60:.1f} minutos")
    print(f"   Total: {total_samples:,} ejemplos")
    print(f"   Archivo: {merged}")
    print(f"\n💡 Para entrenar con estos datos:")
    print(f"   python scripts/train.py --data {merged}")


if __name__ == "__main__":
    main()
