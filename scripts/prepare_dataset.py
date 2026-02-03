# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Descargar y preparar dataset de código para entrenar PAMPAr-Coder.

Datasets soportados:
- codeparrot/github-code-clean (pequeño, rápido)
- bigcode/the-stack (grande, completo)
- Local files
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Iterator
import json

try:
    from datasets import load_dataset, Dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("⚠️ Instala: pip install datasets")


# =============================================================================
# Configuración
# =============================================================================

SUPPORTED_LANGUAGES = {
    'python': ['.py'],
    'javascript': ['.js', '.jsx'],
    'typescript': ['.ts', '.tsx'],
    'rust': ['.rs'],
    'go': ['.go'],
    'java': ['.java'],
    'c': ['.c', '.h'],
    'cpp': ['.cpp', '.hpp', '.cc', '.cxx'],
}

DEFAULT_LANGUAGES = ['python', 'javascript', 'typescript', 'rust']


# =============================================================================
# Dataset Sources
# =============================================================================

def download_codeparrot(
    output_dir: Path,
    languages: list = None,
    max_samples: int = 100000,
    max_size_mb: int = 500
) -> Path:
    """
    Descarga codeparrot/github-code-clean.
    Relativamente pequeño y rápido de descargar.
    """
    print("📥 Descargando codeparrot/github-code-clean...")
    languages = languages or DEFAULT_LANGUAGES
    
    ds = load_dataset(
        "codeparrot/github-code-clean",
        split="train",
        streaming=True,
        trust_remote_code=True
    )
    
    output_file = output_dir / "train.jsonl"
    stats = {lang: 0 for lang in languages}
    total_bytes = 0
    max_bytes = max_size_mb * 1024 * 1024
    
    # Mapear nombres de lenguaje
    lang_map = {
        'python': 'Python',
        'javascript': 'JavaScript', 
        'typescript': 'TypeScript',
        'rust': 'Rust',
        'go': 'Go',
        'java': 'Java',
        'c': 'C',
        'cpp': 'C++',
    }
    target_langs = {lang_map.get(l, l) for l in languages}
    
    with open(output_file, 'w', encoding='utf-8') as f:
        count = 0
        for sample in ds:
            lang = sample.get('language', '')
            if lang in target_langs:
                code = sample.get('code', '')
                
                # Filtros básicos
                if len(code) < 50 or len(code) > 50000:
                    continue
                
                # Guardar
                record = {
                    'text': code,
                    'language': lang.lower(),
                    'path': sample.get('path', ''),
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                
                stats[lang.lower()] = stats.get(lang.lower(), 0) + 1
                total_bytes += len(code.encode('utf-8'))
                count += 1
                
                if count % 10000 == 0:
                    mb = total_bytes / (1024 * 1024)
                    print(f"   {count:,} samples, {mb:.1f} MB")
                
                if count >= max_samples or total_bytes >= max_bytes:
                    break
    
    print(f"\n✅ Dataset guardado: {output_file}")
    print(f"   Total: {count:,} samples, {total_bytes / (1024*1024):.1f} MB")
    print(f"   Por lenguaje: {stats}")
    
    return output_file


def download_the_stack(
    output_dir: Path,
    languages: list = None,
    max_samples: int = 50000,
    max_size_mb: int = 200
) -> Path:
    """
    Descarga bigcode/the-stack (subconjunto).
    Más grande y diverso pero más lento.
    """
    print("📥 Descargando bigcode/the-stack...")
    languages = languages or DEFAULT_LANGUAGES
    
    output_file = output_dir / "train.jsonl"
    total_bytes = 0
    max_bytes = max_size_mb * 1024 * 1024
    count = 0
    stats = {lang: 0 for lang in languages}
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for lang in languages:
            print(f"   Descargando {lang}...")
            
            try:
                ds = load_dataset(
                    "bigcode/the-stack",
                    data_dir=f"data/{lang}",
                    split="train",
                    streaming=True,
                    trust_remote_code=True
                )
                
                lang_count = 0
                for sample in ds:
                    code = sample.get('content', '')
                    
                    if len(code) < 50 or len(code) > 50000:
                        continue
                    
                    record = {
                        'text': code,
                        'language': lang,
                        'path': sample.get('path', ''),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    
                    lang_count += 1
                    total_bytes += len(code.encode('utf-8'))
                    count += 1
                    
                    if lang_count >= max_samples // len(languages):
                        break
                    if total_bytes >= max_bytes:
                        break
                
                stats[lang] = lang_count
                
            except Exception as e:
                print(f"   ⚠️ Error con {lang}: {e}")
            
            if total_bytes >= max_bytes:
                break
    
    print(f"\n✅ Dataset guardado: {output_file}")
    print(f"   Total: {count:,} samples, {total_bytes / (1024*1024):.1f} MB")
    print(f"   Por lenguaje: {stats}")
    
    return output_file


def create_local_dataset(
    source_dir: Path,
    output_dir: Path,
    languages: list = None,
    max_size_mb: int = 100
) -> Path:
    """
    Crea dataset desde archivos locales.
    """
    print(f"📁 Creando dataset desde {source_dir}...")
    languages = languages or DEFAULT_LANGUAGES
    
    output_file = output_dir / "train.jsonl"
    
    # Extensiones a buscar
    extensions = []
    for lang in languages:
        extensions.extend(SUPPORTED_LANGUAGES.get(lang, []))
    
    total_bytes = 0
    max_bytes = max_size_mb * 1024 * 1024
    count = 0
    stats = {lang: 0 for lang in languages}
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for root, dirs, files in os.walk(source_dir):
            # Ignorar carpetas comunes
            dirs[:] = [d for d in dirs if d not in {
                'node_modules', '.git', '__pycache__', 'venv', 
                '.venv', 'build', 'dist', 'target'
            }]
            
            for file in files:
                ext = Path(file).suffix.lower()
                if ext not in extensions:
                    continue
                
                filepath = Path(root) / file
                try:
                    code = filepath.read_text(encoding='utf-8', errors='ignore')
                except:
                    continue
                
                if len(code) < 50 or len(code) > 50000:
                    continue
                
                # Detectar lenguaje
                lang = None
                for l, exts in SUPPORTED_LANGUAGES.items():
                    if ext in exts:
                        lang = l
                        break
                
                if lang and lang in languages:
                    record = {
                        'text': code,
                        'language': lang,
                        'path': str(filepath.relative_to(source_dir)),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    
                    stats[lang] = stats.get(lang, 0) + 1
                    total_bytes += len(code.encode('utf-8'))
                    count += 1
                    
                    if total_bytes >= max_bytes:
                        break
            
            if total_bytes >= max_bytes:
                break
    
    print(f"\n✅ Dataset guardado: {output_file}")
    print(f"   Total: {count:,} samples, {total_bytes / (1024*1024):.1f} MB")
    print(f"   Por lenguaje: {stats}")
    
    return output_file


# =============================================================================
# Dataset para PyTorch
# =============================================================================

def create_train_val_split(dataset_file: Path, val_ratio: float = 0.05):
    """Divide dataset en train/validation."""
    print(f"\n📊 Dividiendo dataset (val_ratio={val_ratio})...")
    
    # Leer todos los samples
    samples = []
    with open(dataset_file, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(line)
    
    # Shuffle
    import random
    random.seed(42)
    random.shuffle(samples)
    
    # Split
    n_val = int(len(samples) * val_ratio)
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]
    
    # Guardar
    train_file = dataset_file.parent / "train.jsonl"
    val_file = dataset_file.parent / "val.jsonl"
    
    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(train_samples)
    
    with open(val_file, 'w', encoding='utf-8') as f:
        f.writelines(val_samples)
    
    print(f"   Train: {len(train_samples):,} samples → {train_file}")
    print(f"   Val:   {len(val_samples):,} samples → {val_file}")
    
    return train_file, val_file


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Preparar dataset de código')
    parser.add_argument('--source', type=str, default='codeparrot',
                        choices=['codeparrot', 'the-stack', 'local'],
                        help='Fuente del dataset')
    parser.add_argument('--local-dir', type=str, default=None,
                        help='Directorio local con código (si source=local)')
    parser.add_argument('--output-dir', type=str, default='data/code',
                        help='Directorio de salida')
    parser.add_argument('--languages', type=str, nargs='+',
                        default=['python', 'javascript', 'typescript', 'rust'],
                        help='Lenguajes a incluir')
    parser.add_argument('--max-samples', type=int, default=100000,
                        help='Máximo número de samples')
    parser.add_argument('--max-size-mb', type=int, default=200,
                        help='Tamaño máximo en MB')
    args = parser.parse_args()
    
    print("=" * 70)
    print("📚 PAMPAr-Coder - Preparar Dataset")
    print("=" * 70)
    print(f"   Fuente:    {args.source}")
    print(f"   Lenguajes: {args.languages}")
    print(f"   Max size:  {args.max_size_mb} MB")
    print()
    
    # Crear directorio
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Descargar/crear dataset
    if args.source == 'codeparrot':
        if not HAS_DATASETS:
            print("❌ Necesitas: pip install datasets")
            return
        dataset_file = download_codeparrot(
            output_dir, args.languages, args.max_samples, args.max_size_mb
        )
    elif args.source == 'the-stack':
        if not HAS_DATASETS:
            print("❌ Necesitas: pip install datasets")
            return
        dataset_file = download_the_stack(
            output_dir, args.languages, args.max_samples, args.max_size_mb
        )
    elif args.source == 'local':
        if not args.local_dir:
            print("❌ Especifica --local-dir")
            return
        dataset_file = create_local_dataset(
            Path(args.local_dir), output_dir, args.languages, args.max_size_mb
        )
    
    # Split train/val
    create_train_val_split(dataset_file)
    
    print("\n" + "=" * 70)
    print("✅ Dataset listo!")
    print(f"   Directorio: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
