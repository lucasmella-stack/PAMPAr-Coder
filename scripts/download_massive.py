# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
PAMPAr-Coder - Descarga MASIVA de código de alta calidad.

Descarga de múltiples fuentes:
1. CodeSearchNet (funciones con docstrings)
2. Code Contests (problemas de programación competitiva)
3. Repos populares de GitHub (clonado directo)
4. Código local del usuario

Objetivo: Crear un dataset de millones de líneas de código
para entrenar PAMPAr-Coder seriamente.
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from typing import List, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

# Repos de Python de alta calidad para clonar
PYTHON_REPOS = [
    # Librerías fundamentales
    'https://github.com/python/cpython',
    'https://github.com/pallets/flask',
    'https://github.com/django/django',
    'https://github.com/fastapi/fastapi',
    'https://github.com/psf/requests',
    'https://github.com/numpy/numpy',
    'https://github.com/pandas-dev/pandas',
    'https://github.com/scikit-learn/scikit-learn',
    'https://github.com/pytorch/pytorch',
    'https://github.com/tensorflow/tensorflow',
    
    # Herramientas y utilidades
    'https://github.com/psf/black',
    'https://github.com/PyCQA/pylint',
    'https://github.com/python/mypy',
    'https://github.com/pytest-dev/pytest',
    'https://github.com/tqdm/tqdm',
    'https://github.com/pallets/click',
    'https://github.com/tiangolo/typer',
    
    # Async y networking
    'https://github.com/aio-libs/aiohttp',
    'https://github.com/encode/httpx',
    'https://github.com/encode/starlette',
    
    # Data science
    'https://github.com/matplotlib/matplotlib',
    'https://github.com/plotly/plotly.py',
    'https://github.com/huggingface/transformers',
    'https://github.com/langchain-ai/langchain',
]

# Repos de JavaScript/TypeScript
JS_REPOS = [
    'https://github.com/facebook/react',
    'https://github.com/vuejs/vue',
    'https://github.com/angular/angular',
    'https://github.com/vercel/next.js',
    'https://github.com/nodejs/node',
    'https://github.com/expressjs/express',
    'https://github.com/nestjs/nest',
    'https://github.com/microsoft/TypeScript',
    'https://github.com/denoland/deno',
]

# Repos de Rust (buen código estructurado)
RUST_REPOS = [
    'https://github.com/rust-lang/rust',
    'https://github.com/tokio-rs/tokio',
    'https://github.com/serde-rs/serde',
    'https://github.com/rayon-rs/rayon',
    'https://github.com/BurntSushi/ripgrep',
]


# =============================================================================
# FUNCIONES
# =============================================================================

def clone_repo(url: str, repos_dir: Path, shallow: bool = True) -> bool:
    """Clona un repositorio."""
    name = url.split('/')[-1].replace('.git', '')
    target = repos_dir / name
    
    if target.exists():
        print(f"   ⏭️ {name} ya existe")
        return True
    
    try:
        cmd = ['git', 'clone', '--depth=1' if shallow else '', url, str(target)]
        cmd = [c for c in cmd if c]  # Eliminar vacíos
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            print(f"   ✅ {name}")
            return True
        else:
            print(f"   ❌ {name}: {result.stderr[:100]}")
            return False
    except Exception as e:
        print(f"   ❌ {name}: {str(e)[:50]}")
        return False


def extract_code_from_repo(repo_path: Path, extensions: Set[str], output_file, 
                           min_lines: int = 10, max_lines: int = 500) -> int:
    """Extrae código de un repositorio."""
    count = 0
    
    for root, dirs, files in os.walk(repo_path):
        # Skip directorios no deseados
        dirs[:] = [d for d in dirs if d not in {
            '.git', 'node_modules', '__pycache__', 'venv', 'env',
            '.tox', 'build', 'dist', 'egg-info', 'test', 'tests',
            'docs', 'examples', 'benchmarks'
        }]
        
        for file in files:
            ext = Path(file).suffix.lower()
            if ext not in extensions:
                continue
            
            file_path = Path(root) / file
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                lines = content.count('\n')
                if lines < min_lines or lines > max_lines:
                    continue
                
                # Verificar que no sea binario/generado
                if '\x00' in content or 'Generated' in content[:100]:
                    continue
                
                # Guardar
                json.dump({
                    'code': content,
                    'language': ext[1:],  # Sin el punto
                    'repo': repo_path.name,
                    'file': file
                }, output_file)
                output_file.write('\n')
                count += 1
                
            except Exception as e:
                continue
    
    return count


def download_from_huggingface(output_file, max_samples: int = 100000) -> int:
    """Descarga código de datasets de HuggingFace."""
    from datasets import load_dataset
    
    count = 0
    
    # CodeSearchNet Python
    print("\n📥 Descargando CodeSearchNet Python...")
    try:
        ds = load_dataset('Nan-Do/code-search-net-python', split='train', streaming=True)
        
        for sample in ds:
            if count >= max_samples:
                break
            
            code = sample.get('code', '')
            if len(code) < 50 or len(code) > 10000:
                continue
            
            json.dump({
                'code': code,
                'language': 'python',
                'source': 'codesearchnet'
            }, output_file)
            output_file.write('\n')
            count += 1
            
            if count % 10000 == 0:
                print(f"   {count:,} samples...")
        
        print(f"   ✅ {count:,} samples de CodeSearchNet")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Code Contests
    print("\n📥 Descargando Code Contests...")
    try:
        ds = load_dataset('deepmind/code_contests', split='train', streaming=True)
        cc_count = 0
        
        for sample in ds:
            if count >= max_samples:
                break
            
            # Soluciones en Python
            solutions = sample.get('solutions', {})
            if isinstance(solutions, dict):
                for lang, codes in solutions.items():
                    if 'python' in lang.lower():
                        for code in (codes if isinstance(codes, list) else [codes]):
                            if isinstance(code, str) and 50 < len(code) < 10000:
                                json.dump({
                                    'code': code,
                                    'language': 'python',
                                    'source': 'code_contests',
                                    'problem': sample.get('name', '')[:100]
                                }, output_file)
                                output_file.write('\n')
                                count += 1
                                cc_count += 1
            
            if cc_count % 1000 == 0 and cc_count > 0:
                print(f"   {cc_count:,} soluciones...")
        
        print(f"   ✅ {cc_count:,} samples de Code Contests")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    return count


def main():
    parser = argparse.ArgumentParser(description='Descargar datos de código masivamente')
    parser.add_argument('--output', type=str, default='data/code/train_massive.jsonl')
    parser.add_argument('--repos-dir', type=str, default='data/repos')
    parser.add_argument('--max-hf-samples', type=int, default=100000,
                        help='Máximo samples de HuggingFace')
    parser.add_argument('--clone-repos', action='store_true',
                        help='Clonar repos de GitHub')
    parser.add_argument('--languages', type=str, default='python,javascript,rust',
                        help='Lenguajes a descargar')
    args = parser.parse_args()
    
    print("=" * 70)
    print("🐸 PAMPAr-Coder - Descarga MASIVA de Código")
    print("=" * 70)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    repos_dir = Path(args.repos_dir)
    repos_dir.mkdir(parents=True, exist_ok=True)
    
    languages = set(args.languages.split(','))
    
    total_samples = 0
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # 1. Descargar de HuggingFace
        print("\n" + "=" * 50)
        print("📦 FASE 1: Datasets de HuggingFace")
        print("=" * 50)
        
        hf_count = download_from_huggingface(f, args.max_hf_samples)
        total_samples += hf_count
        
        # 2. Clonar repos
        if args.clone_repos:
            print("\n" + "=" * 50)
            print("📦 FASE 2: Clonando Repos de GitHub")
            print("=" * 50)
            
            all_repos = []
            if 'python' in languages:
                all_repos.extend(PYTHON_REPOS)
            if 'javascript' in languages or 'typescript' in languages:
                all_repos.extend(JS_REPOS)
            if 'rust' in languages:
                all_repos.extend(RUST_REPOS)
            
            print(f"\n📥 Clonando {len(all_repos)} repositorios...")
            for url in all_repos:
                clone_repo(url, repos_dir)
            
            # 3. Extraer código de repos
            print("\n" + "=" * 50)
            print("📦 FASE 3: Extrayendo Código de Repos")
            print("=" * 50)
            
            extensions = set()
            if 'python' in languages:
                extensions.add('.py')
            if 'javascript' in languages:
                extensions.update(['.js', '.jsx'])
            if 'typescript' in languages:
                extensions.update(['.ts', '.tsx'])
            if 'rust' in languages:
                extensions.add('.rs')
            
            for repo_path in repos_dir.iterdir():
                if repo_path.is_dir():
                    count = extract_code_from_repo(repo_path, extensions, f)
                    total_samples += count
                    print(f"   {repo_path.name}: {count:,} archivos")
    
    # Resumen
    print("\n" + "=" * 70)
    print("✅ DESCARGA COMPLETADA")
    print("=" * 70)
    print(f"   Total samples: {total_samples:,}")
    print(f"   Archivo: {output_path}")
    
    # Tamaño del archivo
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"   Tamaño: {size_mb:.1f} MB")
    
    # Verificar
    print("\n📊 Verificando dataset...")
    with open(output_path, 'r') as f:
        line_count = sum(1 for _ in f)
    print(f"   Líneas: {line_count:,}")


if __name__ == '__main__':
    main()
