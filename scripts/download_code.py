# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Descargador de código REAL para entrenar PAMPAr-Coder.

Fuentes:
1. GitHub repos populares (descarga directa sin API key)
2. Código local del usuario
3. The Stack v2 (si está disponible)

Este script es robusto y no depende de datasets deprecados.
"""

import os
import sys
import json
import time
import zipfile
import tempfile
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Generator
from urllib.request import urlretrieve, urlopen
from urllib.error import URLError
from concurrent.futures import ThreadPoolExecutor, as_completed

# =============================================================================
# Configuración
# =============================================================================

# Repos populares de GitHub para descargar (owner/repo, branch)
GITHUB_REPOS = [
    # Python
    ("python/cpython", "main", "python"),
    ("pallets/flask", "main", "python"),
    ("django/django", "main", "python"),
    ("psf/requests", "main", "python"),
    ("numpy/numpy", "main", "python"),
    ("pandas-dev/pandas", "main", "python"),
    ("pytorch/pytorch", "main", "python"),
    ("huggingface/transformers", "main", "python"),
    ("fastapi/fastapi", "main", "python"),
    ("tiangolo/sqlmodel", "main", "python"),
    
    # JavaScript/TypeScript
    ("microsoft/vscode", "main", "typescript"),
    ("vercel/next.js", "canary", "javascript"),
    ("facebook/react", "main", "javascript"),
    ("vuejs/vue", "main", "javascript"),
    ("sveltejs/svelte", "main", "javascript"),
    ("expressjs/express", "master", "javascript"),
    ("nestjs/nest", "master", "typescript"),
    
    # Rust
    ("rust-lang/rust", "master", "rust"),
    ("denoland/deno", "main", "rust"),
    ("tauri-apps/tauri", "dev", "rust"),
    ("tokio-rs/tokio", "master", "rust"),
    
    # Go
    ("golang/go", "master", "go"),
    ("kubernetes/kubernetes", "master", "go"),
    ("docker/cli", "master", "go"),
]

# Extensiones por lenguaje
EXTENSIONS = {
    'python': ['.py'],
    'javascript': ['.js', '.jsx', '.mjs'],
    'typescript': ['.ts', '.tsx'],
    'rust': ['.rs'],
    'go': ['.go'],
    'java': ['.java'],
    'c': ['.c', '.h'],
    'cpp': ['.cpp', '.hpp', '.cc', '.cxx', '.h'],
}

# Carpetas a ignorar
IGNORE_DIRS = {
    'node_modules', '__pycache__', '.git', 'venv', 'env',
    'build', 'dist', 'target', '.idea', '.vscode',
    'test', 'tests', 'testing', '__tests__',
    'vendor', 'third_party', 'external',
    'docs', 'doc', 'documentation',
    'examples', 'example', 'samples',
}


# =============================================================================
# Descargadores
# =============================================================================

def download_github_repo(
    owner: str,
    repo: str,
    branch: str = "main",
    temp_dir: Path = None
) -> Optional[Path]:
    """Descarga un repo de GitHub como ZIP."""
    url = f"https://github.com/{owner}/{repo}/archive/refs/heads/{branch}.zip"
    
    if temp_dir is None:
        temp_dir = Path(tempfile.mkdtemp())
    
    zip_path = temp_dir / f"{repo}.zip"
    
    try:
        print(f"   📥 Descargando {owner}/{repo}...", end=" ", flush=True)
        urlretrieve(url, zip_path)
        
        # Extraer
        extract_dir = temp_dir / repo
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(temp_dir)
        
        # El zip extrae a {repo}-{branch}
        extracted = temp_dir / f"{repo}-{branch}"
        if extracted.exists():
            shutil.move(str(extracted), str(extract_dir))
        
        os.remove(zip_path)
        print("✓")
        return extract_dir
        
    except Exception as e:
        print(f"✗ ({e})")
        return None


def extract_code_from_dir(
    source_dir: Path,
    languages: List[str],
    min_lines: int = 10,
    max_lines: int = 1000,
    max_files: int = 500
) -> Generator[Dict, None, None]:
    """Extrae archivos de código de un directorio."""
    
    # Extensiones a buscar
    target_exts = set()
    for lang in languages:
        target_exts.update(EXTENSIONS.get(lang, []))
    
    file_count = 0
    
    for root, dirs, files in os.walk(source_dir):
        # Filtrar directorios a ignorar
        dirs[:] = [d for d in dirs if d.lower() not in IGNORE_DIRS]
        
        for filename in files:
            if file_count >= max_files:
                return
            
            ext = Path(filename).suffix.lower()
            if ext not in target_exts:
                continue
            
            filepath = Path(root) / filename
            
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Filtros
                lines = content.count('\n')
                if lines < min_lines or lines > max_lines:
                    continue
                
                # Detectar lenguaje
                lang = None
                for l, exts in EXTENSIONS.items():
                    if ext in exts:
                        lang = l
                        break
                
                if lang is None:
                    continue
                
                file_count += 1
                
                yield {
                    'text': content,
                    'language': lang,
                    'filename': filename,
                    'lines': lines,
                    'source': str(source_dir.name)
                }
                
            except Exception:
                continue


def collect_local_code(
    directories: List[Path],
    languages: List[str]
) -> Generator[Dict, None, None]:
    """Recolecta código de directorios locales."""
    
    for directory in directories:
        if not directory.exists():
            print(f"   ⚠️ No existe: {directory}")
            continue
        
        print(f"   📂 Escaneando {directory.name}...")
        yield from extract_code_from_dir(directory, languages)


def download_github_code(
    repos: List[tuple],
    languages: List[str],
    output_dir: Path,
    max_repos: int = 10
) -> int:
    """Descarga código de repos de GitHub."""
    
    temp_base = Path(tempfile.mkdtemp())
    total_samples = 0
    output_file = output_dir / "github_code.jsonl"
    
    # Filtrar repos por lenguajes solicitados
    filtered_repos = [r for r in repos if r[2] in languages][:max_repos]
    
    print(f"\n🐙 Descargando {len(filtered_repos)} repos de GitHub...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for owner_repo, branch, lang in filtered_repos:
            owner, repo = owner_repo.split('/')
            
            repo_dir = download_github_repo(owner, repo, branch, temp_base)
            if repo_dir is None:
                continue
            
            # Extraer código
            count = 0
            for sample in extract_code_from_dir(repo_dir, [lang]):
                sample['repo'] = f"{owner}/{repo}"
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                count += 1
                total_samples += 1
            
            print(f"      → {count} archivos extraídos")
            
            # Limpiar
            shutil.rmtree(repo_dir, ignore_errors=True)
    
    # Limpiar temp
    shutil.rmtree(temp_base, ignore_errors=True)
    
    print(f"\n✅ GitHub: {total_samples:,} samples guardados en {output_file.name}")
    return total_samples


# =============================================================================
# Dataset combinado
# =============================================================================

def create_combined_dataset(
    output_dir: Path,
    languages: List[str] = None,
    include_github: bool = True,
    include_local: bool = True,
    local_dirs: List[Path] = None,
    max_github_repos: int = 10,
    max_samples: int = 100000
) -> Path:
    """Crea dataset combinado de múltiples fuentes."""
    
    languages = languages or ['python', 'javascript', 'typescript', 'rust']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    final_file = output_dir / "train.jsonl"
    total = 0
    
    print("=" * 60)
    print("🚀 PAMPAr-Coder Dataset Builder")
    print("=" * 60)
    print(f"Lenguajes: {', '.join(languages)}")
    print(f"Output: {output_dir}")
    print()
    
    with open(final_file, 'w', encoding='utf-8') as f:
        
        # 1. Código de GitHub
        if include_github:
            github_file = output_dir / "github_code.jsonl"
            download_github_code(GITHUB_REPOS, languages, output_dir, max_github_repos)
            
            if github_file.exists():
                print(f"\n📦 Integrando código de GitHub...")
                with open(github_file, 'r', encoding='utf-8') as gf:
                    for line in gf:
                        if total >= max_samples:
                            break
                        f.write(line)
                        total += 1
        
        # 2. Código local
        if include_local and local_dirs:
            print(f"\n📂 Integrando código local...")
            for sample in collect_local_code(local_dirs, languages):
                if total >= max_samples:
                    break
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                total += 1
    
    # Estadísticas
    print("\n" + "=" * 60)
    print("📊 Estadísticas del Dataset")
    print("=" * 60)
    
    lang_counts = {}
    with open(final_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
                lang = record.get('language', 'unknown')
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
            except:
                pass
    
    for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
        print(f"  {lang:15} {count:>8,} samples")
    
    print(f"\n  {'TOTAL':15} {total:>8,} samples")
    print(f"\n✅ Dataset guardado en: {final_file}")
    
    return final_file


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Descarga código real para entrenar PAMPAr-Coder"
    )
    
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path(__file__).parent.parent / 'data' / 'code',
        help='Directorio de salida'
    )
    
    parser.add_argument(
        '--languages', '-l',
        nargs='+',
        default=['python', 'javascript', 'typescript', 'rust'],
        help='Lenguajes a incluir'
    )
    
    parser.add_argument(
        '--max-repos', '-r',
        type=int,
        default=10,
        help='Máximo de repos de GitHub a descargar'
    )
    
    parser.add_argument(
        '--max-samples', '-s',
        type=int,
        default=50000,
        help='Máximo de samples totales'
    )
    
    parser.add_argument(
        '--local-dirs',
        nargs='+',
        type=Path,
        default=[],
        help='Directorios locales con código'
    )
    
    parser.add_argument(
        '--no-github',
        action='store_true',
        help='No descargar de GitHub'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Modo rápido: solo 3 repos, 5000 samples'
    )
    
    args = parser.parse_args()
    
    # Modo quick para testing
    if args.quick:
        args.max_repos = 3
        args.max_samples = 5000
        print("⚡ Modo rápido activado")
    
    # Crear dataset
    create_combined_dataset(
        output_dir=args.output,
        languages=args.languages,
        include_github=not args.no_github,
        include_local=bool(args.local_dirs),
        local_dirs=args.local_dirs,
        max_github_repos=args.max_repos,
        max_samples=args.max_samples
    )


if __name__ == '__main__':
    main()
