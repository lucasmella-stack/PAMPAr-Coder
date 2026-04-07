#!/usr/bin/env python3
# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Descarga y normaliza los mejores datasets de código 100% open source.

Todos los datos provienen de fuentes con licencias permisivas:
- Generados por modelos abiertos (StarCoder2)
- Extraídos de código real con licencias MIT/Apache/BSD
- Ninguna dependencia de modelos propietarios

Datasets incluidos:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. bigcode/self-oss-instruct-sc2-exec-filter-50k   (ODC-BY)
   50K instrucciones Python generadas por StarCoder2 y
   validadas por ejecución real. El más limpio disponible.
   Usado por SmolLM (HuggingFace) — calidad verificada.

2. bigcode/commitpackft — Python subset              (MIT/Apache/etc.)
   ~56K commits reales de GitHub que explican cambios de código.
   "Antes → Después" con descripción natural. Único dataset
   que enseña razonamiento de refactor.

3. HuggingFaceFW/smollm-corpus — Python subset       (ODC-BY)
   Código Python educativo de alta calidad extraído de The Stack
   por clasificador entrenado en datos humanos. Ideal para
   aprender patrones idiomáticos de Python.

4. codeparrot/apps                                    (MIT)
   10K+ problemas de competencia algorítmica con soluciones
   verificadas. Enseña razonamiento paso a paso.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Uso:
    python scripts/download_open_datasets.py
    python scripts/download_open_datasets.py --output data/curated --max-per-dataset 20000
    python scripts/download_open_datasets.py --datasets sc2 commits --no-pareto
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Generator, Dict, Optional

# Agregar path del proyecto
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from datasets import load_dataset
except ImportError:
    print("❌ Instalar: pip install datasets")
    sys.exit(1)


# =============================================================================
# NORMALIZADOR — todos los datasets al formato PAMPAr uniforme
# =============================================================================

def _fmt(instruction: str, response: str, context: str = "") -> str:
    """
    Formato estándar PAMPAr:
        ### Instruction:\\n<instruccion>\\n\\n### Response:\\n<respuesta>

    Si hay contexto (ej: código original antes del refactor):
        ### Context:\\n<contexto>\\n\\n### Instruction:\\n...
    """
    parts = []
    if context.strip():
        parts.append(f"### Context:\n{context.strip()}")
    parts.append(f"### Instruction:\n{instruction.strip()}")
    parts.append(f"### Response:\n{response.strip()}")
    return "\n\n".join(parts)


# =============================================================================
# DATASET 1: bigcode/self-oss-instruct-sc2-exec-filter-50k
# Campos: instruction, response, seed, concepts
# Licencia: ODC-BY — generado por StarCoder2 (modelo abierto)
# =============================================================================

def stream_sc2_instruct(max_samples: int) -> Generator[Dict, None, None]:
    """StarCoder2 self-instruct — 50K Python validados por ejecución."""
    print("📥 [1/4] bigcode/self-oss-instruct-sc2-exec-filter-50k...")
    ds = load_dataset(
        "bigcode/self-oss-instruct-sc2-exec-filter-50k",
        split="train",
        streaming=True,
    )
    count = 0
    for row in ds:
        if count >= max_samples:
            break
        instruction = row.get("instruction", "").strip()
        response = row.get("response", "").strip()
        if not instruction or not response:
            continue
        yield {
            "text": _fmt(instruction, response),
            "source": "bigcode/sc2-exec-instruct",
            "license": "odc-by",
            "lang": "python",
        }
        count += 1
    print(f"   ✅ {count:,} samples normalizados")


# =============================================================================
# DATASET 2: bigcode/commitpackft — Python subset
# Campos: subject/message (instrucción), old_contents (contexto),
#         new_contents (respuesta), license, lang
# Licencia: MIT/Apache/BSD — código real de GitHub
# =============================================================================

def stream_commitpackft(max_samples: int) -> Generator[Dict, None, None]:
    """Commits reales de GitHub explicando refactors de código."""
    print("📥 [2/4] bigcode/commitpackft (Python)...")
    ds = load_dataset(
        "bigcode/commitpackft",
        "python",
        split="train",
        streaming=True,
    )
    count = 0
    for row in ds:
        if count >= max_samples:
            break
        instruction = (row.get("subject") or row.get("message", "")).strip()
        old_code = row.get("old_contents", "").strip()
        new_code = row.get("new_contents", "").strip()

        # Filtrar commits triviales
        if not instruction or not new_code:
            continue
        if len(instruction) < 10:
            continue
        if old_code == new_code:
            continue

        # Formatear como refactor con contexto
        instruction_full = f"Refactor the following Python code: {instruction}"
        context = f"```python\n{old_code}\n```" if old_code else ""
        response = f"```python\n{new_code}\n```"

        yield {
            "text": _fmt(instruction_full, response, context),
            "source": "bigcode/commitpackft",
            "license": row.get("license", "unknown"),
            "lang": "python",
        }
        count += 1
    print(f"   ✅ {count:,} samples normalizados")


# =============================================================================
# DATASET 3: HuggingFaceFW/smollm-corpus — Python subset
# Campos: text (código Python educativo, sin formato instrucción)
# Licencia: ODC-BY — extraído de The Stack por clasificador abierto
# Nota: formato de preentrenamiento (completion), no instrucción
# =============================================================================

def stream_smollm_python(max_samples: int) -> Generator[Dict, None, None]:
    """Código Python educativo de alta calidad (SmolLM corpus)."""
    print("📥 [3/4] HuggingFaceFW/smollm-corpus (Python-Edu)...")
    ds = load_dataset(
        "HuggingFaceFW/smollm-corpus",
        "python-edu",
        split="train",
        streaming=True,
    )
    count = 0
    for row in ds:
        if count >= max_samples:
            break
        text = row.get("text", "").strip()
        if not text or len(text) < 100:
            continue

        # Convertir código standalone → formato instrucción
        # Extraer primera función/clase como "qué hace este código"
        lines = text.split("\n")
        first_def = next(
            (l for l in lines if l.startswith("def ") or l.startswith("class ")),
            None,
        )
        if first_def:
            instruction = f"Implement and explain the following Python code:\n```python\n{first_def}\n```"
        else:
            instruction = "Read and understand the following Python code:"

        yield {
            "text": _fmt(instruction, f"```python\n{text[:3000]}\n```"),
            "source": "HuggingFaceFW/smollm-corpus/python-edu",
            "license": "odc-by",
            "lang": "python",
        }
        count += 1
    print(f"   ✅ {count:,} samples normalizados")


# =============================================================================
# DATASET 4: codeparrot/apps
# Campos: question, solutions (JSON list of solution strings)
# Licencia: MIT — problemas de competencia algorítmica
# =============================================================================

def stream_apps(max_samples: int) -> Generator[Dict, None, None]:
    """APPS: problemas de competencia algorítmica con soluciones reales."""
    print("📥 [4/4] codeparrot/apps...")
    ds = load_dataset(
        "codeparrot/apps",
        split="train",
        streaming=True,
    )
    count = 0
    for row in ds:
        if count >= max_samples:
            break
        question = row.get("question", "").strip()
        solutions_raw = row.get("solutions", "")

        if not question or not solutions_raw:
            continue

        # solutions es un JSON string con lista de soluciones
        try:
            solutions = json.loads(solutions_raw)
            if not solutions:
                continue
            # Tomar la primera solución (suelen estar ordenadas por calidad)
            solution = solutions[0].strip()
        except (json.JSONDecodeError, IndexError, TypeError):
            continue

        if len(solution) < 50:
            continue

        yield {
            "text": _fmt(question, f"```python\n{solution}\n```"),
            "source": "codeparrot/apps",
            "license": "mit",
            "lang": "python",
        }
        count += 1
    print(f"   ✅ {count:,} samples normalizados")


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

DATASETS = {
    "sc2": stream_sc2_instruct,
    "commits": stream_commitpackft,
    "smollm": stream_smollm_python,
    "apps": stream_apps,
}


def main():
    parser = argparse.ArgumentParser(
        description="Descarga datasets open source para PAMPAr",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output", type=str, default="data/curated",
        help="Directorio de salida (default: data/curated)",
    )
    parser.add_argument(
        "--max-per-dataset", type=int, default=50000,
        help="Máximo de samples por dataset (default: 50000)",
    )
    parser.add_argument(
        "--datasets", nargs="+",
        choices=list(DATASETS.keys()),
        default=list(DATASETS.keys()),
        help="Datasets a descargar (default: todos)",
    )
    parser.add_argument(
        "--no-pareto", action="store_true",
        help="No aplicar filtro Pareto (guardar todo)",
    )
    parser.add_argument(
        "--pareto-ratio", type=float, default=0.5,
        help="Ratio Pareto a retener (default: 0.5 = top 50%% de cada dataset)",
    )
    parser.add_argument(
        "--min-score", type=float, default=2.0,
        help="Score mínimo de calidad (default: 2.0)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Importar scorer si se usa Pareto
    scorer = None
    if not args.no_pareto:
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from curar_datos import score_sample
            scorer = score_sample
            print("🔍 Filtro Pareto activado\n")
        except ImportError:
            print("⚠️  curar_datos.py no encontrado, continuando sin filtro\n")

    print(f"{'='*60}")
    print(f"🧠 PAMPAr — Descarga de Datasets Open Source")
    print(f"{'='*60}")
    print(f"   Datasets: {', '.join(args.datasets)}")
    print(f"   Max por dataset: {args.max_per_dataset:,}")
    print(f"   Salida: {output_dir}/")
    if scorer:
        print(f"   Pareto ratio: {args.pareto_ratio} | min_score: {args.min_score}")
    print()

    total_global = 0
    stats_global = {}

    for ds_name in args.datasets:
        stream_fn = DATASETS[ds_name]
        output_file = output_dir / f"{ds_name}.jsonl"

        # Recolectar todos los samples del dataset
        all_samples = list(stream_fn(args.max_per_dataset))

        if not all_samples:
            print(f"   ⚠️  {ds_name}: sin samples\n")
            continue

        # Aplicar filtro Pareto
        if scorer:
            scored = []
            for s in all_samples:
                score, _ = scorer(s["text"])
                if score >= args.min_score:
                    scored.append((score, s))

            # Ordenar por score y retener top ratio
            scored.sort(key=lambda x: x[0], reverse=True)
            n_keep = max(1, int(len(scored) * args.pareto_ratio))
            final_samples = [s for _, s in scored[:n_keep]]

            print(f"   📊 {ds_name}: {len(all_samples):,} → "
                  f"{len(final_samples):,} tras Pareto "
                  f"(descartados: {len(all_samples) - len(final_samples):,})")
        else:
            final_samples = all_samples

        # Escritura
        with open(output_file, "w", encoding="utf-8") as f:
            for sample in final_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        size_mb = output_file.stat().st_size / 1024 / 1024
        print(f"   💾 {output_file.name}: {len(final_samples):,} samples ({size_mb:.1f} MB)\n")

        stats_global[ds_name] = len(final_samples)
        total_global += len(final_samples)

    # Combinar en un único archivo curado
    combined_path = output_dir / "pampar_curated.jsonl"
    print(f"{'='*60}")
    print(f"🔗 Combinando en {combined_path.name}...")

    with open(combined_path, "w", encoding="utf-8") as out:
        for ds_name in args.datasets:
            individual = output_dir / f"{ds_name}.jsonl"
            if individual.exists():
                with open(individual, "r", encoding="utf-8") as f:
                    for line in f:
                        out.write(line)

    combined_size = combined_path.stat().st_size / 1024 / 1024
    print(f"\n✅ Dataset final: {combined_path}")
    print(f"   Total samples: {total_global:,}")
    print(f"   Tamaño: {combined_size:.1f} MB")
    print(f"\n   Por dataset:")
    for ds_name, count in stats_global.items():
        print(f"   • {ds_name:<12}: {count:>8,} samples")

    print(f"\n{'='*60}")
    print(f"Siguiente paso:")
    print(f"  python scripts/curar_datos.py \\")
    print(f"    --input {output_dir}/ \\")
    print(f"    --output data/final/ \\")
    print(f"    --ratio 0.2 --recursivo")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
