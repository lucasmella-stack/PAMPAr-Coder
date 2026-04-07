#!/usr/bin/env python3
# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Curación de datos con Principio de Pareto.

Filtra los datasets existentes para quedarse solo con el 20% más
valioso. Mide la "importancia" de cada sample usando heurísticas
que NO requieren el modelo (pre-training curation).

Criterios de importancia:
1. Complejidad del código (tokens únicos, profundidad, largo)
2. Diversidad de estructuras (loops, clases, exceptions, etc.)
3. Calidad de la respuesta (largo, presencia de docstrings)
4. No-trivialidad (filtrar "arr = [1,2,3]" y similares)

Uso:
    python scripts/curar_datos.py --input data/distillation/ --output data/curated/

    # Con ratio custom
    python scripts/curar_datos.py --ratio 0.3 --min-tokens 50
"""

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# =============================================================================
# SCORING DE CALIDAD
# =============================================================================

# Patrones que indican código complejo/interesante
_COMPLEX_PATTERNS = [
    (r"\bclass\b", 3.0, "define clase"),
    (r"\bdef\b.*\bdef\b", 4.0, "funciones anidadas"),
    (r"\btry\b.*\bexcept\b", 2.5, "manejo de errores"),
    (r"\bfor\b.*\bfor\b", 2.0, "loops anidados"),
    (r"\blambda\b", 1.5, "lambdas"),
    (r"\byield\b", 2.5, "generadores"),
    (r"\basync\b|\bawait\b", 3.0, "async"),
    (r"\bwith\b", 1.5, "context managers"),
    (r"@\w+", 2.0, "decoradores"),
    (r'"""[\s\S]+?"""', 2.0, "docstrings"),
    (r"#\s*\w+", 0.5, "comentarios"),
    (r"\bimport\b", 0.5, "imports"),
    (r"\bself\.\w+", 1.0, "atributos"),
    (r"List\[|Dict\[|Optional\[|Tuple\[", 2.0, "type hints"),
    (r"\bif\b.*\belse\b", 1.0, "condicional"),
    (r"\breturn\b", 0.5, "return"),
    (r"assert\b", 1.5, "assertions"),
    (r"raise\b", 1.5, "raise exceptions"),
]

# Patrones que indican baja calidad
_LOW_QUALITY_PATTERNS = [
    (r"^arr\s*=\s*\[", "asignación trivial"),
    (r"^print\(", "solo print"),
    (r"^x\s*=\s*\d+$", "variable trivial"),
    (r'^"[^"]*"$', "solo string"),
    (r"^\d+$", "solo número"),
]

# Keywords de Python para medir diversidad
_PYTHON_KEYWORDS = {
    "def", "class", "if", "elif", "else", "for", "while", "try",
    "except", "finally", "with", "as", "import", "from", "return",
    "yield", "raise", "pass", "break", "continue", "lambda",
    "async", "await", "global", "nonlocal", "assert", "del",
}


def score_sample(text: str) -> Tuple[float, Dict]:
    """
    Calcula un score de calidad para un sample de código.

    Returns:
        (score, detalles) donde score ∈ [0, 10+]
    """
    detalles: Dict = {}

    # Extraer la parte de respuesta/código
    code = _extract_code(text)
    if not code:
        return 0.0, {"razon": "sin código"}

    # 1. Longitud (bonus por código sustancial, penalidad por trivial)
    tokens = code.split()
    n_tokens = len(tokens)
    if n_tokens < 10:
        return 0.1, {"razon": f"muy corto ({n_tokens} tokens)"}
    length_score = min(math.log2(n_tokens) / 3, 2.0)  # Cap at 2.0
    detalles["length_score"] = round(length_score, 2)

    # 2. Diversidad de tokens
    unique_ratio = len(set(tokens)) / max(n_tokens, 1)
    diversity_score = unique_ratio * 2  # 0-2
    detalles["diversity_score"] = round(diversity_score, 2)

    # 3. Complejidad de estructuras
    complexity_score = 0.0
    patterns_found = []
    for pattern, weight, name in _COMPLEX_PATTERNS:
        matches = len(re.findall(pattern, code))
        if matches > 0:
            complexity_score += weight * min(matches, 3)
            patterns_found.append(name)
    detalles["complexity_score"] = round(complexity_score, 2)
    detalles["patterns"] = patterns_found

    # 4. Diversidad de keywords Python
    words_set = set(code.split())
    keywords_used = words_set & _PYTHON_KEYWORDS
    keyword_score = len(keywords_used) * 0.3
    detalles["keywords_used"] = len(keywords_used)

    # 5. Profundidad de indentación (indica estructura)
    lines = code.split("\n")
    max_indent = 0
    for line in lines:
        stripped = line.lstrip()
        if stripped:
            indent = len(line) - len(stripped)
            max_indent = max(max_indent, indent)
    indent_score = min(max_indent / 4, 2.0)  # Hasta 8 niveles
    detalles["max_indent"] = max_indent

    # 6. Penalización por baja calidad
    penalty = 0.0
    for pattern, reason in _LOW_QUALITY_PATTERNS:
        if re.match(pattern, code.strip()):
            penalty += 3.0
            detalles["low_quality"] = reason

    # 7. Presencia de docstring/comentarios (calidad de documentación)
    doc_score = 0.0
    if '"""' in code or "'''" in code:
        doc_score += 1.5
    if re.search(r"#\s*\w{3,}", code):
        doc_score += 0.5
    detalles["doc_score"] = round(doc_score, 2)

    # Score final
    total = (
        length_score
        + diversity_score
        + complexity_score
        + keyword_score
        + indent_score
        + doc_score
        - penalty
    )
    total = max(total, 0.0)
    detalles["total"] = round(total, 2)

    return total, detalles


def _extract_code(text: str) -> str:
    """Extrae la porción de código de un sample instruction/response."""
    # Buscar después de "### Response:", "### Solution:", "```python"
    for marker in ["### Response:", "### Solution:", "```python", "```"]:
        idx = text.find(marker)
        if idx != -1:
            code = text[idx + len(marker):]
            # Limpiar cierre de code block
            end_idx = code.find("```")
            if end_idx != -1:
                code = code[:end_idx]
            return code.strip()

    # Si no hay markers, usar todo el texto
    return text.strip()


# =============================================================================
# FILTRADO PARETO
# =============================================================================

def filtrar_pareto(
    input_path: str,
    output_path: str,
    ratio: float = 0.2,
    min_score: float = 2.0,
    min_tokens: int = 20,
    max_samples: Optional[int] = None,
) -> Dict:
    """
    Aplica filtrado Pareto a un dataset JSONL.

    1. Lee todos los samples y calcula score.
    2. Ordena por score descendente.
    3. Retiene el top `ratio` (20% por defecto).
    4. Guarda el dataset filtrado.

    Args:
        input_path: Ruta al JSONL de entrada.
        output_path: Ruta al JSONL filtrado.
        ratio: Porcentaje a retener (0.2 = 20%).
        min_score: Score mínimo absoluto.
        min_tokens: Tokens mínimos en la respuesta.
        max_samples: Máximo de samples a procesar (None = todos).

    Returns:
        Dict con estadísticas del filtrado.
    """
    scores: List[Tuple[float, str]] = []
    total = 0
    skipped_short = 0

    input_file = Path(input_path)
    if not input_file.exists():
        return {"error": f"No existe: {input_path}"}

    print(f"📖 Leyendo {input_file.name}...")

    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break

            total += 1
            if total % 100000 == 0:
                print(f"   ... {total:,} samples procesados")

            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            text = data.get("text", "")
            if not text:
                continue

            # Filtro rápido por longitud
            if len(text.split()) < min_tokens:
                skipped_short += 1
                continue

            score, _ = score_sample(text)
            if score >= min_score:
                scores.append((score, line.strip()))

    # Ordenar por score
    scores.sort(key=lambda x: x[0], reverse=True)

    # Aplicar ratio Pareto
    n_retener = max(1, int(len(scores) * ratio))
    selected = scores[:n_retener]

    # Guardar
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for _, line in selected:
            f.write(line + "\n")

    # Estadísticas
    all_scores = [s for s, _ in scores]
    stats = {
        "input": str(input_path),
        "output": str(output_path),
        "total_leidos": total,
        "pasaron_min_score": len(scores),
        "skipped_short": skipped_short,
        "seleccionados": n_retener,
        "ratio_efectivo": round(n_retener / max(total, 1) * 100, 2),
        "score_min_selected": round(selected[-1][0], 2) if selected else 0,
        "score_max": round(all_scores[0], 2) if all_scores else 0,
        "score_medio_selected": round(
            sum(s for s, _ in selected) / max(len(selected), 1), 2
        ),
    }

    print(f"✅ {input_file.name}: {total:,} → {n_retener:,} "
          f"({stats['ratio_efectivo']}%)")
    print(f"   Score: min={stats['score_min_selected']}, "
          f"max={stats['score_max']}, "
          f"media={stats['score_medio_selected']}")

    return stats


# =============================================================================
# CURACIÓN MULTI-ARCHIVO
# =============================================================================

def curar_directorio(
    input_dir: str,
    output_dir: str,
    ratio: float = 0.2,
    min_score: float = 2.0,
    min_tokens: int = 20,
    max_samples_per_file: Optional[int] = None,
) -> Dict:
    """
    Aplica curación Pareto a todos los JSONL en un directorio.

    Returns:
        Dict con estadísticas por archivo y totales.
    """
    input_path = Path(input_dir)
    jsonl_files = sorted(input_path.glob("*.jsonl"))

    if not jsonl_files:
        print(f"❌ No se encontraron archivos JSONL en {input_dir}")
        return {"error": "sin archivos"}

    print(f"\n🔬 Curación Pareto de {len(jsonl_files)} archivos")
    print(f"   Ratio: {ratio*100:.0f}% | Min score: {min_score} | "
          f"Min tokens: {min_tokens}")
    print("=" * 60)

    resultados: Dict = {"archivos": {}, "totales": {}}
    total_in = 0
    total_out = 0

    for jsonl_file in jsonl_files:
        output_file = Path(output_dir) / jsonl_file.name
        stats = filtrar_pareto(
            str(jsonl_file),
            str(output_file),
            ratio=ratio,
            min_score=min_score,
            min_tokens=min_tokens,
            max_samples=max_samples_per_file,
        )
        resultados["archivos"][jsonl_file.name] = stats
        total_in += stats.get("total_leidos", 0)
        total_out += stats.get("seleccionados", 0)

    resultados["totales"] = {
        "total_input": total_in,
        "total_output": total_out,
        "ratio_global": round(total_out / max(total_in, 1) * 100, 2),
    }

    print("=" * 60)
    print(f"\n📊 TOTAL: {total_in:,} → {total_out:,} "
          f"({resultados['totales']['ratio_global']}%)")

    # Guardar reporte
    report_path = Path(output_dir) / "curation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(resultados, indent=2, ensure_ascii=False))
    print(f"📝 Reporte guardado en {report_path}")

    return resultados


# =============================================================================
# SEGUNDA PASADA: PARETO DEL PARETO
# =============================================================================

def pareto_recursivo(
    input_dir: str,
    output_dir: str,
    n_pasadas: int = 3,
    ratio: float = 0.2,
    min_score: float = 2.0,
) -> Dict:
    """
    Aplica compresión Pareto recursiva:
    Pasada 1: 100% → 20%
    Pasada 2: 20% → 4%
    Pasada 3: 4% → 0.8%

    Cada pasada aumenta el min_score para ser más selectiva.

    Args:
        input_dir: Directorio con datos originales.
        output_dir: Directorio base para outputs.
        n_pasadas: Cuántas compresiones aplicar.
        ratio: Ratio Pareto por pasada.
        min_score: Score mínimo inicial (sube en cada pasada).

    Returns:
        Dict con resultados de cada pasada.
    """
    print("\n" + "=" * 60)
    print("🧠 COMPRESIÓN PARETO RECURSIVA")
    print(f"   {n_pasadas} pasadas × {ratio*100:.0f}% = "
          f"{ratio**n_pasadas*100:.2f}% final")
    print("=" * 60)

    resultados_pasadas: Dict = {}
    current_input = input_dir

    for pasada in range(1, n_pasadas + 1):
        # Cada pasada es más exigente
        current_min_score = min_score + (pasada - 1) * 1.0
        current_output = str(Path(output_dir) / f"pareto_L{pasada}")

        print(f"\n--- Pasada {pasada}/{n_pasadas} "
              f"(min_score={current_min_score}) ---")

        stats = curar_directorio(
            input_dir=current_input,
            output_dir=current_output,
            ratio=ratio,
            min_score=current_min_score,
        )

        resultados_pasadas[f"pasada_{pasada}"] = stats
        current_input = current_output

    # Resumen final
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE COMPRESIÓN:")
    for pasada, stats in resultados_pasadas.items():
        totales = stats.get("totales", {})
        print(f"   {pasada}: {totales.get('total_input', 0):,} → "
              f"{totales.get('total_output', 0):,} "
              f"({totales.get('ratio_global', 0)}%)")

    return resultados_pasadas


# =============================================================================
# CLI
# =============================================================================

def main():
    """Entry point CLI."""
    parser = argparse.ArgumentParser(
        description="Curación de datos con Principio de Pareto"
    )
    parser.add_argument(
        "--input", "-i",
        default="data/distillation",
        help="Directorio de datos de entrada",
    )
    parser.add_argument(
        "--output", "-o",
        default="data/curated",
        help="Directorio de salida",
    )
    parser.add_argument(
        "--ratio", "-r",
        type=float,
        default=0.2,
        help="Ratio Pareto (0.2 = retener 20%%)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=2.0,
        help="Score mínimo para considerar un sample",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=20,
        help="Tokens mínimos en la respuesta",
    )
    parser.add_argument(
        "--recursivo",
        action="store_true",
        help="Aplicar compresión Pareto recursiva (3 pasadas)",
    )
    parser.add_argument(
        "--pasadas",
        type=int,
        default=3,
        help="Número de pasadas recursivas",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Máximo de samples por archivo (para testing)",
    )

    args = parser.parse_args()

    if args.recursivo:
        pareto_recursivo(
            input_dir=args.input,
            output_dir=args.output,
            n_pasadas=args.pasadas,
            ratio=args.ratio,
            min_score=args.min_score,
        )
    else:
        curar_directorio(
            input_dir=args.input,
            output_dir=args.output,
            ratio=args.ratio,
            min_score=args.min_score,
            min_tokens=args.min_tokens,
            max_samples_per_file=args.max_samples,
        )


if __name__ == "__main__":
    main()
