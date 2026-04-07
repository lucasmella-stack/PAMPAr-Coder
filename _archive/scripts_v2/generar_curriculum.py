# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Generador de Curriculum — Prepara datos por nivel de dificultad

Toma los archivos JSONL existentes y los clasifica en 6 niveles:
  nivel_1_basico.jsonl     → Variables, print, asignaciones
  nivel_2_control.jsonl    → if/else, for, while
  nivel_3_funciones.jsonl  → def, return, parámetros
  nivel_4_clases.jsonl     → class, herencia, métodos
  nivel_5_algoritmos.jsonl → Sorting, búsqueda, recursión
  nivel_6_patrones.jsonl   → Design patterns, código complejo

Uso:
  python scripts/generar_curriculum.py
  python scripts/generar_curriculum.py --max-por-nivel 50000
  python scripts/generar_curriculum.py --data-dir data/distillation --output-dir data/curriculum
"""

import argparse
import json
import sys
from pathlib import Path

# Ajustar path
script_dir = Path(__file__).parent
project_dir = script_dir.parent
sys.path.insert(0, str(project_dir))

from pampar.coder.v2.aprendizaje.curriculum import (
    NivelDificultad,
    clasificar_dificultad,
    crear_curriculum_desde_jsonl,
)


def main():
    parser = argparse.ArgumentParser(
        description="Genera datos organizados por curriculum (nivel de dificultad)"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Directorio raíz de datos"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/curriculum",
        help="Directorio de salida para datos clasificados"
    )
    parser.add_argument(
        "--max-por-nivel", type=int, default=50000,
        help="Máximo ejemplos por nivel"
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Solo muestra distribución sin crear archivos"
    )
    args = parser.parse_args()

    data_dir = Path(project_dir / args.data_dir)
    output_dir = Path(project_dir / args.output_dir)

    # Encontrar archivos JSONL
    archivos = sorted(str(p) for p in data_dir.rglob("*.jsonl"))
    
    # Excluir archivos de curriculum existentes
    archivos = [a for a in archivos if "curriculum" not in a and "nivel_" not in a]

    if not archivos:
        print(f"❌ No se encontraron archivos JSONL en {data_dir}")
        return

    print(f"📚 Generador de Curriculum PAMPAr-Coder")
    print(f"  Directorio de datos: {data_dir}")
    print(f"  Archivos JSONL encontrados: {len(archivos)}")
    for a in archivos:
        p = Path(a)
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"    - {p.name} ({size_mb:.1f} MB)")

    if args.preview:
        # Solo preview: clasificar una muestra
        print(f"\n  Clasificando muestra de cada archivo...")
        conteos = {n: 0 for n in NivelDificultad}
        total = 0

        for archivo in archivos:
            with open(archivo, "r", encoding="utf-8") as f:
                for i, linea in enumerate(f):
                    if i >= 1000:  # Muestra de 1000 por archivo
                        break
                    try:
                        data = json.loads(linea.strip())
                    except json.JSONDecodeError:
                        continue

                    texto = data.get("text", data.get("output", data.get("code", "")))
                    if not texto or len(texto) < 10:
                        continue

                    nivel, conf = clasificar_dificultad(texto)
                    conteos[nivel] += 1
                    total += 1

        print(f"\n  📊 Distribución estimada (muestra de {total}):")
        for nivel in NivelDificultad:
            pct = 100 * conteos[nivel] / max(total, 1)
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            print(f"    {nivel.value}. {nivel.name:12s}: {conteos[nivel]:5d} ({pct:5.1f}%) {bar}")
        return

    # Crear curriculum
    print(f"\n  Creando curriculum en {output_dir}...")
    print(f"  Máximo por nivel: {args.max_por_nivel}")

    conteos = crear_curriculum_desde_jsonl(
        archivos=archivos,
        output_dir=str(output_dir),
        max_por_nivel=args.max_por_nivel,
    )

    # Verificar tamaños
    print(f"\n📊 Archivos creados:")
    for nivel in NivelDificultad:
        nombre = f"nivel_{nivel.value}_{nivel.name.lower()}.jsonl"
        path = output_dir / nombre
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  {nombre}: {conteos[nivel]} ejemplos ({size_mb:.1f} MB)")
        else:
            print(f"  {nombre}: no creado")

    total = sum(conteos.values())
    print(f"\n✅ Curriculum generado: {total} ejemplos en {output_dir}")
    print(f"\nSiguiente paso:")
    print(f"  python scripts/train_cerebral.py --fase 1")


if __name__ == "__main__":
    main()
