"""
Integra los datasets grandes en biblioteca/ dividiéndolos en chunks temáticos.

Fuentes:
  - data/distillation/distillation_data.jsonl   (1.48M líneas)
  - data/distillation/codeexercises_python.jsonl (1.46M líneas)
  - data/textbook_v3/textbook_pretrain.jsonl     (3.5K  líneas)
  - data/distillation/codealpaca_20k.jsonl       (20K   líneas)

Estrategia:
  - Dividir en chunks de CHUNK_SIZE líneas
  - Crear carpetas en biblioteca/ por categoría
  - Actualizar biblioteca/indice.json
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
BIBLIOTECA = ROOT / "biblioteca"
CHUNK_SIZE = 25_000  # ~25K líneas por "tema" — comparable con los temas existentes

# Fuentes a integrar: (archivo, categoria, nombre_base, nivel)
FUENTES = [
    (
        ROOT / "data/distillation/distillation_data.jsonl",
        "instruccion",
        "instruccion",
        3,
    ),
    (
        ROOT / "data/distillation/codeexercises_python.jsonl",
        "ejercicios",
        "ejercicio",
        2,
    ),
    (
        ROOT / "data/textbook_v3/textbook_pretrain.jsonl",
        "textbook",
        "textbook",
        2,
    ),
    (
        ROOT / "data/distillation/codealpaca_20k.jsonl",
        "instruccion",
        "codealpaca",
        3,
    ),
]


def dividir_en_chunks(
    fuente: Path, categoria: str, nombre_base: str, nivel: int
) -> list[dict]:
    """Lee el archivo y lo divide en chunks de CHUNK_SIZE líneas."""
    dest_dir = BIBLIOTECA / categoria
    dest_dir.mkdir(parents=True, exist_ok=True)

    entradas = []
    chunk_idx = 0
    buffer = []

    print(f"Procesando {fuente.name}...")
    with fuente.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # Normalizar: asegurar que tenga campo "text"
                if "text" not in obj:
                    # Intentar extraer texto de campos alternativos
                    text = (
                        obj.get("content")
                        or obj.get("output")
                        or obj.get("response")
                        or str(obj)
                    )
                    obj = {"text": text}
                buffer.append(json.dumps(obj, ensure_ascii=False))
            except json.JSONDecodeError:
                continue

            if len(buffer) >= CHUNK_SIZE:
                nombre_chunk = f"{nombre_base}_{chunk_idx + 1:03d}"
                archivo_rel = f"{categoria}/{nombre_chunk}.jsonl"
                dest_file = BIBLIOTECA / archivo_rel
                dest_file.write_text("\n".join(buffer), encoding="utf-8")
                entradas.append(
                    {
                        "nombre": nombre_chunk,
                        "nivel": nivel,
                        "archivo": archivo_rel,
                        "tiene_datos": True,
                        "n_samples": len(buffer),
                    }
                )
                print(
                    f"  chunk {chunk_idx + 1:03d}: {len(buffer)} líneas → {archivo_rel}"
                )
                buffer = []
                chunk_idx += 1

    # Último chunk (si quedan líneas)
    if buffer:
        nombre_chunk = f"{nombre_base}_{chunk_idx + 1:03d}"
        archivo_rel = f"{categoria}/{nombre_chunk}.jsonl"
        dest_file = BIBLIOTECA / archivo_rel
        dest_file.write_text("\n".join(buffer), encoding="utf-8")
        entradas.append(
            {
                "nombre": nombre_chunk,
                "nivel": nivel,
                "archivo": archivo_rel,
                "tiene_datos": True,
                "n_samples": len(buffer),
            }
        )
        print(f"  chunk {chunk_idx + 1:03d}: {len(buffer)} líneas → {archivo_rel}")

    print(f"  Total: {chunk_idx + 1} chunks de {fuente.name}")
    return entradas


def main() -> None:
    # Cargar indice.json actual
    indice_path = BIBLIOTECA / "indice.json"
    with indice_path.open("r", encoding="utf-8") as f:
        indice: dict = json.load(f)

    temas_nuevos_total = 0

    for fuente, categoria, nombre_base, nivel in FUENTES:
        if not fuente.exists():
            print(f"SKIP: {fuente} no existe")
            continue

        entradas = dividir_en_chunks(fuente, categoria, nombre_base, nivel)

        if categoria not in indice:
            indice[categoria] = []

        # Evitar duplicados si se corre el script dos veces
        nombres_existentes = {e["nombre"] for e in indice[categoria]}
        nuevas = [e for e in entradas if e["nombre"] not in nombres_existentes]
        indice[categoria].extend(nuevas)
        temas_nuevos_total += len(nuevas)

    # Guardar indice actualizado
    indice_path.write_text(
        json.dumps(indice, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    total_temas = sum(len(v) for v in indice.values())
    print(f"\n✓ Listo — {temas_nuevos_total} temas nuevos agregados")
    print(f"  Total temas en indice.json: {total_temas}")
    print(f"  Categorías: {list(indice.keys())}")


if __name__ == "__main__":
    main()
