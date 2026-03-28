#!/usr/bin/env python3
"""
Construye master_sft_v2.jsonl: dataset expandido con datos de distilación filtrados.

Criterios de filtrado:
  - Solo Python (contiene 'def ', 'class ', 'import ', 'print(', etc.)
  - Longitud: 100-2000 chars (evita triviales y mega-ejemplos)
  - Deduplicación por hash de texto
  - Mezcla balanceada: datos originales (1253) + distilación curada
"""

import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PYTHON_SIGNALS = [
    "def ", "class ", "import ", "from ", "print(", "return ",
    "for ", "while ", "if ", "elif ", "else:", "try:", "except",
    "lambda ", "yield ", "async ", "await ", "with ",
    "__init__", "self.", "range(", "len(", ".append(",
]

# Señales de NO-Python (filtrar)
NON_PYTHON = [
    "public static void", "System.out.println", "#include",
    "cout <<", "Console.WriteLine", "SELECT ", "CREATE TABLE",
    "function(", "const ", "let ", "var ", "=>",  # JS
    "<html", "<div", "<!DOCTYPE",  # HTML
]


def is_python(text: str) -> bool:
    """Heurística: es código Python?"""
    text_lower = text.lower()
    # Rechazar NO-Python
    for np in NON_PYTHON:
        if np.lower() in text_lower:
            return False
    # Requiere al menos 2 señales Python
    count = sum(1 for s in PYTHON_SIGNALS if s in text)
    return count >= 2


def text_hash(text: str) -> str:
    """Hash para deduplicación."""
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.md5(normalized.encode()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    """Carga JSONL, retorna lista de {text, source}."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            if text:
                items.append({"text": text, "source": obj.get("source", path.stem)})
    return items


def main() -> None:
    random.seed(42)

    # 1. Cargar datos originales (master_sft.jsonl)
    original_path = ROOT / "data" / "master_sft.jsonl"
    originals = load_jsonl(original_path)
    print(f"Originales: {len(originals)} ejemplos")

    # 2. Cargar distilación
    distill_dir = ROOT / "data" / "distillation"
    distill_items: list[dict] = []

    for jsonl in sorted(distill_dir.glob("*.jsonl")):
        if jsonl.name == "distillation_data.jsonl":
            continue  # Skip combined (evitar duplicados)
        items = load_jsonl(jsonl)
        print(f"  {jsonl.name}: {len(items)} ejemplos")
        distill_items.extend(items)

    print(f"Total distilación raw: {len(distill_items)}")

    # 3. Filtrar por Python y longitud
    filtered = []
    for item in distill_items:
        text = item["text"]
        if len(text) < 100 or len(text) > 2000:
            continue
        if not is_python(text):
            continue
        filtered.append(item)

    print(f"Filtrados (Python, 100-2000 chars): {len(filtered)}")

    # 4. Deduplicar (entre sí y vs originales)
    seen_hashes: set[str] = set()
    for item in originals:
        seen_hashes.add(text_hash(item["text"]))

    unique_distill: list[dict] = []
    for item in filtered:
        h = text_hash(item["text"])
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_distill.append(item)

    print(f"Únicos (sin duplicar originales): {len(unique_distill)}")

    # 5. Seleccionar subconjunto: ~5K de distilación
    # (108M params, ~256 tokens/ejemplo → ~1.3M tokens, buen ratio)
    max_distill = 5000
    if len(unique_distill) > max_distill:
        random.shuffle(unique_distill)
        unique_distill = unique_distill[:max_distill]
        print(f"Seleccionados (muestra): {len(unique_distill)}")

    # 6. Combinar: originales + distilación
    combined = originals + unique_distill
    random.shuffle(combined)

    # 7. Guardar
    out_path = ROOT / "data" / "master_sft_v2.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for item in combined:
            json.dump({"text": item["text"], "source": item["source"]}, f, ensure_ascii=False)
            f.write("\n")

    print(f"\n{'=' * 50}")
    print(f"master_sft_v2.jsonl: {len(combined)} ejemplos")
    print(f"  Originales: {len(originals)}")
    print(f"  Distilación: {len(unique_distill)}")
    print(f"  Path: {out_path}")

    # Stats por source
    from collections import Counter
    sources = Counter(item["source"] for item in combined)
    print(f"\nPor source:")
    for src, cnt in sources.most_common():
        print(f"  {src}: {cnt}")

    # Longitud promedio
    lengths = [len(item["text"]) for item in combined]
    print(f"\nLongitud: avg={sum(lengths)/len(lengths):.0f} min={min(lengths)} max={max(lengths)}")


if __name__ == "__main__":
    main()
