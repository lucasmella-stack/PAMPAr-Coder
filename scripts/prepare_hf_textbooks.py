#!/usr/bin/env python3
"""Descarga open-phi/textbooks, filtra CS/programming, chunkea y guarda JSONL."""

import json
import random
import re
from pathlib import Path

from datasets import load_dataset

RELEVANT_FIELDS = {"computer_science", "systems_engineering"}
RELEVANT_SUBFIELDS = {
    "programming",
    "algorithms_and_data_structures",
    "programming_languages",
    "software_design_and_engineering",
    "artificial_intelligence",
    "computational_science_and_engineering",
    "data_mining",
    "computational_modeling_and_simulation",
    "graphics_and_visualization",
    "computer_networks",
    "human-computer_interfaces",
}

MIN_CHARS = 500
MAX_CHARS = 8000


def chunk_markdown(text: str) -> list[str]:
    """Divide markdown en chunks por headers ##, cada uno entre MIN y MAX chars."""
    sections = re.split(r"\n(?=## )", text)

    chunks: list[str] = []
    current = ""

    for section in sections:
        section = section.strip()
        if not section:
            continue

        if len(current) + len(section) < MAX_CHARS:
            current += "\n\n" + section if current else section
        else:
            if len(current) >= MIN_CHARS:
                chunks.append(current.strip())
            current = section

    if current and len(current) >= MIN_CHARS:
        chunks.append(current.strip())

    return chunks


def main() -> None:
    print("Descargando open-phi/textbooks...")
    ds = load_dataset("open-phi/textbooks", split="train")
    print(f"Total filas: {len(ds)}")

    # Filtrar solo CS/programming
    filtered = [
        r
        for r in ds
        if r["field"] in RELEVANT_FIELDS and r["subfield"] in RELEVANT_SUBFIELDS
    ]
    print(f"Filtrados: {len(filtered)} filas de {len(ds)}")

    output_file = Path("data/textbook_v3/textbook_pretrain.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    all_chunks: list[dict[str, str]] = []
    for row in filtered:
        text = row["markdown"]
        topic = row.get("topic", "")

        chunks = chunk_markdown(text)
        for chunk in chunks:
            # Skipear chunks que son mayormente headers
            lines = chunk.split("\n")
            header_lines = sum(1 for l in lines if l.startswith("#"))
            if header_lines > len(lines) * 0.5:
                continue

            all_chunks.append(
                {
                    "text": chunk,
                    "source": "open-phi/textbooks",
                    "topic": topic,
                }
            )

    print(f"Total chunks: {len(all_chunks)}")
    with_code = sum(1 for c in all_chunks if "```" in c["text"])
    print(f"Con código: {with_code} ({100 * with_code // len(all_chunks)}%)")

    lengths = [len(c["text"]) for c in all_chunks]
    print(f"Longitud promedio: {sum(lengths) // len(lengths)} chars")
    print(f"Min: {min(lengths)}, Max: {max(lengths)}")

    # Shuffle y guardar
    random.seed(42)
    random.shuffle(all_chunks)

    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            json.dump(chunk, f, ensure_ascii=False)
            f.write("\n")

    total_chars = sum(lengths)
    print(f"\nGuardado: {output_file}")
    print(f"{len(all_chunks)} textos, {total_chars:,} chars totales")
    print(f"~{total_chars // 4:,} tokens estimados")


if __name__ == "__main__":
    main()
