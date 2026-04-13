#!/usr/bin/env python3
"""Diagnóstico: mide tokenización de biblioteca sin construir el array."""

import json
import sys
import time
from pathlib import Path

import sentencepiece as spm

bib = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("biblioteca")
tok_path = sys.argv[2] if len(sys.argv) > 2 else "data/tokenizer/pampar_48k.model"

tok = spm.SentencePieceProcessor()
tok.Load(tok_path)
print(f"Tokenizer OK: vocab={tok.GetPieceSize()}", flush=True)

indice = json.load(open(bib / "indice.json", encoding="utf-8"))
archivos = []
for _cat, temas in indice.items():
    if not isinstance(temas, list):
        continue
    for t in temas:
        a = t.get("archivo", "")
        if a and (bib / a).exists():
            archivos.append(bib / a)
archivos.sort()
print(f"Archivos: {len(archivos)}", flush=True)

total_ids = 0
t0 = time.time()
for i, ruta in enumerate(archivos):
    n = 0
    with open(ruta, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get("text", obj.get("content", ""))
                if not text and "instruction" in obj:
                    text = obj["instruction"] + "\n" + obj.get("output", "")
                if text:
                    ids = tok.Encode(text)
                    n += len(ids)
            except Exception:
                pass
    total_ids += n
    if (i + 1) % 20 == 0 or i == len(archivos) - 1:
        elapsed = time.time() - t0
        print(
            f"  {i + 1}/{len(archivos)} archivos — {total_ids / 1e6:.1f}M tokens — {elapsed:.0f}s",
            flush=True,
        )

elapsed = time.time() - t0
seq_len = 513
stride = 256
n_chunks = max(0, (total_ids - seq_len) // stride)
mem_chunks_gb = n_chunks * seq_len * 4 / 1e9
mem_list_gb = total_ids * 28 / 1e9
mem_np_tmp_gb = total_ids * 4 / 1e9
peak_gb = mem_list_gb + mem_np_tmp_gb + mem_chunks_gb

print(f"\n=== RESULTADO ===", flush=True)
print(f"Total tokens: {total_ids:,} ({total_ids / 1e6:.1f}M)", flush=True)
print(f"Tiempo tokenización: {elapsed:.0f}s", flush=True)
print(f"Chunks: {n_chunks:,}", flush=True)
print(f"Memory estimates:", flush=True)
print(f"  Python list[int]:  {mem_list_gb:.1f} GB", flush=True)
print(f"  np.array temporal: {mem_np_tmp_gb:.1f} GB", flush=True)
print(f"  Chunks array:      {mem_chunks_gb:.1f} GB", flush=True)
print(f"  Peak estimado:     {peak_gb:.1f} GB", flush=True)
print("DIAG_OK", flush=True)
