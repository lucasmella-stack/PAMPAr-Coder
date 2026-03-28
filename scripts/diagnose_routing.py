#!/usr/bin/env python3
"""Diagnóstico detallado de errores de routing por token."""
import torch
import sys
import sentencepiece as spm
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pampar.coder.v3.config import PRESET_V3
from pampar.coder.v3.modelo import PamparV3
from pampar.coder.v3.llaves import clasificar_token
from pampar.coder.v3.zonas import ZONA_TERRITORIO, Territorio

TERR_NAMES = ["SINT", "SEMA", "LOGI", "ESTR"]

WEAK_SAMPLES = [
    ("numeros", "pi = 3.14159"),
    ("with", "with open('file.txt') as f:"),
    ("excepcion", "try:\n    result = 1 / 0\nexcept ZeroDivisionError:"),
    ("comparacion", "x == y or x != z"),
    ("lambda", "fn = lambda x: x * 2"),
    ("imports", "from pathlib import Path"),
    ("literals", "name = 'hello world'"),
    ("decorador", "@staticmethod\ndef create():"),
]


def main() -> None:
    tok = spm.SentencePieceProcessor()
    tok.Load(str(ROOT / "data" / "tokenizer" / "pampar_48k.model"))

    vocab_size = tok.GetPieceSize()
    territory_table = torch.zeros(vocab_size, dtype=torch.long)
    for tid in range(vocab_size):
        piece = tok.IdToPiece(tid)
        z, _c = clasificar_token(piece)
        territory_table[tid] = ZONA_TERRITORIO[z].value

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PamparV3(PRESET_V3).to(device)
    ckpt = torch.load(
        str(ROOT / "checkpoints" / "v3_ghidra_v4.pt"),
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(ckpt["modelo"])
    model.registrar_tokenizer(tok)
    model.eval()
    del ckpt

    # Import forward_instrumentado from brain_scanner
    from brain_scanner import forward_instrumentado

    all_errors: list[dict] = []

    for label, code in WEAK_SAMPLES:
        tids = tok.Encode(code, out_type=int)
        pieces = [tok.IdToPiece(t) for t in tids]
        inp = torch.tensor([tids], dtype=torch.long, device=device)

        with torch.no_grad():
            info = forward_instrumentado(model, inp)

        terr_last = info["terr_por_nivel"][-1]

        print(f"\n{'='*60}")
        print(f"  {label}: {repr(code[:50])} ({len(tids)} tokens)")
        print(f"{'='*60}")

        for i, (tid, piece) in enumerate(zip(tids, pieces)):
            expected = territory_table[tid].item()
            actual = terr_last[i].argmax().item()
            zona, conf = clasificar_token(piece)
            acts = terr_last[i].tolist()
            mark = "✓" if actual == expected else "✗"

            if actual != expected:
                all_errors.append({
                    "sample": label,
                    "token": piece,
                    "zona": zona.name,
                    "conf": conf,
                    "expected": TERR_NAMES[expected],
                    "actual": TERR_NAMES[actual],
                    "acts": acts,
                })

            acts_str = " ".join(
                f"{TERR_NAMES[j]}={a:.3f}" for j, a in enumerate(acts)
            )
            print(
                f"  {mark} {piece:20s} exp={TERR_NAMES[expected]:4s} "
                f"act={TERR_NAMES[actual]:4s} zona={zona.name:20s} "
                f"conf={conf:.0%}  [{acts_str}]"
            )

    print(f"\n{'='*60}")
    print(f"  RESUMEN DE ERRORES: {len(all_errors)} tokens mal routeados")
    print(f"{'='*60}")

    # Group errors by pattern
    from collections import Counter
    patterns = Counter()
    for e in all_errors:
        key = f"{e['expected']}->{e['actual']}"
        patterns[key] += 1

    print("\n  Patrones de error:")
    for pattern, count in patterns.most_common():
        print(f"    {pattern}: {count} tokens")

    print("\n  Detalle de cada error:")
    for e in all_errors:
        margin = sorted(e["acts"], reverse=True)
        m = margin[0] - margin[1]
        print(
            f"    [{e['sample']:12s}] {e['token']:20s} "
            f"{e['expected']:4s}->{e['actual']:4s}  "
            f"zona={e['zona']:20s} conf={e['conf']:.0%}  "
            f"margin={m:.4f}"
        )


if __name__ == "__main__":
    main()
