#!/usr/bin/env python3
"""Simula el impacto de correcciones en ZONA_TERRITORIO sobre el Brain Scanner score."""
import copy
import torch
import sys
import sentencepiece as spm
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pampar.coder.v3.config import PRESET_V3
from pampar.coder.v3.modelo import PamparV3
from pampar.coder.v3.llaves import clasificar_token
from pampar.coder.v3.zonas import Zona, Territorio, ZONA_TERRITORIO

TERR_NAMES = ["SINT", "SEMA", "LOGI", "ESTR"]

CODE_SUITE = [
    ("keywords", "def fibonacci(n):"),
    ("clase", "class DataProcessor:"),
    ("imports", "from pathlib import Path"),
    ("loop", "for i in range(10):"),
    ("condicional", "if x > 0 and y < 10:"),
    ("excepcion", "try:\n    result = 1 / 0\nexcept ZeroDivisionError:"),
    ("async", "async def fetch(url):"),
    ("aritmetica", "result = a + b * c - d / e"),
    ("comparacion", "x == y or x != z"),
    ("asignacion", "total += price * quantity"),
    ("literals", "name = 'hello world'"),
    ("numeros", "pi = 3.14159"),
    ("tipos", "items: list[int] = []"),
    ("builtins", "print(len(range(10)))"),
    ("magic", "def __init__(self, value):"),
    ("comprehension", "squares = [x**2 for x in range(10)]"),
    ("lambda", "fn = lambda x: x * 2"),
    ("decorador", "@staticmethod\ndef create():"),
    ("return", "return sorted(data, key=lambda x: x.name)"),
    ("with", "with open('file.txt') as f:"),
]

# Possible corrections with linguistic justification
CORRECTIONS = {
    "assign_to_sint": {
        "desc": "B35_OP_ASSIGN (=, +=, etc.): LOGICO → SINTAXIS. Assignment is statement syntax, not logic.",
        "changes": {Zona.B35_OP_ASSIGN: Territorio.SINTAXIS},
    },
    "member_to_estr": {
        "desc": "B36_OP_MEMBER (.) : LOGICO → ESTRUCTURAL. Member access navigates structure.",
        "changes": {Zona.B36_OP_MEMBER: Territorio.ESTRUCTURAL},
    },
    "ternary_to_sint": {
        "desc": "B37_OP_TERNARY (?:) : LOGICO → SINTAXIS. Ternary is syntactic form.",
        "changes": {Zona.B37_OP_TERNARY: Territorio.SINTAXIS},
    },
}


def build_territory_table(
    tokenizer: spm.SentencePieceProcessor,
    overrides: dict[Zona, Territorio] | None = None,
) -> torch.Tensor:
    """Build territory table with optional zone-to-territory overrides."""
    mapping = dict(ZONA_TERRITORIO)
    if overrides:
        mapping.update(overrides)

    vocab_size = tokenizer.GetPieceSize()
    table = torch.zeros(vocab_size, dtype=torch.long)
    for tid in range(vocab_size):
        piece = tokenizer.IdToPiece(tid)
        z, _c = clasificar_token(piece)
        table[tid] = mapping[z].value
    return table


def evaluate_suite(
    model: PamparV3,
    tokenizer: spm.SentencePieceProcessor,
    territory_table: torch.Tensor,
    device: torch.device,
) -> dict:
    """Run suite and return detailed metrics."""
    from brain_scanner import forward_instrumentado

    all_correct = 0
    all_total = 0
    all_margins: list[float] = []
    all_max_conf: list[float] = []
    per_sample: list[dict] = []

    for label, code in CODE_SUITE:
        tids = tokenizer.Encode(code, out_type=int)
        inp = torch.tensor([tids], dtype=torch.long, device=device)

        with torch.no_grad():
            info = forward_instrumentado(model, inp)

        n_levels = len(info["confianza"])
        terr_last = info["terr_por_nivel"][n_levels]

        correct = 0
        margins: list[float] = []
        for i, tid in enumerate(tids):
            expected = territory_table[tid].item()
            actual = terr_last[i].argmax().item()
            if actual == expected:
                correct += 1
            acts = terr_last[i].tolist()
            s = sorted(acts, reverse=True)
            margins.append(s[0] - s[1])

        n = len(tids)
        acc = correct / n * 100
        avg_margin = sum(margins) / len(margins)
        max_conf = max(info["confianza"])

        all_correct += correct
        all_total += n
        all_margins.extend(margins)
        all_max_conf.append(max_conf)

        per_sample.append({
            "label": label,
            "tokens": n,
            "acc": acc,
            "margin": avg_margin,
            "max_conf": max_conf,
        })

    global_acc = all_correct / all_total * 100
    global_margin = sum(all_margins) / len(all_margins)
    global_exit = sum(all_max_conf) / len(all_max_conf)
    min_acc = min(s["acc"] for s in per_sample)
    worst = [s for s in per_sample if s["acc"] == min_acc][0]

    score = (
        min(global_acc, 100) * 0.40
        + min(global_margin * 1000, 100) * 0.25
        + min(global_exit * 100, 100) * 0.15
        + min(min_acc, 100) * 0.20
    )

    return {
        "score": score,
        "acc": global_acc,
        "margin": global_margin,
        "exit": global_exit,
        "min_acc": min_acc,
        "worst": worst["label"],
        "per_sample": per_sample,
    }


def main() -> None:
    tok = spm.SentencePieceProcessor()
    tok.Load(str(ROOT / "data" / "tokenizer" / "pampar_48k.model"))

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

    # ── Baseline (current LLAVES) ──
    print("=" * 70)
    print("  SIMULACIÓN DE CORRECCIONES EN ZONA_TERRITORIO")
    print("=" * 70)

    table_base = build_territory_table(tok)
    result_base = evaluate_suite(model, tok, table_base, device)

    print(f"\n  BASELINE (actual):")
    print(f"    Score: {result_base['score']:.1f}")
    print(f"    AccN5: {result_base['acc']:.1f}%")
    print(f"    Margin: {result_base['margin']:.4f}")
    print(f"    Exit: {result_base['exit']:.1%}")
    print(f"    Min acc: {result_base['min_acc']:.1f}% ({result_base['worst']})")

    # ── Test each correction individually ──
    for name, corr in CORRECTIONS.items():
        table = build_territory_table(tok, corr["changes"])
        result = evaluate_suite(model, tok, table, device)

        delta_score = result["score"] - result_base["score"]
        delta_acc = result["acc"] - result_base["acc"]
        sign = "+" if delta_score >= 0 else ""

        print(f"\n  CORRECCIÓN: {name}")
        print(f"    {corr['desc']}")
        print(f"    Score: {result['score']:.1f} ({sign}{delta_score:.1f})")
        print(f"    AccN5: {result['acc']:.1f}% ({'+' if delta_acc >= 0 else ''}{delta_acc:.1f}%)")
        print(f"    Min acc: {result['min_acc']:.1f}% ({result['worst']})")

        # Show per-sample changes
        changed = []
        for s, s_base in zip(result["per_sample"], result_base["per_sample"]):
            d = s["acc"] - s_base["acc"]
            if abs(d) > 0.1:
                changed.append(f"      {s['label']:16s} {s_base['acc']:5.1f}% → {s['acc']:5.1f}% ({'+' if d >= 0 else ''}{d:.1f}%)")
        if changed:
            print("    Samples affected:")
            for c in changed:
                print(c)

    # ── Test ALL corrections combined ──
    all_overrides: dict[Zona, Territorio] = {}
    for corr in CORRECTIONS.values():
        all_overrides.update(corr["changes"])

    table_all = build_territory_table(tok, all_overrides)
    result_all = evaluate_suite(model, tok, table_all, device)

    delta = result_all["score"] - result_base["score"]
    print(f"\n  TODAS LAS CORRECCIONES COMBINADAS:")
    print(f"    Score: {result_all['score']:.1f} ({'+' if delta >= 0 else ''}{delta:.1f})")
    print(f"    AccN5: {result_all['acc']:.1f}%")
    print(f"    Min acc: {result_all['min_acc']:.1f}% ({result_all['worst']})")

    for s, s_base in zip(result_all["per_sample"], result_base["per_sample"]):
        d = s["acc"] - s_base["acc"]
        emoji = "↑" if d > 0.1 else "↓" if d < -0.1 else "="
        print(f"    {emoji} {s['label']:16s} {s['acc']:5.1f}% (was {s_base['acc']:.1f}%)")


if __name__ == "__main__":
    main()
