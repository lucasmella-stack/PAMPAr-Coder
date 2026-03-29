#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
PAMPAr Benchmark — Evaluación objetiva y comparable entre runs.

Métricas:
  - Perplexity    : en datos held-out de biblioteca/ (objetivo, comparable)
  - Syntax valid  : % de código generado que pasa ast.parse()
  - Top-1 accuracy: acierto del modelo en next-token sobre test fijo
  - Code score    : heurística de coherencia (return, indent, operadores)

Guarda resultados en benchmarks/history.jsonl (una línea por run).
Compara automáticamente contra el run anterior.

Uso:
  python scripts/benchmark.py --checkpoint checkpoints/pampar_v2_best.pt
  python scripts/benchmark.py --checkpoint checkpoints/pampar_v2_best.pt --tag "paso-5000"
  python scripts/benchmark.py --history   # Ver historial sin evaluar
"""

import argparse
import ast
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

# =============================================================================
# PROMPTS FIJOS — no cambiar entre runs (son la "regla" del benchmark)
# =============================================================================

BENCHMARK_PROMPTS = [
    # Nivel 1 — funciones básicas
    "def suma(a, b):\n    ",
    "def es_par(n):\n    ",
    "def maximo(lista):\n    ",
    # Nivel 2 — algoritmos
    "def factorial(n):\n    if n <= 1:\n        return 1\n    return ",
    "def busqueda_binaria(arr, objetivo):\n    izq, der = 0, len(arr) - 1\n    while izq <= der:\n        ",
    "def invertir_lista(lst):\n    ",
    # Nivel 3 — OOP
    "class Pila:\n    def __init__(self):\n        self.items = []\n\n    def push(self, item):\n        ",
    "class NodoBST:\n    def __init__(self, valor):\n        self.valor = valor\n        self.izq = ",
    # Nivel 4 — patrones y typing
    "from typing import Generator\n\ndef fibonacci() -> Generator[int, None, None]:\n    a, b = 0, 1\n    while True:\n        ",
    "from functools import wraps\n\ndef decorador_tiempo(func):\n    @wraps(func)\n    def wrapper(*args, **kwargs):\n        ",
    # Nivel 5 — avanzado
    "from typing import TypeVar, Generic, Optional\nT = TypeVar('T')\n\nclass Resultado(Generic[T]):\n    def __init__(self, valor: Optional[T], error: str = ''):\n        ",
    "def merge_sort(arr: list) -> list:\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    izq = merge_sort(arr[:mid])\n    der = merge_sort(arr[mid:])\n    return ",
]

# Respuestas esperadas (para top-1 accuracy) — solo los primeros tokens
EXPECTED_CONTINUATIONS = [
    "return a + b",
    "return n % 2 == 0",
    "return max(lista)",
    "n * factorial(n - 1)",
    "mid = (izq + der) // 2",
    "return lst[::-1]",
    "self.items.append(item)",
    "None\n        self.der = None",
    "yield a\n        a, b = b, a + b",
    "import time\n        inicio = time.time()",
    "self.valor = valor\n        self.error = error",
    "merge(izq, der)",
]

# =============================================================================
# Cargadores
# =============================================================================


def cargar_modelo(checkpoint: Path, device: torch.device):
    from pampar.coder.v2.modelo import PampaRCoderV2
    from pampar.coder.v2.config import (
        ConfigV2, PRESET_4GB, PRESET_8GB, PRESET_24GB, PRESET_1_5B,
    )
    import dataclasses

    PRESET_MAP = {
        "4GB": PRESET_4GB, "8GB": PRESET_8GB,
        "24GB": PRESET_24GB, "1_5B": PRESET_1_5B,
    }

    config = PRESET_4GB
    if checkpoint.exists():
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        raw_cfg = ckpt.get("config", {})
        state = ckpt.get("modelo", ckpt.get("model", ckpt))

        if isinstance(raw_cfg, ConfigV2):
            config = raw_cfg
        elif isinstance(raw_cfg, dict):
            preset_name = raw_cfg.get("preset")
            if preset_name in PRESET_MAP:
                config = PRESET_MAP[preset_name]
            else:
                valid = {f.name for f in dataclasses.fields(ConfigV2)}
                filtered = {k: v for k, v in raw_cfg.items() if k in valid}
                if filtered:
                    config = ConfigV2(**filtered)

        # Inferir desde pesos si hace falta
        emb = state.get("tok_emb.weight")
        if emb is not None and emb.shape != (config.vocab_size, config.dim):
            n_capas = sum(
                1 for k in state
                if k.startswith("capas.") and k.endswith(".ln1.weight")
            )
            config = ConfigV2(
                vocab_size=int(emb.shape[0]),
                dim=int(emb.shape[1]),
                n_capas=n_capas or config.n_capas,
            )

        modelo = PampaRCoderV2(config).to(device)
        modelo.load_state_dict(state, strict=False)
        modelo.eval()
    else:
        raise FileNotFoundError(f"Checkpoint no encontrado: {checkpoint}")

    n_params = sum(p.numel() for p in modelo.parameters()) / 1e6
    return modelo, config, n_params


def cargar_tokenizer(vocab_size: int):
    import sentencepiece as spm

    tok_map = {
        16000: Path("data/tokenizer/code_tokenizer.model"),
        48000: Path("data/tokenizer/pampar_48k.model"),
    }
    path = tok_map.get(vocab_size)
    if path is None or not path.exists():
        # Fallback
        for p in tok_map.values():
            if p.exists():
                path = p
                break
    tok = spm.SentencePieceProcessor()
    tok.Load(str(path))
    return tok


# =============================================================================
# Métricas
# =============================================================================


def calcular_perplexity(
    modelo,
    tokenizer,
    biblioteca_path: Path,
    device: torch.device,
    max_tokens_total: int = 20_000,
    held_out_pct: float = 0.1,
) -> float:
    """
    Perplexity en el último held_out_pct de cada archivo de la biblioteca.
    Usa datos que el modelo NUNCA entrena directamente (últimas líneas).
    """
    total_loss = 0.0
    total_tokens = 0
    archivos = list(biblioteca_path.rglob("*.jsonl"))  # rglob para buscar en subcarpetas

    if not archivos:
        return float("nan")

    with torch.no_grad():
        for archivo in archivos:
            lineas = archivo.read_text(encoding="utf-8", errors="ignore").splitlines()
            if len(lineas) < 2:  # Mínimo 2 líneas para tener held-out
                continue
            # Held-out: último 10%  (mínimo 1 línea)
            inicio_heldout = max(1, int(len(lineas) * (1 - held_out_pct)))
            lineas_test = lineas[inicio_heldout:]

            for linea in lineas_test:
                if total_tokens >= max_tokens_total:
                    break
                try:
                    obj = json.loads(linea)
                    texto = obj.get("text", obj.get("content", str(obj)))
                except Exception:
                    texto = linea.strip()

                if not texto or len(texto) < 10:
                    continue

                ids = tokenizer.Encode(texto)
                if len(ids) < 2:
                    continue

                # Truncar a 512
                ids = ids[:513]
                tokens = torch.tensor([ids], device=device)
                inp = tokens[:, :-1]
                tgt = tokens[:, 1:]

                logits, _, _ = modelo(inp)
                B, L, V = logits.shape
                loss = F.cross_entropy(
                    logits.reshape(B * L, V),
                    tgt.reshape(B * L),
                    ignore_index=0,
                )
                if not math.isfinite(loss.item()):
                    continue
                n = (tgt != 0).sum().item()
                if n == 0:
                    continue
                total_loss += loss.item() * n
                total_tokens += n

    if total_tokens == 0:
        return float("nan")
    return math.exp(total_loss / total_tokens)


def calcular_syntax_validity(
    modelo,
    tokenizer,
    device: torch.device,
    max_tokens: int = 120,
    temperature: float = 0.5,
) -> tuple[float, list[dict]]:
    """
    Genera código para cada prompt y verifica si es Python válido con ast.parse.
    Retorna (% válido, detalles).
    """
    validos = 0
    detalles = []

    with torch.no_grad():
        for prompt in BENCHMARK_PROMPTS:
            ids = tokenizer.Encode(prompt)
            generados = list(ids)

            for _ in range(max_tokens):
                ctx = torch.tensor([generados[-512:]], device=device)
                logits, _, _ = modelo(ctx)
                next_logits = logits[0, -1] / temperature
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                generados.append(next_token)
                # Parar en doble newline con suficiente contenido
                decoded_so_far = tokenizer.Decode(generados[len(ids):])
                if "\n\n" in decoded_so_far and len(decoded_so_far) > 20:
                    break

            generado = tokenizer.Decode(generados)
            codigo_completo = generado.strip()

            valido = False
            try:
                ast.parse(codigo_completo)
                valido = True
                validos += 1
            except SyntaxError:
                # Intentar completar el bloque con pass (código incompleto al final)
                try:
                    lineas = codigo_completo.splitlines()
                    if lineas and (lineas[-1].endswith(":") or lineas[-1].strip() == ""):
                        ast.parse(codigo_completo + "\n    pass")
                    else:
                        # Agregar return/pass al último bloque abierto
                        ast.parse(codigo_completo + "\npass")
                    valido = True
                    validos += 1
                except SyntaxError:
                    pass

            detalles.append({
                "prompt": prompt[:50].replace("\n", "↵"),
                "generado": generado[len(prompt):60].replace("\n", "↵"),
                "valido": valido,
            })

    pct = validos / len(BENCHMARK_PROMPTS) * 100
    return pct, detalles, validos


def calcular_topk_accuracy(
    modelo,
    tokenizer,
    device: torch.device,
    k: int = 5,
) -> dict:
    """
    Para cada prompt, compara los primeros tokens generados (greedy)
    contra la continuación esperada. Mide top-1 y top-5 accuracy.
    """
    top1_hits = 0
    top5_hits = 0
    total = 0

    with torch.no_grad():
        for prompt, expected in zip(BENCHMARK_PROMPTS, EXPECTED_CONTINUATIONS):
            expected_ids = tokenizer.Encode(expected)
            if not expected_ids:
                continue

            prompt_ids = tokenizer.Encode(prompt)
            ctx = torch.tensor([prompt_ids[-512:]], device=device)
            logits, _, _ = modelo(ctx)
            next_logits = logits[0, -1]

            # Top-k tokens predichos
            topk = torch.topk(next_logits, k).indices.tolist()
            top1 = topk[0]
            expected_first = expected_ids[0]

            if top1 == expected_first:
                top1_hits += 1
            if expected_first in topk:
                top5_hits += 1
            total += 1

    return {
        "top1_accuracy": top1_hits / total * 100 if total else 0,
        "top5_accuracy": top5_hits / total * 100 if total else 0,
        "total_prompts": total,
    }


# =============================================================================
# Historial y comparación
# =============================================================================

HISTORY_FILE = Path("benchmarks/history.jsonl")
RESULTS_FILE = Path("benchmarks/results.json")


def cargar_ultimo_run() -> Optional[dict]:
    if not HISTORY_FILE.exists():
        return None
    lineas = HISTORY_FILE.read_text(encoding="utf-8").splitlines()
    for linea in reversed(lineas):
        if linea.strip():
            try:
                return json.loads(linea)
            except Exception:
                continue
    return None


def guardar_run(resultado: dict) -> None:
    HISTORY_FILE.parent.mkdir(exist_ok=True)
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(resultado, ensure_ascii=False) + "\n")


def mostrar_comparacion(actual: dict, anterior: Optional[dict]) -> None:
    def delta(key: str, mejor_es: str = "menor") -> str:
        if anterior is None or key not in anterior:
            return ""
        a = actual.get(key)
        p = anterior.get(key)
        if a is None or p is None:
            return ""
        diff = a - p
        if mejor_es == "menor":
            emoji = "✅" if diff < 0 else ("❌" if diff > 0 else "━")
            signo = "+" if diff > 0 else ""
        else:
            emoji = "✅" if diff > 0 else ("❌" if diff < 0 else "━")
            signo = "+" if diff > 0 else ""
        return f"  {emoji} {signo}{diff:.2f} vs anterior"

    print(f"\n{'═'*60}")
    print(f"  BENCHMARK — {actual['tag']}  ({actual['timestamp'][:16]})")
    if anterior:
        print(f"  (anterior: {anterior['tag']}  {anterior['timestamp'][:16]})")
    print(f"{'═'*60}")
    print(f"  Checkpoint : {actual['checkpoint']}")
    print(f"  Params     : {actual['params_M']:.0f}M  |  vocab: {actual['vocab_size']}")
    print()
    print(f"  Perplexity      : {actual['perplexity']:.2f}{delta('perplexity', 'menor')}")
    print(f"  Syntax valid    : {actual['syntax_validity_pct']:.1f}%{delta('syntax_validity_pct', 'mayor')}")
    print(f"  Top-1 accuracy  : {actual['top1_accuracy']:.1f}%{delta('top1_accuracy', 'mayor')}")
    print(f"  Top-5 accuracy  : {actual['top5_accuracy']:.1f}%{delta('top5_accuracy', 'mayor')}")
    print(f"  Tiempo          : {actual['tiempo_s']:.1f}s")
    print(f"{'═'*60}\n")


def mostrar_historial() -> None:
    if not HISTORY_FILE.exists():
        print("Sin historial aún — corre el benchmark primero.")
        return

    runs = []
    for linea in HISTORY_FILE.read_text(encoding="utf-8").splitlines():
        if linea.strip():
            try:
                runs.append(json.loads(linea))
            except Exception:
                pass

    if not runs:
        print("Sin runs en historial.")
        return

    print(f"\n{'─'*80}")
    print(f"  {'Tag':<30} {'PPL':>8} {'Syntax':>8} {'Top1':>7} {'Top5':>7}  Fecha")
    print(f"{'─'*80}")
    for r in runs:
        ppl = f"{r['perplexity']:.1f}" if r["perplexity"] == r["perplexity"] else "NaN"
        print(
            f"  {r['tag']:<30} {ppl:>8} "
            f"{r['syntax_validity_pct']:>7.1f}% "
            f"{r['top1_accuracy']:>6.1f}% "
            f"{r['top5_accuracy']:>6.1f}%  "
            f"{r['timestamp'][:16]}"
        )
    print(f"{'─'*80}\n")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PAMPAr Benchmark — evaluación objetiva y reproducible"
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("checkpoints/pampar_v2_best.pt"),
        help="Checkpoint a evaluar",
    )
    parser.add_argument(
        "--tag", type=str, default=None,
        help="Etiqueta para identificar este run (ej: 'paso-5000', 'con-distil')",
    )
    parser.add_argument(
        "--biblioteca", type=Path, default=Path("biblioteca"),
    )
    parser.add_argument(
        "--device", type=str, default="auto",
    )
    parser.add_argument(
        "--history", action="store_true",
        help="Solo mostrar el historial de runs anteriores",
    )
    parser.add_argument(
        "--max-tokens-ppl", type=int, default=20_000,
        help="Máximo tokens para calcular perplexity (más = más preciso, más lento)",
    )
    args = parser.parse_args()

    if args.history:
        mostrar_historial()
        return

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    tag = args.tag or f"paso-manual-{datetime.now().strftime('%H%M')}"

    print(f"\n  PAMPAr Benchmark")
    print(f"  Device: {device} | Checkpoint: {args.checkpoint.name}\n")

    # Cargar modelo
    print("  [1/4] Cargando modelo...")
    modelo, config, n_params = cargar_modelo(args.checkpoint, device)
    tokenizer = cargar_tokenizer(config.vocab_size)
    print(f"        {n_params:.0f}M params | vocab={config.vocab_size}")

    t0 = time.time()

    # Perplexity
    print("  [2/4] Calculando perplexity en held-out...")
    ppl = calcular_perplexity(
        modelo, tokenizer, args.biblioteca, device,
        max_tokens_total=args.max_tokens_ppl,
    )
    print(f"        Perplexity: {ppl:.2f}")

    # Syntax validity
    print("  [3/4] Evaluando validez sintáctica...")
    syntax_pct, syntax_detalles, syntax_n = calcular_syntax_validity(modelo, tokenizer, device)
    print(f"        Syntax valid: {syntax_pct:.1f}% ({syntax_n}/{len(BENCHMARK_PROMPTS)})")

    # Top-k accuracy
    print("  [4/4] Calculando top-k accuracy...")
    topk = calcular_topk_accuracy(modelo, tokenizer, device)
    print(f"        Top-1: {topk['top1_accuracy']:.1f}% | Top-5: {topk['top5_accuracy']:.1f}%")

    tiempo = time.time() - t0

    resultado = {
        "tag": tag,
        "timestamp": datetime.now().isoformat(),
        "checkpoint": str(args.checkpoint),
        "params_M": n_params,
        "vocab_size": config.vocab_size,
        "perplexity": ppl,
        "syntax_validity_pct": syntax_pct,
        "top1_accuracy": topk["top1_accuracy"],
        "top5_accuracy": topk["top5_accuracy"],
        "tiempo_s": tiempo,
        "syntax_detalles": syntax_detalles,
    }

    anterior = cargar_ultimo_run()
    mostrar_comparacion(resultado, anterior)

    # Guardar (sin detalles verbosos en el historial principal)
    resultado_compact = {k: v for k, v in resultado.items() if k != "syntax_detalles"}
    guardar_run(resultado_compact)

    print(f"  Guardado en {HISTORY_FILE}")
    print(f"  Ver historial: python scripts/benchmark.py --history\n")


if __name__ == "__main__":
    main()
