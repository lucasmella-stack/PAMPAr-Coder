#!/usr/bin/env python3
# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2025-2026 Lucas Ricardo Mella Chillemi
"""
benchmark_humaneval.py — Evaluación HumanEval (pass@1) para PamparV3.

Descarga el dataset HumanEval de OpenAI (164 problemas), genera soluciones
con el modelo y ejecuta los tests unitarios oficiales.

Uso:
  python -X utf8 scripts/benchmark_humaneval.py
  python -X utf8 scripts/benchmark_humaneval.py --checkpoint checkpoints/v3_train.pt
  python -X utf8 scripts/benchmark_humaneval.py --samples-per-task 5 --temp 0.4
  python -X utf8 scripts/benchmark_humaneval.py --device cpu --verbose

Requiere:
  pip install datasets  (para descargar HumanEval desde HuggingFace)

Resultados se guardan en benchmarks/humaneval_results.json
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Descarga del dataset
# =============================================================================


def cargar_humaneval() -> list[dict[str, Any]]:
    """Descarga HumanEval desde HuggingFace datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: instalar datasets → pip install datasets")
        sys.exit(1)

    ds = load_dataset("openai_humaneval", split="test")
    problemas = []
    for row in ds:
        problemas.append(
            {
                "task_id": row["task_id"],
                "prompt": row["prompt"],
                "canonical_solution": row["canonical_solution"],
                "test": row["test"],
                "entry_point": row["entry_point"],
            }
        )
    print(f"  HumanEval cargado: {len(problemas)} problemas")
    return problemas


# =============================================================================
# Carga del modelo
# =============================================================================


def cargar_modelo(checkpoint: Path, device: torch.device):
    """Carga PamparV3 desde checkpoint."""
    import dataclasses

    from pampar.coder.v3.config import PRESET_V3, ConfigV3
    from pampar.coder.v3.modelo import PamparV3

    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    raw_cfg = ckpt.get("config", {})
    state = ckpt.get("modelo", ckpt)

    if isinstance(raw_cfg, dict) and "dim" in raw_cfg:
        campos = {f.name for f in dataclasses.fields(ConfigV3)}
        kwargs = {k: v for k, v in raw_cfg.items() if k in campos}
        try:
            config = ConfigV3(**kwargs)
        except TypeError:
            config = PRESET_V3
    elif isinstance(raw_cfg, ConfigV3):
        config = raw_cfg
    else:
        config = PRESET_V3

    modelo = PamparV3(config).to(device)
    modelo.load_state_dict(state, strict=False)
    modelo.eval()
    return modelo, config


def cargar_tokenizer(vocab_size: int = 48000):
    """Busca y carga el tokenizer SentencePiece."""
    import sentencepiece as spm

    candidates = [
        Path("data/tokenizer/pampar_48k.model"),
        Path("data/tokenizer/code_tokenizer.model"),
    ]
    for p in candidates:
        if p.exists():
            tok = spm.SentencePieceProcessor()
            tok.Load(str(p))
            return tok
    raise FileNotFoundError("Tokenizer no encontrado")


# =============================================================================
# Generación
# =============================================================================


@torch.no_grad()
def generar(
    modelo,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.95,
) -> str:
    """Genera código completando el prompt con nucleus sampling."""
    ids = tokenizer.Encode(prompt)
    generados = list(ids)

    for _ in range(max_tokens):
        ctx = torch.tensor([generados[-512:]], dtype=torch.long, device=device)
        logits, _, _ = modelo(ctx)
        next_logits = logits[0, -1]

        if temperature <= 0.0:
            next_token = int(next_logits.argmax())
        else:
            next_logits = next_logits / temperature
            # Top-p (nucleus) sampling
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[mask] = float("-inf")
            probs = F.softmax(sorted_logits, dim=-1)
            idx = int(torch.multinomial(probs, 1))
            next_token = int(sorted_indices[idx])

        generados.append(next_token)
        decoded = tokenizer.Decode(generados[len(ids) :])

        # Stop conditions
        if "\n\nclass " in decoded or "\n\ndef " in decoded:
            # Truncar antes de la siguiente definición top-level
            for stop in ["\n\nclass ", "\n\ndef "]:
                if stop in decoded:
                    decoded = decoded[: decoded.index(stop)]
            return decoded

        # Demasiadas líneas vacías seguidas → terminó
        if "\n\n\n" in decoded:
            decoded = decoded[: decoded.index("\n\n\n")]
            return decoded

    return tokenizer.Decode(generados[len(ids) :])


# =============================================================================
# Ejecución segura con timeout
# =============================================================================


class TimeoutError(Exception):
    """Señal de timeout."""


def _timeout_handler(signum, frame):
    raise TimeoutError("Timeout")


def ejecutar_con_timeout(code: str, timeout_sec: int = 5) -> tuple[bool, str]:
    """
    Ejecuta código Python con timeout.

    Returns:
        (passed, error_msg)
    """
    # En Windows no hay signal.SIGALRM, usar threading
    if sys.platform == "win32":
        import threading

        result: dict[str, Any] = {"passed": False, "error": "timeout"}

        def _run():
            try:
                ns: dict[str, Any] = {}
                exec(compile(code, "<humaneval>", "exec"), ns)
                result["passed"] = True
                result["error"] = ""
            except AssertionError as e:
                result["error"] = f"AssertionError: {e}"
            except Exception as e:
                result["error"] = f"{type(e).__name__}: {e}"

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=timeout_sec)
        if t.is_alive():
            return False, "timeout"
        return result["passed"], result["error"]
    else:
        # Unix: usar SIGALRM
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_sec)
        try:
            ns: dict[str, Any] = {}
            exec(compile(code, "<humaneval>", "exec"), ns)
            signal.alarm(0)
            return True, ""
        except TimeoutError:
            return False, "timeout"
        except AssertionError as e:
            signal.alarm(0)
            return False, f"AssertionError: {e}"
        except Exception as e:
            signal.alarm(0)
            return False, f"{type(e).__name__}: {e}"
        finally:
            signal.signal(signal.SIGALRM, old_handler)


# =============================================================================
# Evaluación pass@k
# =============================================================================


def evaluar_problema(
    modelo,
    tokenizer,
    device: torch.device,
    problema: dict,
    n_samples: int = 1,
    temperature: float = 0.2,
    max_tokens: int = 512,
    verbose: bool = False,
) -> dict:
    """Genera n_samples completions y evalúa cada una."""
    task_id = problema["task_id"]
    prompt = problema["prompt"]
    test_code = problema["test"]
    entry_point = problema["entry_point"]

    resultados = []

    for s in range(n_samples):
        temp = 0.0 if n_samples == 1 else temperature
        completion = generar(modelo, tokenizer, prompt, device, max_tokens, temp)

        # Construir código completo: prompt + completion + tests
        full_code = prompt + completion + "\n\n" + test_code
        # HumanEval tests llaman check(entry_point), agregar la llamada
        full_code += f"\n\ncheck({entry_point})\n"

        passed, error = ejecutar_con_timeout(full_code, timeout_sec=10)

        if verbose:
            status = "PASS" if passed else f"FAIL ({error[:60]})"
            print(f"    sample {s}: {status}")
            if not passed:
                # Mostrar solo el completion generado
                for line in completion.split("\n")[:15]:
                    print(f"      {line}")

        resultados.append(
            {
                "sample": s,
                "passed": passed,
                "error": error,
                "completion_len": len(completion),
            }
        )

    any_passed = any(r["passed"] for r in resultados)
    return {
        "task_id": task_id,
        "entry_point": entry_point,
        "passed": any_passed,
        "n_samples": n_samples,
        "pass_count": sum(1 for r in resultados if r["passed"]),
        "samples": resultados,
    }


# =============================================================================
# pass@k estimator (unbiased, from Chen et al. 2021)
# =============================================================================


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Estimador insesgado de pass@k.

    n: total de muestras por problema
    c: cantidad de muestras correctas
    k: k para pass@k
    """
    if n - c < k:
        return 1.0
    result = 1.0
    for i in range(k):
        result *= (n - c - i) / (n - i)
    return 1.0 - result


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="HumanEval benchmark para PamparV3")
    parser.add_argument("--checkpoint", default="checkpoints/v3_train.pt")
    parser.add_argument(
        "--samples-per-task",
        type=int,
        default=1,
        help="Muestras por problema (1=pass@1 determinista)",
    )
    parser.add_argument("--temp", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--limit", type=int, default=0, help="Limitar a N problemas (0=todos)"
    )
    parser.add_argument("--output", default="benchmarks/humaneval_results.json")
    args = parser.parse_args()

    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint = Path(args.checkpoint)

    print(f"\n{'═' * 65}")
    print(f"  BENCHMARK HumanEval — PamparV3")
    print(f"  Checkpoint : {checkpoint.name}")
    print(f"  Device     : {device}")
    print(f"  Samples/task: {args.samples_per_task}  |  Temp: {args.temp}")
    print(f"{'═' * 65}\n")

    # Cargar modelo
    print("  Cargando modelo...", end=" ", flush=True)
    t0 = time.time()
    modelo, config = cargar_modelo(checkpoint, device)
    tokenizer = cargar_tokenizer(config.vocab_size)
    n_params = sum(p.numel() for p in modelo.parameters()) / 1e6
    print(f"OK ({n_params:.1f}M params, {time.time() - t0:.1f}s)")

    # Cargar HumanEval
    print("  Descargando HumanEval...", end=" ", flush=True)
    problemas = cargar_humaneval()
    if args.limit > 0:
        problemas = problemas[: args.limit]
        print(f"(limitado a {args.limit})")

    # Evaluar
    print(f"\n  Evaluando {len(problemas)} problemas...\n")
    results = []
    t_start = time.time()

    for i, prob in enumerate(problemas, 1):
        print(
            f"  [{i:03d}/{len(problemas)}] {prob['task_id']:30s}",
            end="  ",
            flush=True,
        )
        t_prob = time.time()

        try:
            res = evaluar_problema(
                modelo,
                tokenizer,
                device,
                prob,
                n_samples=args.samples_per_task,
                temperature=args.temp,
                max_tokens=args.max_tokens,
                verbose=args.verbose,
            )
        except Exception as e:
            traceback.print_exc()
            res = {
                "task_id": prob["task_id"],
                "entry_point": prob["entry_point"],
                "passed": False,
                "n_samples": args.samples_per_task,
                "pass_count": 0,
                "samples": [],
                "error": str(e),
            }

        dt = time.time() - t_prob
        icon = "✅" if res["passed"] else "❌"
        pc = res["pass_count"]
        ns = res["n_samples"]
        print(f"[{dt:.1f}s] {icon} {pc}/{ns}")
        results.append(res)

    elapsed = time.time() - t_start

    # Calcular pass@k
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    pass1 = passed / total * 100 if total > 0 else 0.0

    # pass@k insesgado si n_samples > 1
    if args.samples_per_task > 1:
        pass_at_1_unbiased = (
            sum(pass_at_k(r["n_samples"], r["pass_count"], 1) for r in results)
            / total
            * 100
        )
    else:
        pass_at_1_unbiased = pass1

    # Resultados
    print(f"\n{'═' * 65}")
    print(f"  RESULTADO HumanEval — {elapsed:.0f}s total")
    print(f"{'═' * 65}")
    print(f"  Problemas evaluados : {total}")
    print(f"  pass@1              : {pass1:.1f}%  ({passed}/{total})")
    if args.samples_per_task > 1:
        print(f"  pass@1 (unbiased)   : {pass_at_1_unbiased:.1f}%")
    print(f"  Modelo              : {n_params:.1f}M params")
    print(f"  Device              : {device}")
    print(f"{'═' * 65}\n")

    # Guardar resultados
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "model": "PAMPAr-Coder V3",
        "params_m": round(n_params, 1),
        "checkpoint": str(checkpoint),
        "benchmark": "HumanEval",
        "n_problems": total,
        "samples_per_task": args.samples_per_task,
        "temperature": args.temp,
        "pass_at_1_pct": round(pass1, 2),
        "pass_at_1_unbiased_pct": round(pass_at_1_unbiased, 2),
        "passed": passed,
        "total": total,
        "elapsed_sec": round(elapsed, 1),
        "device": str(device),
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "results": results,
    }

    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"  Resultados guardados en: {output_path}\n")


if __name__ == "__main__":
    main()
