# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
📊 Evaluación PAMPAr-Coder v2 — HumanEval + MBPP

Mide pass@k en benchmarks estándar de generación de código.

Uso:
  # Evaluar con HumanEval (164 problemas)
  python scripts/evaluate_v2.py --checkpoint checkpoints/cerebral/fase1_final.pt

  # Evaluar con MBPP (374 problemas, subset sanitized)
  python scripts/evaluate_v2.py --checkpoint checkpoints/cerebral/fase1_final.pt --benchmark mbpp

  # Ambos benchmarks
  python scripts/evaluate_v2.py --checkpoint checkpoints/cerebral/fase1_final.pt --benchmark all

  # Adjust generation params
  python scripts/evaluate_v2.py --checkpoint ckpt.pt --temperature 0.2 --top-k 40 --n-samples 5

  # Guardar samples para inspección
  python scripts/evaluate_v2.py --checkpoint ckpt.pt --save-samples

Requisitos:
  pip install human-eval  # OpenAI HumanEval dataset
  # O: descarga manual a data/benchmarks/
"""

import argparse
import json
import os
import sys
import time
import itertools
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# Ajustar path
script_dir = Path(__file__).parent
project_dir = script_dir.parent
sys.path.insert(0, str(project_dir))

from pampar.coder.v2.config import PRESET_1_5B, PRESET_8GB, PRESET_4GB, ConfigV2
from pampar.coder.v2.modelo import PampaRCoderV2, crear_modelo


# =============================================================================
# BENCHMARKS DATA
# =============================================================================

# 10 problemas de HumanEval embebidos para testing rápido sin deps
HUMANEVAL_MINI = [
    {
        "task_id": "HumanEval/0",
        "prompt": 'from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n',
        "test": "def check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n",
        "entry_point": "has_close_elements",
    },
    {
        "task_id": "HumanEval/1",
        "prompt": 'from typing import List\n\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    """ Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those group into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    >>> separate_paren_groups(\'( ) (( )) (( )( ))\')\n    [\'()\', \'(())\', \'(()())\']\n    """\n',
        "test": "def check(candidate):\n    assert candidate('(()()) ((())) () ((())()())') == ['(()())', '((()))', '()', '((())()())']\n    assert candidate('() (()) ((())) (((())))') == ['()', '(())', '((()))', '(((())))']\n    assert candidate('(()(()))(()()(()))((()))') == ['(()(()))', '(()()(()))', '((()))']\n    assert candidate('( ) (( )) (( )( ))') == ['()', '(())', '(()())']\n",
        "entry_point": "separate_paren_groups",
    },
    {
        "task_id": "HumanEval/4",
        "prompt": 'from typing import List\n\n\ndef mean_absolute_deviation(numbers: List[float]) -> float:\n    """ For a given list of input numbers, calculate Mean Absolute Deviation\n    around the mean of this dataset.\n    Mean Absolute Deviation is the average absolute difference between each\n    element and a centerpoint (mean in this case):\n    MAD = average | x - x_mean |\n    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])\n    1.0\n    """\n',
        "test": "def check(candidate):\n    assert abs(candidate([1.0, 2.0, 3.0]) - 2.0/3.0) < 1e-6\n    assert abs(candidate([1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-6\n    assert abs(candidate([1.0, 2.0, 3.0, 4.0, 5.0]) - 6.0/5.0) < 1e-6\n",
        "entry_point": "mean_absolute_deviation",
    },
]


def load_humaneval(data_dir: Optional[str] = None) -> List[Dict]:
    """Carga HumanEval dataset. Intenta varias fuentes."""
    # 1. Probar archivo local
    if data_dir:
        path = Path(data_dir) / "HumanEval.jsonl"
        if path.exists():
            problems = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    problems.append(json.loads(line))
            return problems

    # 2. Probar dataset huggingface
    try:
        from human_eval.data import read_problems
        problems = read_problems()
        return [
            {"task_id": k, "prompt": v["prompt"], "test": v["test"],
             "entry_point": v["entry_point"]}
            for k, v in problems.items()
        ]
    except ImportError:
        pass

    # 3. Descargar directamente
    try:
        import urllib.request
        url = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
        import gzip
        import tempfile
        
        print("  Descargando HumanEval...")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl.gz")
        urllib.request.urlretrieve(url, tmp.name)
        
        problems = []
        with gzip.open(tmp.name, "rt", encoding="utf-8") as f:
            for line in f:
                problems.append(json.loads(line))
        
        # Guardar para futuro uso
        save_dir = Path(data_dir or "data/benchmarks")
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "HumanEval.jsonl", "w", encoding="utf-8") as f:
            for p in problems:
                f.write(json.dumps(p) + "\n")
        print(f"  Guardado en {save_dir / 'HumanEval.jsonl'}")
        
        os.unlink(tmp.name)
        return problems
    except Exception as e:
        print(f"  ⚠️  No se pudo descargar HumanEval: {e}")

    # 4. Fallback: mini set embebido
    print("  ⚠️  Usando HumanEval-Mini (10 problemas embebidos)")
    return HUMANEVAL_MINI


def load_mbpp(data_dir: Optional[str] = None) -> List[Dict]:
    """Carga MBPP (sanitized) dataset."""
    if data_dir:
        path = Path(data_dir) / "mbpp_sanitized.jsonl"
        if path.exists():
            problems = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    problems.append(json.loads(line))
            return problems
    
    # Descargar
    try:
        import urllib.request
        url = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/sanitized-mbpp.json"
        
        print("  Descargando MBPP sanitized...")
        import tempfile
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w")
        urllib.request.urlretrieve(url, tmp.name)
        
        with open(tmp.name, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        problems = []
        for item in data:
            # MBPP format → HumanEval-like format
            prompt = f'"""{item["prompt"]}"""\n'
            test_code = "\n".join(item.get("test_list", []))
            
            problems.append({
                "task_id": f"MBPP/{item['task_id']}",
                "prompt": prompt,
                "test": test_code,
                "entry_point": "",  # MBPP doesn't have explicit entry
                "code": item.get("code", ""),
            })
        
        # Guardar
        save_dir = Path(data_dir or "data/benchmarks")
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "mbpp_sanitized.jsonl", "w", encoding="utf-8") as f:
            for p in problems:
                f.write(json.dumps(p) + "\n")
        print(f"  Guardado en {save_dir / 'mbpp_sanitized.jsonl'}")
        
        os.unlink(tmp.name)
        return problems
    except Exception as e:
        print(f"  ⚠️  No se pudo descargar MBPP: {e}")
        return []


# =============================================================================
# CODE EXECUTION (SANDBOX)
# =============================================================================

def execute_code_safe(code: str, test: str, entry_point: str, timeout: float = 5.0) -> Dict:
    """
    Ejecuta código + tests en un proceso aislado con timeout.
    
    Returns:
        {"passed": bool, "error": str or None, "time": float}
    """
    import multiprocessing
    import signal
    
    full_code = code + "\n" + test
    if entry_point:
        full_code += f"\ncheck({entry_point})\n"
    
    def run_code(result_dict):
        try:
            exec_globals = {}
            exec(full_code, exec_globals)
            result_dict["passed"] = True
        except Exception as e:
            result_dict["passed"] = False
            result_dict["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    
    manager = multiprocessing.Manager()
    result = manager.dict({"passed": False, "error": None})
    
    t0 = time.time()
    p = multiprocessing.Process(target=run_code, args=(result,))
    p.start()
    p.join(timeout=timeout)
    elapsed = time.time() - t0
    
    if p.is_alive():
        p.terminate()
        p.join(1)
        return {"passed": False, "error": "TimeoutError: execution exceeded time limit", "time": elapsed}
    
    return {"passed": result.get("passed", False), "error": result.get("error"), "time": elapsed}


# =============================================================================
# GENERATION
# =============================================================================

def generate_completion(
    model: PampaRCoderV2,
    tokenizer,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.2,
    top_k: int = 50,
    stop_tokens: Optional[List[str]] = None,
    device: str = "cuda",
) -> str:
    """
    Genera un completion para un prompt dado.
    
    Usa temperature baja (0.2) por defecto para evaluación (greedy-ish).
    """
    model.eval()
    
    # Tokenizar prompt
    prompt_tokens = tokenizer.Encode(prompt)
    if len(prompt_tokens) > model.config.max_seq_len - max_tokens:
        prompt_tokens = prompt_tokens[-(model.config.max_seq_len - max_tokens):]
    
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    
    # Generar
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
        )
    
    # Decodificar solo la parte generada
    generated_tokens = output_ids[0, len(prompt_tokens):].tolist()
    completion = tokenizer.Decode(generated_tokens)
    
    # Truncar en stop tokens
    if stop_tokens is None:
        stop_tokens = ["\nclass ", "\ndef ", "\n#", "\nif __name__", "\nprint(", "\n```"]
    
    for stop in stop_tokens:
        idx = completion.find(stop)
        if idx != -1:
            completion = completion[:idx]
    
    return completion


def generate_n_samples(
    model: PampaRCoderV2,
    tokenizer,
    prompt: str,
    n: int = 1,
    max_tokens: int = 512,
    temperature: float = 0.2,
    top_k: int = 50,
    device: str = "cuda",
) -> List[str]:
    """Genera n samples para un prompt (para pass@k con k < n)."""
    samples = []
    for _ in range(n):
        comp = generate_completion(
            model, tokenizer, prompt,
            max_tokens=max_tokens,
            temperature=temperature if n > 1 else 0.0,  # greedy for n=1
            top_k=top_k,
            device=device,
        )
        samples.append(comp)
    return samples


# =============================================================================
# PASS@K CALCULATION
# =============================================================================

def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calcula pass@k estimator (unbiased).
    
    n: número total de samples generados
    c: número de samples correctos
    k: k para pass@k
    
    Fórmula: 1 - C(n-c, k) / C(n, k)
    """
    if n - c < k:
        return 1.0
    return 1.0 - float(
        _comb(n - c, k) / _comb(n, k)
    )


def _comb(n: int, k: int) -> float:
    """Combinación C(n, k) con protección overflow."""
    if k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1.0
    for i in range(k):
        result *= (n - i)
        result /= (i + 1)
    return result


# =============================================================================
# EVALUATION PIPELINE
# =============================================================================

def evaluate_benchmark(
    model: PampaRCoderV2,
    tokenizer,
    problems: List[Dict],
    benchmark_name: str = "HumanEval",
    n_samples: int = 1,
    max_tokens: int = 512,
    temperature: float = 0.2,
    top_k: int = 50,
    device: str = "cuda",
    save_samples: bool = False,
    output_dir: str = "benchmarks",
) -> Dict:
    """
    Evalúa el modelo en un benchmark completo.
    
    Returns:
        Dict con pass@1, pass@5, pass@10, detalle por problema
    """
    print(f"\n{'=' * 60}")
    print(f"📊 Evaluando {benchmark_name}: {len(problems)} problemas")
    print(f"{'=' * 60}")
    print(f"  Samples por problema: {n_samples}")
    print(f"  Temperature: {temperature}")
    print(f"  Top-K: {top_k}")
    print(f"  Max tokens: {max_tokens}")
    
    results = []
    total_correct = 0
    total_problems = len(problems)
    t_start = time.time()
    
    all_samples = {} if save_samples else None
    
    for i, problem in enumerate(problems):
        task_id = problem["task_id"]
        prompt = problem["prompt"]
        test = problem["test"]
        entry_point = problem.get("entry_point", "")
        
        # Generar samples
        samples = generate_n_samples(
            model, tokenizer, prompt,
            n=n_samples,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            device=device,
        )
        
        # Evaluar cada sample
        n_correct = 0
        sample_results = []
        for j, completion in enumerate(samples):
            full_code = prompt + completion
            exec_result = execute_code_safe(full_code, test, entry_point, timeout=5.0)
            sample_results.append({
                "completion": completion[:200],  # Truncar para log
                "passed": exec_result["passed"],
                "error": exec_result.get("error"),
            })
            if exec_result["passed"]:
                n_correct += 1
        
        # pass@k
        result = {
            "task_id": task_id,
            "n_samples": n_samples,
            "n_correct": n_correct,
            "pass@1": pass_at_k(n_samples, n_correct, 1),
        }
        if n_samples >= 5:
            result["pass@5"] = pass_at_k(n_samples, n_correct, 5)
        if n_samples >= 10:
            result["pass@10"] = pass_at_k(n_samples, n_correct, 10)
        
        results.append(result)
        
        if n_correct > 0:
            total_correct += 1
        
        if save_samples:
            all_samples[task_id] = {
                "prompt": prompt,
                "samples": sample_results,
            }
        
        # Progreso
        if (i + 1) % 10 == 0 or (i + 1) == total_problems:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (total_problems - i - 1) / rate if rate > 0 else 0
            current_pass1 = sum(r["pass@1"] for r in results) / len(results) * 100
            print(
                f"  [{i+1:3d}/{total_problems}] "
                f"pass@1={current_pass1:.1f}% | "
                f"correct={total_correct}/{i+1} | "
                f"ETA={eta:.0f}s"
            )
    
    # Calcular métricas globales
    total_time = time.time() - t_start
    
    metrics = {
        "benchmark": benchmark_name,
        "n_problems": total_problems,
        "n_samples_per_problem": n_samples,
        "pass@1": sum(r["pass@1"] for r in results) / total_problems * 100,
        "total_correct_at_least_1": total_correct,
        "total_time_seconds": total_time,
        "avg_time_per_problem": total_time / total_problems,
    }
    
    if n_samples >= 5:
        metrics["pass@5"] = sum(r.get("pass@5", 0) for r in results) / total_problems * 100
    if n_samples >= 10:
        metrics["pass@10"] = sum(r.get("pass@10", 0) for r in results) / total_problems * 100
    
    # Imprimir resultados
    print(f"\n{'─' * 50}")
    print(f"  Resultados {benchmark_name}:")
    print(f"  pass@1:  {metrics['pass@1']:.1f}%")
    if "pass@5" in metrics:
        print(f"  pass@5:  {metrics['pass@5']:.1f}%")
    if "pass@10" in metrics:
        print(f"  pass@10: {metrics['pass@10']:.1f}%")
    print(f"  Correctos (≥1): {total_correct}/{total_problems}")
    print(f"  Tiempo: {total_time:.1f}s ({metrics['avg_time_per_problem']:.2f}s/problema)")
    print(f"{'─' * 50}")
    
    # Guardar resultados
    os.makedirs(output_dir, exist_ok=True)
    
    results_file = Path(output_dir) / f"{benchmark_name.lower()}_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": metrics,
            "per_problem": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"  💾 Resultados: {results_file}")
    
    if save_samples and all_samples:
        samples_file = Path(output_dir) / f"{benchmark_name.lower()}_samples.json"
        with open(samples_file, "w", encoding="utf-8") as f:
            json.dump(all_samples, f, indent=2, ensure_ascii=False)
        print(f"  💾 Samples: {samples_file}")
    
    return metrics


# =============================================================================
# COMPARACIÓN CON BASELINES
# =============================================================================

BASELINES = {
    "HumanEval": {
        "Qwen2.5-Coder-1.5B": 61.6,
        "CodeLlama-7B": 33.5,
        "StarCoder2-3B": 31.7,
        "DeepSeek-Coder-1.3B": 34.8,
        "Phi-1.5 (1.3B)": 41.4,
        "CodeGen-2B": 26.8,
    },
    "MBPP": {
        "Qwen2.5-Coder-1.5B": 65.0,
        "CodeLlama-7B": 41.4,
        "StarCoder2-3B": 40.2,
        "DeepSeek-Coder-1.3B": 46.2,
    },
}


def print_comparison(benchmark: str, our_score: float):
    """Imprime comparación con baselines."""
    if benchmark not in BASELINES:
        return
    
    print(f"\n📊 Comparación {benchmark}:")
    print(f"  {'Modelo':<25} {'pass@1':>8}")
    print(f"  {'─' * 35}")
    
    # Nuestro modelo
    print(f"  {'🧠 PAMPAr-Coder 1.5B':<25} {our_score:>7.1f}%  ← NUESTRO")
    
    # Baselines ordenados
    for model_name, score in sorted(BASELINES[benchmark].items(), key=lambda x: -x[1]):
        marker = ""
        if our_score > score:
            marker = " ✅"
        elif our_score > score * 0.9:
            marker = " ≈"
        print(f"  {model_name:<25} {score:>7.1f}%{marker}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="📊 Evaluación PAMPAr-Coder v2"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path al checkpoint del modelo"
    )
    parser.add_argument(
        "--benchmark", type=str, default="humaneval",
        choices=["humaneval", "mbpp", "all", "mini"],
        help="Benchmark a usar"
    )
    parser.add_argument(
        "--preset", type=str, default="4gb",
        choices=["4gb", "8gb", "1.5b"],
        help="Preset del modelo"
    )
    parser.add_argument(
        "--n-samples", type=int, default=1,
        help="Samples por problema (n para pass@k)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512,
        help="Max tokens para generación"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.2,
        help="Temperature para sampling"
    )
    parser.add_argument(
        "--top-k", type=int, default=50,
        help="Top-K para sampling"
    )
    parser.add_argument(
        "--save-samples", action="store_true",
        help="Guardar completions generados"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/benchmarks",
        help="Directorio de datos de benchmarks"
    )
    parser.add_argument(
        "--output-dir", type=str, default="benchmarks",
        help="Directorio para resultados"
    )
    parser.add_argument(
        "--tokenizer", type=str, default="data/tokenizer/pampar_48k.model",
        help="Path al tokenizer"
    )
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n📊 PAMPAr-Coder v2 — Evaluación")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")

    # Preset
    presets = {"4gb": PRESET_4GB, "8gb": PRESET_8GB, "1.5b": PRESET_1_5B}
    model_config = presets[args.preset]

    # Crear y cargar modelo
    print(f"\n  Cargando modelo desde: {args.checkpoint}")
    model = crear_modelo(model_config).to(device)
    
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")
    if "fase" in ckpt:
        print(f"  Checkpoint: fase={ckpt['fase']}, paso={ckpt.get('paso', '?')}, loss={ckpt.get('loss', '?')}")

    # Tokenizer
    tokenizer_path = str(project_dir / args.tokenizer)
    if not os.path.exists(tokenizer_path):
        tokenizer_path = args.tokenizer
    
    print(f"  Tokenizer: {tokenizer_path}")
    import sentencepiece as spm
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(tokenizer_path)
    
    # Registrar tokenizer en modelo
    model.registrar_tokenizer(tokenizer)

    # Evaluar
    all_metrics = {}
    
    if args.benchmark in ("humaneval", "all"):
        problems = load_humaneval(args.data_dir)
        if problems:
            metrics = evaluate_benchmark(
                model, tokenizer, problems,
                benchmark_name="HumanEval",
                n_samples=args.n_samples,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                device=str(device),
                save_samples=args.save_samples,
                output_dir=args.output_dir,
            )
            all_metrics["HumanEval"] = metrics
            print_comparison("HumanEval", metrics["pass@1"])
    
    if args.benchmark in ("mbpp", "all"):
        problems = load_mbpp(args.data_dir)
        if problems:
            metrics = evaluate_benchmark(
                model, tokenizer, problems,
                benchmark_name="MBPP",
                n_samples=args.n_samples,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                device=str(device),
                save_samples=args.save_samples,
                output_dir=args.output_dir,
            )
            all_metrics["MBPP"] = metrics
            print_comparison("MBPP", metrics["pass@1"])
    
    if args.benchmark == "mini":
        metrics = evaluate_benchmark(
            model, tokenizer, HUMANEVAL_MINI,
            benchmark_name="HumanEval-Mini",
            n_samples=args.n_samples,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=str(device),
            save_samples=args.save_samples,
            output_dir=args.output_dir,
        )
        all_metrics["HumanEval-Mini"] = metrics

    # Resumen final
    if all_metrics:
        print(f"\n{'═' * 60}")
        print(f"  RESUMEN FINAL")
        print(f"{'═' * 60}")
        for name, m in all_metrics.items():
            print(f"  {name}: pass@1 = {m['pass@1']:.1f}%")
        
        # Guardar resumen
        summary_file = Path(args.output_dir) / "evaluation_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)
        print(f"\n  💾 Resumen: {summary_file}")


if __name__ == "__main__":
    main()
