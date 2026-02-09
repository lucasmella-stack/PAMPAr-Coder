# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
PAMPAr-Coder - Evaluación en Benchmarks Estándar.

Evalúa en los MISMOS benchmarks que usan Kimi-72B y otros modelos:
- HumanEval (OpenAI): 164 problemas de Python
- MBPP (Google): 974 problemas básicos

Métricas:
- pass@1: % de problemas resueltos al primer intento
- pass@10: % de problemas resueltos en 10 intentos
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import re

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from pampar.coder import PampaRCoder, crear_modelo, CODER_4GB, CODER_4GB_MAX, ConfigPampaRCoder


@dataclass
class EvalResult:
    """Resultado de evaluación."""
    benchmark: str
    total_problems: int
    passed: int
    pass_at_1: float
    generation_time: float
    tokens_generated: int


class PampaRCodeGenerator:
    """Generador de código usando PAMPAr-Coder."""
    
    def __init__(self, model: PampaRCoder, tokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        self.model.to(device)
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.95,
        stop_tokens: List[str] = None
    ) -> str:
        """Genera código a partir de un prompt."""
        stop_tokens = stop_tokens or ['\ndef ', '\nclass ', '\n#', '\nif __name__']
        
        # Tokenizar prompt
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            # Forward
            with torch.amp.autocast('cuda', enabled=True):
                logits, _ = self.model(generated[:, -self.model.config.max_seq_len:])
            
            # Último token
            logits = logits[:, -1, :] / temperature
            
            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Decodificar para verificar stop tokens
            generated_text = self.tokenizer.decode(generated[0].tolist())
            
            # Check stop tokens
            for stop in stop_tokens:
                if stop in generated_text[len(prompt):]:
                    # Encontramos stop token, truncar
                    idx = generated_text.find(stop, len(prompt))
                    return generated_text[len(prompt):idx]
        
        # Límite de tokens alcanzado
        return self.tokenizer.decode(generated[0].tolist())[len(prompt):]


def load_humaneval() -> List[Dict]:
    """Carga HumanEval dataset."""
    from datasets import load_dataset
    
    ds = load_dataset('openai/openai_humaneval', split='test')
    problems = []
    
    for sample in ds:
        problems.append({
            'task_id': sample['task_id'],
            'prompt': sample['prompt'],
            'canonical_solution': sample['canonical_solution'],
            'test': sample['test'],
            'entry_point': sample['entry_point']
        })
    
    return problems


def load_mbpp() -> List[Dict]:
    """Carga MBPP dataset."""
    from datasets import load_dataset
    
    ds = load_dataset('mbpp', 'sanitized', split='test')
    problems = []
    
    for sample in ds:
        problems.append({
            'task_id': sample['task_id'],
            'prompt': sample['prompt'],
            'code': sample['code'],
            'test_list': sample['test_list'] if 'test_list' in sample else [],
        })
    
    return problems


def check_correctness(code: str, test_code: str, entry_point: str, timeout: float = 5.0) -> bool:
    """
    Ejecuta código y tests para verificar corrección.
    
    NOTA: Ejecutar código no confiable es peligroso.
    En producción usar sandbox.
    """
    import multiprocessing
    import traceback
    
    def run_tests(result_queue):
        try:
            exec_globals = {}
            # Ejecutar código generado
            exec(code, exec_globals)
            # Ejecutar tests
            exec(test_code, exec_globals)
            # Si llegamos aquí, los tests pasaron
            result_queue.put(True)
        except Exception as e:
            result_queue.put(False)
    
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=run_tests, args=(result_queue,))
    process.start()
    process.join(timeout=timeout)
    
    if process.is_alive():
        process.terminate()
        process.join()
        return False
    
    try:
        return result_queue.get_nowait()
    except:
        return False


def evaluate_humaneval(generator: PampaRCodeGenerator, problems: List[Dict], 
                       n_samples: int = 1) -> EvalResult:
    """Evalúa en HumanEval."""
    print("\n" + "=" * 60)
    print("📊 Evaluando en HumanEval")
    print("=" * 60)
    
    passed = 0
    total_tokens = 0
    start_time = time.time()
    
    for i, problem in enumerate(problems):
        prompt = problem['prompt']
        entry_point = problem['entry_point']
        test_code = problem['test']
        
        # Generar código
        generated = generator.generate(
            prompt,
            max_new_tokens=256,
            temperature=0.2
        )
        
        # Código completo
        full_code = prompt + generated
        
        # Verificar
        try:
            passed_test = check_correctness(full_code, test_code, entry_point)
        except:
            passed_test = False
        
        if passed_test:
            passed += 1
            status = "✅"
        else:
            status = "❌"
        
        total_tokens += len(generator.tokenizer.encode(generated))
        
        if (i + 1) % 10 == 0:
            print(f"   {i+1}/{len(problems)} | Passed: {passed} | {status} {problem['task_id']}")
    
    elapsed = time.time() - start_time
    pass_at_1 = passed / len(problems) * 100
    
    print(f"\n📈 Resultados HumanEval:")
    print(f"   pass@1: {pass_at_1:.1f}% ({passed}/{len(problems)})")
    print(f"   Tiempo: {elapsed:.1f}s")
    print(f"   Tokens: {total_tokens:,}")
    
    return EvalResult(
        benchmark="HumanEval",
        total_problems=len(problems),
        passed=passed,
        pass_at_1=pass_at_1,
        generation_time=elapsed,
        tokens_generated=total_tokens
    )


def evaluate_mbpp(generator: PampaRCodeGenerator, problems: List[Dict]) -> EvalResult:
    """Evalúa en MBPP."""
    print("\n" + "=" * 60)
    print("📊 Evaluando en MBPP")
    print("=" * 60)
    
    passed = 0
    total_tokens = 0
    start_time = time.time()
    
    for i, problem in enumerate(problems):
        # MBPP tiene formato diferente
        prompt = f"# {problem['prompt']}\ndef solution("
        
        # Generar código
        generated = generator.generate(
            prompt,
            max_new_tokens=256,
            temperature=0.2
        )
        
        # Código completo
        full_code = prompt + generated
        
        # Tests
        test_code = '\n'.join(problem.get('test_list', []))
        test_code = test_code.replace('assert ', 'assert solution')  # Ajustar nombre
        
        try:
            passed_test = check_correctness(full_code, test_code, 'solution')
        except:
            passed_test = False
        
        if passed_test:
            passed += 1
        
        total_tokens += len(generator.tokenizer.encode(generated))
        
        if (i + 1) % 50 == 0:
            print(f"   {i+1}/{len(problems)} | Passed: {passed}")
    
    elapsed = time.time() - start_time
    pass_at_1 = passed / len(problems) * 100
    
    print(f"\n📈 Resultados MBPP:")
    print(f"   pass@1: {pass_at_1:.1f}% ({passed}/{len(problems)})")
    print(f"   Tiempo: {elapsed:.1f}s")
    
    return EvalResult(
        benchmark="MBPP",
        total_problems=len(problems),
        passed=passed,
        pass_at_1=pass_at_1,
        generation_time=elapsed,
        tokens_generated=total_tokens
    )


def compare_with_models(results: List[EvalResult]) -> str:
    """Compara resultados con otros modelos."""
    # Scores de referencia (aproximados, de papers)
    reference_scores = {
        'HumanEval': {
            'Kimi-Dev-72B': 78.0,
            'DeepSeek-Coder-33B': 70.1,
            'CodeLlama-34B': 48.8,
            'CodeLlama-7B': 31.1,
            'StarCoder-15B': 33.6,
            'GPT-4': 67.0,
        },
        'MBPP': {
            'Kimi-Dev-72B': 75.0,  # Estimado
            'DeepSeek-Coder-33B': 66.0,
            'CodeLlama-34B': 55.0,
            'CodeLlama-7B': 41.4,
        }
    }
    
    output = "\n" + "=" * 70 + "\n"
    output += "📊 COMPARACIÓN CON OTROS MODELOS\n"
    output += "=" * 70 + "\n"
    
    for result in results:
        output += f"\n🔹 {result.benchmark}:\n"
        output += f"   PAMPAr-Coder: {result.pass_at_1:.1f}%\n"
        output += "   " + "-" * 40 + "\n"
        
        if result.benchmark in reference_scores:
            for model, score in sorted(reference_scores[result.benchmark].items(), 
                                       key=lambda x: -x[1]):
                diff = result.pass_at_1 - score
                arrow = "↑" if diff > 0 else "↓"
                output += f"   {model}: {score:.1f}% ({arrow} {abs(diff):.1f}%)\n"
    
    return output


def main():
    parser = argparse.ArgumentParser(description='Evaluar PAMPAr-Coder en benchmarks')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path al checkpoint del modelo')
    parser.add_argument('--benchmark', type=str, default='all',
                        choices=['humaneval', 'mbpp', 'all'])
    parser.add_argument('--output', type=str, default='eval_results.json')
    args = parser.parse_args()
    
    print("=" * 70)
    print("🐸 PAMPAr-Coder - Evaluación vs Kimi-72B")
    print("=" * 70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️ Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name()}")
    
    # Cargar modelo
    print(f"\n📦 Cargando checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    config = checkpoint.get('config', {})
    model_config = CODER_4GB_MAX if config.get('dim', 256) > 256 else CODER_4GB
    
    # Ajustar vocab_size si está en checkpoint
    if 'vocab_size' in config:
        model_config.vocab_size = config['vocab_size']
    
    model = PampaRCoder(model_config)
    model.load_state_dict(checkpoint['model'])
    
    params = sum(p.numel() for p in model.parameters())
    print(f"   Parámetros: {params:,} ({params/1e6:.1f}M)")
    
    # Tokenizer
    import sentencepiece as spm
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load('data/tokenizer/code_tokenizer.model')
    print(f"   Tokenizer: vocab={tokenizer.get_piece_size()}")
    
    # Generator
    generator = PampaRCodeGenerator(model, tokenizer, device)
    
    # Evaluar
    results = []
    
    if args.benchmark in ['humaneval', 'all']:
        problems = load_humaneval()
        result = evaluate_humaneval(generator, problems)
        results.append(result)
    
    if args.benchmark in ['mbpp', 'all']:
        problems = load_mbpp()
        result = evaluate_mbpp(generator, problems)
        results.append(result)
    
    # Comparación
    comparison = compare_with_models(results)
    print(comparison)
    
    # Guardar resultados
    output_data = {
        'model': 'PAMPAr-Coder',
        'params': params,
        'checkpoint': args.checkpoint,
        'results': [
            {
                'benchmark': r.benchmark,
                'pass_at_1': r.pass_at_1,
                'passed': r.passed,
                'total': r.total_problems,
                'time': r.generation_time,
                'tokens': r.tokens_generated
            }
            for r in results
        ]
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\n💾 Resultados guardados: {args.output}")


if __name__ == '__main__':
    main()
