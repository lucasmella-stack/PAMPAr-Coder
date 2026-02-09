# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Benchmark suite para PAMPAr-Coder.

Compara PAMPAr-Coder contra:
1. Transformer vanilla (mismo tamaño)
2. Sí mismo con/sin Early Exit
3. Sí mismo con diferentes pesos de LLAVES

Genera gráficas de:
- Velocidad de inferencia (tokens/sec)
- Calidad (perplexity)
- Uso de VRAM
- Distribución de territorios
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from pampar.coder import PampaRCoder, crear_modelo, CODER_4GB, ConfigPampaRCoder

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Para sistemas sin display
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️ matplotlib no encontrado, gráficas deshabilitadas")
    print("   Instala: pip install matplotlib")


# =============================================================================
# Transformer Vanilla (baseline)
# =============================================================================

class TransformerVanilla(nn.Module):
    """
    Transformer vanilla para comparación.
    Mismo número de parámetros que PAMPAr-Coder pero arquitectura estándar.
    """
    
    def __init__(self, vocab_size: int, dim: int, n_heads: int, n_layers: int, 
                 max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # LM Head
        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # Weight tying
        
        # Causal mask
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer('causal_mask', mask)
    
    def forward(self, input_ids, targets=None):
        B, L = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        tok_emb = self.tok_emb(input_ids)
        pos = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_emb(pos)
        x = self.dropout(tok_emb + pos_emb)
        
        # Transformer
        mask = self.causal_mask[:L, :L]
        x = self.transformer(x, mask=mask)
        
        # LM Head
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-100
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=50, temperature=0.8):
        self.eval()
        generated = prompt_ids.clone()
        
        for _ in range(max_new_tokens):
            context = generated[:, -self.max_seq_len:]
            logits, _ = self.forward(context)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Benchmark Tests
# =============================================================================

@dataclass
class BenchmarkResult:
    """Resultado de un benchmark."""
    name: str
    tokens_per_sec: float
    perplexity: float
    vram_mb: float
    params: int
    extra: Dict = None


def benchmark_inference_speed(
    model: nn.Module,
    device: torch.device,
    vocab_size: int,
    seq_len: int = 64,
    new_tokens: int = 50,
    n_runs: int = 5,
    warmup_runs: int = 2
) -> float:
    """Mide velocidad de inferencia en tokens/segundo."""
    model.eval()
    model.to(device)
    
    # Warmup
    prompt = torch.randint(0, vocab_size, (1, seq_len), device=device)
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model.generate(prompt, max_new_tokens=10)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(n_runs):
        prompt = torch.randint(0, vocab_size, (1, seq_len), device=device)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(prompt, max_new_tokens=new_tokens)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    tokens_per_sec = new_tokens / avg_time
    
    return tokens_per_sec


def benchmark_perplexity(
    model: nn.Module,
    device: torch.device,
    vocab_size: int,
    n_samples: int = 100,
    seq_len: int = 128
) -> float:
    """Calcula perplexity en datos sintéticos."""
    model.eval()
    model.to(device)
    
    total_loss = 0
    n_batches = 0
    
    for _ in range(n_samples):
        input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)
        targets = torch.randint(0, vocab_size, (1, seq_len), device=device)
        
        with torch.no_grad():
            _, loss = model(input_ids, targets)
        
        if loss is not None:
            total_loss += loss.item()
            n_batches += 1
    
    avg_loss = total_loss / n_batches
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    
    return perplexity


def benchmark_vram(
    model: nn.Module,
    device: torch.device,
    vocab_size: int,
    batch_size: int = 4,
    seq_len: int = 128
) -> float:
    """Mide uso de VRAM."""
    if device.type != 'cuda':
        return 0.0
    
    model.to(device)
    torch.cuda.reset_peak_memory_stats()
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # Forward + backward
    logits, loss = model(input_ids, targets)
    loss.backward()
    
    vram_bytes = torch.cuda.max_memory_allocated()
    vram_mb = vram_bytes / (1024 * 1024)
    
    return vram_mb


def cargar_checkpoint(model, checkpoint_path: str, device: torch.device):
    """Carga pesos desde checkpoint."""
    if not os.path.exists(checkpoint_path):
        print(f"   ⚠️ Checkpoint no encontrado: {checkpoint_path}")
        return False
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    print(f"   ✅ Checkpoint cargado: {checkpoint_path}")
    return True


def run_benchmarks(device: torch.device, checkpoint_path: str = None) -> List[BenchmarkResult]:
    """Ejecuta todos los benchmarks."""
    results = []
    
    config = CODER_4GB
    vocab_size = config.vocab_size
    
    print("\n" + "=" * 70)
    print("🔬 Ejecutando Benchmarks")
    print("=" * 70)
    
    # =========================================================================
    # 1. PAMPAr-Coder ENTRENADO (si hay checkpoint)
    # =========================================================================
    if checkpoint_path and os.path.exists(checkpoint_path):
        print("\n📊 1. PAMPAr-Coder ENTRENADO")
        model_trained = crear_modelo("4GB")
        model_trained.config.usar_early_exit = True
        cargar_checkpoint(model_trained, checkpoint_path, device)
        
        speed_trained = benchmark_inference_speed(model_trained, device, vocab_size)
        ppl_trained = benchmark_perplexity(model_trained, device, vocab_size)
        vram_trained = benchmark_vram(model_trained, device, vocab_size) if device.type == 'cuda' else 0
        
        results.append(BenchmarkResult(
            name="PAMPAr-Coder (Entrenado)",
            tokens_per_sec=speed_trained,
            perplexity=ppl_trained,
            vram_mb=vram_trained,
            params=model_trained.count_parameters()['total']
        ))
        print(f"   Speed: {speed_trained:.1f} tok/s | PPL: {ppl_trained:.2f} | VRAM: {vram_trained:.0f} MB")
        
        del model_trained
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # =========================================================================
    # 2. PAMPAr-Coder SIN ENTRENAR (baseline)
    # =========================================================================
    print("\n📊 2. PAMPAr-Coder (Sin Entrenar)")
    model_ee = crear_modelo("4GB")
    model_ee.config.usar_early_exit = True
    
    speed_ee = benchmark_inference_speed(model_ee, device, vocab_size)
    ppl_ee = benchmark_perplexity(model_ee, device, vocab_size)
    vram_ee = benchmark_vram(model_ee, device, vocab_size) if device.type == 'cuda' else 0
    
    results.append(BenchmarkResult(
        name="PAMPAr-Coder (Sin Entrenar)",
        tokens_per_sec=speed_ee,
        perplexity=ppl_ee,
        vram_mb=vram_ee,
        params=model_ee.count_parameters()['total']
    ))
    print(f"   Speed: {speed_ee:.1f} tok/s | PPL: {ppl_ee:.2f} | VRAM: {vram_ee:.0f} MB")
    
    del model_ee
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # =========================================================================
    # 3. PAMPAr-Coder sin Early Exit
    # =========================================================================
    print("\n📊 3. PAMPAr-Coder (Early Exit OFF)")
    model_no_ee = crear_modelo("4GB")
    model_no_ee.config.usar_early_exit = False
    
    speed_no_ee = benchmark_inference_speed(model_no_ee, device, vocab_size)
    ppl_no_ee = benchmark_perplexity(model_no_ee, device, vocab_size)
    vram_no_ee = benchmark_vram(model_no_ee, device, vocab_size) if device.type == 'cuda' else 0
    
    results.append(BenchmarkResult(
        name="PAMPAr-Coder (No Early Exit)",
        tokens_per_sec=speed_no_ee,
        perplexity=ppl_no_ee,
        vram_mb=vram_no_ee,
        params=model_no_ee.count_parameters()['total']
    ))
    print(f"   Speed: {speed_no_ee:.1f} tok/s | PPL: {ppl_no_ee:.2f} | VRAM: {vram_no_ee:.0f} MB")
    
    del model_no_ee
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # =========================================================================
    # 4. Transformer Vanilla (baseline)
    # =========================================================================
    print("\n📊 4. Transformer Vanilla (baseline)")
    model_vanilla = TransformerVanilla(
        vocab_size=vocab_size,
        dim=config.dim,
        n_heads=config.n_heads,
        n_layers=config.n_capas * 2,  # Más capas para igualar parámetros aprox
        max_seq_len=config.max_seq_len,
        dropout=config.dropout
    )
    
    speed_vanilla = benchmark_inference_speed(model_vanilla, device, vocab_size)
    ppl_vanilla = benchmark_perplexity(model_vanilla, device, vocab_size)
    vram_vanilla = benchmark_vram(model_vanilla, device, vocab_size) if device.type == 'cuda' else 0
    
    results.append(BenchmarkResult(
        name="Transformer Vanilla",
        tokens_per_sec=speed_vanilla,
        perplexity=ppl_vanilla,
        vram_mb=vram_vanilla,
        params=model_vanilla.count_parameters()
    ))
    print(f"   Speed: {speed_vanilla:.1f} tok/s | PPL: {ppl_vanilla:.2f} | VRAM: {vram_vanilla:.0f} MB")
    
    del model_vanilla
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # =========================================================================
    # 5. PAMPAr-Coder con diferentes pesos de LLAVES
    # =========================================================================
    for llaves_peso in [0.5, 0.8, 0.95]:
        print(f"\n📊 5. PAMPAr-Coder (LLAVES {int(llaves_peso*100)}%)")
        
        custom_config = ConfigPampaRCoder(
            vocab_size=vocab_size,
            dim=config.dim,
            n_heads=config.n_heads,
            n_capas=config.n_capas,
            max_seq_len=config.max_seq_len,
            peso_llaves=llaves_peso,
            usar_early_exit=True,
        )
        model_llaves = PampaRCoder(custom_config)
        
        speed = benchmark_inference_speed(model_llaves, device, vocab_size)
        ppl = benchmark_perplexity(model_llaves, device, vocab_size)
        
        results.append(BenchmarkResult(
            name=f"PAMPAr (LLAVES {int(llaves_peso*100)}%)",
            tokens_per_sec=speed,
            perplexity=ppl,
            vram_mb=0,
            params=model_llaves.count_parameters()['total'],
            extra={'llaves_peso': llaves_peso}
        ))
        print(f"   Speed: {speed:.1f} tok/s | PPL: {ppl:.2f}")
        
        del model_llaves
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return results


# =============================================================================
# Plotting
# =============================================================================

def plot_results(results: List[BenchmarkResult], output_dir: Path):
    """Genera gráficas de los resultados."""
    if not HAS_MATPLOTLIB:
        print("⚠️ matplotlib no disponible, saltando gráficas")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Colores
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c']
    
    # =========================================================================
    # 1. Velocidad de inferencia
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = [r.name for r in results]
    speeds = [r.tokens_per_sec for r in results]
    
    bars = ax.barh(names, speeds, color=colors[:len(names)])
    ax.set_xlabel('Tokens por segundo', fontsize=12)
    ax.set_title('🚀 Velocidad de Inferencia', fontsize=14, fontweight='bold')
    
    # Añadir valores en las barras
    for bar, speed in zip(bars, speeds):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{speed:.1f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'speed_comparison.png', dpi=150)
    plt.close()
    print(f"   📈 Guardado: {output_dir / 'speed_comparison.png'}")
    
    # =========================================================================
    # 2. Comparación Speed vs Parameters
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    params = [r.params / 1e6 for r in results]  # En millones
    
    scatter = ax.scatter(params, speeds, c=colors[:len(results)], s=200, alpha=0.7)
    
    for i, r in enumerate(results):
        ax.annotate(r.name, (params[i], speeds[i]), 
                   textcoords="offset points", xytext=(5, 5), fontsize=9)
    
    ax.set_xlabel('Parámetros (millones)', fontsize=12)
    ax.set_ylabel('Tokens por segundo', fontsize=12)
    ax.set_title('⚡ Eficiencia: Velocidad vs Tamaño', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'efficiency_comparison.png', dpi=150)
    plt.close()
    print(f"   📈 Guardado: {output_dir / 'efficiency_comparison.png'}")
    
    # =========================================================================
    # 3. Impacto de LLAVES
    # =========================================================================
    llaves_results = [r for r in results if r.extra and 'llaves_peso' in r.extra]
    
    if llaves_results:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        llaves_pesos = [r.extra['llaves_peso'] * 100 for r in llaves_results]
        llaves_speeds = [r.tokens_per_sec for r in llaves_results]
        
        ax.plot(llaves_pesos, llaves_speeds, 'o-', linewidth=2, markersize=10, color='#2ecc71')
        ax.fill_between(llaves_pesos, llaves_speeds, alpha=0.3, color='#2ecc71')
        
        ax.set_xlabel('Peso LLAVES (%)', fontsize=12)
        ax.set_ylabel('Tokens por segundo', fontsize=12)
        ax.set_title('🔑 Impacto del Sistema LLAVES', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'llaves_impact.png', dpi=150)
        plt.close()
        print(f"   📈 Guardado: {output_dir / 'llaves_impact.png'}")
    
    # =========================================================================
    # 4. Resumen general
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Speed comparison
    ax1 = axes[0]
    ax1.barh(names[:3], speeds[:3], color=colors[:3])
    ax1.set_xlabel('Tokens/segundo')
    ax1.set_title('Velocidad')
    
    # VRAM comparison (si hay datos)
    ax2 = axes[1]
    vrams = [r.vram_mb for r in results[:3]]
    if any(v > 0 for v in vrams):
        ax2.barh(names[:3], vrams, color=colors[:3])
        ax2.set_xlabel('VRAM (MB)')
        ax2.set_title('Uso de Memoria')
    else:
        ax2.text(0.5, 0.5, 'VRAM data\nnot available\n(CPU mode)', 
                ha='center', va='center', fontsize=12)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
    
    plt.suptitle('📊 PAMPAr-Coder Benchmark Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'summary.png', dpi=150)
    plt.close()
    print(f"   📈 Guardado: {output_dir / 'summary.png'}")


def print_results_table(results: List[BenchmarkResult]):
    """Imprime tabla de resultados."""
    print("\n" + "=" * 80)
    print("📊 RESULTADOS DEL BENCHMARK")
    print("=" * 80)
    
    print(f"\n{'Modelo':<35} {'Tok/s':>10} {'PPL':>10} {'VRAM':>10} {'Params':>12}")
    print("-" * 80)
    
    for r in results:
        vram_str = f"{r.vram_mb:.0f} MB" if r.vram_mb > 0 else "N/A"
        params_str = f"{r.params/1e6:.1f}M"
        print(f"{r.name:<35} {r.tokens_per_sec:>10.1f} {r.perplexity:>10.2f} {vram_str:>10} {params_str:>12}")
    
    print("-" * 80)
    
    # Análisis
    print("\n📈 ANÁLISIS:")
    
    # Mejor velocidad
    fastest = max(results, key=lambda r: r.tokens_per_sec)
    print(f"   🚀 Más rápido: {fastest.name} ({fastest.tokens_per_sec:.1f} tok/s)")
    
    # Comparación Early Exit
    ee_results = [r for r in results if 'Early Exit' in r.name]
    no_ee_results = [r for r in results if 'No Early Exit' in r.name]
    if ee_results and no_ee_results:
        speedup = ee_results[0].tokens_per_sec / no_ee_results[0].tokens_per_sec
        print(f"   ⚡ Speedup Early Exit: {speedup:.2f}x")
    
    # vs Vanilla
    pampar_results = [r for r in results if 'PAMPAr' in r.name and 'LLAVES' not in r.name]
    vanilla_results = [r for r in results if 'Vanilla' in r.name]
    if pampar_results and vanilla_results:
        speedup = pampar_results[0].tokens_per_sec / vanilla_results[0].tokens_per_sec
        print(f"   🧠 PAMPAr vs Vanilla: {speedup:.2f}x")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Benchmark PAMPAr-Coder')
    parser.add_argument('--output-dir', type=str, default='benchmarks',
                        help='Directorio para guardar resultados')
    parser.add_argument('--cpu', action='store_true',
                        help='Forzar uso de CPU')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint del modelo entrenado')
    args = parser.parse_args()
    
    print("=" * 70)
    print("🔬 PAMPAr-Coder - Benchmark Suite")
    print("=" * 70)
    
    # Device
    if args.cpu or not torch.cuda.is_available():
        device = torch.device('cpu')
        print(f"🖥️ Usando CPU")
    else:
        device = torch.device('cuda')
        print(f"🎮 Usando GPU: {torch.cuda.get_device_name()}")
    
    # Auto-detectar checkpoint si no se especifica
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        default_checkpoint = Path('checkpoints/pampar_coder_final.pt')
        if default_checkpoint.exists():
            checkpoint_path = str(default_checkpoint)
            print(f"📦 Checkpoint encontrado: {checkpoint_path}")
    
    # Run benchmarks
    results = run_benchmarks(device, checkpoint_path)
    
    # Print results
    print_results_table(results)
    
    # Plot results
    output_dir = Path(args.output_dir)
    plot_results(results, output_dir)
    
    # Save JSON
    results_json = [
        {
            'name': r.name,
            'tokens_per_sec': r.tokens_per_sec,
            'perplexity': r.perplexity,
            'vram_mb': r.vram_mb,
            'params': r.params,
            'extra': r.extra
        }
        for r in results
    ]
    
    json_path = output_dir / 'results.json'
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\n💾 Resultados guardados: {json_path}")
    
    print("\n" + "=" * 70)
    print("✅ Benchmark completado!")
    print(f"   Gráficas en: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
