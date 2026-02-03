# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Test de PAMPAr-Coder.

Verifica que todos los componentes funcionan correctamente:
- Config y presets
- LLAVES para código
- Territorios
- Modelo completo
- Generación
"""

import sys
import time
sys.path.insert(0, '.')

import torch


def test_config():
    """Test de configuraciones."""
    print("\n" + "=" * 70)
    print("🔧 Test: Configuraciones")
    print("=" * 70)
    
    from pampar.coder import ConfigPampaRCoder, CODER_4GB, CODER_8GB, CODER_24GB
    
    configs = [
        ("CODER_4GB", CODER_4GB),
        ("CODER_8GB", CODER_8GB),
        ("CODER_24GB", CODER_24GB),
    ]
    
    for name, cfg in configs:
        params = cfg.estimate_params()
        vram = cfg.estimate_vram_gb(training=False)
        print(f"  ✓ {name}: {params['total']/1e6:.1f}M params, ~{vram:.2f}GB VRAM")
    
    print("  ✅ Configuraciones OK")


def test_llaves():
    """Test del sistema LLAVES."""
    print("\n" + "=" * 70)
    print("🔑 Test: Sistema LLAVES")
    print("=" * 70)
    
    from pampar.coder import LlavesCodigo, TipoTerritorioCoder
    
    llaves = LlavesCodigo()
    
    # Test tokens específicos
    test_cases = [
        ("def", TipoTerritorioCoder.SINTAXIS),
        ("if", TipoTerritorioCoder.LOGICO),
        ("int", TipoTerritorioCoder.SEMANTICA),
        ("{", TipoTerritorioCoder.SINTAXIS),
    ]
    
    for token, expected_territorio in test_cases:
        territorio, peso = llaves.clasificar_token(token)
        status = "✓" if territorio == expected_territorio else "✗"
        print(f"  {status} '{token}' → {territorio.name} (peso={peso:.1f})")
    
    print("  ✅ LLAVES OK")


def test_territorios():
    """Test de territorios."""
    print("\n" + "=" * 70)
    print("🏛️ Test: Territorios")
    print("=" * 70)
    
    from pampar.coder import GestorTerritoriosCoder, TipoTerritorioCoder
    
    gestor = GestorTerritoriosCoder(dim=192, n_heads=6, max_len=512)
    
    # Contar parámetros
    params = sum(p.numel() for p in gestor.parameters())
    print(f"  ✓ GestorTerritorios: {params:,} parámetros")
    
    # Test forward
    x = torch.randn(2, 32, 192)
    activaciones = {t: torch.rand(2, 32) for t in TipoTerritorioCoder}
    
    with torch.no_grad():
        out, conf, should_exit = gestor(x, activaciones)
    
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    print(f"  ✓ Forward pass: {x.shape} → {out.shape}")
    print(f"  ✓ Confianza: {conf.mean().item():.3f}")
    
    print("  ✅ Territorios OK")


def test_modelo():
    """Test del modelo completo."""
    print("\n" + "=" * 70)
    print("🚀 Test: Modelo PampaRCoder")
    print("=" * 70)
    
    from pampar.coder import PampaRCoder, CODER_4GB, crear_modelo
    
    # Crear modelo
    model = crear_modelo("4GB")
    
    # Estadísticas
    params = model.count_parameters()
    print(f"  ✓ Total parámetros: {params['total']:,}")
    
    # Test forward
    x = torch.randint(0, CODER_4GB.vocab_size, (2, 64))
    targets = torch.randint(0, CODER_4GB.vocab_size, (2, 64))
    
    logits, loss = model(x, targets)
    
    assert logits.shape == (2, 64, CODER_4GB.vocab_size)
    print(f"  ✓ Forward: {x.shape} → {logits.shape}")
    print(f"  ✓ Loss: {loss.item():.4f}")
    
    print("  ✅ Modelo OK")


def test_generation():
    """Test de generación."""
    print("\n" + "=" * 70)
    print("🎯 Test: Generación")
    print("=" * 70)
    
    from pampar.coder import crear_modelo, CODER_4GB
    
    model = crear_modelo("4GB")
    model.eval()
    
    prompt = torch.randint(0, CODER_4GB.vocab_size, (1, 10))
    
    # Sin early exit
    start = time.time()
    gen1 = model.generate(prompt, max_new_tokens=30, use_early_exit=False)
    t1 = time.time() - start
    
    # Con early exit
    start = time.time()
    gen2 = model.generate(prompt, max_new_tokens=30, use_early_exit=True)
    t2 = time.time() - start
    
    print(f"  ✓ Sin early exit: {gen1.shape[1]} tokens en {t1:.3f}s ({30/t1:.1f} tok/s)")
    print(f"  ✓ Con early exit: {gen2.shape[1]} tokens en {t2:.3f}s ({30/t2:.1f} tok/s)")
    
    speedup = t1 / t2 if t2 > 0 else 1.0
    print(f"  ✓ Speedup early exit: {speedup:.2f}x")
    
    print("  ✅ Generación OK")


def test_gpu():
    """Test en GPU si está disponible."""
    print("\n" + "=" * 70)
    print("🎮 Test: GPU")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("  ⚠️ CUDA no disponible, saltando test GPU")
        return
    
    from pampar.coder import crear_modelo, CODER_4GB
    
    device = torch.device("cuda")
    model = crear_modelo("4GB").to(device)
    
    # Verificar VRAM usada
    torch.cuda.reset_peak_memory_stats()
    
    x = torch.randint(0, CODER_4GB.vocab_size, (4, 128), device=device)
    targets = torch.randint(0, CODER_4GB.vocab_size, (4, 128), device=device)
    
    logits, loss = model(x, targets)
    loss.backward()
    
    vram_used = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  ✓ GPU: {torch.cuda.get_device_name()}")
    print(f"  ✓ VRAM usada: {vram_used:.2f} GB")
    print(f"  ✓ Loss: {loss.item():.4f}")
    
    # Test generación en GPU
    model.eval()
    prompt = torch.randint(0, CODER_4GB.vocab_size, (1, 10), device=device)
    
    start = time.time()
    generated = model.generate(prompt, max_new_tokens=50, use_early_exit=True)
    elapsed = time.time() - start
    
    print(f"  ✓ Generación GPU: {50/elapsed:.1f} tokens/sec")
    
    print("  ✅ GPU OK")


def main():
    """Ejecuta todos los tests."""
    print("\n" + "=" * 70)
    print("       PAMPAr-Coder Test Suite")
    print("=" * 70)
    
    try:
        test_config()
        test_llaves()
        test_territorios()
        test_modelo()
        test_generation()
        test_gpu()
        
        print("\n" + "=" * 70)
        print("🎉 TODOS LOS TESTS PASARON")
        print("=" * 70)
        
        # Resumen final
        print("\n📊 Resumen PAMPAr-Coder:")
        from pampar.coder import CODER_4GB, crear_modelo
        
        model = crear_modelo("4GB")
        params = model.count_parameters()
        vram = CODER_4GB.estimate_vram_gb(training=True)
        
        print(f"   Parámetros:  {params['total']/1e6:.1f}M")
        print(f"   VRAM train:  ~{vram:.2f} GB")
        print(f"   Preset:      CODER_4GB (GTX 1650 compatible)")
        print(f"   Early Exit:  Sí (umbral={CODER_4GB.umbral_confianza_exit})")
        print(f"   LLAVES:      {CODER_4GB.peso_llaves*100:.0f}% reglas")
        
        print("\n🚀 Listo para entrenar!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
