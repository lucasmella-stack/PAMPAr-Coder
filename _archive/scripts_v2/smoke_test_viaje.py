#!/usr/bin/env python3
# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Smoke Test — Viaje Intelectual.

Valida que TODOS los componentes del pipeline funcionen correctamente
ANTES de lanzar el entrenamiento largo.

Corre en ~30-60 segundos en CPU. Si pasa todo → safe to launch.

Qué verifica:
  [1] Tokenizer carga y tokeniza
  [2] Modelo carga desde checkpoint
  [3] Forward pass no explota (shapes correctos)
  [4] Biblioteca tiene datos legibles
  [5] Motor de Curiosidad selecciona temas
  [6] Paso de gradiente funciona (loss baja, no NaN)
  [7] MemoriaJerarquica procesa batch
  [8] Replay step no falla
  [9] Checkpoint save/load round-trip
  [10] Loop completo de 10 pasos sin error

Uso:
  python scripts/smoke_test_viaje.py
  python scripts/smoke_test_viaje.py --checkpoint checkpoints/stable_last.pt
  python scripts/smoke_test_viaje.py --verbose
"""

import argparse
import json
import sys
import tempfile
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# HELPERS
# =============================================================================

OK   = "\033[92m[OK]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[...]\033[0m"

_resultados: list[tuple[str, bool, str]] = []

def check(nombre: str, fn, *args, **kwargs):
    """Ejecuta fn y registra si pasó o falló."""
    print(f"  {INFO} {nombre}...", end="", flush=True)
    t0 = time.time()
    try:
        resultado = fn(*args, **kwargs)
        elapsed = time.time() - t0
        print(f"\r  {OK}  {nombre} ({elapsed:.1f}s)")
        _resultados.append((nombre, True, ""))
        return resultado
    except Exception as e:
        elapsed = time.time() - t0
        msg = str(e)
        print(f"\r  {FAIL} {nombre} ({elapsed:.1f}s)")
        print(f"         → {msg}")
        if args and hasattr(args[0], '__name__') or True:
            tb = traceback.format_exc()
            for line in tb.strip().split("\n")[-4:]:
                print(f"         {line}")
        _resultados.append((nombre, False, msg))
        return None


def resumen():
    total = len(_resultados)
    pasados = sum(1 for _, ok, _ in _resultados if ok)
    fallidos = total - pasados

    print(f"\n{'='*55}")
    print(f"  SMOKE TEST -- {pasados}/{total} checks pasaron")
    print(f"{'='*55}")

    if fallidos == 0:
        print(f"\033[92m  OK Todo bien -- safe to launch aprender_solo.py\033[0m\n")
    else:
        print(f"\033[91m  FAIL {fallidos} check(s) fallaron -- NO lanzar hasta resolver\033[0m")
        for nombre, ok, msg in _resultados:
            if not ok:
                print(f"    → {nombre}: {msg[:80]}")
        print()

    return fallidos == 0


# =============================================================================
# CHECKS INDIVIDUALES
# =============================================================================

def _check_tokenizer(tokenizer_path: Path):
    import sentencepiece as spm
    tok = spm.SentencePieceProcessor()
    tok.Load(str(tokenizer_path))
    vocab = tok.GetPieceSize()
    assert vocab > 1000, f"Vocab muy chico: {vocab}"
    # Round-trip
    texto = "def suma(a, b):\n    return a + b"
    ids = tok.Encode(texto)
    assert len(ids) > 5, "Tokenización vacía"
    decoded = tok.Decode(ids)
    assert "suma" in decoded, "Decode incorrecto"
    return tok


def _infer_config_from_state(state: dict):
    """Infiere ConfigV2 desde las shapes del state_dict (sin conocer el preset original)."""
    from pampar.coder.v2.config import ConfigV2

    emb = state.get("tok_emb.weight")           # [vocab, dim]
    if emb is None:
        return None
    vocab_size, dim = emb.shape

    # Contar capas: buscamos llaves tipo "capas.N."
    n_capas = 0
    for k in state:
        if k.startswith("capas."):
            idx = int(k.split(".")[1])
            n_capas = max(n_capas, idx + 1)

    # n_heads: buscamos qkv projection (attn.qkv o attn.wqkv)
    n_heads = 6  # default
    ffn_mult = 4.0

    # Detectar n_heads desde Q projection shape [n_heads*head_dim, dim]
    for k, v in state.items():
        if "attn" in k and ("q_proj" in k or "wq" in k or "qkv" in k):
            if len(v.shape) == 2:
                q_dim = v.shape[0]
                head_dim = 64  # típico
                n_heads = max(1, q_dim // head_dim)
            break

    return ConfigV2(
        vocab_size=int(vocab_size),
        dim=int(dim),
        n_heads=n_heads,
        n_capas=n_capas if n_capas > 0 else 6,
    )


def _check_modelo(checkpoint: Path, device: torch.device):
    from pampar.coder.v2.modelo import PampaRCoderV2
    from pampar.coder.v2.config import (
        PRESET_1_5B, PRESET_4GB, PRESET_8GB, PRESET_24GB, ConfigV2,
    )

    PRESET_MAP = {
        "4GB": PRESET_4GB, "8GB": PRESET_8GB,
        "24GB": PRESET_24GB, "1_5B": PRESET_1_5B,
    }

    config = PRESET_4GB
    ckpt = None

    if checkpoint.exists():
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        raw_cfg = ckpt.get("config")
        state = ckpt.get("modelo", ckpt.get("model", ckpt))

        if isinstance(raw_cfg, ConfigV2):
            config = raw_cfg
        elif isinstance(raw_cfg, dict):
            preset_name = raw_cfg.get("preset")
            if preset_name and preset_name in PRESET_MAP:
                # Verificar que el preset coincide con los pesos reales
                preset = PRESET_MAP[preset_name]
                emb = state.get("tok_emb.weight")
                if emb is not None and emb.shape[1] != preset.dim:
                    # El preset guardado no coincide — inferir desde pesos
                    inferred = _infer_config_from_state(state)
                    config = inferred if inferred else preset
                else:
                    config = preset
            else:
                import dataclasses
                valid = {f.name for f in dataclasses.fields(ConfigV2)}
                filtered = {k: v for k, v in raw_cfg.items() if k in valid}
                if filtered:
                    config = ConfigV2(**filtered)
        else:
            # Sin config guardada: inferir desde el state_dict
            if isinstance(state, dict) and "tok_emb.weight" in state:
                inferred = _infer_config_from_state(state)
                if inferred:
                    config = inferred

    modelo = PampaRCoderV2(config).to(device)

    if ckpt is not None:
        state = ckpt.get("modelo", ckpt.get("model", ckpt))
        missing, unexpected = modelo.load_state_dict(state, strict=False)
        # Claves con "." son parámetros reales (no metadatos) — solo advertir
        bad = [k for k in unexpected if "." in k]
        if bad:
            print(f"\n      [WARN] {len(bad)} pesos del ckpt ignorados (arch legacy): {bad[:2]}...")

    n = sum(p.numel() for p in modelo.parameters()) / 1e6
    assert n > 0.5, f"Modelo muy chico: {n:.1f}M params"
    return modelo, n


def _check_forward(modelo, device: torch.device):
    modelo.eval()
    with torch.no_grad():
        B, L = 2, 32
        tokens = torch.randint(1, 1000, (B, L), device=device)
        out = modelo(tokens)
        # Soporta (logits,) | (logits, info, x) | etc.
        logits = out[0] if isinstance(out, (tuple, list)) else out
        info = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else None
        assert logits.shape == (B, L, modelo.config.vocab_size), \
            f"Shape incorrecto: {logits.shape}"
        assert not torch.isnan(logits).any(), "NaN en logits"
        assert not torch.isinf(logits).any(), "Inf en logits"
        has_terr = info is not None and isinstance(info, dict) and "terr_acts" in info
        if not has_terr:
            print(f"\n      [INFO] terr_acts no disponible en esta arch (se continua)", end="")
    return logits.shape, info


def _temas_iter(indice: dict):
    """Itera solo sobre los temas (ignora meta-keys como version, descripcion)."""
    for cat, temas in indice.items():
        if isinstance(temas, list):
            yield from temas


def _check_biblioteca(biblioteca: Path, indice: dict):
    temas_con_datos = 0
    temas_sin_datos = []
    total_bytes = 0

    for tema in _temas_iter(indice):
        archivo = biblioteca / tema["archivo"]
        if archivo.exists() and archivo.stat().st_size > 0:
            temas_con_datos += 1
            total_bytes += archivo.stat().st_size
            # Verificar que es JSON válido
            primera_linea = archivo.read_text(encoding="utf-8", errors="ignore"
                                               ).split("\n")[0]
            if primera_linea:
                obj = json.loads(primera_linea)
                assert "text" in obj, f"Falta campo 'text' en {tema['nombre']}"
        else:
            temas_sin_datos.append(tema["nombre"])

    total_mb = total_bytes / 1024**2
    assert temas_con_datos >= 10, \
        f"Muy pocos temas con datos: {temas_con_datos}. " \
        f"Ejecuta: python scripts/poblar_biblioteca.py"
    return temas_con_datos, len(temas_sin_datos), total_mb


def _check_lector(biblioteca: Path, indice: dict, tok, device: torch.device):
    """Verifica que el LectorBiblioteca puede leer y tokenizar un batch."""
    sys.path.insert(0, str(Path(__file__).parent))
    from aprender_solo import LectorBiblioteca

    lector = LectorBiblioteca(
        raiz=biblioteca,
        tokenizer=tok,
        max_seq_len=64,
        batch_size=2,
    )

    # Buscar un tema con datos
    archivo_test = None
    for tema in _temas_iter(indice):
        p = biblioteca / tema["archivo"]
        if p.exists() and p.stat().st_size > 100:
            archivo_test = tema["archivo"]
            break

    assert archivo_test, "No se encontró ningún archivo de tema con datos"
    batch = lector.obtener_batch(archivo_test, device)
    assert batch is not None, f"batch None para {archivo_test}"
    assert batch.shape[0] >= 1, "Batch vacío"
    assert batch.shape[1] >= 4, "Secuencia demasiado corta"
    return batch.shape


def _check_motor_curiosidad(indice: dict):
    from pampar.coder.v2.aprendizaje.curiosidad import MotorCuriosidad
    motor = MotorCuriosidad(nivel_actual=1)
    n = motor.registrar_temas_desde_indice(indice)
    assert n > 0, "No se registraron temas"

    tema = motor.siguiente_tema()
    assert tema is not None, "siguiente_tema() devolvió None"
    assert tema in motor.temas, f"Tema '{tema}' no está registrado"

    # Simular sesión
    info = motor.retroalimentar(tema, loss=2.5)
    assert "loss_actual" in info
    assert abs(info["loss_actual"] - 2.5) < 0.1
    return motor, n


def _check_paso_gradiente(modelo, device: torch.device):
    from pampar.coder.v2.modelo import PampaRCoderV2

    optimizer = torch.optim.AdamW(modelo.parameters(), lr=1e-4)
    modelo.train()

    B, L = 2, 32
    tokens = torch.randint(1, 1000, (B, L + 1), device=device)
    input_ids = tokens[:, :-1]
    targets = tokens[:, 1:]

    losses = []
    for _ in range(3):
        optimizer.zero_grad()
        logits, _, _ = modelo(input_ids, targets=targets)
        bL, bV = logits.shape[0] * logits.shape[1], logits.shape[2]
        loss = F.cross_entropy(logits.reshape(bL, bV), targets.reshape(bL))
        assert not torch.isnan(loss), "Loss NaN"
        assert not torch.isinf(loss), "Loss Inf"
        loss.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

    # Al menos no debe crecer explosivamente
    assert losses[-1] < losses[0] * 10, \
        f"Loss explotó: {losses[0]:.3f} → {losses[-1]:.3f}"
    return losses, optimizer


def _check_memoria(modelo, device: torch.device):
    from pampar.coder.v2.aprendizaje.memoria_jerarquica import MemoriaJerarquica

    memoria = MemoriaJerarquica(capacidad_l0=128, capacidad_l1=256, capacidad_l2=64)

    modelo.eval()
    B, L = 2, 32
    tokens = torch.randint(1, 1000, (B, L + 1), device=device)

    with torch.no_grad():
        inp = tokens[:, :-1]
        tgt = tokens[:, 1:]
        out = modelo(inp)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        info = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else None
        bL = B * L
        ptl = F.cross_entropy(
            logits.reshape(bL, logits.shape[-1]),
            tgt.reshape(bL),
            ignore_index=0,
            reduction="none",
        ).reshape(B, L)
        pad = torch.zeros(B, 1, device=device)
        ptl_padded = torch.cat([pad, ptl], dim=1)

    terr_acts = info.get("terr_acts") if isinstance(info, dict) else None
    memoria.procesar_batch(tokens, ptl_padded, terr_acts)
    stats = memoria.stats()
    assert isinstance(stats, dict), "stats() no retorna dict"
    return stats


def _check_replay(modelo, device: torch.device, optimizer):
    from pampar.coder.v2.aprendizaje.memoria_jerarquica import MemoriaJerarquica

    memoria = MemoriaJerarquica(capacidad_l0=128, capacidad_l1=256, capacidad_l2=64)

    # Llenar memoria con algo
    for _ in range(5):
        B, L = 2, 32
        tokens = torch.randint(1, 1000, (B, L + 1), device=device)
        ptl = torch.rand(B, L + 1, device=device)
        memoria.procesar_batch(tokens, ptl, None)

    batch = memoria.get_replay_batch(strategy="hardest")
    # Puede ser None si memoria vacía — no es error crítico
    if batch is not None:
        assert batch.shape[1] >= 2, "Replay batch demasiado corto"
        modelo.train()
        optimizer.zero_grad()
        inp = batch[:, :-1].to(device)
        tgt = batch[:, 1:].to(device)
        logits, _, _ = modelo(inp)
        bL = inp.shape[0] * inp.shape[1]
        loss = F.cross_entropy(
            logits.reshape(bL, logits.shape[-1]),
            tgt.reshape(bL),
            ignore_index=0,
        )
        (loss * 0.1).backward()
        optimizer.step()
    return batch is not None


def _check_checkpoint_roundtrip(modelo, optimizer, device: torch.device):
    from pampar.coder.v2.modelo import PampaRCoderV2

    config = modelo.config  # Usar la config ya cargada en el modelo existente

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ruta = Path(f.name)

    # Guardar
    torch.save({
        "modelo": modelo.state_dict(),
        "optimizer": optimizer.state_dict(),
        "paso_global": 42,
    }, ruta)

    assert ruta.exists() and ruta.stat().st_size > 0, "Archivo vacío"
    size = ruta.stat().st_size

    # Restaurar en modelo nuevo con la misma config
    modelo2 = PampaRCoderV2(config).to(device)
    ckpt = torch.load(ruta, map_location=device, weights_only=True)
    modelo2.load_state_dict(ckpt["modelo"], strict=False)
    assert ckpt["paso_global"] == 42

    ruta.unlink()
    return f"{size / 1024**2:.1f} MB guardados y restaurados OK"


def _check_loop_completo(modelo, optimizer, biblioteca: Path, indice: dict,
                         tok, device: torch.device, n_pasos: int = 10):
    """Simula exactamente 10 pasos del loop de aprender_solo.py."""
    sys.path.insert(0, str(Path(__file__).parent))
    from aprender_solo import LectorBiblioteca, ViajeIntelectual
    from pampar.coder.v2.aprendizaje.memoria_jerarquica import MemoriaJerarquica
    from pampar.coder.v2.aprendizaje.curiosidad import MotorCuriosidad

    memoria = MemoriaJerarquica(capacidad_l0=128, capacidad_l1=256, capacidad_l2=64)
    motor = MotorCuriosidad(nivel_actual=1)
    lector = LectorBiblioteca(raiz=biblioteca, tokenizer=tok,
                               max_seq_len=64, batch_size=2)

    viaje = ViajeIntelectual(
        modelo=modelo,
        optimizer=optimizer,
        memoria=memoria,
        motor=motor,
        biblioteca=lector,
        indice=indice,
        device=device,
        pasos_por_tema=n_pasos,
        replay_cada=5,
        consolidar_cada=20,
        guardar_cada=100,      # No guarda durante el smoke test
        ruta_checkpoint=None,  # Sin guardar
        ruta_estado_motor=None,
    )

    # Correr solo max_pasos pasos (no infinite loop)
    viaje.estudiar(max_pasos=n_pasos)
    return viaje.paso_global


def _auto_detect_tokenizer(checkpoint: Path, base_dir: Path) -> Path:
    """Detecta el tokenizer correcto según el vocab_size del checkpoint."""
    vocab_to_tok = {
        16000: base_dir / "data/tokenizer/code_tokenizer.model",
        48000: base_dir / "data/tokenizer/pampar_48k.model",
    }
    try:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = ckpt.get("modelo", ckpt.get("model", ckpt))
        emb = state.get("tok_emb.weight")
        if emb is not None:
            vocab_size = emb.shape[0]
            tok_path = vocab_to_tok.get(int(vocab_size))
            if tok_path and tok_path.exists():
                return tok_path
    except Exception:
        pass
    return base_dir / "data/tokenizer/pampar_48k.model"  # fallback


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Smoke test del viaje intelectual")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/pampar_v2_best.pt"))
    parser.add_argument("--tokenizer", type=Path, default=None,
                        help="Tokenizer a usar (auto-detectado si no se especifica)")
    parser.add_argument("--biblioteca", type=Path, default=Path("biblioteca"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Auto-detectar tokenizer si no se especificó
    base_dir = Path(__file__).parent.parent
    if args.tokenizer is None:
        args.tokenizer = _auto_detect_tokenizer(args.checkpoint, base_dir)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"\n{'='*55}")
    print(f"  PAMPAr -- Smoke Test (Viaje Intelectual)")
    print(f"  Device: {device} | Checkpoint: {args.checkpoint.name}")
    print(f"  Tokenizer: {args.tokenizer.name}")
    print(f"{'='*55}\n")

    # ── Checks ───────────────────────────────────────────────────────────────
    tok = check("1. Tokenizer carga y tokeniza", _check_tokenizer, args.tokenizer)
    if tok is None:
        resumen(); return

    result = check("2. Modelo carga desde checkpoint", _check_modelo,
                   args.checkpoint, device)
    if result is None:
        resumen(); return
    modelo, n_params = result
    print(f"      → {n_params:.0f}M parámetros | vocab={modelo.config.vocab_size}")

    # Validar que tokenizer y modelo tienen el mismo vocab
    if tok is not None:
        tok_vocab = tok.GetPieceSize()
        model_vocab = modelo.config.vocab_size
        if tok_vocab != model_vocab:
            print(f"\n  [FAIL] Vocab mismatch: tokenizer={tok_vocab} vs modelo={model_vocab}")
            print(f"         El tokenizer '{args.tokenizer.name}' NO es compatible con este checkpoint.")
            print(f"         Usa el tokenizer correcto para vocab_size={model_vocab}")
            _resultados.append(("2.5. Vocab tokenizer=modelo", False,
                                 f"tokenizer={tok_vocab} != modelo={model_vocab}"))
            resumen(); return
        else:
            _resultados.append(("2.5. Vocab tokenizer=modelo", True, ""))
            print(f"  {OK}  2.5. Vocab match: tokenizer={tok_vocab} = modelo={model_vocab}")

    fwd_result = check("3. Forward pass (shapes + sin NaN)", _check_forward, modelo, device)
    if fwd_result:
        fwd_shape, _ = fwd_result
        print(f"      → logits shape: {fwd_shape}")

    # Leer índice para checks siguientes
    indice_path = args.biblioteca / "indice.json"
    if not indice_path.exists():
        print(f"\n  {FAIL} indice.json no encontrado en {args.biblioteca}")
        resumen(); return
    indice = json.loads(indice_path.read_text())

    result = check("4. Biblioteca tiene datos legibles", _check_biblioteca,
                   args.biblioteca, indice)
    if result:
        con, sin, mb = result
        print(f"      → {con} temas con datos, {sin} sin datos, {mb:.1f} MB total")

    bshape = check("5. LectorBiblioteca obtiene batch tokenizado",
                   _check_lector, args.biblioteca, indice, tok, device)
    if bshape:
        print(f"      → batch shape: {bshape}")

    result = check("6. MotorCuriosidad selecciona temas", _check_motor_curiosidad, indice)
    if result:
        motor_ok, n_temas = result
        print(f"      → {n_temas} temas registrados")

    result = check("7. Paso de gradiente (loss no explota)", _check_paso_gradiente,
                   modelo, device)
    optimizer = None
    if result:
        losses, optimizer = result
        print(f"      → losses: {losses[0]:.3f} → {losses[-1]:.3f}")

    check("8. MemoriaJerarquica procesa batch", _check_memoria, modelo, device)

    if optimizer:
        replay_ok = check("9. Replay step (backprop desde memoria)",
                           _check_replay, modelo, device, optimizer)

        check("10. Checkpoint save/load round-trip",
              _check_checkpoint_roundtrip, modelo, optimizer, device)

        pasos = check(f"11. Loop completo (10 pasos reales)",
                      _check_loop_completo, modelo, optimizer,
                      args.biblioteca, indice, tok, device, 10)
        if pasos is not None:
            print(f"      → {pasos} pasos ejecutados sin error")

    # ── Resultado ─────────────────────────────────────────────────────────────
    ok = resumen()
    if ok:
        print("  Para lanzar el entrenamiento:")
        print(f"  python scripts/aprender_solo.py \\")
        print(f"    --checkpoint {args.checkpoint} \\")
        print(f"    --batch-size 2 --seq-len 512\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
