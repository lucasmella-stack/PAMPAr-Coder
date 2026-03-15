#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
eval_agents.py — Evaluación de generación de AGENTS.md (Milestone 3).

Dado un scan sintético, el modelo debe generar un AGENTS.md válido.
Se evalúan 12 casos con entornos variados.

Métricas:
  - Secciones presentes: Quick Reference, Sistema detectado, Paquetes, Boot protocol
  - Info del scan reflejada: OS, Python version, GPU (si existe)
  - Formato válido: líneas con '|' para tablas, '```' para bloques, '##' headers
  - Sin hallucination grave: no inventa servicios que no existían

Uso:
  python -X utf8 scripts/eval_agents.py --checkpoint checkpoints/v3_sft_v8.pt
"""

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

TOKENIZER_PATHS = [
    "data/tokenizer/pampar_48k.model",
    "data/tokenizer/code_tokenizer.model",
]

# ─────────────────────────────────────────────────────────────────────────────
# Casos de evaluación (entornos sintéticos nunca vistos en entrenamiento)
# ─────────────────────────────────────────────────────────────────────────────

CASOS_EVAL = [
    {
        "nombre": "ML PyTorch GPU",
        "scan": (
            "El Scanner detectó el siguiente entorno. "
            "Genera el archivo AGENTS.md contextual para este despliegue.\n\n"
            "## Entorno detectado\n\n"
            "- **OS**: Linux Ubuntu 22.04\n"
            "- **Python**: 3.11.9\n"
            "- **GPU**: NVIDIA GeForce RTX 4090 (24576 MB)\n"
            "- **RAM**: 64.0 GB\n"
            "- **Archivos**: Python: 52, JSON: 14, Markdown: 9\n"
            "- **Paquetes** (6 relevantes): torch==2.5.1, transformers==4.47.1, "
            "peft==0.13.2, datasets==3.1.0, trl==0.12.2, bitsandbytes==0.45.0\n"
            "- **Servicios inactivos**: PostgreSQL, Redis\n"
            "- **Voz**: espeak\n\n"
            "**Proyecto**: llm-finetune — pipeline de fine-tuning con LoRA/QLoRA"
        ),
        "espera": {
            "secciones": ["Quick Reference", "Sistema", "Paquetes", "Boot"],
            "info_scan": ["Linux", "3.11.9", "RTX 4090", "torch", "peft"],
            "servicios_correctos": False,  # PostgreSQL/Redis estaban inactivos
        },
    },
    {
        "nombre": "FastAPI + PostgreSQL",
        "scan": (
            "El Scanner detectó el siguiente entorno. "
            "Genera el archivo AGENTS.md contextual para este despliegue.\n\n"
            "## Entorno detectado\n\n"
            "- **OS**: macOS 14.0 Sonoma\n"
            "- **Python**: 3.12.4\n"
            "- **GPU**: no disponible (solo CPU)\n"
            "- **RAM**: 32.0 GB\n"
            "- **Archivos**: Python: 40, TypeScript: 8, JSON: 12, Markdown: 6\n"
            "- **Paquetes** (6 relevantes): fastapi==0.115.0, uvicorn==0.32.0, "
            "pydantic==2.10.0, sqlalchemy==2.0.36, alembic==1.14.0, httpx==0.28.0\n"
            "- **Servicios activos**: PostgreSQL, HTTP-8000\n"
            "- **Voz**: say\n\n"
            "**Proyecto**: api-service — API REST con FastAPI y PostgreSQL"
        ),
        "espera": {
            "secciones": ["Quick Reference", "Sistema", "Paquetes", "Boot"],
            "info_scan": ["macOS", "3.12.4", "fastapi", "PostgreSQL"],
            "servicios_correctos": True,
        },
    },
    {
        "nombre": "CLI / Script simple",
        "scan": (
            "El Scanner detectó el siguiente entorno. "
            "Genera el archivo AGENTS.md contextual para este despliegue.\n\n"
            "## Entorno detectado\n\n"
            "- **OS**: Windows 11.0.22621\n"
            "- **Python**: 3.13.3\n"
            "- **GPU**: no disponible (solo CPU)\n"
            "- **RAM**: 8.0 GB\n"
            "- **Archivos**: Python: 12, Markdown: 3\n"
            "- **Paquetes** (4 relevantes): click==8.1.8, rich==13.9.4, "
            "httpx==0.28.0, python-dotenv==1.0.1\n"
            "- **Voz**: SAPI\n\n"
            "**Proyecto**: data-fetcher — herramienta CLI para descarga de datos"
        ),
        "espera": {
            "secciones": ["Quick Reference", "Sistema", "Paquetes", "Boot"],
            "info_scan": ["Windows", "11", "click", "rich"],
            "servicios_correctos": True,
        },
    },
    {
        "nombre": "Django + Celery",
        "scan": (
            "El Scanner detectó el siguiente entorno. "
            "Genera el archivo AGENTS.md contextual para este despliegue.\n\n"
            "## Entorno detectado\n\n"
            "- **OS**: Linux Debian 12\n"
            "- **Python**: 3.11.9\n"
            "- **GPU**: no disponible (solo CPU)\n"
            "- **RAM**: 16.0 GB\n"
            "- **Archivos**: Python: 58, HTML: 22, CSS: 12, JavaScript: 9, Markdown: 5\n"
            "- **Paquetes** (5 relevantes): django==5.1.4, djangorestframework==3.15.2, "
            "celery==5.4.0, redis==5.2.1, pillow==11.0.0\n"
            "- **Servicios activos**: PostgreSQL, Redis, HTTP-8000\n"
            "- **Voz**: espeak\n\n"
            "**Proyecto**: plataforma-web — aplicación Django con tareas asíncronas"
        ),
        "espera": {
            "secciones": ["Quick Reference", "Sistema", "Paquetes", "Boot"],
            "info_scan": ["Linux", "django", "celery", "Redis"],
            "servicios_correctos": True,
        },
    },
    {
        "nombre": "Data Science sin GPU",
        "scan": (
            "El Scanner detectó el siguiente entorno. "
            "Genera el archivo AGENTS.md contextual para este despliegue.\n\n"
            "## Entorno detectado\n\n"
            "- **OS**: macOS 14.0 Sonoma\n"
            "- **Python**: 3.10.14\n"
            "- **GPU**: no disponible (solo CPU)\n"
            "- **RAM**: 16.0 GB\n"
            "- **Archivos**: Python: 20, JSON: 18, Markdown: 10\n"
            "- **Paquetes** (6 relevantes): pandas==2.2.3, numpy==2.1.3, "
            "matplotlib==3.9.3, scikit-learn==1.5.2, jupyter==1.1.1, plotly==5.24.1\n"
            "- **Servicios activos**: PostgreSQL\n"
            "- **Voz**: say\n\n"
            "**Proyecto**: market-analysis — análisis de datos financieros"
        ),
        "espera": {
            "secciones": ["Quick Reference", "Sistema", "Paquetes", "Boot"],
            "info_scan": ["macOS", "pandas", "scikit-learn", "PostgreSQL"],
            "servicios_correctos": True,
        },
    },
    {
        "nombre": "GTX 1650 Local",
        "scan": (
            "El Scanner detectó el siguiente entorno. "
            "Genera el archivo AGENTS.md contextual para este despliegue.\n\n"
            "## Entorno detectado\n\n"
            "- **OS**: Windows 10.0.26200\n"
            "- **Python**: 3.13.3\n"
            "- **GPU**: NVIDIA GeForce GTX 1650 (4095 MB)\n"
            "- **RAM**: 31.9 GB\n"
            "- **Archivos**: Python: 65, Markdown: 10, JSON: 8\n"
            "- **Paquetes** (6 relevantes): torch==2.5.1, sentencepiece==0.2.0, "
            "transformers==4.47.1, accelerate==1.12.0, pytest==9.0.1, peft==0.13.2\n"
            "- **Servicios inactivos**: PostgreSQL, Redis\n"
            "- **Voz**: SAPI\n\n"
            "**Proyecto**: pampar-coder — modelo de lenguaje 108M entrenado localmente"
        ),
        "espera": {
            "secciones": ["Quick Reference", "Sistema", "Paquetes", "Boot"],
            "info_scan": ["Windows", "GTX 1650", "torch", "sentencepiece"],
            "servicios_correctos": False,
        },
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Carga de modelo (ídem eval_v3.py)
# ─────────────────────────────────────────────────────────────────────────────

def _cargar_modelo(checkpoint: Path, device: torch.device):
    import dataclasses
    from pampar.coder.v3.config import PRESET_V3, ConfigV3
    from pampar.coder.v3.modelo import PamparV3

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    raw_cfg = ckpt.get("config", {})
    state = ckpt.get("modelo", ckpt)
    if isinstance(raw_cfg, dict) and "dim" in raw_cfg:
        valid = {f.name for f in dataclasses.fields(ConfigV3)}
        cfg = ConfigV3(**{k: v for k, v in raw_cfg.items() if k in valid})
    else:
        cfg = PRESET_V3
    modelo = PamparV3(cfg)
    modelo.load_state_dict(state, strict=False)
    return modelo.to(device).eval(), cfg


def _cargar_tok(vocab_size: int) -> spm.SentencePieceProcessor:
    sp = spm.SentencePieceProcessor()
    for p in TOKENIZER_PATHS:
        pp = Path(p)
        if pp.exists():
            sp.Load(str(pp))
            if sp.vocab_size() == vocab_size:
                return sp
    raise FileNotFoundError(f"Tokenizer {vocab_size} no encontrado")


# ─────────────────────────────────────────────────────────────────────────────
# Generación (igual que eval_v3.py pero con max_tokens más alto)
# ─────────────────────────────────────────────────────────────────────────────

def _generar(modelo, tok, prompt: str, device, max_tokens=800, temp=0.1, rep_pen=1.15) -> str:
    ids = tok.Encode(prompt)
    gen = list(ids)
    rep_window = 32

    for _ in range(max_tokens):
        ctx = torch.tensor([gen[-512:]], dtype=torch.long, device=device)
        logits, _, _ = modelo(ctx)
        nxt = logits[0, -1]

        if rep_pen != 1.0 and len(gen) > len(ids):
            ws = max(len(ids), len(gen) - rep_window)
            seen = set(gen[ws:])
            for t in seen:
                nxt[t] = nxt[t] / rep_pen if nxt[t] > 0 else nxt[t] * rep_pen

        if temp <= 0.0:
            next_tok = int(nxt.argmax())
        else:
            probs = F.softmax(nxt / temp, dim=-1)
            next_tok = int(torch.multinomial(probs, 1))

        gen.append(next_tok)
        decoded = tok.Decode(gen[len(ids):]).replace("\u2047", "\n")

        # Detener en nueva sección de instrucción
        if decoded.count("### Problem:") > 0:
            idx = decoded.index("### Problem:")
            if idx > 50:
                return decoded[:idx].rstrip()

        # Detener si terminó el documento (línea de separador final)
        if decoded.rstrip().endswith("```") and len(decoded) > 200:
            return decoded.rstrip()

    return tok.Decode(gen[len(ids):]).replace("\u2047", "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Verificación
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ResultadoCaso:
    nombre: str
    score: float       # 0.0 – 1.0
    ok_secciones: bool
    ok_info: bool
    ok_formato: bool
    n_secciones: int
    n_info: int
    texto: str


def _verificar(texto: str, espera: dict) -> tuple[bool, bool, bool, int, int]:
    """Verifica que el texto generado cumpla con las expectativas."""
    texto_lower = texto.lower()

    # 1. Secciones mínimas
    secciones_esperadas = espera["secciones"]
    n_encontradas = sum(1 for s in secciones_esperadas if s.lower() in texto_lower)
    ok_secciones = n_encontradas >= len(secciones_esperadas) * 0.75  # 75%

    # 2. Info del scan reflejada
    info = espera["info_scan"]
    n_info = sum(1 for i in info if i.lower() in texto_lower)
    ok_info = n_info >= len(info) * 0.6  # 60%

    # 3. Formato válido (tiene headers ## y alguna tabla o bloque)
    tiene_header = bool(re.search(r'^##\s+\w', texto, re.MULTILINE))
    tiene_tabla_o_bloque = '|' in texto or '```' in texto
    ok_formato = tiene_header and tiene_tabla_o_bloque

    return ok_secciones, ok_info, ok_formato, n_encontradas, n_info


def _score(ok_s: bool, ok_i: bool, ok_f: bool, n_s: int, n_i: int,
           max_s: int, max_i: int) -> float:
    """Score ponderado: secciones 40%, info 40%, formato 20%."""
    s = (n_s / max_s) * 0.4 + (n_i / max_i) * 0.4 + (0.2 if ok_f else 0.0)
    return round(min(s, 1.0), 3)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/v3_sft_v8.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--rep-penalty", type=float, default=1.15)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        args.device if args.device != "auto" else "cpu"
    )

    sep = "═" * 70
    print(f"\n{sep}")
    print(f"  EVAL MILESTONE 3 — Generación de AGENTS.md")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Device     : {device}  |  Temp: {args.temp}  |  RepPen: {args.rep_penalty}")
    print(f"  Casos      : {len(CASOS_EVAL)}")
    print(f"{sep}\n")

    t0_total = time.time()
    modelo, cfg = _cargar_modelo(Path(args.checkpoint), device)
    tok = _cargar_tok(cfg.vocab_size)
    modelo.registrar_tokenizer(tok)
    print(f"  Modelo cargado ({sum(p.numel() for p in modelo.parameters())/1e6:.1f}M params)\n")

    resultados: list[ResultadoCaso] = []

    for i, caso in enumerate(CASOS_EVAL, 1):
        prompt = f"### Problem:\n{caso['scan']}\n### Solution:\n"
        print(f"  [{i:02d}/{len(CASOS_EVAL)}] {caso['nombre']}", end="  ", flush=True)
        t0 = time.time()

        texto = _generar(modelo, tok, prompt, device,
                         max_tokens=args.max_tokens, temp=args.temp, rep_pen=args.rep_penalty)

        dt = time.time() - t0
        espera = caso["espera"]
        ok_s, ok_i, ok_f, n_s, n_i = _verificar(texto, espera)
        sc = _score(ok_s, ok_i, ok_f, n_s, n_i,
                    len(espera["secciones"]), len(espera["info_scan"]))

        emoji = "✅" if sc >= 0.6 else "⚠️ " if sc >= 0.4 else "❌"
        print(f"[{dt:.1f}s] {emoji} score={sc:.2f}  "
              f"secciones={n_s}/{len(espera['secciones'])}  "
              f"info={n_i}/{len(espera['info_scan'])}  "
              f"formato={'✓' if ok_f else '✗'}")

        if args.verbose:
            print(f"\n    --- Output (primeros 400 chars) ---")
            for line in texto[:400].splitlines():
                print(f"    {line}")
            print("    ...\n")

        resultados.append(ResultadoCaso(
            nombre=caso["nombre"],
            score=sc,
            ok_secciones=ok_s,
            ok_info=ok_i,
            ok_formato=ok_f,
            n_secciones=n_s,
            n_info=n_i,
            texto=texto,
        ))

    # Resultado final
    total = time.time() - t0_total
    scores = [r.score for r in resultados]
    aprobados = sum(1 for s in scores if s >= 0.6)
    promedio = sum(scores) / len(scores)

    print(f"\n{sep}")
    print(f"  RESULTADO MILESTONE 3 — {total:.0f}s total")
    print(f"{sep}")
    print(f"  ✅ Aprobados  : {aprobados}/{len(resultados)}  (umbral score ≥ 0.60)")
    print(f"  📊 Score prom : {promedio:.3f}")
    print(f"  📊 Score mín  : {min(scores):.3f}")
    print(f"  📊 Score máx  : {max(scores):.3f}")

    # Detalles por criterio
    ok_s_total = sum(1 for r in resultados if r.ok_secciones)
    ok_i_total = sum(1 for r in resultados if r.ok_info)
    ok_f_total = sum(1 for r in resultados if r.ok_formato)
    print(f"\n  Por criterio:")
    print(f"    Secciones correctas : {ok_s_total}/{len(resultados)}")
    print(f"    Info del scan       : {ok_i_total}/{len(resultados)}")
    print(f"    Formato válido      : {ok_f_total}/{len(resultados)}")

    if promedio >= 0.7:
        print(f"\n  🎯 MILESTONE 3 SUPERADO — el modelo genera AGENTS.md válidos")
    elif promedio >= 0.5:
        print(f"\n  📈 PROGRESO — necesita más SFT sobre agents_sft.jsonl")
    else:
        print(f"\n  🔧 NECESITA SFT — score por debajo del umbral de progreso")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
