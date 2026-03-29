# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
🎓 Script de Destilación — Aprender de un Profesor

Genera datos de entrenamiento de alta calidad usando un modelo
profesor (GPT-4o-mini, Claude Haiku, Qwen local, CodeLlama local).

Modos:
  A) GENERAR datos offline (crear dataset destilado)
  B) CORREGIR (alumno genera → profesor corrige → pares DPO)

Uso:
  # Generar dataset con GPT-4o-mini (más barato, ~$0.15 total)
  python scripts/destilar.py --profesor gpt4o-mini --modo codigo --n 1000

  # Con Qwen-7B local via Ollama (gratis)
  python scripts/destilar.py --profesor qwen-7b --modo cot --n 500

  # Modo corrección (necesita checkpoint del alumno)
  python scripts/destilar.py --profesor gpt4o-mini --modo correccion \\
      --checkpoint checkpoints/cerebral/fase1_best.pt --n 200

  # Entrenar con dataset destilado existente
  python scripts/destilar.py --entrenar --dataset data/distillation/destilado.jsonl

  # Generar + entrenar en un solo paso
  python scripts/destilar.py --profesor qwen-7b --modo codigo --n 200 --entrenar

Requisitos:
  - Para OpenAI: OPENAI_API_KEY en environment
  - Para Anthropic: ANTHROPIC_API_KEY en environment
  - Para Ollama: ollama corriendo en localhost:11434
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

# Ajustar path
script_dir = Path(__file__).parent
project_dir = script_dir.parent
sys.path.insert(0, str(project_dir))

from pampar.coder.v2.config import PRESET_1_5B, PRESET_8GB, PRESET_4GB
from pampar.coder.v2.modelo import PampaRCoderV2, crear_modelo
from pampar.coder.v2.aprendizaje.destilacion import (
    PROFESORES,
    ConfigProfesor,
    ClienteProfesor,
    GeneradorDestilacion,
    distillation_loss,
    territory_aware_distillation,
)


def cargar_tokenizer(path: str):
    """Carga el tokenizer SentencePiece."""
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(path)
    return sp


def generar_dataset(args):
    """Genera un dataset destilado usando un profesor."""
    print(f"\n{'=' * 60}")
    print(f"🎓 GENERACIÓN DE DATOS DESTILADOS")
    print(f"{'=' * 60}")

    # Seleccionar profesor
    if args.profesor in PROFESORES:
        config_prof = PROFESORES[args.profesor]
    else:
        print(f"  ❌ Profesor '{args.profesor}' no encontrado")
        print(f"  Disponibles: {', '.join(PROFESORES.keys())}")
        return None

    # API key
    if args.api_key:
        config_prof.api_key = args.api_key

    print(f"  Profesor: {config_prof.nombre} ({config_prof.modelo})")
    print(f"  Tipo: {config_prof.tipo}")
    print(f"  Modo: {args.modo}")
    print(f"  Ejemplos a generar: {args.n}")

    # Estimar costo
    tokens_estimados = args.n * 500  # ~500 tokens/ejemplo promedio
    costo_input = (tokens_estimados / 1000) * config_prof.costo_por_1k_input
    costo_output = (tokens_estimados / 1000) * config_prof.costo_por_1k_output
    costo_total = costo_input + costo_output

    if costo_total > 0:
        print(f"\n  💰 Costo estimado: ${costo_total:.4f}")
        print(f"     Input: ${costo_input:.4f} | Output: ${costo_output:.4f}")
        if not args.si:
            respuesta = input("  ¿Continuar? (s/n): ")
            if respuesta.lower() not in ("s", "si", "sí", "y", "yes"):
                print("  Cancelado.")
                return None
    else:
        print(f"  💰 Modelo local — ¡GRATIS!")

    # Crear cliente
    cliente = ClienteProfesor(config_prof)

    # Crear generador
    generador = GeneradorDestilacion(cliente, modo=args.modo)

    # Para modo corrección, cargar modelo alumno
    alumno_modelo = None
    alumno_tokenizer = None
    if args.modo == "correccion":
        if not args.checkpoint:
            print("  ❌ Modo corrección necesita --checkpoint del alumno")
            return None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        preset = {"4gb": PRESET_4GB, "8gb": PRESET_8GB, "1.5b": PRESET_1_5B}[args.preset]
        alumno_modelo = crear_modelo(preset).to(device)

        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        alumno_modelo.load_state_dict(ckpt["model"], strict=False)
        alumno_modelo.eval()

        tokenizer_path = str(project_dir / "data" / "tokenizer" / "pampar_48k.model")
        alumno_tokenizer = cargar_tokenizer(tokenizer_path)
        print(f"  Alumno cargado desde: {args.checkpoint}")

    # Generar datos
    output_path = args.output or f"data/distillation/destilado_{args.profesor}_{args.modo}.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Determinar niveles a generar
    niveles = list(range(1, 7))
    if args.niveles:
        niveles = [int(n) for n in args.niveles.split(",")]

    print(f"\n  Generando datos...")
    print(f"  Output: {output_path}")
    print(f"  Niveles: {niveles}")
    print(f"  {'─' * 40}")

    n_generados = 0
    n_errores = 0
    costo_real = 0.0
    t_start = time.time()

    with open(output_path, "w", encoding="utf-8") as f:
        for nivel in niveles:
            n_por_nivel = args.n // len(niveles)
            print(f"\n  📚 Nivel {nivel}: generando {n_por_nivel} ejemplos...")

            for i in range(n_por_nivel):
                try:
                    if args.modo == "correccion" and alumno_modelo is not None:
                        # Modo corrección: alumno genera, profesor corrige
                        prompt = PROMPTS_DESTILACION[nivel][i % len(PROMPTS_DESTILACION[nivel])]

                        # Generar con alumno
                        tokens = alumno_tokenizer.Encode(prompt)
                        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
                        with torch.no_grad():
                            output = alumno_modelo.generate(input_ids, max_tokens=256, temperature=0.8)
                        codigo_alumno = alumno_tokenizer.Decode(output[0, len(tokens):].tolist())

                        # Profesor corrige
                        prompt_correccion = (
                            f"Un estudiante intentó resolver:\n{prompt}\n\n"
                            f"Código del estudiante:\n```python\n{codigo_alumno}\n```\n\n"
                            f"Corrige el código y explica."
                        )
                        correccion_text = cliente.generar(prompt_correccion)
                        resultado = {
                            "instruction": prompt,
                            "output": correccion_text,
                            "alumno": codigo_alumno,
                            "nivel": nivel,
                            "tipo": "correccion",
                        }
                    else:
                        # Generar normalmente
                        resultado = generador.generar(nivel=nivel)

                    if resultado:
                        f.write(json.dumps(resultado, ensure_ascii=False) + "\n")
                        n_generados += 1

                        # Estimar costo real
                        tokens_usados = len(resultado.get("output", "").split()) * 1.3
                        costo_real += (tokens_usados / 1000) * (
                            config_prof.costo_por_1k_output
                        )

                except Exception as e:
                    n_errores += 1
                    if n_errores <= 5:
                        print(f"    ⚠️  Error: {e}")
                    elif n_errores == 6:
                        print(f"    ⚠️  Suprimiendo errores adicionales...")

                # Progreso
                total = n_generados + n_errores
                if total % 50 == 0 and total > 0:
                    elapsed = time.time() - t_start
                    rate = n_generados / elapsed if elapsed > 0 else 0
                    print(
                        f"    {n_generados}/{args.n} generados "
                        f"({n_errores} errores) | "
                        f"{rate:.1f}/seg | "
                        f"Costo: ${costo_real:.4f}"
                    )

                # Rate limiting
                if config_prof.tipo in ("openai", "anthropic"):
                    time.sleep(0.2)  # Evitar rate limits

    elapsed = time.time() - t_start
    print(f"\n  {'=' * 40}")
    print(f"  ✅ Generación completada")
    print(f"  Ejemplos: {n_generados} ({n_errores} errores)")
    print(f"  Tiempo: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Costo real: ${costo_real:.4f}")
    print(f"  Output: {output_path}")

    return output_path


def entrenar_con_destilacion(args, dataset_path):
    """Entrena PAMPAr con datos destilados."""
    print(f"\n{'=' * 60}")
    print(f"📖 ENTRENAMIENTO CON DATOS DESTILADOS")
    print(f"{'=' * 60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Cargar config
    preset = {"4gb": PRESET_4GB, "8gb": PRESET_8GB, "1.5b": PRESET_1_5B}[args.preset]

    # Crear o cargar modelo
    model = crear_modelo(preset).to(device)
    if args.checkpoint:
        print(f"  Cargando checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)

    # Tokenizer
    tokenizer_path = str(project_dir / "data" / "tokenizer" / "pampar_48k.model")
    tokenizer = cargar_tokenizer(tokenizer_path)
    model.registrar_tokenizer(tokenizer)

    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Dataset: {dataset_path}")

    # Cargar y tokenizar datos
    datos = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                datos.append(json.loads(line))

    print(f"  Ejemplos cargados: {len(datos)}")

    if not datos:
        print("  ❌ Dataset vacío")
        return

    # Preparar optimizer
    lr = args.lr or 1e-4
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=0.1,
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # Entrenamiento
    batch_size = args.batch_size or 4
    grad_accum = args.grad_accum or 8
    max_steps = args.max_steps or len(datos) * 3  # 3 epochs
    max_seq_len = 2048

    model.train()
    paso = 0
    mejor_loss = float("inf")
    loss_accum = 0.0
    n_accum = 0

    import random
    random.shuffle(datos)

    print(f"  LR: {lr}")
    print(f"  Batch: {batch_size} × {grad_accum} = {batch_size * grad_accum}")
    print(f"  Max steps: {max_steps}")
    print(f"\n  Entrenando...")

    for epoch in range(10):  # Múltiples epochs si hace falta
        for ejemplo in datos:
            # Tokenizar: input + output concatenados
            texto = ejemplo.get("input", "") + "\n" + ejemplo.get("output", "")
            tokens = tokenizer.Encode(texto)[:max_seq_len]

            if len(tokens) < 4:
                continue

            input_ids = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
            targets = torch.tensor([tokens[1:]], dtype=torch.long, device=device)

            # Forward
            with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                logits, loss, info = model(input_ids, targets)

                # Si tiene razonamiento CoT, aplicar territory-aware distillation
                if "razonamiento" in ejemplo:
                    loss_terr = territory_aware_distillation(
                        model, tokenizer, ejemplo["output"],
                        ejemplo.get("nivel", 3),
                    )
                    loss = loss + 0.1 * loss_terr

                loss = loss / grad_accum

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            loss_accum += loss.item() * grad_accum
            n_accum += 1

            if n_accum >= grad_accum:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad()

                paso += 1
                avg_loss = loss_accum / n_accum

                if avg_loss < mejor_loss:
                    mejor_loss = avg_loss

                if paso % 50 == 0:
                    print(
                        f"    Paso {paso:5d} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Mejor: {mejor_loss:.4f} | "
                        f"Epoch: {epoch}"
                    )

                if paso % 200 == 0 and avg_loss <= mejor_loss:
                    ckpt_path = f"checkpoints/cerebral/destilacion_best.pt"
                    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                    torch.save({
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "paso": paso,
                        "loss": avg_loss,
                        "config": {
                            k: getattr(model.config, k)
                            for k in model.config.__dataclass_fields__
                        },
                    }, ckpt_path)
                    print(f"    💾 Best checkpoint: {ckpt_path}")

                loss_accum = 0.0
                n_accum = 0

                if paso >= max_steps:
                    break

        if paso >= max_steps:
            break

    # Guardar checkpoint final
    ckpt_path = f"checkpoints/cerebral/destilacion_final.pt"
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "paso": paso,
        "loss": mejor_loss,
        "config": {k: getattr(model.config, k) for k in model.config.__dataclass_fields__},
    }, ckpt_path)

    print(f"\n  ✅ Entrenamiento completado")
    print(f"  Pasos: {paso}, Mejor loss: {mejor_loss:.4f}")
    print(f"  Checkpoint: {ckpt_path}")


def main():
    parser = argparse.ArgumentParser(
        description="🎓 Destilación de Conocimiento para PAMPAr-Coder"
    )

    # Modo generación
    parser.add_argument(
        "--profesor", type=str, default=None,
        help=f"Profesor a usar: {', '.join(PROFESORES.keys())}"
    )
    parser.add_argument(
        "--modo", type=str, default="codigo",
        choices=["codigo", "cot", "correccion"],
        help="Modo de destilación"
    )
    parser.add_argument(
        "--n", type=int, default=100,
        help="Número de ejemplos a generar"
    )
    parser.add_argument(
        "--niveles", type=str, default=None,
        help="Niveles a generar (ej: 1,2,3)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path de salida para datos destilados"
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="API key del profesor"
    )
    parser.add_argument(
        "--si", "-y", action="store_true",
        help="Aceptar costo sin preguntar"
    )

    # Modo entrenamiento
    parser.add_argument(
        "--entrenar", action="store_true",
        help="Entrenar con dataset destilado"
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Path al dataset destilado (si no generar)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Checkpoint del modelo"
    )
    parser.add_argument(
        "--preset", type=str, default="4gb",
        choices=["4gb", "8gb", "1.5b"],
    )
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)

    args = parser.parse_args()

    # Determinar qué hacer
    dataset_path = args.dataset

    if args.profesor:
        # Generar datos
        dataset_path = generar_dataset(args)
        if dataset_path is None and not args.dataset:
            return

    if args.entrenar:
        if dataset_path is None:
            print("  ❌ Se necesita --dataset o --profesor para entrenar")
            return
        entrenar_con_destilacion(args, dataset_path)

    if not args.profesor and not args.entrenar:
        parser.print_help()
        print(f"\n  Profesores disponibles:")
        for key, prof in PROFESORES.items():
            costo = prof.costo_por_1k_output
            tag = f"${costo:.4f}/1K tok" if costo > 0 else "GRATIS (local)"
            print(f"    {key:15s} — {prof.nombre} [{tag}]")


if __name__ == "__main__":
    main()
