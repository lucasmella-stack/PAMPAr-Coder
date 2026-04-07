#!/usr/bin/env python3
# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
chat.py — Loop interactivo de PAMPAr-Coder.

El agente autónomo:
  1. Recibe un problema de programación en lenguaje natural
  2. Genera código Python usando el modelo (formato SFT)
  3. Ejecuta el código automáticamente con EjecutorCodigo
  4. Si falla → muestra el error y reintenta con el error como contexto
  5. Agotados los reintentos → guarda el fallo en ColaFinetune
  6. Cuando la cola alcanza el umbral → ofrece lanzar un mini-SFT

Uso:
  python -X utf8 scripts/chat.py
  python -X utf8 scripts/chat.py --checkpoint checkpoints/v3_sft_v7.pt
  python -X utf8 scripts/chat.py --temp 0.2 --max-tokens 600
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from pampar.memoria.clasificador import ClasificadorPareto, EntradaMemoria
from pampar.memoria.cola_finetune import ColaFinetune
from pampar.skills.ejecutar_codigo import EjecutorCodigo


ROOT = Path(__file__).resolve().parent.parent
SFT_SCRIPT = ROOT / "scripts" / "sft_v5.py"
GOOD_DATASET = ROOT / "data" / "final_sft.jsonl"

# =============================================================================
# Constantes
# =============================================================================

MAX_REINTENTOS = 2
MEMORIA_DIR = "memoria/data"
TOKENIZER_PATHS = [
    "data/tokenizer/pampar_48k.model",
    "data/tokenizer/code_tokenizer.model",
]


# =============================================================================
# Carga de modelo y tokenizer (igual que eval_v3.py)
# =============================================================================

def cargar_modelo(checkpoint: Path, device: torch.device):
    """Carga PamparV3 desde un checkpoint .pt."""
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
    modelo.to(device).eval()
    return modelo, cfg


def cargar_tokenizer(vocab_size: int) -> spm.SentencePieceProcessor:
    """Busca el tokenizer correcto según vocab_size."""
    sp = spm.SentencePieceProcessor()
    for path in TOKENIZER_PATHS:
        p = Path(path)
        if p.exists():
            sp.Load(str(p))
            if sp.vocab_size() == vocab_size:
                return sp
    raise FileNotFoundError(
        f"No se encontró tokenizer con vocab_size={vocab_size}. "
        f"Buscado en: {TOKENIZER_PATHS}"
    )


# =============================================================================
# Generación (adaptada de eval_v3.py, misma lógica que funciona)
# =============================================================================

def generar(
    modelo,
    tokenizer: spm.SentencePieceProcessor,
    prompt: str,
    device: torch.device,
    max_tokens: int = 512,
    temperature: float = 0.1,
    repetition_penalty: float = 1.15,
    rep_window: int = 32,
) -> str:
    """
    Genera código a partir de un prompt en formato SFT.

    Devuelve solo la parte generada (no el prompt), limpia y lista para ejecutar.
    """
    ids = tokenizer.Encode(prompt)
    generados = list(ids)

    for _ in range(max_tokens):
        ctx = torch.tensor([generados[-512:]], dtype=torch.long, device=device)
        logits, _, _ = modelo(ctx)
        next_logits = logits[0, -1]

        # Penalizar repetición en ventana local (no destruir keywords de Python)
        if repetition_penalty != 1.0 and len(generados) > len(ids):
            window_start = max(len(ids), len(generados) - rep_window)
            seen = set(generados[window_start:])
            for token_id in seen:
                if next_logits[token_id] > 0:
                    next_logits[token_id] /= repetition_penalty
                else:
                    next_logits[token_id] *= repetition_penalty

        if temperature <= 0.0:
            next_token = int(next_logits.argmax())
        else:
            next_logits = next_logits / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = int(torch.multinomial(probs, 1))

        generados.append(next_token)
        decoded = tokenizer.Decode(generados[len(ids):]).replace("\u2047", "\n")

        # Parar si el modelo empieza una nueva sección
        if "###" in decoded:
            idx = decoded.index("###")
            if idx > 10:
                return decoded[:idx].rstrip()

        # Parar cuando termina la función (línea sin sangría después de contenido)
        lines = decoded.split("\n")
        if len(lines) > 3:
            for i, line in enumerate(lines[2:], 2):
                if line and not line[0].isspace() and line.strip() not in ("", "pass"):
                    return "\n".join(lines[:i])

        # Parar solo cuando el bloque TERMINÓ con línea en blanco (no mid-función)
        if decoded.endswith("\n\n") and len(decoded) > 20:
            return decoded.rstrip()

    return tokenizer.Decode(generados[len(ids):]).replace("\u2047", "\n")


# =============================================================================
# Normalización de indentación (ídem eval_v3.py)
# =============================================================================

def _normalizar_indentacion(codigo: str) -> str:
    """Corrige indentación inconsistente a múltiplos de 4 espacios."""
    lineas = codigo.splitlines()
    normalizadas = []
    for linea in lineas:
        if not linea.strip():
            normalizadas.append("")
            continue
        n_spaces = len(linea) - len(linea.lstrip())
        n_tabs = linea[:n_spaces].count("\t")
        total = n_spaces + n_tabs * 4 - n_tabs
        nivel = round(total / 4)
        normalizadas.append("    " * nivel + linea.lstrip())
    return "\n".join(normalizadas)


# =============================================================================
# Loop de coding autónomo
# =============================================================================

class CodingLoop:
    """
    Loop autónomo: genera → ejecuta → observa error → reintenta → aprende.

    Si el modelo falla repetidamente en un problema, el par (problema, error, código)
    va a la ColaFinetune. Cuando la cola tiene suficientes ejemplos, se ofrece
    lanzar un mini-SFT para mejorar el modelo.
    """

    def __init__(
        self,
        checkpoint: Path,
        device: torch.device,
        max_reintentos: int = MAX_REINTENTOS,
        temperatura: float = 0.1,
        rep_penalty: float = 1.15,
        max_tokens: int = 512,
        memoria_dir: str = MEMORIA_DIR,
    ):
        print(f"\n{'═'*60}")
        print(f"  PAMPAr-Coder — Cargando modelo...")
        print(f"{'═'*60}")

        t0 = time.time()
        print(f"  Checkpoint : {checkpoint.name}", end=" ", flush=True)
        self.modelo, self.cfg = cargar_modelo(checkpoint, device)
        self.tok = cargar_tokenizer(self.cfg.vocab_size)
        self.modelo.registrar_tokenizer(self.tok)
        n_params = sum(p.numel() for p in self.modelo.parameters()) / 1e6
        print(f"({n_params:.1f}M params, {time.time() - t0:.1f}s)")

        self.checkpoint = checkpoint
        self.device = device
        self.max_reintentos = max_reintentos
        self.temp = temperatura
        self.rep_penalty = rep_penalty
        self.max_tokens = max_tokens

        self.ejecutor = EjecutorCodigo(timeout=15)
        self.clasificador = ClasificadorPareto()
        self.cola = ColaFinetune(
            directorio=memoria_dir,
            min_ejemplos=50,
            callback_proponer=self._proponer_finetune,
        )

        print(f"  Dispositivo: {device}")
        print(f"  Cola aprendizaje: {len(self.cola)} ejemplos acumulados")
        print(f"{'═'*60}\n")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _hacer_mini_sft(self) -> bool:
        """
        Lanza sft_v5.py como refresher SFT sobre final_sft.jsonl,
        espera a que termine (síncrono), recarga el modelo en proceso
        y vacía la cola.

        Returns True si el training y la recarga fueron exitosos.
        """
        if not GOOD_DATASET.exists():
            print(f"  ⚠️  No se encontró {GOOD_DATASET}. Cancelando mini-SFT.")
            return False

        # Determinar nombre del nuevo checkpoint sin sobrescribir
        idx = 1
        while (ROOT / "checkpoints" / f"v3_sft_chat_v{idx}.pt").exists():
            idx += 1
        nuevo_ckpt = ROOT / "checkpoints" / f"v3_sft_chat_v{idx}.pt"

        print(f"  Dataset  : {GOOD_DATASET.name} ({GOOD_DATASET.stat().st_size // 1024} KB)")
        print(f"  Entrada  : {self.checkpoint.name}")
        print(f"  Salida   : {nuevo_ckpt.name}")
        print(f"  Entrenando... (puede tardar varios minutos)\n")

        cmd = [
            sys.executable, str(SFT_SCRIPT),
            "--checkpoint-in", str(self.checkpoint),
            "--checkpoint-out", str(nuevo_ckpt),
            "--targeted", str(GOOD_DATASET),
            "--lr", "5e-7",
            "--lr-min", "5e-8",
            "--max-pasos", "200",
            "--epochs", "5",
            "--warmup", "10",
        ]

        result = subprocess.run(cmd, cwd=str(ROOT))

        if result.returncode != 0:
            print(f"\n  ❌ Mini-SFT falló (rc={result.returncode}). Checkpoint no guardado.")
            return False

        if not nuevo_ckpt.exists():
            print(f"\n  ❌ Checkpoint no se creó ({nuevo_ckpt.name}).")
            return False

        # Recargar modelo en proceso
        print(f"\n  ♻️  Recargando modelo desde {nuevo_ckpt.name}...")
        self.modelo, self.cfg = cargar_modelo(nuevo_ckpt, self.device)
        self.tok = cargar_tokenizer(self.cfg.vocab_size)
        self.modelo.registrar_tokenizer(self.tok)
        self.checkpoint = nuevo_ckpt

        n_vaciados = self.cola.vaciar_post_finetune()
        print(f"  ✅ Modelo actualizado. Cola vaciada ({n_vaciados} ejemplos procesados).\n")
        return True

    def _proponer_finetune(self, n: int, stats: dict) -> bool:
        """
        Callback cuando la cola supera el umbral.

        Lanza mini-SFT de forma síncrona si el usuario acepta,
        recarga el modelo y vacía la cola.
        Siempre retorna False para evitar que ColaFinetune llame
        al lanzar_finetune() antiguo (que usa el script obsoleto).
        """
        print(f"\n{'─'*60}")
        print(f"📚 Cola de aprendizaje lista: {n} ejemplos")
        print(f"   Importancia promedio: {stats.get('importancia_promedio', '?')}")
        print(f"   ¿Querés lanzar un mini-SFT para que el modelo mejore?")
        try:
            resp = input("   [s/N] >>> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            resp = "n"

        if resp in ("s", "si", "sí", "y", "yes"):
            print("\n  Lanzando mini-SFT...\n")
            self._hacer_mini_sft()
        else:
            print("   OK. La cola se conserva y sigue acumulando.\n")

        # Siempre False — el lanzar_finetune() de ColaFinetune no debe ejecutarse
        return False

    # ── Persistencia de errores ────────────────────────────────────────────────

    def _guardar_error(self, problema: str, codigo: str, error: str) -> None:
        """Persiste un fallo en la ColaFinetune para futura mejora."""
        texto = (
            f"### Problem:\n{problema}\n"
            f"### Attempted:\n{codigo}\n"
            f"### Error:\n{error}"
        )
        entrada = self.clasificador.clasificar(texto=texto, tipo="error")
        # Los errores del modelo son siempre L3 — siempre queremos aprender de ellos
        entrada.nivel = 3
        entrada.importancia = 0.90
        self.cola.agregar(entrada)

    # ── Ciclo principal ────────────────────────────────────────────────────────

    def responder(self, problema: str) -> None:
        """
        Procesa un problema:
          1. Genera código con el modelo
          2. Lo ejecuta
          3. Si falla → añade el error al contexto y reintenta
          4. Si sigue fallando → guarda en ColaFinetune
        """
        prompt_base = f"### Problem:\n{problema}\n### Solution:\n"
        prompt = prompt_base
        ultimo_codigo = ""
        ultimo_error = ""

        for intento in range(1, self.max_reintentos + 2):
            label = f"intento {intento}/{self.max_reintentos + 1}"
            print(f"  ⚡ Generando ({label})...", end=" ", flush=True)
            t0 = time.time()

            codigo_raw = generar(
                self.modelo, self.tok, prompt, self.device,
                max_tokens=self.max_tokens,
                temperature=self.temp,
                repetition_penalty=self.rep_penalty,
            )
            codigo = _normalizar_indentacion(codigo_raw.strip())
            print(f"{time.time() - t0:.1f}s")

            # Mostrar código generado
            print()
            print("  " + "─" * 50)
            for line in codigo.splitlines():
                print(f"  {line}")
            print("  " + "─" * 50)

            # Ejecutar
            resultado = self.ejecutor.execute(codigo=codigo)
            ultimo_codigo = codigo

            if resultado.exito:
                print(f"\n  ✅ Ejecutado correctamente\n")
                if resultado.contenido.strip():
                    print(f"  Output:")
                    for line in resultado.contenido.strip().splitlines():
                        print(f"    {line}")
                print()
                return

            # Falló
            ultimo_error = resultado.error or "Error desconocido"
            print(f"\n  ⚠️  Error: {ultimo_error}")

            if intento <= self.max_reintentos:
                print(f"  → Reintentando con el error como contexto...\n")
                # El modelo ve su propio error y tiene otra oportunidad
                prompt = (
                    prompt_base
                    + codigo
                    + f"\n\n# ⚠️ Error en el código anterior:\n# {ultimo_error}\n"
                    + "# Corrección:\n"
                )

        # Agotamos todos los reintentos
        print(f"\n  📚 Guardando en cola de aprendizaje...", end=" ", flush=True)
        self._guardar_error(problema, ultimo_codigo, ultimo_error)
        stats = self.cola.stats()
        print(
            f"cola: {stats['total']}/{self.cola.min_ejemplos} "
            f"(faltan {stats['faltan_para_finetune']} para mini-SFT)"
        )
        print()

    # ── REPL ──────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Loop interactivo. Escribe un problema → el modelo lo resuelve."""
        print("─" * 60)
        print("  🦙 PAMPAr-Coder — Chat interactivo")
        print("     Describe un problema de programación en Python.")
        print("     'salir' o Ctrl+C para terminar.")
        print("─" * 60)
        print()

        while True:
            try:
                problema = input(">>> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\nHasta pronto.")
                break

            if not problema:
                continue

            if problema.lower() in ("salir", "quit", "exit"):
                print("Hasta pronto.")
                break

            print()
            self.responder(problema)


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PAMPAr-Coder — Loop interactivo de generación y ejecución de código"
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/v3_sft_v7.pt",
        help="Checkpoint del modelo (default: v3_sft_v7.pt)",
    )
    parser.add_argument("--device", default="auto", help="cuda / cpu / auto")
    parser.add_argument("--temp", type=float, default=0.1, help="Temperatura de generación")
    parser.add_argument("--rep-penalty", type=float, default=1.15, help="Penalización de repetición")
    parser.add_argument("--max-tokens", type=int, default=512, help="Máximo tokens a generar")
    parser.add_argument("--max-reintentos", type=int, default=2, help="Reintentos si falla")
    parser.add_argument("--memoria-dir", default=MEMORIA_DIR, help="Directorio para ColaFinetune")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    agente = CodingLoop(
        checkpoint=Path(args.checkpoint),
        device=device,
        max_reintentos=args.max_reintentos,
        temperatura=args.temp,
        rep_penalty=args.rep_penalty,
        max_tokens=args.max_tokens,
        memoria_dir=args.memoria_dir,
    )
    agente.run()


if __name__ == "__main__":
    main()
