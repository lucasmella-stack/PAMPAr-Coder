#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
train_v3.py — Entrenamiento autónomo de PamparV3.

Arquitectura:
  - MotorCuriosidad  → elige qué tema estudiar (ZDP Vygotsky)
  - LectorBiblioteca → devuelve batches de tokens
  - PamparV3         → calcula logits y loss internamente
  - Replay buffer    → revisión de experiencias duras (deque)

Uso rápido:
  python scripts/train_v3.py
  python scripts/train_v3.py --checkpoint checkpoints/run.pt --max-pasos 0
  python scripts/train_v3.py --lr 3e-4 --seq-len 256 --batch-size 2

Detener limpiamente con Ctrl-C — guarda el checkpoint antes de salir.
"""

import argparse
import dataclasses
import json
import logging
import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import torch
import torch.nn.utils as nn_utils
import sentencepiece as spm

# ── Rutas relativas al script ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent          # PAMPAr-Coder/
sys.path.insert(0, str(ROOT))

from pampar.coder.v3 import PamparV3, ConfigV3, PRESET_V3
from pampar.training import MotorCuriosidad, LectorBiblioteca

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_v3")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cargar_indice(ruta: Path) -> dict:
    """Carga biblioteca/indice.json y valida el formato básico."""
    if not ruta.exists():
        raise FileNotFoundError(f"Índice no encontrado: {ruta}")
    with ruta.open(encoding="utf-8") as f:
        data = json.load(f)
    return data


def _construir_mapa_tema(indice: dict) -> dict[str, str]:
    """
    Devuelve {nombre_tema: ruta_relativa_jsonl}.
    Ignora claves del índice que no sean listas.
    """
    mapa: dict[str, str] = {}
    for _categoria, temas in indice.items():
        if not isinstance(temas, list):
            continue
        for t in temas:
            nombre = t.get("nombre", "")
            archivo = t.get("archivo", "")
            if nombre and archivo:
                mapa[nombre] = archivo
    return mapa


def _pp_loss(loss: float) -> str:
    """Formatea el loss con su perplexity."""
    ppl = math.exp(min(loss, 20.0))
    return f"loss={loss:.3f}  ppl={ppl:.1f}"


# ─────────────────────────────────────────────────────────────────────────────
# ViajeIntelectualV3 — bucle de entrenamiento autónomo
# ─────────────────────────────────────────────────────────────────────────────

class ViajeIntelectualV3:
    """
    Orquesta el ciclo estudiar → retroalimentar → avanzar de PamparV3.

    Args:
        modelo:        Instancia de PamparV3 ya en `device`.
        optimizer:     Optimizer Torch (AdamW recomendado).
        motor:         MotorCuriosidad con temas ya registrados.
        biblioteca:    LectorBiblioteca apuntando a biblioteca/.
        mapa_tema:     {nombre → ruta_relativa_jsonl}
        device:        Dispositivo de entrenamiento.
        ruta_ckpt:     Ruta del archivo de checkpoint.
        paso_global:   Paso inicial (0 si es nuevo entrenamiento).
        replay_cada:   Pasos entre cada sesión de replay. 0 = desactivado.
        guardar_cada:  Pasos entre checkpoints.
        pasos_por_tema: Gradients steps por sesión de tema.
        max_grad_norm: Clip de gradiente.
        replay_size:   Nº muestras máximas en el buffer de replay.
    """

    def __init__(
        self,
        modelo: PamparV3,
        optimizer: torch.optim.Optimizer,
        motor: MotorCuriosidad,
        biblioteca: LectorBiblioteca,
        mapa_tema: dict[str, str],
        device: torch.device,
        ruta_ckpt: Path,
        paso_global: int = 0,
        replay_cada: int = 500,
        guardar_cada: int = 200,
        pasos_por_tema: int = 20,
        max_grad_norm: float = 1.0,
        replay_size: int = 512,
    ) -> None:
        self.modelo = modelo
        self.optimizer = optimizer
        self.motor = motor
        self.biblioteca = biblioteca
        self.mapa_tema = mapa_tema
        self.device = device
        self.ruta_ckpt = ruta_ckpt

        self.paso_global = paso_global
        self.replay_cada = replay_cada
        self.guardar_cada = guardar_cada
        self.pasos_por_tema = pasos_por_tema
        self.max_grad_norm = max_grad_norm

        # Replay buffer: almacena tensores [B, L] de sesiones difíciles
        self._replay_buffer: deque[torch.Tensor] = deque(maxlen=replay_size)

        # Historial de losses para el banner
        self._historial_loss: deque[float] = deque(maxlen=100)

        # Estadísticas
        self._temas_estudiados = 0
        self._tiempo_inicio = time.time()

    # ── Paso de entrenamiento ─────────────────────────────────────────────────

    def _gradiente(self, tokens: torch.Tensor) -> dict:
        """
        Realiza un paso de gradiente y devuelve métricas.

        Args:
            tokens: Tensor [B, L] con tokens de entrada + target.
        Returns:
            Dict con loss, exit_nivel, terr_ratio.
        """
        self.modelo.train()
        self.optimizer.zero_grad(set_to_none=True)

        input_ids = tokens[:, :-1]
        targets = tokens[:, 1:]

        logits, loss, info = self.modelo(input_ids, targets=targets)

        loss.backward()
        nn_utils.clip_grad_norm_(self.modelo.parameters(), self.max_grad_norm)
        self.optimizer.step()

        loss_val = float(loss.detach())
        self._historial_loss.append(loss_val)

        # Extraer métricas de salida temprana
        terr_acts = info.get("terr_acts", None)
        terr_ratio = float(terr_acts.float().mean()) if terr_acts is not None else 0.0

        return {
            "loss": loss_val,
            "exit_nivel": info.get("exit_nivel", -1),
            "terr_ratio": terr_ratio,
        }

    # ── Paso de replay ────────────────────────────────────────────────────────

    def _paso_replay(self) -> Optional[float]:
        """Repasa muestras del buffer de experiencias. None si está vacío."""
        if not self._replay_buffer:
            return None

        batch = self._replay_buffer[
            int(torch.randint(0, len(self._replay_buffer), (1,)))
        ].to(self.device)

        metricas = self._gradiente(batch)
        return metricas["loss"]

    def _agregar_a_replay(self, tokens: torch.Tensor, loss: float, umbral: float = 3.5) -> None:
        """Solo guarda en el buffer si la sesión fue difícil."""
        if loss > umbral:
            self._replay_buffer.append(tokens.cpu())

    # ── Guardado ──────────────────────────────────────────────────────────────

    def _guardar(self, ruta_motor: Path) -> None:
        """Guarda checkpoint del modelo + estado del motor."""
        self.ruta_ckpt.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "modelo": self.modelo.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "paso_global": self.paso_global,
            "config": dataclasses.asdict(self.modelo.config),
        }
        torch.save(payload, self.ruta_ckpt)

        self.motor.guardar(ruta_motor)
        logger.info("✓ Checkpoint guardado → paso %d", self.paso_global)

    # ── Banner ────────────────────────────────────────────────────────────────

    def _banner(self, nombre_tema: str, loss: float) -> None:
        """Imprime un resumen compacto del estado actual."""
        res = self.motor.resumen()
        elapsed = time.time() - self._tiempo_inicio
        hh = int(elapsed // 3600); mm = int((elapsed % 3600) // 60)

        avg_loss = sum(self._historial_loss) / len(self._historial_loss) if self._historial_loss else 0.0

        print(
            f"\n[paso {self.paso_global:>6}  {hh:02d}h{mm:02d}m]"
            f"  nivel={res['nivel_actual']}"
            f"  dom={res['temas_dominados']}/{res['temas_total']}"
            f"  {_pp_loss(loss)}"
            f"  avg100={avg_loss:.3f}"
            f"\n  tema → {nombre_tema}"
        )

    # ── Bucle principal ───────────────────────────────────────────────────────

    def estudiar(self, max_pasos: int = 0, ruta_motor: Optional[Path] = None) -> None:
        """
        Entra en el bucle de entrenamiento autónomo.

        Args:
            max_pasos: Límite de pasos totales. 0 = infinito (Ctrl-C para parar).
            ruta_motor: Ruta donde persistir el estado del motor de curiosidad.
        """
        if ruta_motor is None:
            ruta_motor = self.ruta_ckpt.parent / "motor_v3.json"

        logger.info("Iniciando ViajeIntelectualV3 — %s", "∞ pasos" if max_pasos == 0 else f"{max_pasos} pasos")
        logger.info("Checkpoint → %s", self.ruta_ckpt)

        try:
            self._bucle_principal(max_pasos, ruta_motor)
        except KeyboardInterrupt:
            logger.info("\nInterrumpido por el usuario — guardando...")
        finally:
            self._guardar(ruta_motor)

        res = self.motor.resumen()
        print(
            f"\n── Fin del viaje ──"
            f"\n  Pasos totales : {self.paso_global}"
            f"\n  Temas dominados: {res['temas_dominados']}/{res['temas_total']}"
            f"\n  Nivel actual  : {res['nivel_actual']}"
        )

    def _bucle_principal(self, max_pasos: int, ruta_motor: Path) -> None:
        """Bucle interno de entrenamiento tema a tema."""
        pasos_sin_tema = 0

        while True:
            # ── 1. ELEGIR ────────────────────────────────────────────────────
            nombre_tema = self.motor.siguiente_tema()
            if nombre_tema is None:
                pasos_sin_tema += 1
                if pasos_sin_tema > 10:
                    logger.warning("Sin temas disponibles — esperando 5s...")
                    time.sleep(5)
                continue
            pasos_sin_tema = 0

            archivo = self.mapa_tema.get(nombre_tema)
            if not archivo:
                logger.debug("Tema '%s' sin archivo mapeado, saltando.", nombre_tema)
                self.motor.retroalimentar(nombre_tema, 4.0)  # penalizar
                continue

            # ── 2. SESIÓN DE TEMA ────────────────────────────────────────────
            losses_sesion: list[float] = []

            for _ in range(self.pasos_por_tema):
                # ── 2a. LEER
                tokens = self.biblioteca.obtener_batch(archivo, self.device)
                if tokens is None:
                    break

                # ── 2b. GRADIENTE
                metricas = self._gradiente(tokens)
                loss = metricas["loss"]
                losses_sesion.append(loss)
                self.paso_global += 1

                # ── 2c. REPLAY BUFFER
                self._agregar_a_replay(tokens, loss)

                # ── 2d. REPLAY periódico
                if self.replay_cada > 0 and self.paso_global % self.replay_cada == 0:
                    rl = self._paso_replay()
                    if rl is not None:
                        logger.debug("Replay loss: %.3f", rl)

                # ── 2e. GUARDAR periódico
                if self.paso_global % self.guardar_cada == 0:
                    self._guardar(ruta_motor)

                # ── 2f. Límite de pasos
                if max_pasos > 0 and self.paso_global >= max_pasos:
                    return

            # ── 3. FEEDBACK ──────────────────────────────────────────────────
            if losses_sesion:
                loss_media = sum(losses_sesion) / len(losses_sesion)
                self.motor.retroalimentar(nombre_tema, loss_media)
                self._temas_estudiados += 1

                if self._temas_estudiados % 5 == 0:
                    self._banner(nombre_tema, loss_media)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Entrenamiento autónomo de PamparV3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Rutas
    p.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "v3_train.pt",
                   help="Ruta del checkpoint a guardar/reanudar")
    p.add_argument("--tokenizer", type=Path, default=ROOT / "data" / "tokenizer" / "pampar_48k.model",
                   help="Modelo SentencePiece 48K")
    p.add_argument("--biblioteca", type=Path, default=ROOT / "biblioteca",
                   help="Raíz de la biblioteca de conocimiento")
    p.add_argument("--indice", type=Path, default=None,
                   help="Ruta a indice.json (por defecto: biblioteca/indice.json)")
    p.add_argument("--estado", type=Path, default=None,
                   help="Ruta para el estado del MotorCuriosidad (JSON)")

    # Hiperparámetros
    p.add_argument("--lr", type=float, default=3e-4, help="Tasa de aprendizaje (AdamW)")
    p.add_argument("--pasos-por-tema", type=int, default=20, help="Gradients steps por sesión")
    p.add_argument("--replay-cada", type=int, default=500, help="Pasos entre replay. 0=desactivado")
    p.add_argument("--guardar-cada", type=int, default=200, help="Pasos entre checkpoints")
    p.add_argument("--max-pasos", type=int, default=0, help="Límite total de pasos. 0=infinito")
    p.add_argument("--seq-len", type=int, default=512, help="max_seq_len del lector")
    p.add_argument("--batch-size", type=int, default=2, help="Tamaño de batch")
    p.add_argument("--max-grad-norm", type=float, default=1.0, help="Clip de gradiente")
    p.add_argument("--replay-size", type=int, default=256, help="Tamaño del replay buffer")

    # Dispositivo
    p.add_argument("--device", type=str, default="auto",
                   help="'auto' (cuda si disponible), 'cuda', 'cpu'")

    return p.parse_args()


def _resolver_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def _cargar_o_init_modelo(ruta_ckpt: Path, device: torch.device) -> tuple[PamparV3, int]:
    """
    Carga modelo desde checkpoint o inicializa uno nuevo.

    Returns:
        (modelo, paso_global)
    """
    if ruta_ckpt.exists():
        logger.info("Reanudando desde %s", ruta_ckpt)
        payload = torch.load(ruta_ckpt, map_location=device, weights_only=False)
        config = ConfigV3(**payload["config"]) if "config" in payload else PRESET_V3
        modelo = PamparV3(config).to(device)
        modelo.load_state_dict(payload["modelo"])
        paso = int(payload.get("paso_global", 0))
        logger.info("Modelo cargado — paso %d", paso)
        return modelo, paso
    else:
        logger.info("Nuevo entrenamiento con PRESET_V3")
        modelo = PamparV3(PRESET_V3).to(device)
        return modelo, 0


def main() -> None:
    args = _parse_args()

    device = _resolver_device(args.device)
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s  (%.1f GiB VRAM)", torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1e9)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    if not args.tokenizer.exists():
        logger.error("Tokenizer no encontrado: %s", args.tokenizer)
        sys.exit(1)

    tok = spm.SentencePieceProcessor()
    tok.Load(str(args.tokenizer))
    vocab_size = tok.GetPieceSize()
    logger.info("Tokenizer cargado — vocab=%d", vocab_size)

    # ── Modelo ────────────────────────────────────────────────────────────────
    modelo, paso_global = _cargar_o_init_modelo(args.checkpoint, device)

    if modelo.config.vocab_size != vocab_size:
        logger.error(
            "vocab_size mismatch: modelo=%d, tokenizer=%d",
            modelo.config.vocab_size, vocab_size,
        )
        sys.exit(1)

    n_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    logger.info("PamparV3 — %.1fM parámetros", n_params / 1e6)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        modelo.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        eps=1e-8,
    )
    if args.checkpoint.exists():
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if "optimizer" in payload:
            try:
                optimizer.load_state_dict(payload["optimizer"])
            except Exception as e:
                logger.warning("No se pudo restaurar optimizer: %s", e)

    # ── Biblioteca ────────────────────────────────────────────────────────────
    if not args.biblioteca.exists():
        logger.error("Biblioteca no encontrada: %s", args.biblioteca)
        sys.exit(1)

    indice_path = args.indice or (args.biblioteca / "indice.json")
    indice = _cargar_indice(indice_path)
    mapa_tema = _construir_mapa_tema(indice)
    logger.info("Índice cargado — %d temas mapeados", len(mapa_tema))

    biblioteca = LectorBiblioteca(
        raiz=args.biblioteca,
        tokenizer=tok,
        max_seq_len=args.seq_len,
        batch_size=args.batch_size,
    )

    # ── Motor de curiosidad ────────────────────────────────────────────────────
    ruta_motor = args.estado or (args.checkpoint.parent / "motor_v3.json")
    motor = MotorCuriosidad(ruta_estado=ruta_motor, nivel_actual=1)

    n_nuevos = motor.registrar_temas_desde_indice(indice)
    logger.info("Motor listo — %d temas nuevos registrados", n_nuevos)
    res = motor.resumen()
    logger.info("Estado motor: nivel=%d  dominados=%d/%d",
                res["nivel_actual"], res["temas_dominados"], res["temas_total"])

    # ── Viaje ─────────────────────────────────────────────────────────────────
    viaje = ViajeIntelectualV3(
        modelo=modelo,
        optimizer=optimizer,
        motor=motor,
        biblioteca=biblioteca,
        mapa_tema=mapa_tema,
        device=device,
        ruta_ckpt=args.checkpoint,
        paso_global=paso_global,
        replay_cada=args.replay_cada,
        guardar_cada=args.guardar_cada,
        pasos_por_tema=args.pasos_por_tema,
        max_grad_norm=args.max_grad_norm,
        replay_size=args.replay_size,
    )

    viaje.estudiar(max_pasos=args.max_pasos, ruta_motor=ruta_motor)


if __name__ == "__main__":
    main()
