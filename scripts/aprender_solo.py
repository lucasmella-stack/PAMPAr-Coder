#!/usr/bin/env python3
# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
PAMPAr — Aprendizaje Autónomo Local (v3).

El modelo hace un "viaje intelectual" por la biblioteca de conocimiento,
guiado por su propia curiosidad — aprende lo que NO sabe, consolida lo
que SÍ sabe, y crece progresivamente sin supervisión humana.

Corre en tu computadora local:
  - CPU: ~2-3 tok/s (usable, lento pero funciona)
  - GPU 4GB: ~50-100 tok/s (cómodo)
  - GPU mayor: más rápido

El modelo NUNCA olvida lo aprendido gracias a:
  1. MemoriaJerarquica: replay de momentos clave (L0/L1/L2)
  2. Gradient episódico: re-entrenamiento periódico en temas dominados
  3. Consolidación periódica: protección de patrones aprendidos

Uso:
  python scripts/aprender_solo.py --checkpoint checkpoints/v3_ghidra_v9.pt

  # Con más control:
  python scripts/aprender_solo.py \\
    --checkpoint checkpoints/v3_ghidra_v9.pt \\
    --biblioteca biblioteca/ \\
    --estado curiosidad_estado.json \\
    --lr 5e-5 \\
    --pasos-por-tema 100 \\
    --replay-cada 50 \\
    --consolidar-cada 300 \\
    --guardar-cada 500
"""

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import sentencepiece as spm
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from pampar.coder.v3 import (
    PRESET_V3,
    PRESET_V3_LARGE,
    PRESET_V3_SMALL,
    ConfigV3,
    PamparV3,
)
from pampar.training import LectorBiblioteca, MotorCuriosidad
from pampar.training.memoria_jerarquica import MemoriaJerarquica

# =============================================================================
# COLORES PARA LA TERMINAL (hace lindo el log del viaje)
# =============================================================================


class C:
    AZUL = "\033[94m"
    VERDE = "\033[92m"
    AMARILLO = "\033[93m"
    ROJO = "\033[91m"
    GRIS = "\033[90m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def log(nivel: str, msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    colores = {
        "INFO": C.AZUL,
        "OK": C.VERDE,
        "WARN": C.AMARILLO,
        "ERROR": C.ROJO,
        "DEBUG": C.GRIS,
        "LIBRO": C.BOLD + C.AZUL,
        "NIVEL": C.BOLD + C.VERDE,
    }
    color = colores.get(nivel, "")
    print(f"{C.GRIS}[{ts}]{C.RESET} {color}[{nivel}]{C.RESET} {msg}")


# =============================================================================
# LOOP PRINCIPAL DE APRENDIZAJE AUTÓNOMO
# =============================================================================


class ViajeIntelectual:
    """
    Loop de aprendizaje autónomo de PAMPAr.

    El modelo "estudia" tema por tema, guiado por curiosidad,
    como un estudiante autodidacta con acceso ilimitado a una biblioteca.

    Fases de cada iteración:
      1. ELEGIR    — MotorCuriosidad decide qué estudiar
      2. LEER      — Cargar un batch del tema elegido
      3. ESTUDIAR  — Gradient step sobre el batch
      4. MEDIR     — Loss sin gradiente (¿cuánto aprendió?)
      5. FEEDBACK  — MemoriaJerarquica procesa el batch
      6. REPLAY    — Cada N pasos, repasar lo aprendido antes
      7. CONSOLIDAR — Cada M pasos, L2 → pesos del modelo
      8. GUARDAR   — Checkpoint + estado del motor de curiosidad
    """

    def __init__(
        self,
        modelo: PamparV3,
        optimizer: torch.optim.Optimizer,
        memoria: MemoriaJerarquica,
        motor: MotorCuriosidad,
        biblioteca: LectorBiblioteca,
        indice: dict,
        device: torch.device,
        # Hiperparámetros
        pasos_por_tema: int = 50,
        replay_cada: int = 50,
        consolidar_cada: int = 300,
        guardar_cada: int = 500,
        ruta_checkpoint: Optional[Path] = None,
        ruta_estado_motor: Optional[Path] = None,
    ):
        self.modelo = modelo
        self.optimizer = optimizer
        self.memoria = memoria
        self.motor = motor
        self.biblioteca = biblioteca
        self.indice = indice
        self.device = device

        self.pasos_por_tema = pasos_por_tema
        self.replay_cada = replay_cada
        self.consolidar_cada = consolidar_cada
        self.guardar_cada = guardar_cada
        self.ruta_checkpoint = ruta_checkpoint
        self.ruta_estado_motor = ruta_estado_motor
        self.teacher: Optional[PamparV3] = None  # Se asigna desde fuera
        self.alpha_distil: float = 0.3  # Peso KL vs CE
        self.temp_distil: float = 4.0  # Temperatura de destilación

        self.paso_global = 0
        self.inicio = time.time()

        # Registrar todos los temas de la biblioteca en el motor
        n = self.motor.registrar_temas_desde_indice(indice)
        log("INFO", f"Biblioteca cargada: {n} temas nuevos registrados")

    def _tema_a_archivo(self, nombre_tema: str) -> Optional[str]:
        """Encuentra la ruta del archivo de un tema en el índice."""
        for categoria, temas in self.indice.items():
            if not isinstance(temas, list):
                continue  # Ignorar meta-keys como "version", "descripcion"
            for tema in temas:
                if tema["nombre"] == nombre_tema:
                    return tema["archivo"]
        return None

    def _distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """KL divergence entre student y teacher con temperatura.

        loss_kl = T² × KL(softmax(S/T) ‖ softmax(T_teacher/T))
        Escalado por T² para que los gradientes tengan la misma
        magnitud independientemente de la temperatura.
        """
        T = self.temp_distil
        s = F.log_softmax(student_logits / T, dim=-1)
        t = F.softmax(teacher_logits / T, dim=-1)
        return F.kl_div(s, t, reduction="batchmean") * (T**2)

    def _paso_entrenamiento(self, tokens: torch.Tensor) -> dict:
        """Un paso de gradiente sobre un batch de tokens.

        Si hay teacher cargado, combina:
          loss = (1 - α) * cross_entropy + α * KL_divergence(student, teacher)
        """
        self.modelo.train()
        self.optimizer.zero_grad()

        input_ids = tokens[:, :-1]
        targets = tokens[:, 1:]

        logits, _loss_model, info = self.modelo(input_ids, targets=targets)

        B, L, V = logits.shape
        loss_ce = F.cross_entropy(
            logits.reshape(B * L, V),
            targets.reshape(B * L),
            ignore_index=0,
        )

        if self.teacher is not None:
            with torch.no_grad():
                t_logits, _, _ = self.teacher(input_ids)
            loss_kl = self._distillation_loss(
                logits.reshape(B * L, V),
                t_logits.reshape(B * L, V),
            )
            loss = (1.0 - self.alpha_distil) * loss_ce + self.alpha_distil * loss_kl
        else:
            loss = loss_ce

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.modelo.parameters(), 1.0)
        self.optimizer.step()

        terr_acts = info.get("terr_acts") if isinstance(info, dict) else None
        return {
            "loss": loss_ce.item(),  # Reportar CE puro para comparabilidad
            "loss_total": loss.item(),
            "terr_acts": terr_acts,
        }

    def _paso_replay(self) -> Optional[float]:
        """Replay de MemoriaJerarquica (repasar lo aprendido antes)."""
        batch = self.memoria.get_replay_batch(strategy="hardest")
        if batch is None:
            return None

        self.modelo.train()
        self.optimizer.zero_grad()

        tokens = batch.to(self.device)
        if tokens.shape[1] < 2:
            return None

        input_ids = tokens[:, :-1]
        targets = tokens[:, 1:]

        logits, _, _ = self.modelo(input_ids, targets=targets)
        B, L, V = logits.shape
        loss = F.cross_entropy(
            logits.reshape(B * L, V),
            targets.reshape(B * L),
            ignore_index=0,
        )

        # Replay con peso reducido (no borrar memorias nuevas)
        (loss * 0.15).backward()
        torch.nn.utils.clip_grad_norm_(self.modelo.parameters(), 0.5)
        self.optimizer.step()

        return loss.item()

    def _guardar(self) -> None:
        """Guarda checkpoint del modelo y estado del motor de curiosidad."""
        if self.ruta_checkpoint:
            torch.save(
                {
                    "modelo": self.modelo.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "paso_global": self.paso_global,
                },
                self.ruta_checkpoint,
            )
            # También guardar estado de memoria
            ruta_mem = self.ruta_checkpoint.with_suffix(".memoria.json")
            self.memoria.guardar(str(ruta_mem))

        if self.ruta_estado_motor:
            self.motor.guardar(self.ruta_estado_motor)

    def _banner_progreso(self) -> None:
        """Imprime resumen del viaje intelectual."""
        r = self.motor.resumen()
        elapsed = (time.time() - self.inicio) / 3600
        tops = r["tops_curiosidad"]

        print(f"\n{C.BOLD}{'═' * 60}{C.RESET}")
        print(f"{C.BOLD}  VIAJE INTELECTUAL — Paso {self.paso_global:,}{C.RESET}")
        print(f"{'═' * 60}")
        print(f"  Nivel actual:   {C.BOLD}{r['nivel_actual']}/6{C.RESET}")
        print(
            f"  Temas dominados: {C.VERDE}{r['temas_dominados']}/{r['temas_total']}{C.RESET} "
            f"({r['porcentaje_dominio']:.0f}%)"
        )
        print(f"  Loss global:    {r['loss_promedio_global']:.3f}")
        print(f"  Tiempo activo:  {elapsed:.1f}h")
        print(f"  Próximos temas de mayor curiosidad:")
        for nombre, score in tops:
            perfil = self.motor.temas.get(nombre)
            estado = "✓" if perfil and perfil.dominado else "→"
            print(f"    {estado} {nombre:<30} curiosidad={score:.3f}")
        print(f"{'═' * 60}\n")

    # -------------------------------------------------------------------------
    # LOOP PRINCIPAL
    # -------------------------------------------------------------------------

    def estudiar(self, max_pasos: Optional[int] = None) -> None:
        """
        Inicia el viaje intelectual autónomo.

        Args:
            max_pasos: Número máximo de pasos (None = infinito, hasta Ctrl+C).
        """
        log("LIBRO", "Iniciando viaje intelectual autónomo...")
        if self.teacher is not None:
            log(
                "OK",
                f"Destilación activa: α={self.alpha_distil} T={self.temp_distil} → aprendiendo del teacher",
            )
        log("INFO", f"Device: {self.device} | Temas: {len(self.motor.temas)}")

        try:
            while True:
                if max_pasos and self.paso_global >= max_pasos:
                    break

                # ── 1. ELEGIR TEMA ────────────────────────────────────────────
                nombre_tema = self.motor.siguiente_tema()
                if nombre_tema is None:
                    log("WARN", "No hay temas disponibles. Esperando...")
                    time.sleep(5)
                    continue

                archivo = self._tema_a_archivo(nombre_tema)
                if archivo is None:
                    continue

                perfil = self.motor.temas[nombre_tema]
                log(
                    "LIBRO",
                    f"Estudiando: '{nombre_tema}' | "
                    f"nivel={perfil.nivel_dificultad} | "
                    f"loss_prev={perfil.loss_media:.2f} | "
                    f"sesiones={perfil.n_sesiones}",
                )

                # ── 2-4. LEER → ESTUDIAR → MEDIR ─────────────────────────────
                losses_sesion = []

                for paso_local in range(self.pasos_por_tema):
                    tokens = self.biblioteca.obtener_batch(archivo, self.device)

                    if tokens is None:
                        # Archivo no existe aún — medir con loss alta ficticia
                        log("DEBUG", f"  Sin datos para '{nombre_tema}' aún.")
                        losses_sesion.append(4.0)
                        break

                    # Paso de entrenamiento
                    resultado = self._paso_entrenamiento(tokens)
                    loss = resultado["loss"]
                    losses_sesion.append(loss)
                    self.paso_global += 1

                    # ── 5. FEEDBACK A MEMORIA ─────────────────────────────────
                    with torch.no_grad():
                        per_token_loss = (
                            F.cross_entropy(
                                resultado.get("logits_detach", torch.zeros(1)),
                                tokens[:, 1:].reshape(-1),
                                ignore_index=0,
                                reduction="none",
                            ).reshape(tokens.shape[0], -1)
                            if False
                            else None
                        )

                    if resultado.get("terr_acts") is not None:
                        with torch.no_grad():
                            # Loss por token aproximada
                            self.modelo.eval()
                            inp = tokens[:, :-1].to(self.device)
                            tgt = tokens[:, 1:].to(self.device)
                            lg, _loss_eval, info2 = self.modelo(inp)
                            B2, L2, V2 = lg.shape
                            ptl = F.cross_entropy(
                                lg.reshape(B2 * L2, V2),
                                tgt.reshape(B2 * L2),
                                ignore_index=0,
                                reduction="none",
                            ).reshape(B2, L2)
                            # Pad primera columna
                            pad = torch.zeros(B2, 1, device=self.device)
                            ptl_padded = torch.cat([pad, ptl], dim=1)
                            terr_acts2 = (
                                info2.get("terr_acts")
                                if isinstance(info2, dict)
                                else None
                            )
                            self.memoria.procesar_batch(tokens, ptl_padded, terr_acts2)
                            self.modelo.train()

                    # ── 6. REPLAY ─────────────────────────────────────────────
                    if self.paso_global % self.replay_cada == 0:
                        rl = self._paso_replay()
                        if rl is not None:
                            log("DEBUG", f"  [replay] loss={rl:.3f}")

                    # ── 7. CONSOLIDAR ─────────────────────────────────────────
                    if self.paso_global % self.consolidar_cada == 0:
                        log("INFO", "  [consolidar] Transfiriendo L2 → pesos...")
                        self.memoria.consolidar(self.modelo)

                    # ── 8. GUARDAR ────────────────────────────────────────────
                    if self.paso_global % self.guardar_cada == 0:
                        self._guardar()
                        log("OK", f"  Checkpoint guardado. Paso {self.paso_global:,}")

                    # Banner periódico
                    if self.paso_global % (self.guardar_cada * 2) == 0:
                        self._banner_progreso()

                # ── FEEDBACK POST-SESIÓN ──────────────────────────────────────
                if losses_sesion:
                    loss_media_sesion = sum(losses_sesion) / len(losses_sesion)
                    info_fb = self.motor.retroalimentar(nombre_tema, loss_media_sesion)

                    if info_fb.get("recien_dominado"):
                        log(
                            "NIVEL",
                            f"¡'{nombre_tema}' DOMINADO! "
                            f"loss={loss_media_sesion:.3f} | "
                            f"nivel_actual={info_fb['nivel_actual']}",
                        )
                    elif info_fb.get("mejora", 0) > 0.1:
                        log(
                            "OK",
                            f"  Mejora en '{nombre_tema}': "
                            f"{info_fb['loss_anterior']:.3f} → {loss_media_sesion:.3f}",
                        )

        except KeyboardInterrupt:
            log("INFO", "\nViaje interrumpido por el usuario. Guardando estado...")
            self._guardar()
            self._banner_progreso()
            log("OK", "Estado guardado. Hasta la próxima sesión de estudio.")


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PAMPAr — Aprendizaje autónomo local guiado por curiosidad"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Ruta al checkpoint del modelo (.pt)",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("data/tokenizer/pampar_48k.model"),
        help="Ruta al tokenizer SentencePiece",
    )
    parser.add_argument(
        "--biblioteca",
        type=Path,
        default=Path("biblioteca"),
        help="Ruta a la carpeta biblioteca/",
    )
    parser.add_argument(
        "--estado",
        type=Path,
        default=Path("checkpoints/curiosidad_estado.json"),
        help="Dónde guardar/cargar el estado del motor de curiosidad",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate (bajo para aprendizaje continuo, default=5e-5)",
    )
    parser.add_argument(
        "--pasos-por-tema",
        type=int,
        default=50,
        help="Pasos de gradiente por sesión de cada tema",
    )
    parser.add_argument(
        "--replay-cada",
        type=int,
        default=50,
        help="Replay de memoria cada N pasos",
    )
    parser.add_argument(
        "--consolidar-cada",
        type=int,
        default=300,
        help="Consolidar L2→pesos cada N pasos",
    )
    parser.add_argument(
        "--guardar-cada",
        type=int,
        default=500,
        help="Guardar checkpoint cada N pasos",
    )
    parser.add_argument(
        "--max-pasos",
        type=int,
        default=None,
        help="Máximo de pasos (None = infinito)",
    )
    parser.add_argument(
        "--teacher",
        type=Path,
        default=None,
        help="Checkpoint del modelo teacher para destilación (ej: checkpoints/v3_ghidra_v9.pt)",
    )
    parser.add_argument(
        "--alpha-distil",
        type=float,
        default=0.3,
        help="Peso de la loss de destilación KL vs cross-entropy (0=solo CE, 1=solo KL, default=0.3)",
    )
    parser.add_argument(
        "--temp-distil",
        type=float,
        default=4.0,
        help="Temperatura de destilación — valores más altos dan distribuciones más suaves (default=4.0)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Longitud máxima de secuencia (default=512 para CPU)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size (2-4 para GPU 4GB, 1 para CPU)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Dispositivo: 'auto', 'cpu', 'cuda', 'mps'",
    )
    args = parser.parse_args()

    # ── Device ───────────────────────────────────────────────────────────────
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    log("INFO", f"Device: {device}")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    if not args.tokenizer.exists():
        log("ERROR", f"Tokenizer no encontrado: {args.tokenizer}")
        sys.exit(1)
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(str(args.tokenizer))
    tok_vocab = tokenizer.GetPieceSize()
    log("OK", f"Tokenizer cargado: {tok_vocab:,} vocab")

    # ── Modelo ───────────────────────────────────────────────────────────────
    import dataclasses

    PRESET_MAP = {
        "V3": PRESET_V3,
        "V3_SMALL": PRESET_V3_SMALL,
        "V3_LARGE": PRESET_V3_LARGE,
    }

    config = PRESET_V3
    if args.checkpoint.exists():
        ckpt_meta = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        raw_cfg = ckpt_meta.get("config")
        state_for_infer = ckpt_meta.get("modelo", ckpt_meta.get("model", ckpt_meta))

        if isinstance(raw_cfg, ConfigV3):
            config = raw_cfg
        elif isinstance(raw_cfg, dict):
            valid = {f.name for f in dataclasses.fields(ConfigV3)}
            filtered = {k: v for k, v in raw_cfg.items() if k in valid}
            if filtered:
                config = ConfigV3(**filtered)
            else:
                # Intentar inferir del preset name
                preset_name = raw_cfg.get("preset", "")
                if preset_name in PRESET_MAP:
                    config = PRESET_MAP[preset_name]
        else:
            # Sin config en checkpoint — inferir desde los pesos
            emb = (
                state_for_infer.get("tok_emb.weight")
                if isinstance(state_for_infer, dict)
                else None
            )
            if emb is not None:
                inferred_vocab = int(emb.shape[0])
                inferred_dim = int(emb.shape[1])
                matched = False
                for candidate in (PRESET_V3, PRESET_V3_SMALL, PRESET_V3_LARGE):
                    if candidate.dim == inferred_dim:
                        config = dataclasses.replace(
                            candidate, vocab_size=inferred_vocab
                        )
                        matched = True
                        break
                if not matched:
                    config = ConfigV3(vocab_size=inferred_vocab, dim=inferred_dim)
                log(
                    "WARN",
                    f"Sin 'config' en checkpoint — inferido: dim={inferred_dim}, vocab={inferred_vocab:,}",
                )

    # Validar que tokenizer y modelo tienen el mismo vocab
    if config.vocab_size != tok_vocab:
        log(
            "ERROR",
            f"Vocab mismatch: tokenizer={tok_vocab} vs modelo={config.vocab_size}",
        )
        auto_toks = {
            16000: Path("data/tokenizer/code_tokenizer.model"),
            48000: Path("data/tokenizer/pampar_48k.model"),
        }
        sugerido = auto_toks.get(config.vocab_size)
        if sugerido and sugerido.exists():
            log("INFO", f"Sugerencia: --tokenizer {sugerido}")
        sys.exit(1)

    modelo = PamparV3(config).to(device)

    if args.checkpoint.exists():
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state = ckpt.get("modelo", ckpt.get("model", ckpt))
        missing, unexpected = modelo.load_state_dict(state, strict=False)
        if unexpected:
            log("WARN", f"{len(unexpected)} pesos inesperados: {unexpected[:3]}")
        if missing:
            log("WARN", f"{len(missing)} pesos faltantes: {missing[:3]}")
        log(
            "OK",
            f"Modelo cargado desde {args.checkpoint} "
            f"({config.vocab_size:,} vocab, {sum(p.numel() for p in modelo.parameters()) / 1e6:.0f}M params)",
        )
    else:
        log(
            "WARN",
            f"Checkpoint no encontrado — iniciando desde cero: {args.checkpoint}",
        )

    n_params = sum(p.numel() for p in modelo.parameters()) / 1e6
    log("INFO", f"Parámetros: {n_params:.0f}M")

    # ── Optimizer ────────────────────────────────────────────────────────────
    # LR bajo para aprendizaje continuo — no sobreescribir lo ya aprendido
    optimizer = torch.optim.AdamW(
        modelo.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )

    # ── Memoria ──────────────────────────────────────────────────────────────
    memoria = MemoriaJerarquica(
        capacidad_l0=2048,
        capacidad_l1=8000,
        capacidad_l2=3000,
    )
    ruta_mem = args.checkpoint.with_suffix(".memoria.json")
    if ruta_mem.exists():
        memoria = MemoriaJerarquica.cargar(str(ruta_mem))
        log("OK", "Estado de memoria cargado")

    # ── Motor de curiosidad ───────────────────────────────────────────────────
    motor = MotorCuriosidad(
        ruta_estado=args.estado,
        nivel_actual=1,
    )

    # ── Biblioteca ───────────────────────────────────────────────────────────
    if not args.biblioteca.exists():
        log(
            "WARN",
            f"Biblioteca no encontrada en {args.biblioteca}. Créala o descarga datos.",
        )
        args.biblioteca.mkdir(parents=True, exist_ok=True)

    indice_path = args.biblioteca / "indice.json"
    if not indice_path.exists():
        log("ERROR", f"Índice de biblioteca no encontrado: {indice_path}")
        sys.exit(1)

    indice = json.loads(indice_path.read_text())
    biblioteca = LectorBiblioteca(
        raiz=args.biblioteca,
        tokenizer=tokenizer,
        max_seq_len=args.seq_len,
        batch_size=args.batch_size,
    )

    # ── Viaje Intelectual ────────────────────────────────────────────────────
    # ── Teacher (opcional, para destilación) ────────────────────────────────
    teacher_modelo = None
    if args.teacher is not None:
        if not args.teacher.exists():
            log("ERROR", f"Teacher no encontrado: {args.teacher}")
            sys.exit(1)
        log("INFO", f"Cargando teacher desde {args.teacher}...")
        ck_t = torch.load(args.teacher, map_location=device, weights_only=False)
        state_t = ck_t.get("modelo", ck_t.get("model", ck_t))
        # Inferir config del teacher desde sus pesos
        emb_t = state_t.get("tok_emb.weight") if isinstance(state_t, dict) else None
        if emb_t is not None:
            inferred_dim_t = int(emb_t.shape[1])
            inferred_vocab_t = int(emb_t.shape[0])
            config_t = None
            for cand in (PRESET_V3, PRESET_V3_SMALL, PRESET_V3_LARGE):
                if cand.dim == inferred_dim_t:
                    config_t = dataclasses.replace(cand, vocab_size=inferred_vocab_t)
                    break
            if config_t is None:
                config_t = ConfigV3(vocab_size=inferred_vocab_t, dim=inferred_dim_t)
        else:
            config_t = config  # Asumir misma config
        if config_t.vocab_size != config.vocab_size:
            log(
                "ERROR",
                f"Teacher vocab ({config_t.vocab_size}) != Student vocab ({config.vocab_size}) — "
                f"deben compartir el mismo tokenizer",
            )
            sys.exit(1)
        teacher_modelo = PamparV3(config_t).to(device)
        missing_t, _ = teacher_modelo.load_state_dict(state_t, strict=False)
        teacher_modelo.eval()
        for p in teacher_modelo.parameters():
            p.requires_grad_(False)
        t_params = sum(p.numel() for p in teacher_modelo.parameters()) / 1e6
        log(
            "OK",
            f"Teacher listo: {t_params:.0f}M params | "
            f"α={args.alpha_distil} T={args.temp_distil} | "
            f"Pesos CONGELADOS (no se entrena)",
        )

    viaje = ViajeIntelectual(
        modelo=modelo,
        optimizer=optimizer,
        memoria=memoria,
        motor=motor,
        biblioteca=biblioteca,
        indice=indice,
        device=device,
        pasos_por_tema=args.pasos_por_tema,
        replay_cada=args.replay_cada,
        consolidar_cada=args.consolidar_cada,
        guardar_cada=args.guardar_cada,
        ruta_checkpoint=args.checkpoint,
        ruta_estado_motor=args.estado,
    )
    if teacher_modelo is not None:
        viaje.teacher = teacher_modelo
        viaje.alpha_distil = args.alpha_distil
        viaje.temp_distil = args.temp_distil

    viaje.estudiar(max_pasos=args.max_pasos)


if __name__ == "__main__":
    main()
