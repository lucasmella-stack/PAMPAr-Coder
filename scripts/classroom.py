#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
classroom.py — Motor principal del Classroom (ClassroomEngine).

Orquesta profesor, alumno y entrenamiento bio-inspirado.
Para ejecutar: usar classroom_server.py (CLI/Web).
"""

from __future__ import annotations

import json
import os
import queue
import random
import sys
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from bio_mechanisms import BioOrchestrator, BioState
from classroom_curriculum import CURRICULUM, ClassroomConfig
from classroom_memory import EWC, LessonResult, ReplayBuffer
from classroom_teacher import Teacher

# Leer .env
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())


# =============================================================================
# Classroom Engine — Motor principal
# =============================================================================


class ClassroomEngine:
    """
    Motor del aula — orquesta profesor, alumno y entrenamiento.

    Flujo de una lección:
      1. Seleccionar problema del curriculum (según nivel)
      2. Alumno genera respuesta
      3. Profesor evalúa y da feedback
      4. Si incorrecto: profesor da la solución correcta
      5. Paso de entrenamiento con:
         - Loss CE sobre la solución correcta
         - Penalización EWC (proteger pesos importantes)
         - Replay de 50% ejemplos viejos
         - LR diferencial (LLAVES congelado, FFN aprende)
      6. Si correcto: guardar en replay buffer
      7. Actualizar curriculum (avanzar si accuracy > threshold)
    """

    def __init__(self, config: ClassroomConfig):
        self.config = config
        self.device = self._resolve_device(config.device)
        self.model: Optional[nn.Module] = None
        self.tokenizer = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.teacher: Optional[Teacher] = None
        self.ewc = EWC(nn.Module(), config.ewc_lambda)
        self.replay = ReplayBuffer(config.replay_size)

        # Estado del curriculum
        self.current_level = config.start_level
        self.level_history: deque[bool] = deque(maxlen=config.window_size)
        self.lesson_count = 0
        self.total_correct = 0
        self.used_exercises: dict[int, set[int]] = {i: set() for i in range(1, 6)}

        # Sesión — log completo
        self.session_log: list[LessonResult] = []

        # SSE: cola de eventos para la UI
        self.event_queue: queue.Queue = queue.Queue()

        # Bio-inspired orchestrator (se inicializa después de cargar modelo)
        self.bio: Optional[BioOrchestrator] = None
        self._last_terr_acts: Optional[list[torch.Tensor]] = None

        # Recording — captura TODOS los eventos con timestamps
        self._recording_events: list[dict] = []
        self._recording_start: float = 0.0

    def _resolve_device(self, device_arg: str) -> torch.device:
        if device_arg == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_arg)

    # ── Carga del modelo ────────────────────────────────────────────

    def load(self) -> None:
        """Carga modelo, tokenizer, configura optimizer con LR diferencial."""
        import sentencepiece as spm
        from pampar.coder.v3.config import PRESET_V3
        from pampar.coder.v3.modelo import PamparV3

        self._emit("system", "Cargando modelo...")

        # Tokenizer
        project_root = Path(__file__).parent.parent
        tok_path = project_root / "data" / "tokenizer" / "pampar_48k.model"
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(str(tok_path))

        # Modelo
        self.model = PamparV3(PRESET_V3).to(self.device)
        ckpt_path = project_root / self.config.checkpoint_in
        ckpt = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
        state_dict = ckpt.get("modelo", ckpt.get("model", ckpt))
        self.model.load_state_dict(state_dict, strict=False)
        self.model.registrar_tokenizer(self.tokenizer)

        params = sum(p.numel() for p in self.model.parameters()) / 1e6
        self._emit("system", f"Modelo cargado: {params:.1f}M params en {self.device}")

        # Optimizer con groups de LR diferencial
        self._setup_optimizer()

        # Teacher
        api_key = self.config.api_key
        if not api_key:
            if self.config.teacher_backend == "github":
                api_key = os.environ.get("GITHUB_TOKEN", "")
            else:
                api_key = os.environ.get("OPENROUTER_API_KEY", "")

        if not api_key:
            self._emit(
                "error",
                "No se encontró API key. Configura GITHUB_TOKEN o OPENROUTER_API_KEY en .env",
            )
            return

        self.teacher = Teacher(
            backend=self.config.teacher_backend,
            model=self.config.teacher_model,
            api_key=api_key,
        )
        self._emit(
            "system",
            f"Profesor: {self.config.teacher_model} ({self.config.teacher_backend})",
        )

        # Calcular Fisher Information para EWC
        self._compute_ewc_baseline()

        # Inicializar mecanismos bio-inspirados
        if self.config.bio_enabled:
            from pampar.coder.v3.config import PRESET_V3

            self.bio = BioOrchestrator(
                model=self.model,
                optimizer=self.optimizer,
                replay_buffer=self.replay,
                device=self.device,
                baseline_lr=self._baseline_lr,
                dim=PRESET_V3.dim,
                n_streams=PRESET_V3.n_streams,
                n_levels=PRESET_V3.n_levels,
                sleep_every=self.config.sleep_every,
                prune_every=self.config.prune_every,
            )
            self._emit(
                "system",
                "Bio-mechanisms activados: Neuromod + LTP + Sleep + Neurogenesis + Pruning",
            )

        self._emit("system", "¡Aula lista! Comienza la clase.")

    def _setup_optimizer(self) -> None:
        """Configura optimizer con Learning Rate diferencial (neuromodulación)."""
        cfg = self.config
        param_groups = []
        assigned = set()

        # Grupo 1: LLAVES / Tálamo — casi congelado (simula sinapsis endurecidas)
        llaves_params = []
        for name, param in self.model.named_parameters():
            if any(k in name for k in ["talamo", "llaves", "attn_proj"]):
                if param.requires_grad:
                    llaves_params.append(param)
                    assigned.add(name)
        if llaves_params:
            param_groups.append(
                {
                    "params": llaves_params,
                    "lr": cfg.lr_base * cfg.lr_llaves_mult,
                    "label": "llaves_talamo",
                }
            )

        # Grupo 2: Atención — aprende lento
        attn_params = []
        for name, param in self.model.named_parameters():
            if name not in assigned and any(
                k in name for k in ["attn", "q_proj", "k_proj", "v_proj", "o_proj"]
            ):
                if param.requires_grad:
                    attn_params.append(param)
                    assigned.add(name)
        if attn_params:
            param_groups.append(
                {
                    "params": attn_params,
                    "lr": cfg.lr_base * cfg.lr_attn_mult,
                    "label": "attention",
                }
            )

        # Grupo 3: Embeddings — aprende lento
        embed_params = []
        for name, param in self.model.named_parameters():
            if name not in assigned and any(k in name for k in ["tok_emb", "emb"]):
                if param.requires_grad:
                    embed_params.append(param)
                    assigned.add(name)
        if embed_params:
            param_groups.append(
                {
                    "params": embed_params,
                    "lr": cfg.lr_base * cfg.lr_embed_mult,
                    "label": "embeddings",
                }
            )

        # Grupo 4: FFN / StreamFFN / todo lo demás — aprende normal
        ffn_params = []
        for name, param in self.model.named_parameters():
            if name not in assigned and param.requires_grad:
                ffn_params.append(param)
                assigned.add(name)
        if ffn_params:
            param_groups.append(
                {
                    "params": ffn_params,
                    "lr": cfg.lr_base * cfg.lr_ffn_mult,
                    "label": "ffn_generation",
                }
            )

        self.optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.95),
            weight_decay=0.01,
        )

        # Guardar baseline LR para neuromodulación
        self._baseline_lr = self.config.lr_base

        # Log LR por grupo
        for g in param_groups:
            n = sum(p.numel() for p in g["params"])
            self._emit(
                "system", f"  LR {g['label']}: {g['lr']:.2e} ({n / 1e6:.1f}M params)"
            )

    def _compute_ewc_baseline(self) -> None:
        """Calcula Fisher Information sobre los datos que el modelo ya maneja bien."""
        self._emit("system", "Calculando Fisher Information para EWC...")

        # Generar muestras del modelo actual (lo que "ya sabe")
        baseline_prompts = [
            "def suma(a, b):",
            "for i in range(10):",
            "class Punto:",
            "if x > 0:",
            "import os\n",
            "def fibonacci(n):",
            "return sorted(",
            "try:\n    ",
            "with open('",
            "result = [x for x in",
        ]

        baseline_tokens = []
        self.model.eval()
        for prompt in baseline_prompts:
            ids = self.tokenizer.Encode(prompt)
            if len(ids) < 4:
                continue
            t = torch.tensor(ids, dtype=torch.long, device=self.device)
            # Generar tokens para crear secuencia completa
            for _ in range(20):  # 20 repeticiones con variación
                # Usar subsecuencias aleatorias del prompt como muestras
                if len(ids) > 2:
                    start = random.randint(0, max(0, len(ids) - 3))
                    chunk = ids[
                        start : start + min(self.config.seq_len, len(ids) - start)
                    ]
                    baseline_tokens.append(torch.tensor(chunk, dtype=torch.long))

        if baseline_tokens:
            self.ewc = EWC(self.model, self.config.ewc_lambda)
            self.ewc.compute_fisher(
                self.model, baseline_tokens, self.device, self.config.ewc_samples
            )
            self._emit(
                "system",
                f"EWC listo: Fisher calculada sobre {len(baseline_tokens)} muestras",
            )
        else:
            self._emit("system", "EWC: no se pudieron generar muestras baseline")

    # ── Tokenización ────────────────────────────────────────────────

    def _tokenize_pair(self, problem: str, solution: str) -> torch.Tensor:
        """Tokeniza un par problema→solución en formato training."""
        text = f"### Problem:\n{problem}\n### Solution:\n```python\n{solution}\n```"
        ids = self.tokenizer.Encode(text)
        # Truncar a seq_len
        if len(ids) > self.config.seq_len:
            ids = ids[: self.config.seq_len]
        return torch.tensor(ids, dtype=torch.long)

    # ── Generación del alumno ───────────────────────────────────────

    def _student_generate(self, problem: str) -> str:
        """El alumno (PamparV3) intenta resolver el problema."""
        self.model.eval()
        prompt = f"### Problem:\n{problem}\n### Solution:\n```python\n"
        ids = self.tokenizer.Encode(prompt)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_tokens=200,
                temperature=0.3,
                top_k=40,
                top_p=0.9,
            )

        generated = output[0, len(ids) :].tolist()
        text = self.tokenizer.Decode(generated)

        # Cortar en ``` o ### si aparece
        for stop in ["```", "###", "\n\n\n"]:
            if stop in text:
                text = text[: text.index(stop)]
        return text.strip()

    # ── Paso de entrenamiento ───────────────────────────────────────

    def _train_step(self, tokens_list: list[torch.Tensor]) -> tuple[float, float]:
        """
        Un paso de entrenamiento bio-inspirado.

        Returns: (loss_ce, ewc_penalty)
        """
        self.model.train()
        self.optimizer.zero_grad()

        total_loss = torch.tensor(0.0, device=self.device)
        total_ce = 0.0
        n = 0
        last_info: dict = {}

        for tokens in tokens_list:
            tokens = tokens.to(self.device)
            if tokens.dim() == 1:
                tokens = tokens.unsqueeze(0)
            if tokens.shape[1] < 3:
                continue

            input_ids = tokens[:, :-1]
            targets = tokens[:, 1:]
            logits, _, info = self.model(input_ids, targets=targets)
            last_info = info  # Guardar info para terr_acts

            loss_ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=0,
            )
            total_loss = total_loss + loss_ce
            total_ce += loss_ce.item()
            n += 1

        # Capturar terr_acts para mecanismos bio
        if last_info and "terr_acts" in last_info:
            self._last_terr_acts = [last_info["terr_acts"].detach()]

        if n == 0:
            return 0.0, 0.0

        total_loss = total_loss / n

        # EWC penalty
        ewc_pen = self.ewc.penalty(self.model)
        total_loss = total_loss + ewc_pen

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return total_ce / n, ewc_pen.item()

    # ── Quick brain check ───────────────────────────────────────────

    def _quick_brain_check(self) -> float:
        """Mini brain scan rápido: accuracyN5 sobre 3 muestras."""
        self.model.eval()
        probes = ["def fibonacci(n):", "for i in range(10):", "class DataProcessor:"]
        correct = 0
        total = 0

        with torch.no_grad():
            for probe in probes:
                ids = self.tokenizer.Encode(probe)
                if len(ids) < 3:
                    continue
                input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
                logits, _, _ = self.model(input_ids)

                for pos in range(len(ids) - 1):
                    probs = F.softmax(logits[0, pos], dim=-1)
                    top5 = probs.topk(5).indices.tolist()
                    if ids[pos + 1] in top5:
                        correct += 1
                    total += 1

        return correct / total if total > 0 else 0.0

    # ── Curriculum ──────────────────────────────────────────────────

    def _select_problem(self) -> tuple[int, str]:
        """Selecciona el siguiente problema según el nivel actual."""
        level = self.current_level
        exercises = CURRICULUM[level]["ejercicios"]

        # Encontrar ejercicio no usado
        available = [
            i for i in range(len(exercises)) if i not in self.used_exercises[level]
        ]
        if not available:
            # Todos usados, resetear
            self.used_exercises[level] = set()
            available = list(range(len(exercises)))

        idx = random.choice(available)
        self.used_exercises[level].add(idx)
        return level, exercises[idx]

    def _update_curriculum(self, correct: bool) -> None:
        """Actualiza el nivel según la performance reciente."""
        self.level_history.append(correct)

        if len(self.level_history) >= self.config.window_size:
            accuracy = sum(self.level_history) / len(self.level_history)
            if accuracy >= self.config.advance_threshold and self.current_level < 5:
                self.current_level += 1
                self.level_history.clear()
                self._emit(
                    "level_up",
                    {
                        "new_level": self.current_level,
                        "nombre": CURRICULUM[self.current_level]["nombre"],
                        "accuracy": accuracy,
                    },
                )

    # ── Lección completa ────────────────────────────────────────────

    def run_lesson(self) -> LessonResult:
        """Ejecuta una lección completa."""
        self.lesson_count += 1

        # 1. Seleccionar problema
        level, problem = self._select_problem()
        self._emit(
            "lesson_start",
            {
                "lesson_id": self.lesson_count,
                "level": level,
                "level_name": CURRICULUM[level]["nombre"],
                "problem": problem,
            },
        )

        # 2. Alumno intenta resolver
        self._emit("student_thinking", {"lesson_id": self.lesson_count})
        student_answer = self._student_generate(problem)
        self._emit(
            "student_answer",
            {
                "lesson_id": self.lesson_count,
                "answer": student_answer,
            },
        )

        # 3. Profesor evalúa
        self._emit("teacher_evaluating", {"lesson_id": self.lesson_count})
        eval_result = self.teacher.evaluate_student(problem, student_answer)
        correct = eval_result.get("correct", False)
        feedback = eval_result.get("feedback", "")

        self._emit(
            "teacher_feedback",
            {
                "lesson_id": self.lesson_count,
                "correct": correct,
                "feedback": feedback,
            },
        )

        # 4. Obtener la solución correcta (del profesor)
        if correct:
            teacher_solution = student_answer  # El alumno acertó
            self.total_correct += 1
        else:
            teacher_solution = eval_result.get("fix", "")
            if not teacher_solution:
                teacher_solution = self.teacher.generate_solution(problem)
            if not teacher_solution:
                teacher_solution = student_answer  # Fallback

            self._emit(
                "teacher_solution",
                {
                    "lesson_id": self.lesson_count,
                    "solution": teacher_solution,
                },
            )

        # 5. Paso de entrenamiento
        tokens_new = self._tokenize_pair(problem, teacher_solution)
        train_batch = [tokens_new]

        # Replay buffer: mezclar con ejemplos viejos
        if len(self.replay) > 0:
            n_replay = max(
                1,
                int(
                    len(train_batch)
                    / (1 - self.config.replay_ratio)
                    * self.config.replay_ratio
                ),
            )
            replay_samples = self.replay.sample(n_replay)
            for s in replay_samples:
                train_batch.append(s["tokens"])

        self._emit(
            "training", {"lesson_id": self.lesson_count, "batch_size": len(train_batch)}
        )
        loss_ce, ewc_pen = self._train_step(train_batch)

        # 6. Guardar en replay buffer si fue correcto (o la corrección del profesor)
        self.replay.add(problem, teacher_solution, tokens_new, level)

        # 7. Quick brain check
        brain_score = self._quick_brain_check()

        # 8. Actualizar curriculum
        self._update_curriculum(correct)

        # 8.5 Bio-mechanisms hook
        bio_state = None
        if self.bio is not None:
            bio_state = self.bio.after_lesson(
                correct=correct,
                loss=loss_ce,
                level=level,
                terr_acts_per_level=self._last_terr_acts,
            )
            self._emit(
                "bio_update",
                {
                    "lesson_id": self.lesson_count,
                    "dopamine": round(bio_state.dopamine, 3),
                    "norepinephrine": round(bio_state.norepinephrine, 3),
                    "lr_factor": round(bio_state.lr_factor, 3),
                    "ltp_applied": bio_state.ltp_applied,
                    "sleep_triggered": bio_state.sleep_triggered,
                    "sleep_loss": round(bio_state.sleep_loss, 4)
                    if bio_state.sleep_triggered
                    else 0,
                    "adapters_total": bio_state.adapters_total,
                    "pruned": bool(bio_state.pruned_streams),
                },
            )

        # 9. Resultado
        result = LessonResult(
            lesson_id=self.lesson_count,
            level=level,
            problem=problem,
            student_answer=student_answer,
            teacher_solution=teacher_solution,
            correct=correct,
            feedback=feedback,
            loss=loss_ce,
            ewc_penalty=ewc_pen,
            brain_score=brain_score,
        )
        self.session_log.append(result)

        accuracy = self.total_correct / self.lesson_count
        self._emit(
            "lesson_complete",
            {
                "lesson_id": self.lesson_count,
                "correct": correct,
                "loss": round(loss_ce, 4),
                "ewc_penalty": round(ewc_pen, 6),
                "brain_score": round(brain_score, 4),
                "accuracy": round(accuracy, 4),
                "level": self.current_level,
                "replay_size": len(self.replay),
            },
        )

        # Guardar checkpoint periódicamente
        if self.lesson_count % self.config.guardar_cada == 0:
            self._save_checkpoint()

        return result

    # ── Guardar checkpoint ──────────────────────────────────────────

    def _save_checkpoint(self) -> None:
        """Guarda checkpoint del modelo."""
        project_root = Path(__file__).parent.parent
        ckpt_path = project_root / self.config.checkpoint_out
        torch.save(
            {
                "modelo": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "paso_global": self.lesson_count,
                "config": asdict(self.config),
                "curriculum_level": self.current_level,
                "accuracy": self.total_correct / max(1, self.lesson_count),
            },
            str(ckpt_path),
        )
        self._emit("checkpoint", {"path": str(ckpt_path), "lesson": self.lesson_count})

    # ── Guardar sesión ──────────────────────────────────────────────

    def save_session(self) -> str:
        """Guarda la sesión completa como JSONL."""
        project_root = Path(__file__).parent.parent
        ts = time.strftime("%Y%m%d_%H%M%S")
        session_path = project_root / f"sessions/classroom_{ts}.jsonl"
        session_path.parent.mkdir(parents=True, exist_ok=True)

        with open(session_path, "w", encoding="utf-8") as f:
            for r in self.session_log:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

        self._emit(
            "session_saved",
            {"path": str(session_path), "lessons": len(self.session_log)},
        )
        return str(session_path)

    def save_recording(self) -> str:
        """Guarda la grabación completa de eventos como HTML reproducible."""
        if not self._recording_events:
            return ""

        project_root = Path(__file__).parent.parent
        ts = time.strftime("%Y%m%d_%H%M%S")
        recording_dir = project_root / "sessions"
        recording_dir.mkdir(parents=True, exist_ok=True)

        # Metadata
        meta = {
            "model": "PamparV3 (108M)",
            "teacher_backend": self.config.teacher_backend,
            "teacher_model": self.config.teacher_model,
            "start_time": time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.localtime(self._recording_start),
            ),
            "duration_s": round(time.time() - self._recording_start, 1)
            if self._recording_start
            else 0,
            "total_lessons": self.lesson_count,
            "accuracy": round(self.total_correct / max(1, self.lesson_count), 4),
            "final_level": self.current_level,
            "ewc_lambda": self.config.ewc_lambda,
            "lr_base": self.config.lr_base,
        }

        # Leer template del reproductor
        replay_template = Path(__file__).parent / "classroom_replay.html"
        if replay_template.exists():
            template = replay_template.read_text(encoding="utf-8")
        else:
            template = "<html><body><pre>No replay template found</pre></body></html>"

        # Inyectar datos en el template (archivo autocontenido)
        recording_data = json.dumps(
            {"meta": meta, "events": self._recording_events},
            ensure_ascii=False,
        )

        html = template.replace(
            "/*__RECORDING_DATA__*/",
            f"window.__RECORDING__ = {recording_data};",
        )

        out_path = recording_dir / f"classroom_{ts}.html"
        out_path.write_text(html, encoding="utf-8")

        self._emit(
            "recording_saved",
            {"path": str(out_path), "events": len(self._recording_events)},
        )
        return str(out_path)

    # ── Emitir eventos (SSE) ────────────────────────────────────────

    def _emit(self, event_type: str, data: str | dict = "") -> None:
        """Emite un evento para la UI y lo imprime en consola."""
        if isinstance(data, dict):
            payload = json.dumps(data, ensure_ascii=False)
        else:
            payload = data

        self.event_queue.put({"event": event_type, "data": payload})

        # Grabar evento para reproducción
        if self.config.record:
            if self._recording_start == 0.0:
                self._recording_start = time.time()
            self._recording_events.append(
                {
                    "t": round(time.time() - self._recording_start, 3),
                    "event": event_type,
                    "data": data if isinstance(data, (dict, str)) else str(data),
                }
            )

        # También imprimir en consola
        if event_type == "system":
            print(f"  🏫 {data}")
        elif event_type == "lesson_start":
            d = data if isinstance(data, dict) else {}
            print(
                f"\n  ═══ Lección {d.get('lesson_id', '?')} — Nivel {d.get('level', '?')} ({d.get('level_name', '')}) ═══"
            )
            print(f"  📝 {d.get('problem', '')[:80]}")
        elif event_type == "student_answer":
            d = data if isinstance(data, dict) else {}
            ans = d.get("answer", "")[:100]
            print(f"  🧑‍🎓 Alumno: {ans}")
        elif event_type == "teacher_feedback":
            d = data if isinstance(data, dict) else {}
            icon = "✅" if d.get("correct") else "❌"
            print(f"  👨‍🏫 Profesor: {icon} {d.get('feedback', '')[:100]}")
        elif event_type == "lesson_complete":
            d = data if isinstance(data, dict) else {}
            print(
                f"  📊 Loss: {d.get('loss', 0):.4f} | EWC: {d.get('ewc_penalty', 0):.6f} | Brain: {d.get('brain_score', 0):.2%} | Acc: {d.get('accuracy', 0):.1%} | Replay: {d.get('replay_size', 0)}"
            )
        elif event_type == "level_up":
            d = data if isinstance(data, dict) else {}
            print(
                f"\n  🎉 ¡NIVEL UP! → Nivel {d.get('new_level', '?')}: {d.get('nombre', '')}"
            )
        elif event_type == "checkpoint":
            d = data if isinstance(data, dict) else {}
            print(f"  💾 Checkpoint guardado: lección {d.get('lesson', '?')}")
        elif event_type == "bio_update":
            d = data if isinstance(data, dict) else {}
            parts = [
                f"DA={d.get('dopamine', 0):.2f}",
                f"NE={d.get('norepinephrine', 0):.2f}",
                f"LR×{d.get('lr_factor', 1):.2f}",
            ]
            if d.get("ltp_applied"):
                parts.append("LTP!")
            if d.get("sleep_triggered"):
                parts.append(f"SLEEP(loss={d.get('sleep_loss', 0):.3f})")
            if d.get("adapters_total", 0) > 0:
                parts.append(f"LoRA={d.get('adapters_total', 0)}")
            if d.get("pruned"):
                parts.append("PRUNED")
            print(f"  🧠 Bio: {' | '.join(parts)}")
        elif event_type == "error":
            print(f"  ❗ {data}")

