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
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from bio_mechanisms import BioOrchestrator, BioState
from classroom_curriculum import (
    _CONCEPT_BY_ID,
    ClassroomConfig,
    StudentProfile,
    concept_level,
)
from classroom_events import format_event_to_console
from classroom_memory import EWC, LessonResult, ReplayBuffer, compute_ewc_baseline
from classroom_persistence import (
    save_checkpoint as _persist_checkpoint,
)
from classroom_persistence import (
    save_recording as _persist_recording,
)
from classroom_persistence import (
    save_session as _persist_session,
)
from classroom_teacher import Teacher
from classroom_training import (
    setup_optimizer,
    tokenize_pair,
    tokenize_teaching,
    train_step,
)

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
    Motor del aula — orquesta mentor, alumno y entrenamiento.

    Flujo conversacional de una lección:
      1. Seleccionar concepto via StudentProfile (adaptativo)
      2. Mentor (Qwen) genera lección: explicación + ejemplo + ejercicio + solución
      3. Phase A — Absorber: entrenar en explicación+ejemplo (todos los tokens)
      4. Phase B — Practicar: alumno genera respuesta al ejercicio
      5. Phase C — Corregir: mentor evalúa, entrenar en solución correcta + replay
      6. Actualizar perfil del alumno (mastery por concepto)
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

        # Perfil adaptativo del alumno (árbol de conceptos)
        self.student_profile = StudentProfile()

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
            elif self.config.teacher_backend == "qwen":
                api_key = os.environ.get("QWEN_API_KEY", "")
            else:
                api_key = os.environ.get("OPENROUTER_API_KEY", "")

        if not api_key:
            self._emit(
                "error",
                "No se encontró API key. Configura GITHUB_TOKEN, OPENROUTER_API_KEY o QWEN_API_KEY en .env",
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
        """Configura optimizer con Learning Rate diferencial."""
        self.optimizer, self._baseline_lr, info = setup_optimizer(
            self.model,
            self.config,
        )
        for g in info:
            self._emit(
                "system",
                f"  LR {g['label']}: {g['lr']:.2e} ({g['n_params'] / 1e6:.1f}M params)",
            )

    def _compute_ewc_baseline(self) -> None:
        """Calcula Fisher Information sobre datos que el modelo ya maneja bien."""
        self._emit("system", "Calculando Fisher Information para EWC...")
        self.ewc = compute_ewc_baseline(
            self.model,
            self.tokenizer,
            self.config.ewc_lambda,
            self.config.ewc_samples,
            self.config.seq_len,
            self.device,
        )
        n_samples = len(self.ewc.fisher)
        self._emit("system", f"EWC listo: Fisher calculada sobre {n_samples} params")

    # ── Tokenización ────────────────────────────────────────────────

    def _tokenize_pair(
        self, problem: str, solution: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokeniza problema→solución con máscara de loss."""
        return tokenize_pair(self.tokenizer, problem, solution, self.config.seq_len)

    def _tokenize_teaching(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokeniza contenido del mentor (todos los tokens entrenables)."""
        return tokenize_teaching(self.tokenizer, text, self.config.seq_len)

    # ── Generación del alumno ───────────────────────────────────────

    def _student_generate(self, problem: str, concept_type: str = "coding") -> str:
        """El alumno (PamparV3) intenta resolver el problema.

        Para conceptos conceptual/bridge usa formato conversacional en español.
        Para coding usa el formato de código Python.
        """
        self.model.eval()

        if concept_type in ("conceptual", "bridge"):
            # Formato conversacional: pregunta → respuesta en español (sin tildes)
            from classroom_training import _norm_for_tok
            prompt = _norm_for_tok(f"### Pregunta:\n{problem}\n### Respuesta:\n")
            stops = ["###", "\n\n\n", "\n"]
            max_tokens = 30   # una frase corta, no 80
            temperature = 0.2  # más determinístico
        else:
            # Formato código Python
            from classroom_training import _norm_for_tok
            prompt = f"### Problem:\n{problem}\n### Solution:\n```python\n"
            stops = ["```", "###", "\n\n\n"]
            max_tokens = 200
            temperature = 0.3

        ids = self.tokenizer.Encode(prompt)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=10 if concept_type in ("conceptual", "bridge") else 40,
                top_p=0.9,
            )

        generated = output[0, len(ids) :].tolist()
        text = self.tokenizer.Decode(generated)

        for stop in stops:
            if stop in text:
                text = text[: text.index(stop)]
        return text.strip()

    # ── Paso de entrenamiento ───────────────────────────────────────

    def _train_step(
        self, examples: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[float, float]:
        """Delega al módulo classroom_training y captura terr_acts."""
        loss_ce, ewc_pen, last_info = train_step(
            self.model,
            self.optimizer,
            self.ewc,
            examples,
            self.device,
        )
        if last_info and "terr_acts" in last_info:
            self._last_terr_acts = [last_info["terr_acts"].detach()]
        return loss_ce, ewc_pen

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

    def _select_concept(self) -> tuple[str, dict]:
        """Selecciona el concepto y genera lección via mentor.

        Returns:
            (concept_id, lesson_dict) donde lesson_dict tiene
            keys: explain, example, exercise, solution.
        """
        concept_id = self.student_profile.select_next_concept()
        concept = _CONCEPT_BY_ID[concept_id]
        concept_type = concept.get("type", "coding")
        profile_summary = self.student_profile.summary()

        self._emit(
            "system", f"Mentor preparando: {concept['name']} [{concept_type}]..."
        )
        lesson = self.teacher.generate_lesson(
            profile_summary, concept["name"], concept_type=concept_type
        )

        if not lesson:
            self._emit("system", "Reintentando generación de lección...")
            lesson = self.teacher.generate_lesson(
                profile_summary, concept["name"], concept_type=concept_type
            )

        if not lesson:
            # Fallback según tipo
            if concept_type == "conceptual":
                fallback_exercise = (
                    f"¿Puedes explicar con tus palabras qué es: {concept['desc']}?"
                )
            elif concept_type == "bridge":
                fallback_exercise = (
                    f"Muestra en Python el concepto de: {concept['desc']}"
                )
            else:
                fallback_exercise = (
                    f"Write a Python function demonstrating: {concept['desc']}"
                )
            lesson = {
                "explain": "",
                "example": "",
                "exercise": fallback_exercise,
                "solution": "",
            }

        return concept_id, lesson

    # ── Lección completa ────────────────────────────────────────────

    def run_lesson(self) -> LessonResult:
        """Ejecuta una lección conversacional completa.

        Flujo según tipo de concepto:
          conceptual/bridge:
            Phase A — Absorber: entrenar en explicación + ejemplo (todos los tokens)
            Phase B — Responder: alumno responde en lenguaje natural
            Phase C — Corregir: entrenar en pregunta→respuesta correcta (sin máscara)
          coding:
            Phase A — Absorber: entrenar en explicación + ejemplo
            Phase B — Practicar: alumno intenta el ejercicio en Python
            Phase C — Corregir: entrenar en ejercicio→solución (con máscara de prompt)
        """
        self.lesson_count += 1

        # 1. Seleccionar concepto y generar lección
        concept_id, lesson = self._select_concept()
        concept = _CONCEPT_BY_ID[concept_id]
        concept_type = concept.get("type", "coding")
        level = concept_level(concept_id)
        self.current_level = level

        self._emit(
            "lesson_start",
            {
                "lesson_id": self.lesson_count,
                "level": level,
                "level_name": concept["name"],
                "concept": concept_id,
                "problem": lesson.get("exercise", concept["desc"]),
            },
        )

        # 2. Mostrar lo que el mentor enseña
        if lesson.get("explain"):
            self._emit(
                "mentor_explain",
                {
                    "lesson_id": self.lesson_count,
                    "explain": lesson["explain"],
                },
            )
        if lesson.get("example"):
            self._emit(
                "mentor_example",
                {
                    "lesson_id": self.lesson_count,
                    "example": lesson["example"],
                },
            )

        # 3. Phase A — Absorber: entrenar en contenido del mentor
        #    Para conceptos conceptuales el text incluye la conversación natural.
        #    Para coding incluye explicación + código de ejemplo.
        teaching_text = ""
        if lesson.get("explain"):
            teaching_text += lesson["explain"] + "\n\n"
        if lesson.get("example"):
            teaching_text += lesson["example"]

        teach_loss = 0.0
        if teaching_text.strip():
            teach_ids, teach_labels = self._tokenize_teaching(teaching_text)
            teach_loss, _ = self._train_step([(teach_ids, teach_labels)])
            self._emit("system", f"Absorción completada (loss={teach_loss:.4f})")

        # 4. Phase B — El alumno intenta responder
        exercise = lesson.get("exercise", "")
        teacher_solution = lesson.get("solution", "")
        student_answer = ""
        correct = False
        feedback = ""
        loss_ce = teach_loss
        ewc_pen = 0.0

        if exercise:
            self._emit("student_thinking", {"lesson_id": self.lesson_count})
            student_answer = self._student_generate(exercise, concept_type=concept_type)
            self._emit(
                "student_answer",
                {"lesson_id": self.lesson_count, "answer": student_answer},
            )

            # 5. Phase C — Mentor evalúa el intento (lenguaje o código según tipo)
            self._emit("teacher_evaluating", {"lesson_id": self.lesson_count})
            profile_summary = self.student_profile.summary()
            eval_result = self.teacher.respond_to_attempt(
                exercise,
                student_answer,
                profile_summary,
                concept_type=concept_type,
            )
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

            if correct:
                teacher_solution = student_answer
                self.total_correct += 1
            else:
                fix = eval_result.get("fix", "")
                if fix:
                    teacher_solution = fix
                if not teacher_solution and concept_type == "coding":
                    teacher_solution = (
                        self.teacher.generate_solution(exercise) or student_answer
                    )
                self._emit(
                    "teacher_solution",
                    {"lesson_id": self.lesson_count, "solution": teacher_solution},
                )

            # Entrenar en ejercicio→solución
            if teacher_solution:
                if concept_type in ("conceptual", "bridge"):
                    # Sin máscara de prompt: todo el par es señal de aprendizaje
                    full_text = (
                        f"### Pregunta:\n{exercise}\n### Respuesta:\n{teacher_solution}"
                    )
                    ex_ids, ex_labels = self._tokenize_teaching(full_text)
                else:
                    # Con máscara: solo la solución genera loss
                    ex_ids, ex_labels = self._tokenize_pair(exercise, teacher_solution)

                train_batch: list[tuple[torch.Tensor, torch.Tensor]] = [
                    (ex_ids, ex_labels),
                ]

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
                        train_batch.append((s["input_ids"], s["labels"]))

                self._emit(
                    "training",
                    {"lesson_id": self.lesson_count, "batch_size": len(train_batch)},
                )

                loss_ce, ewc_pen = self._train_step(train_batch)

                # Guardar en replay buffer (todos los tipos)
                self.replay.add(
                    exercise,
                    teacher_solution,
                    ex_ids,
                    ex_labels,
                    level,
                )
        else:
            correct = True
            feedback = "Lección absorbida (sin ejercicio)"

        # 6. Actualizar perfil del alumno
        error_desc = feedback if not correct else ""
        self.student_profile.record(concept_id, correct, error_desc)

        # 7. Quick brain check
        brain_score = self._quick_brain_check()

        # 8. Bio-mechanisms hook
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
            problem=exercise or concept["desc"],
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
                "concept": concept_id,
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
        path = _persist_checkpoint(
            self.model,
            self.optimizer,
            self.config,
            self.lesson_count,
            self.current_level,
            self.total_correct,
        )
        self._emit("checkpoint", {"path": path, "lesson": self.lesson_count})

    # ── Guardar sesión ──────────────────────────────────────────────

    def save_session(self) -> str:
        """Guarda la sesión completa como JSONL."""
        path = _persist_session(self.session_log)
        self._emit(
            "session_saved",
            {"path": path, "lessons": len(self.session_log)},
        )
        return path

    def save_recording(self) -> str:
        """Guarda la grabación completa de eventos como HTML reproducible."""
        path = _persist_recording(
            self._recording_events,
            self._recording_start,
            self.config,
            self.lesson_count,
            self.total_correct,
            self.current_level,
        )
        if path:
            self._emit(
                "recording_saved",
                {"path": path, "events": len(self._recording_events)},
            )
        return path

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

        # Imprimir en consola
        format_event_to_console(event_type, data)
