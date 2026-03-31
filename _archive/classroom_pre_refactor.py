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
from classroom_curriculum import (
    CURRICULUM,
    ClassroomConfig,
    StudentProfile,
    _CONCEPT_BY_ID,
)
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

    def _tokenize_pair(
        self, problem: str, solution: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokeniza un par problema→solución con máscara de loss.

        Returns:
            (input_ids, labels) donde labels tiene -100 en el prompt
            para que el loss solo se compute sobre la solución.
        """
        prompt = f"### Problem:\n{problem}\n### Solution:\n```python\n"
        prompt_ids = self.tokenizer.Encode(prompt)
        solution_ids = self.tokenizer.Encode(solution + "\n```")

        all_ids = prompt_ids + solution_ids
        # Truncar a seq_len
        if len(all_ids) > self.config.seq_len:
            all_ids = all_ids[: self.config.seq_len]
            # Recalcular cuántos tokens de solución quedan
            n_prompt = min(len(prompt_ids), len(all_ids))
        else:
            n_prompt = len(prompt_ids)

        input_ids = torch.tensor(all_ids, dtype=torch.long)
        labels = input_ids.clone()
        # Maskear el prompt: el modelo NO debe entrenar en predecir el enunciado
        labels[:n_prompt] = -100

        return input_ids, labels

    def _tokenize_teaching(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokeniza contenido de enseñanza donde TODOS los tokens son entrenables.

        Se usa para que el alumno absorba explicaciones y ejemplos del mentor.
        A diferencia de _tokenize_pair, no hay máscara: el modelo aprende a
        predecir cada token de la conversación del mentor.
        """
        ids = self.tokenizer.Encode(text)
        if len(ids) > self.config.seq_len:
            ids = ids[: self.config.seq_len]
        input_ids = torch.tensor(ids, dtype=torch.long)
        labels = input_ids.clone()
        return input_ids, labels

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

    def _train_step(
        self, examples: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[float, float]:
        """
        Un paso de entrenamiento con loss masking.

        Args:
            examples: lista de (input_ids, labels) donde labels=-100 en prompt

        Returns: (loss_ce, ewc_penalty)
        """
        self.model.train()
        self.optimizer.zero_grad()

        total_loss = torch.tensor(0.0, device=self.device)
        total_ce = 0.0
        n = 0
        last_info: dict = {}

        for input_ids, labels in examples:
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                labels = labels.unsqueeze(0)
            if input_ids.shape[1] < 3:
                continue

            # Shift: predecir el siguiente token
            inp = input_ids[:, :-1]
            tgt = labels[:, 1:]  # labels ya tiene -100 en el prompt
            logits, _, info = self.model(inp)
            last_info = info

            loss_ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt.reshape(-1),
                ignore_index=-100,
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

    @staticmethod
    def _concept_level(concept_id: str) -> int:
        """Mapea concepto a nivel del curriculum (1-5)."""
        _level_map = {
            "arithmetic": 1, "variables_types": 1, "conditionals": 1,
            "strings": 1, "functions_basic": 1,
            "loops_for": 2, "loops_while": 2, "lists": 2,
            "tuples_sets": 2, "dicts": 2,
            "recursion": 3, "higher_order": 3, "generators": 3,
            "error_handling": 3,
            "classes_basic": 4, "inheritance": 4, "dunder_methods": 4,
            "decorators": 5, "context_managers": 5, "algorithms": 5,
            "file_io": 5,
        }
        return _level_map.get(concept_id, 1)

    def _select_concept(self) -> tuple[str, dict]:
        """Selecciona el concepto y genera lección via mentor.

        Returns:
            (concept_id, lesson_dict) donde lesson_dict tiene
            keys: explain, example, exercise, solution.
        """
        concept_id = self.student_profile.select_next_concept()
        concept = _CONCEPT_BY_ID[concept_id]
        profile_summary = self.student_profile.summary()

        self._emit("system", f"Mentor preparando: {concept['name']}...")
        lesson = self.teacher.generate_lesson(profile_summary, concept["name"])

        if not lesson:
            self._emit("system", "Reintentando generación de lección...")
            lesson = self.teacher.generate_lesson(profile_summary, concept["name"])

        if not lesson:
            # Fallback mínimo
            lesson = {
                "explain": "",
                "example": "",
                "exercise": f"Write a Python function demonstrating: {concept['desc']}",
                "solution": "",
            }

        return concept_id, lesson

    # ── Lección completa ────────────────────────────────────────────

    def run_lesson(self) -> LessonResult:
        """Ejecuta una lección conversacional completa.

        Flujo mentor:
          1. Seleccionar concepto via StudentProfile (adaptativo)
          2. Mentor genera lección: explicación + ejemplo + ejercicio + solución
          3. Phase A — Absorber: entrenar en explicación + ejemplo (todos los tokens)
          4. Phase B — Practicar: alumno intenta el ejercicio
          5. Phase C — Corregir: mentor evalúa, entrenar en solución correcta + replay
          6. Actualizar perfil del alumno
        """
        self.lesson_count += 1

        # 1. Seleccionar concepto y generar lección
        concept_id, lesson = self._select_concept()
        concept = _CONCEPT_BY_ID[concept_id]
        level = self._concept_level(concept_id)
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
            self._emit("mentor_explain", {
                "lesson_id": self.lesson_count,
                "explain": lesson["explain"],
            })
        if lesson.get("example"):
            self._emit("mentor_example", {
                "lesson_id": self.lesson_count,
                "example": lesson["example"],
            })

        # 3. Phase A — Absorber: entrenar en contenido del mentor
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

        # 4. Phase B — Practicar: alumno intenta el ejercicio
        exercise = lesson.get("exercise", "")
        teacher_solution = lesson.get("solution", "")
        student_answer = ""
        correct = False
        feedback = ""
        loss_ce = teach_loss
        ewc_pen = 0.0

        if exercise:
            self._emit("student_thinking", {"lesson_id": self.lesson_count})
            student_answer = self._student_generate(exercise)
            self._emit(
                "student_answer",
                {"lesson_id": self.lesson_count, "answer": student_answer},
            )

            # 5. Phase C — Mentor evalúa el intento
            self._emit("teacher_evaluating", {"lesson_id": self.lesson_count})
            profile_summary = self.student_profile.summary()
            eval_result = self.teacher.respond_to_attempt(
                exercise, student_answer, profile_summary,
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
                if not teacher_solution:
                    teacher_solution = (
                        self.teacher.generate_solution(exercise) or student_answer
                    )
                self._emit(
                    "teacher_solution",
                    {"lesson_id": self.lesson_count, "solution": teacher_solution},
                )

            # Entrenar en ejercicio→solución + replay
            if teacher_solution:
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

                # Guardar en replay buffer
                self.replay.add(
                    exercise, teacher_solution, ex_ids, ex_labels, level,
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
            concept = d.get("concept", "")
            print(
                f"\n  ═══ Lección {d.get('lesson_id', '?')} — {d.get('level_name', '')} [{concept}] (Nivel {d.get('level', '?')}) ═══"
            )
            print(f"  📝 {d.get('problem', '')[:100]}")
        elif event_type == "mentor_explain":
            d = data if isinstance(data, dict) else {}
            print(f"  📖 Mentor explica: {d.get('explain', '')[:120]}")
        elif event_type == "mentor_example":
            d = data if isinstance(data, dict) else {}
            example = d.get("example", "")
            lines = example.split("\n")
            preview = lines[0][:80] if lines else ""
            print(f"  💻 Mentor ejemplo: {preview}{'...' if len(lines) > 1 else ''}")
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
            concept = d.get("concept", "")
            print(
                f"  📊 Loss: {d.get('loss', 0):.4f} | EWC: {d.get('ewc_penalty', 0):.6f} | Brain: {d.get('brain_score', 0):.2%} | Acc: {d.get('accuracy', 0):.1%} | Replay: {d.get('replay_size', 0)} | Concepto: {concept}"
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
