#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
classroom.py — Aula simulada: modelo profesor enseña a PamparV3.

Implementa aprendizaje bio-inspirado:
  - EWC (Elastic Weight Consolidation): protege pesos importantes
  - Replay Buffer: mezcla ejemplos nuevos con viejos (simula sueño)
  - LR diferencial: LLAVES casi congelado, generación aprende
  - Curriculum progresivo: de trivial a complejo

Uso:
  python scripts/classroom.py --checkpoint checkpoints/v3_ghidra_v9.pt

  # Con GitHub Models API:
  python scripts/classroom.py --checkpoint checkpoints/v3_ghidra_v9.pt --teacher github

  # Con OpenRouter:
  python scripts/classroom.py --checkpoint checkpoints/v3_ghidra_v9.pt --teacher openrouter
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import queue
import random
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from collections import deque
from dataclasses import asdict, dataclass, field
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from bio_mechanisms import BioOrchestrator, BioState

# Leer .env
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())


# =============================================================================
# Configuración
# =============================================================================


@dataclass
class ClassroomConfig:
    """Configuración del aula."""

    # Modelo
    checkpoint_in: str = "checkpoints/v3_ghidra_v9.pt"
    checkpoint_out: str = "checkpoints/v3_classroom.pt"
    device: str = "auto"

    # Teacher
    teacher_backend: str = "github"  # "github" | "openrouter"
    teacher_model: str = "openai/gpt-4o-mini"
    api_key: str = ""

    # Entrenamiento bio-inspirado
    lr_base: float = 5e-6  # LR base (conservador)
    lr_llaves_mult: float = 0.01  # LLAVES/Tálamo: 1% del LR base
    lr_attn_mult: float = 0.1  # Atención: 10% del LR base
    lr_embed_mult: float = 0.1  # Embeddings: 10% del LR base
    lr_ffn_mult: float = 1.0  # FFN/StreamFFN: 100% del LR base

    # EWC
    ewc_lambda: float = 500.0  # Fuerza de la penalización EWC
    ewc_samples: int = 200  # Muestras para calcular Fisher

    # Replay buffer
    replay_size: int = 100  # Tamaño del buffer
    replay_ratio: float = 0.5  # 50% replay, 50% nuevo

    # Curriculum
    start_level: int = 1  # Nivel inicial (1-5)
    advance_threshold: float = 0.7  # 70% correcto para avanzar
    window_size: int = 10  # Ventana para calcular accuracy

    # Sesión
    max_lessons: int = 200  # Máximo de lecciones por sesión
    guardar_cada: int = 20  # Guardar checkpoint cada N lecciones
    seq_len: int = 256  # Longitud máx de secuencia para training

    # Bio-inspired mechanisms
    bio_enabled: bool = True  # Activar mecanismos bio-inspirados
    sleep_every: int = 15  # Consolidación de sueño cada N lecciones
    prune_every: int = 30  # Poda sináptica cada N lecciones

    # Server
    port: int = 8888

    # Recording
    record: bool = True  # Grabar sesión como video reproducible


# =============================================================================
# Teacher — Modelo profesor via API
# =============================================================================

CURRICULUM = {
    1: {
        "nombre": "Fundamentos",
        "desc": "Variables, funciones simples, operaciones básicas",
        "ejercicios": [
            "Write a Python function `suma(a, b)` that returns the sum of two numbers.",
            "Write a Python function `es_par(n)` that returns True if n is even, False otherwise.",
            "Write a Python function `longitud(texto)` that returns the length of a string without using len().",
            "Write a Python function `invertir(texto)` that returns the reversed string.",
            "Write a Python function `contar_vocales(texto)` that counts vowels (a,e,i,o,u) case-insensitive.",
            "Write a Python function `suma_digitos(n)` that returns the sum of all digits of a non-negative integer.",
            "Write a Python function `es_palindromo(s)` that returns True if the string is a palindrome.",
            "Write a Python function `maximo(a, b)` that returns the larger of two numbers without using max().",
            "Write a Python function `absoluto(n)` that returns the absolute value without using abs().",
            "Write a Python function `celsius_a_fahrenheit(c)` that converts Celsius to Fahrenheit.",
            "Write a Python function `factorial(n)` that returns n! using a loop.",
            "Write a Python function `potencia(base, exp)` that returns base**exp using a loop.",
            "Write a Python function `duplicar_lista(lst)` that returns a new list with each element doubled.",
            "Write a Python function `minimo_lista(lst)` that returns the smallest element without using min().",
            "Write a Python function `contar_mayusculas(texto)` that counts uppercase letters in a string.",
        ],
    },
    2: {
        "nombre": "Estructuras de control",
        "desc": "Loops, condicionales, listas, diccionarios",
        "ejercicios": [
            "Write a Python function `fizzbuzz(n)` that returns 'FizzBuzz' if n divisible by 3 and 5, 'Fizz' if by 3, 'Buzz' if by 5, else str(n).",
            "Write a Python function `fibonacci(n)` that returns the n-th Fibonacci number (0-indexed).",
            "Write a Python function `frecuencia(lista)` that returns a dict mapping each element to its count.",
            "Write a Python function `aplanar(lista)` that flattens a list of lists by one level.",
            "Write a Python function `cuadrados_pares(n)` that returns squares of all even numbers from 2 to n.",
            "Write a Python function `invertir_dict(d)` that returns a new dict with keys and values swapped.",
            "Write a Python function `busqueda_lineal(lista, objetivo)` that returns the index or -1 if not found.",
            "Write a Python function `eliminar_duplicados(lista)` that returns a list without duplicates, preserving order.",
            "Write a Python function `es_primo(n)` that returns True if n is prime.",
            "Write a Python function `ordenar_burbuja(lista)` that sorts a list using bubble sort.",
            "Write a Python function `interseccion(a, b)` that returns elements common to both lists.",
            "Write a Python function `rotar_lista(lst, k)` that rotates list left by k positions.",
        ],
    },
    3: {
        "nombre": "Funciones avanzadas",
        "desc": "Recursión, generadores, comprensiones complejas",
        "ejercicios": [
            "Write a Python function `merge_sort(lista)` that returns a new sorted list using merge sort.",
            "Write a Python function `busqueda_binaria(lista, objetivo)` that returns the index or -1.",
            "Write a Python function `primos_hasta(n)` that yields all primes up to n using a generator.",
            "Write a Python function `memoize(fn)` that returns a cached version of fn.",
            "Write a Python function `aplanar_profundo(lst)` that recursively flattens nested lists.",
            "Write a Python function `permutaciones(lst)` that returns all permutations of a list.",
            "Write a Python function `cifrado_cesar(texto, k)` that shifts each letter by k positions.",
            "Write a Python function `potencia_recursiva(base, exp)` that calculates power recursively.",
            "Write a Python function `torre_hanoi(n, origen, destino, auxiliar)` that prints the moves.",
            "Write a Python function `zip_manual(a, b)` that zips two lists without using zip().",
        ],
    },
    4: {
        "nombre": "Clases y OOP",
        "desc": "Clases, herencia, métodos especiales",
        "ejercicios": [
            "Write a Python class `Stack` with methods `push(item)`, `pop()`, `is_empty()`, `peek()`.",
            "Write a Python class `Punto` with x, y attributes and a method `distancia(otro)` for Euclidean distance.",
            "Write a Python class `Cola` implementing a FIFO queue with `enqueue(item)` and `dequeue()`.",
            "Write a Python class `Fraccion` with add, sub, mul, and __str__ using GCD simplification.",
            "Write a Python class `Contador` that counts how many times it has been called (using __call__).",
            "Write a Python class `Vector` with __add__, __sub__, __mul__ (scalar) and __repr__.",
            "Write a Python class `ListaEnlazada` with `agregar(valor)`, `buscar(valor)`, `__len__`.",
            "Write a Python class `Matriz` with __add__ and __mul__ for 2D matrix operations.",
        ],
    },
    5: {
        "nombre": "Patrones avanzados",
        "desc": "Decoradores, context managers, algoritmos complejos",
        "ejercicios": [
            "Write a Python decorator `cronometrar` that prints how long a function takes to execute.",
            "Write a Python context manager class `TempFile` that creates a temp file and deletes it on exit.",
            "Write a Python function `lru_cache(maxsize)` decorator that caches the last maxsize unique calls.",
            "Write a Python function `dijkstra(grafo, inicio)` that returns shortest distances from inicio.",
            "Write a Python async function `fetch_all(urls)` that fetches URLs concurrently with asyncio.",
            "Write a Python function `quick_sort(lista)` implementing quicksort with median-of-three pivot.",
        ],
    },
}


class Teacher:
    """Modelo profesor via API (GitHub Models o OpenRouter)."""

    ENDPOINTS = {
        "github": "https://models.inference.ai.azure.com/chat/completions",
        "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    }

    def __init__(self, backend: str, model: str, api_key: str):
        self.backend = backend
        self.model = model
        self.api_key = api_key
        self.endpoint = self.ENDPOINTS[backend]

    def _headers(self) -> dict[str, str]:
        h = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.backend == "openrouter":
            h["HTTP-Referer"] = "https://github.com/lucasmella-stack/PAMPAr-Coder"
            h["X-Title"] = "PAMPAr Classroom"
        return h

    def _call(
        self, messages: list[dict], max_tokens: int = 800, temperature: float = 0.3
    ) -> str | None:
        """Llama a la API del profesor."""
        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers=self._headers(),
            method="POST",
        )

        for intento in range(3):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    time.sleep(10 * (intento + 1))
                    continue
                body = e.read().decode("utf-8", errors="ignore")[:200]
                print(f"  [Teacher API {e.code}] {body}")
                return None
            except Exception as e:
                print(f"  [Teacher error] {e}")
                time.sleep(5)
        return None

    def generate_solution(self, problem: str) -> str | None:
        """Pide al profesor la solución correcta para un problema."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Python expert teacher. When given a coding problem, "
                    "respond with ONLY the Python code solution. No explanations, "
                    "no markdown, no ```python blocks. Just clean, correct Python code. "
                    "Use the EXACT function/class names specified in the problem."
                ),
            },
            {"role": "user", "content": problem},
        ]
        return self._call(messages, max_tokens=500, temperature=0.2)

    def evaluate_student(self, problem: str, student_code: str) -> dict:
        """El profesor evalúa el código del alumno."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Python teacher evaluating a student's code. "
                    "Respond with a JSON object with these fields:\n"
                    '  "correct": true/false,\n'
                    '  "feedback": "brief feedback in Spanish",\n'
                    '  "fix": "corrected code if wrong, empty if correct"\n'
                    "Respond ONLY with the JSON object, no other text."
                ),
            },
            {
                "role": "user",
                "content": f"Problem:\n{problem}\n\nStudent's code:\n{student_code}",
            },
        ]
        raw = self._call(messages, max_tokens=600, temperature=0.1)
        if not raw:
            return {"correct": False, "feedback": "Error de comunicación", "fix": ""}
        try:
            # Extraer JSON de la respuesta
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"correct": False, "feedback": raw[:200], "fix": ""}

    def generate_hint(self, problem: str, level: int) -> str | None:
        """Genera una pista adaptada al nivel del alumno."""
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are teaching a beginner (level {level}/5). "
                    "Give a helpful hint for solving this problem in Spanish. "
                    "Be encouraging but brief (2-3 sentences max). "
                    "Do NOT give the solution, just a hint about the approach."
                ),
            },
            {"role": "user", "content": problem},
        ]
        return self._call(messages, max_tokens=150, temperature=0.5)


# =============================================================================
# EWC — Elastic Weight Consolidation
# =============================================================================


class EWC:
    """
    Elastic Weight Consolidation (Kirkpatrick et al., 2017).

    Simula LTP biológica: identifica pesos importantes (alta Fisher info)
    y penaliza moverlos durante entrenamiento nuevo.

    L_total = L_task + (λ/2) * Σ F_i * (θ_i - θ*_i)²
    """

    def __init__(self, model: nn.Module, lam: float = 500.0):
        self.lam = lam
        self.params_star: dict[str, torch.Tensor] = {}
        self.fisher: dict[str, torch.Tensor] = {}

    def compute_fisher(
        self,
        model: nn.Module,
        data_loader: list[torch.Tensor],
        device: torch.device,
        n_samples: int = 200,
    ) -> None:
        """Calcula la Diagonal Fisher Information Matrix sobre datos existentes."""
        model.eval()

        # Guardar pesos originales (θ*)
        self.params_star = {
            n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad
        }

        # Inicializar Fisher a cero
        self.fisher = {
            n: torch.zeros_like(p.data)
            for n, p in model.named_parameters()
            if p.requires_grad
        }

        n = min(n_samples, len(data_loader))
        samples = random.sample(data_loader, n) if len(data_loader) > n else data_loader

        for tokens in samples:
            model.zero_grad()
            tokens = tokens.to(device)
            if tokens.dim() == 1:
                tokens = tokens.unsqueeze(0)

            input_ids = tokens[:, :-1]
            targets = tokens[:, 1:]
            logits, _, _ = model(input_ids, targets=targets)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=0,
            )
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher[name] += param.grad.data.pow(2) / n

        model.zero_grad()

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Calcula la penalización EWC: (λ/2) * Σ F_i * (θ_i - θ*_i)²"""
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if name in self.fisher:
                loss += (
                    self.fisher[name] * (param - self.params_star[name]).pow(2)
                ).sum()
        return (self.lam / 2.0) * loss


# =============================================================================
# Replay Buffer — Simula consolidación durante sueño
# =============================================================================


class ReplayBuffer:
    """
    Buffer circular de ejemplos exitosos.

    Mezcla ejemplos nuevos con viejos para evitar olvido catastrófico.
    Como el replay neuronal durante el sueño: reactiva memorias viejas
    mientras integra las nuevas.
    """

    def __init__(self, maxsize: int = 100):
        self.buffer: deque[dict] = deque(maxlen=maxsize)

    def add(
        self, problem: str, solution: str, tokens: torch.Tensor, level: int
    ) -> None:
        self.buffer.append(
            {
                "problem": problem,
                "solution": solution,
                "tokens": tokens.cpu(),
                "level": level,
                "timestamp": time.time(),
            }
        )

    def sample(self, n: int) -> list[dict]:
        if len(self.buffer) == 0:
            return []
        n = min(n, len(self.buffer))
        return random.sample(list(self.buffer), n)

    def __len__(self) -> int:
        return len(self.buffer)


# =============================================================================
# Classroom Engine — Motor principal
# =============================================================================


@dataclass
class LessonResult:
    """Resultado de una lección."""

    lesson_id: int
    level: int
    problem: str
    student_answer: str
    teacher_solution: str
    correct: bool
    feedback: str
    loss: float
    ewc_penalty: float
    brain_score: float  # AccN5 simplificado
    timestamp: float = field(default_factory=time.time)


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


# =============================================================================
# HTTP Server — SSE para la UI
# =============================================================================


class ClassroomHandler(SimpleHTTPRequestHandler):
    """Handler HTTP con SSE para la UI del classroom."""

    engine: ClassroomEngine = None  # type: ignore
    ui_path: str = ""

    def do_GET(self) -> None:
        if self.path == "/" or self.path == "/index.html":
            self._serve_ui()
        elif self.path == "/events":
            self._serve_sse()
        elif self.path == "/status":
            self._serve_status()
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        if self.path == "/start":
            self._handle_start()
        elif self.path == "/stop":
            self._handle_stop()
        elif self.path == "/save":
            self._handle_save()
        else:
            self.send_error(404)

    def _serve_ui(self) -> None:
        """Sirve el archivo HTML de la UI."""
        try:
            ui_file = Path(self.ui_path)
            content = ui_file.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, str(e))

    def _serve_sse(self) -> None:
        """Server-Sent Events stream."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        try:
            while True:
                try:
                    event = self.engine.event_queue.get(timeout=1.0)
                    msg = f"event: {event['event']}\ndata: {event['data']}\n\n"
                    self.wfile.write(msg.encode("utf-8"))
                    self.wfile.flush()
                except queue.Empty:
                    # Heartbeat
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _serve_status(self) -> None:
        """Estado actual del aula."""
        e = self.engine
        status = {
            "lesson_count": e.lesson_count,
            "level": e.current_level,
            "accuracy": e.total_correct / max(1, e.lesson_count),
            "replay_size": len(e.replay),
            "session_log_size": len(e.session_log),
        }
        body = json.dumps(status).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_start(self) -> None:
        """Inicia las lecciones en background."""
        threading.Thread(target=self._run_lessons, daemon=True).start()
        self._json_response({"status": "started"})

    def _handle_stop(self) -> None:
        """Detiene las lecciones."""
        self.engine._running = False
        self._json_response({"status": "stopped"})

    def _handle_save(self) -> None:
        """Guarda la sesión."""
        path = self.engine.save_session()
        rec_path = self.engine.save_recording()
        self.engine._save_checkpoint()
        self._json_response({"status": "saved", "path": path, "recording": rec_path})

    def _run_lessons(self) -> None:
        """Loop principal de lecciones."""
        self.engine._running = True
        try:
            while (
                self.engine._running
                and self.engine.lesson_count < self.engine.config.max_lessons
            ):
                self.engine.run_lesson()
                time.sleep(1)  # Pausa entre lecciones
        except Exception as e:
            self.engine._emit("error", f"Error: {e}\n{traceback.format_exc()}")
        finally:
            self.engine._emit("system", "Sesión finalizada.")
            self.engine.save_session()
            self.engine.save_recording()
            self.engine._save_checkpoint()

    def _json_response(self, data: dict, code: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args) -> None:
        """Silenciar logs del HTTP server."""
        pass


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="PAMPAr Classroom — Aula simulada")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/v3_ghidra_v9.pt",
        help="Checkpoint del alumno",
    )
    parser.add_argument(
        "--checkpoint-out",
        default="checkpoints/v3_classroom.pt",
        help="Donde guardar el progreso",
    )
    parser.add_argument(
        "--teacher",
        choices=["github", "openrouter"],
        default="github",
        help="Backend del profesor",
    )
    parser.add_argument(
        "--model", default="openai/gpt-4o-mini", help="Modelo del profesor"
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="API key (o usa GITHUB_TOKEN / OPENROUTER_API_KEY)",
    )
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate base")
    parser.add_argument("--ewc-lambda", type=float, default=500.0, help="Fuerza EWC")
    parser.add_argument(
        "--max-lessons", type=int, default=200, help="Máximo de lecciones"
    )
    parser.add_argument(
        "--port", type=int, default=8888, help="Puerto del servidor web"
    )
    parser.add_argument("--no-ui", action="store_true", help="Solo consola, sin UI web")
    parser.add_argument(
        "--no-bio", action="store_true", help="Desactivar mecanismos bio-inspirados"
    )
    parser.add_argument("--level", type=int, default=1, help="Nivel inicial (1-5)")

    args = parser.parse_args()

    config = ClassroomConfig(
        checkpoint_in=args.checkpoint,
        checkpoint_out=args.checkpoint_out,
        teacher_backend=args.teacher,
        teacher_model=args.model,
        api_key=args.api_key,
        lr_base=args.lr,
        ewc_lambda=args.ewc_lambda,
        max_lessons=args.max_lessons,
        port=args.port,
        start_level=args.level,
        bio_enabled=not args.no_bio,
    )

    engine = ClassroomEngine(config)
    engine.load()

    if not engine.teacher:
        print("\n❌ No se pudo configurar el profesor. Revisa tu API key.")
        sys.exit(1)

    if args.no_ui:
        # Modo consola
        print("\n" + "=" * 60)
        print("  🏫 PAMPAr CLASSROOM — Modo consola")
        print("=" * 60)
        engine._running = True
        try:
            while engine._running and engine.lesson_count < config.max_lessons:
                engine.run_lesson()
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n\n  Interrumpido por usuario.")
        finally:
            engine.save_session()
            rec = engine.save_recording()
            engine._save_checkpoint()
            accuracy = engine.total_correct / max(1, engine.lesson_count)
            print(
                f"\n  📊 Resumen: {engine.lesson_count} lecciones, {accuracy:.1%} accuracy, nivel {engine.current_level}"
            )
            if rec:
                print(f"  🎥 Grabación guardada: {rec}")
    else:
        # Modo UI web
        ui_path = Path(__file__).parent / "classroom.html"
        ClassroomHandler.engine = engine
        ClassroomHandler.ui_path = str(ui_path)

        server = HTTPServer(("127.0.0.1", config.port), ClassroomHandler)
        print(f"\n  🏫 PAMPAr CLASSROOM — UI en http://localhost:{config.port}")
        print(f"  Presiona Ctrl+C para detener\n")

        # Abrir navegador automáticamente
        try:
            import webbrowser

            webbrowser.open(f"http://localhost:{config.port}")
        except Exception:
            pass

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n  Detenido.")
            engine.save_session()
            engine.save_recording()
            engine._save_checkpoint()
            server.server_close()


if __name__ == "__main__":
    main()
