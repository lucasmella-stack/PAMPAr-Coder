"""classroom_teacher.py — Mentor conversacional via API (GitHub Models / OpenRouter / Qwen).

El mentor genera lecciones completas como un tutor en un chat:
explicaciones, ejemplos, ejercicios y correcciones, todo en un flujo
conversacional que el alumno (PamparV3) absorbe via gradient descent.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

# ── System prompts ──────────────────────────────────────────────────────────

_META_CONTEXT = (
    "You are a senior AI mentor. Your student is PamparV3, a 108M parameter "
    "language model learning to write Python code. PamparV3 learns via gradient "
    "descent from your responses — every token you produce directly shapes its "
    "weights.\n\n"
    "Teaching guidelines:\n"
    "- Write CLEAN, CORRECT, IDIOMATIC Python — the student learns patterns from your exact tokens\n"
    "- Consistent formatting — the tokenizer is sensitive to whitespace/structure\n"
    "- Prefer simple, readable solutions over clever one-liners\n"
    "- Include type hints and brief docstrings — they teach structure\n"
    "- No unnecessary imports or abstractions — high signal-to-noise ratio\n"
)

# Prompt para generar una lección conversacional completa
_SYSTEM_MENTOR = (
    _META_CONTEXT
    + "\nYou are having a teaching CONVERSATION with PamparV3. Generate a lesson "
    "that flows naturally, as if you were chatting with a student who is learning.\n\n"
    "Structure your lesson as:\n"
    "1. Brief concept explanation (2-3 sentences, connect to what they already know)\n"
    "2. A simple example with code showing the concept\n"
    "3. A practice exercise for the student to try\n\n"
    "Format EXACTLY like this (the markers are required):\n"
    "---EXPLAIN---\n"
    "[Your explanation of the concept]\n"
    "---EXAMPLE---\n"
    "[A complete working code example demonstrating the concept]\n"
    "---EXERCISE---\n"
    "[A clear problem statement for the student to solve]\n"
    "---SOLUTION---\n"
    "[The correct solution to the exercise]\n\n"
    "Rules:\n"
    "- Code must be clean Python, NO markdown, NO ```python blocks\n"
    "- Each example/solution must be a complete, runnable function\n"
    "- Use the EXACT function name you specify in the exercise\n"
    "- Explanations in SPANISH, code in English\n"
    "- Keep explanations SHORT — the student learns more from code than words\n"
    "- Build on previously mastered concepts when possible\n"
)

# Prompt para continuar la conversación después de ver el intento del alumno
_SYSTEM_RESPOND = (
    _META_CONTEXT
    + "\nThe student just attempted an exercise. Continue the teaching conversation.\n\n"
    "Respond with a JSON object:\n"
    '  "correct": true/false,\n'
    '  "feedback": "1-2 sentences in Spanish about what went right/wrong",\n'
    '  "fix": "corrected code if wrong, empty string if correct",\n'
    '  "next_concept": "what concept to teach next based on student performance"\n'
    "\nRespond ONLY with the JSON object.\n"
    "Be strict: wrong function name, broken syntax, or incorrect logic = incorrect."
)

# Prompt legacy para evaluar (mantener compatibilidad)
_SYSTEM_SOLVE = (
    _META_CONTEXT
    + "When given a coding problem, respond with ONLY the Python code solution. "
    "No explanations, no markdown, no ```python blocks. Just clean, correct Python code."
)


class Teacher:
    """Modelo profesor via API (GitHub Models, OpenRouter o Qwen/DashScope)."""

    ENDPOINTS = {
        "github": "https://models.inference.ai.azure.com/chat/completions",
        "openrouter": "https://openrouter.ai/api/v1/chat/completions",
        "qwen": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
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
            {"role": "system", "content": _SYSTEM_SOLVE},
            {"role": "user", "content": problem},
        ]
        return self._call(messages, max_tokens=500, temperature=0.2)

    def generate_lesson(self, student_profile: str, concept: str) -> dict | None:
        """Genera una lección conversacional completa.

        Args:
            student_profile: Resumen del perfil del alumno (qué sabe, qué le cuesta).
            concept: Concepto a enseñar en esta lección.

        Returns:
            Dict con keys: explain, example, exercise, solution. None si falla.
        """
        user_msg = (
            f"Student profile:\n{student_profile}\n\n"
            f"Teach a lesson about: {concept}\n"
            f"Generate the lesson now."
        )
        messages = [
            {"role": "system", "content": _SYSTEM_MENTOR},
            {"role": "user", "content": user_msg},
        ]
        raw = self._call(messages, max_tokens=1200, temperature=0.4)
        if not raw:
            return None
        return self._parse_lesson(raw)

    def respond_to_attempt(
        self, exercise: str, student_code: str, student_profile: str
    ) -> dict:
        """Evalúa el intento del alumno y sugiere qué enseñar después.

        Returns:
            Dict con: correct, feedback, fix, next_concept.
        """
        messages = [
            {"role": "system", "content": _SYSTEM_RESPOND},
            {
                "role": "user",
                "content": (
                    f"Student profile:\n{student_profile}\n\n"
                    f"Exercise:\n{exercise}\n\n"
                    f"Student's attempt:\n{student_code}"
                ),
            },
        ]
        raw = self._call(messages, max_tokens=600, temperature=0.1)
        if not raw:
            return {
                "correct": False,
                "feedback": "Error de comunicación",
                "fix": "",
                "next_concept": "",
            }
        try:
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            result = json.loads(raw)
            result.setdefault("next_concept", "")
            return result
        except json.JSONDecodeError:
            return {
                "correct": False,
                "feedback": raw[:200],
                "fix": "",
                "next_concept": "",
            }

    def _parse_lesson(self, raw: str) -> dict | None:
        """Parsea la respuesta del mentor en secciones."""
        sections: dict[str, str] = {}
        markers = {
            "---EXPLAIN---": "explain",
            "---EXAMPLE---": "example",
            "---EXERCISE---": "exercise",
            "---SOLUTION---": "solution",
        }

        current_key: str | None = None
        current_lines: list[str] = []

        for line in raw.split("\n"):
            stripped = line.strip()
            if stripped in markers:
                if current_key:
                    sections[current_key] = "\n".join(current_lines).strip()
                current_key = markers[stripped]
                current_lines = []
            elif current_key is not None:
                current_lines.append(line)

        if current_key:
            sections[current_key] = "\n".join(current_lines).strip()

        # Validar que tenemos al menos ejemplo y solución
        if "example" not in sections or "solution" not in sections:
            # Fallback: tratar todo como ejemplo
            return {
                "explain": "",
                "example": raw.strip(),
                "exercise": "",
                "solution": raw.strip(),
            }

        sections.setdefault("explain", "")
        sections.setdefault("exercise", "")
        return sections
