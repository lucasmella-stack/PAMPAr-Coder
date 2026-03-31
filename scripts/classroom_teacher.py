"""classroom_teacher.py — Mentor conversacional via API (GitHub Models / OpenRouter / Qwen).

El mentor genera lecciones completas como un tutor en un chat:
explicaciones, ejemplos, ejercicios y correcciones, todo en un flujo
conversacional que el alumno (PamparV3) absorbe via gradient descent.

Tipos de lección según etapa del curriculum:
  conceptual — sin código, lenguaje natural, analogías cotidianas
  bridge     — concepto + correspondencia Python
  coding     — Python puro (comportamiento original)
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

# ── System prompts ──────────────────────────────────────────────────────────

_META_CONTEXT = (
    "You are a senior AI mentor. Your student is PamparV3, a 108M parameter "
    "language model learning to understand the world and eventually write Python code. "
    "PamparV3 learns via gradient descent from your responses — every token you "
    "produce directly shapes its weights.\n\n"
)

# ── Prompt CONCEPTUAL (etapas 0-3): sin código, lenguaje natural ────────────
_SYSTEM_CONCEPTUAL = (
    _META_CONTEXT + "RIGHT NOW your student is in the CONCEPTUAL stage. This means:\n"
    "- NO CODE whatsoever — not a single line of Python\n"
    "- Teach like a primary school teacher starting from absolute basics\n"
    "- Use everyday analogies, real-world examples, questions\n"
    "- Use SPANISH. Keep it warm, simple, conversational.\n"
    "- The student is learning what things ARE before learning to code them.\n\n"
    "Format EXACTLY:\n"
    "---EXPLAIN---\n"
    "[Explanation of the concept in simple Spanish, using everyday analogies]\n"
    "---EXAMPLE---\n"
    "[2-3 concrete real-world examples that illustrate the concept. No code.]\n"
    "---CLAVE---\n"
    "[3 bullet points in Spanish starting with '- ': the exact ideas the student MUST retain from this lesson]\n"
    "---EXERCISE---\n"
    "[A simple question in Spanish the student can answer in natural language]\n"
    "---SOLUTION---\n"
    "[The ideal answer the student should give, in Spanish]\n\n"
    "Rules:\n"
    "- ZERO code. If the concept is 'number', talk about apples and people, not int.\n"
    "- Be warm and encouraging, like talking to a curious child.\n"
    "- Explanations AND examples AND exercise in SPANISH.\n"
    "- Keep the exercise answerable in 1-3 sentences.\n"
    "- The ---CLAVE--- section is the most important: distill the lesson into 3 ideas to retain.\n"
)

# ── Prompt BRIDGE (etapa 4): concepto cotidiano → Python ────────────────────
_SYSTEM_BRIDGE = (
    _META_CONTEXT
    + "Your student is in the BRIDGE stage: they understand everyday concepts and "
    "are now learning that Python is just a way to WRITE those concepts precisely.\n\n"
    "Teaching approach:\n"
    "- Always START with the everyday analogy they already know\n"
    "- THEN show the Python equivalent side-by-side\n"
    "- Use Spanish for explanations, Python for code\n"
    "- Keep code ultra-simple — 1-3 lines max\n\n"
    "Format EXACTLY:\n"
    "---EXPLAIN---\n"
    "[Everyday analogy first, then Python equivalent. Spanish + minimal Python.]\n"
    "---EXAMPLE---\n"
    "[Side by side: 'In real life: X ... In Python: Y'. Very short code.]\n"
    "---CLAVE---\n"
    "[3 bullet points in Spanish starting with '- ': what the student MUST retain from this bridge lesson]\n"
    "---EXERCISE---\n"
    "[A simple question that can be answered in Python + 1 sentence of explanation]\n"
    "---SOLUTION---\n"
    "[Correct Python + brief Spanish explanation of why it's correct]\n\n"
    "Rules:\n"
    "- Code blocks max 3 lines. No imports. No complex structures.\n"
    "- ALWAYS connect to the everyday concept first.\n"
    "- If the concept is 'variables in Python': start with 'Una caja con nombre...'\n"
    "- The ---CLAVE--- section bridges real-world and Python: make it memorable.\n"
)

# ── Prompt CODING (etapas 5+): Python puro ──────────────────────────────────
_SYSTEM_MENTOR = (
    _META_CONTEXT
    + "Your student understands concepts and is now learning Python deeply.\n\n"
    "Teaching guidelines:\n"
    "- Write CLEAN, CORRECT, IDIOMATIC Python\n"
    "- Consistent formatting (the tokenizer is sensitive to whitespace)\n"
    "- Prefer simple, readable solutions over clever one-liners\n"
    "- Include type hints and brief docstrings\n"
    "- No unnecessary imports or abstractions\n\n"
    "Structure your lesson as:\n"
    "---EXPLAIN---\n"
    "[Brief concept explanation in Spanish, 2-3 sentences]\n"
    "---EXAMPLE---\n"
    "[A complete working code example demonstrating the concept]\n"
    "---CLAVE---\n"
    "[3 bullet points in Spanish starting with '- ': the exact patterns/rules the student must memorize]\n"
    "---EXERCISE---\n"
    "[A clear problem statement for the student to solve]\n"
    "---SOLUTION---\n"
    "[The correct Python solution]\n\n"
    "Rules:\n"
    "- Code must be clean Python, NO markdown, NO ```python blocks\n"
    "- Each example/solution must be a complete, runnable function\n"
    "- Use the EXACT function name you specify in the exercise\n"
    "- Explanations in SPANISH, code in English\n"
    "- The ---CLAVE--- section is critical: distill the 3 most important patterns to remember.\n"
)

# ── Prompt de evaluación de respuestas conceptuales ────────────────────────
_SYSTEM_RESPOND_CONCEPTUAL = (
    _META_CONTEXT
    + "The student just answered a conceptual question (no code involved). "
    "Evaluate whether they demonstrated understanding of the concept.\n\n"
    "Respond with a JSON object:\n"
    '  "correct": true if the student showed understanding, false if confused,\n'
    '  "feedback": "1-2 sentences in Spanish acknowledging what was right/wrong",\n'
    '  "fix": "the ideal answer in Spanish if wrong, empty string if correct",\n'
    '  "next_concept": ""\n'
    "\nBe generous — if the student said ANYTHING related to the concept, mark correct.\n"
    "Respond ONLY with the JSON object.\n"
)

# ── Prompt de evaluación de respuestas puente ───────────────────────────────
_SYSTEM_RESPOND_BRIDGE = (
    _META_CONTEXT
    + "The student just attempted a bridge exercise that mixes everyday concepts "
    "with simple Python. Evaluate their understanding.\n\n"
    "Respond with a JSON object:\n"
    '  "correct": true/false,\n'
    '  "feedback": "1-2 sentences in Spanish",\n'
    '  "fix": "corrected code + explanation if wrong, empty if correct",\n'
    '  "next_concept": ""\n'
    "\nBe lenient with syntax — focus on whether the IDEA is correct.\n"
    "Respond ONLY with the JSON object.\n"
)

# ── Prompt de evaluación de código ─────────────────────────────────────────
_SYSTEM_RESPOND = (
    _META_CONTEXT
    + "The student just attempted a Python coding exercise. Continue the teaching conversation.\n\n"
    "Respond with a JSON object:\n"
    '  "correct": true/false,\n'
    '  "feedback": "1-2 sentences in Spanish about what went right/wrong",\n'
    '  "fix": "corrected code if wrong, empty string if correct",\n'
    '  "next_concept": "what concept to teach next based on student performance"\n'
    "\nRespond ONLY with the JSON object.\n"
    "Be strict: wrong function name, broken syntax, or incorrect logic = incorrect."
)

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

    def generate_lesson(
        self, student_profile: str, concept: str, concept_type: str = "coding"
    ) -> dict | None:
        """Genera una lección completa según el tipo de concepto.

        Args:
            student_profile: Resumen del perfil del alumno.
            concept: Nombre del concepto a enseñar.
            concept_type: "conceptual" | "bridge" | "coding"

        Returns:
            Dict con keys: explain, example, exercise, solution. None si falla.
        """
        if concept_type == "conceptual":
            system = _SYSTEM_CONCEPTUAL
            max_tokens = 800
        elif concept_type == "bridge":
            system = _SYSTEM_BRIDGE
            max_tokens = 1000
        else:
            system = _SYSTEM_MENTOR
            max_tokens = 1200

        user_msg = (
            f"Student profile:\n{student_profile}\n\n"
            f"Teach a lesson about: {concept}\n"
            f"Generate the lesson now."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ]
        raw = self._call(messages, max_tokens=max_tokens, temperature=0.4)
        if not raw:
            return None
        return self._parse_lesson(raw)

    def respond_to_attempt(
        self,
        exercise: str,
        student_code: str,
        student_profile: str,
        concept_type: str = "coding",
    ) -> dict:
        """Evalúa el intento del alumno según el tipo de concepto.

        Args:
            exercise: El ejercicio o pregunta planteada.
            student_code: La respuesta del alumno (código o texto).
            student_profile: Resumen del perfil.
            concept_type: "conceptual" | "bridge" | "coding"

        Returns:
            Dict con: correct, feedback, fix, next_concept.
        """
        if concept_type == "conceptual":
            system = _SYSTEM_RESPOND_CONCEPTUAL
        elif concept_type == "bridge":
            system = _SYSTEM_RESPOND_BRIDGE
        else:
            system = _SYSTEM_RESPOND

        messages = [
            {"role": "system", "content": system},
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
            "---CLAVE---": "clave",
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
        sections.setdefault("clave", "")
        sections.setdefault("exercise", "")
        return sections
