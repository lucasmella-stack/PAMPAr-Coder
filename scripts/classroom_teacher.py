"""classroom_teacher.py — Modelo profesor via API (GitHub Models / OpenRouter)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request


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
