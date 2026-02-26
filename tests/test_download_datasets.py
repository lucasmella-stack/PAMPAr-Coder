# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""Tests del pipeline de descarga de datasets open source."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


# =============================================================================
# HELPERS — datos sintéticos que emulan cada dataset
# =============================================================================

def _make_sc2_rows(n: int):
    """Emula filas de bigcode/self-oss-instruct-sc2-exec-filter-50k."""
    for i in range(n):
        yield {
            "instruction": f"Write a Python function that computes fibonacci({i})",
            "response": (
                f"def fibonacci(n: int) -> int:\n"
                f"    \"\"\"Compute fibonacci number recursively.\"\"\"\n"
                f"    if n <= 1:\n        return n\n"
                f"    return fibonacci(n-1) + fibonacci(n-2)\n"
            ),
            "seed": f"def fib_{i}(): pass",
            "concepts": ["recursion", "dynamic programming"],
        }


def _make_commit_rows(n: int):
    """Emula filas de bigcode/commitpackft."""
    for i in range(n):
        yield {
            "subject": f"Refactor loop to use list comprehension (iteration {i})",
            "message": f"Refactor loop to use list comprehension (iteration {i})\n",
            "old_contents": (
                f"result = []\nfor x in range({i+1}):\n    result.append(x * 2)\n"
            ),
            "new_contents": f"result = [x * 2 for x in range({i+1})]\n",
            "lang": "Python",
            "license": "mit",
            "repos": f"user/repo-{i}",
            "commit": f"abc{i:03d}",
            "old_file": "main.py",
            "new_file": "main.py",
        }


def _make_smollm_rows(n: int):
    """Emula filas de HuggingFaceFW/smollm-corpus."""
    for i in range(n):
        yield {
            "text": (
                f"def calculate_metrics_{i}(data: list) -> dict:\n"
                f"    \"\"\"Calculate statistical metrics for given data.\"\"\"\n"
                f"    return {{\n"
                f"        'mean': sum(data) / len(data),\n"
                f"        'min': min(data),\n"
                f"        'max': max(data),\n"
                f"    }}\n"
            ),
        }


def _make_apps_rows(n: int):
    """Emula filas de codeparrot/apps."""
    for i in range(n):
        yield {
            "question": (
                f"Problem {i}: Given a list of integers, find the maximum subarray sum.\n"
                f"Input: A list of n integers\nOutput: Maximum sum of contiguous subarray"
            ),
            "solutions": json.dumps([
                (
                    f"def max_subarray_{i}(nums):\n"
                    f"    max_sum = nums[0]\n"
                    f"    current = nums[0]\n"
                    f"    for num in nums[1:]:\n"
                    f"        current = max(num, current + num)\n"
                    f"        max_sum = max(max_sum, current)\n"
                    f"    return max_sum\n"
                ),
                f"def brute_{i}(nums):\n    return max(sum(nums[i:j]) for i in range(len(nums)) for j in range(i+1, len(nums)+1))\n",
            ]),
            "difficulty": "interview",
            "url": f"https://example.com/problem/{i}",
        }


# =============================================================================
# TESTS: normalizadores
# =============================================================================

class TestNormalizacion:
    """Verifica que cada dataset se normaliza al formato correcto."""

    def test_sc2_formato_instruction_response(self):
        from download_open_datasets import stream_sc2_instruct

        mock_ds = list(_make_sc2_rows(3))

        with patch("download_open_datasets.load_dataset") as mock_ld:
            mock_ld.return_value = iter(mock_ds)
            results = list(stream_sc2_instruct(3))

        assert len(results) == 3
        for r in results:
            assert "### Instruction:" in r["text"]
            assert "### Response:" in r["text"]
            assert r["source"] == "bigcode/sc2-exec-instruct"
            assert r["license"] == "odc-by"
            assert r["lang"] == "python"

    def test_commits_formato_context_instruction_response(self):
        from download_open_datasets import stream_commitpackft

        mock_ds = list(_make_commit_rows(3))

        with patch("download_open_datasets.load_dataset") as mock_ld:
            mock_ld.return_value = iter(mock_ds)
            results = list(stream_commitpackft(3))

        assert len(results) == 3
        for r in results:
            assert "### Context:" in r["text"]
            assert "### Instruction:" in r["text"]
            assert "### Response:" in r["text"]
            assert "```python" in r["text"]
            assert r["source"] == "bigcode/commitpackft"
            assert r["license"] == "mit"

    def test_commits_filtra_old_igual_new(self):
        """Commits donde old_contents == new_contents se descartan."""
        from download_open_datasets import stream_commitpackft

        rows = [
            {"subject": "no change", "old_contents": "x = 1", "new_contents": "x = 1",
             "lang": "Python", "license": "mit", "message": "no change\n"},
        ]

        with patch("download_open_datasets.load_dataset") as mock_ld:
            mock_ld.return_value = iter(rows)
            results = list(stream_commitpackft(10))

        assert len(results) == 0

    def test_commits_filtra_subject_corto(self):
        """Commits con subject < 10 chars se descartan."""
        from download_open_datasets import stream_commitpackft

        rows = [
            {"subject": "fix", "old_contents": "x = 1", "new_contents": "x = 2",
             "lang": "Python", "license": "mit", "message": "fix\n"},
        ]

        with patch("download_open_datasets.load_dataset") as mock_ld:
            mock_ld.return_value = iter(rows)
            results = list(stream_commitpackft(10))

        assert len(results) == 0

    def test_smollm_formato(self):
        from download_open_datasets import stream_smollm_python

        mock_ds = list(_make_smollm_rows(3))

        with patch("download_open_datasets.load_dataset") as mock_ld:
            mock_ld.return_value = iter(mock_ds)
            results = list(stream_smollm_python(3))

        assert len(results) == 3
        for r in results:
            assert "### Instruction:" in r["text"]
            assert "### Response:" in r["text"]
            assert r["source"].startswith("HuggingFaceFW")
            assert r["license"] == "odc-by"

    def test_smollm_filtra_textos_cortos(self):
        """Textos < 100 chars se descartan."""
        from download_open_datasets import stream_smollm_python

        rows = [{"text": "x = 1"}]

        with patch("download_open_datasets.load_dataset") as mock_ld:
            mock_ld.return_value = iter(rows)
            results = list(stream_smollm_python(10))

        assert len(results) == 0

    def test_apps_formato(self):
        from download_open_datasets import stream_apps

        mock_ds = list(_make_apps_rows(3))

        with patch("download_open_datasets.load_dataset") as mock_ld:
            mock_ld.return_value = iter(mock_ds)
            results = list(stream_apps(3))

        assert len(results) == 3
        for r in results:
            assert "### Instruction:" in r["text"]
            assert "```python" in r["text"]
            assert r["source"] == "codeparrot/apps"
            assert r["license"] == "mit"

    def test_apps_filtra_sin_soluciones(self):
        """Problemas sin soluciones se descartan."""
        from download_open_datasets import stream_apps

        rows = [{"question": "A long question " * 10, "solutions": "[]"}]

        with patch("download_open_datasets.load_dataset") as mock_ld:
            mock_ld.return_value = iter(rows)
            results = list(stream_apps(10))

        assert len(results) == 0

    def test_apps_soluciones_invalidas(self):
        """solutions no-JSON se descartan."""
        from download_open_datasets import stream_apps

        rows = [{"question": "A good question " * 5, "solutions": "not-json"}]

        with patch("download_open_datasets.load_dataset") as mock_ld:
            mock_ld.return_value = iter(rows)
            results = list(stream_apps(10))

        assert len(results) == 0


# =============================================================================
# TESTS: formato de salida
# =============================================================================

class TestFormatoSalida:
    def test_fmt_basico(self):
        from download_open_datasets import _fmt
        result = _fmt("Write a function", "def f(): pass")
        assert result == "### Instruction:\nWrite a function\n\n### Response:\ndef f(): pass"

    def test_fmt_con_context(self):
        from download_open_datasets import _fmt
        result = _fmt("Refactor this", "new code", "old code")
        assert "### Context:\nold code" in result
        assert "### Instruction:\nRefactor this" in result
        assert "### Response:\nnew code" in result
        # Context debe ir primero
        assert result.index("### Context:") < result.index("### Instruction:")

    def test_fmt_context_vacio_omitido(self):
        from download_open_datasets import _fmt
        result = _fmt("Write a function", "def f(): pass", "")
        assert "### Context:" not in result

    def test_max_samples_respetado(self):
        """El streamer no entrega más de max_samples."""
        from download_open_datasets import stream_sc2_instruct

        mock_ds = list(_make_sc2_rows(100))

        with patch("download_open_datasets.load_dataset") as mock_ld:
            mock_ld.return_value = iter(mock_ds)
            results = list(stream_sc2_instruct(5))

        assert len(results) == 5


# =============================================================================
# TESTS: pipeline completo (con filesystem real)
# =============================================================================

class TestPipelineCompleto:
    def test_descarga_y_guarda_jsonl(self, tmp_path):
        """Simula el pipeline completo y verifica output válido."""
        from download_open_datasets import stream_sc2_instruct

        output_file = tmp_path / "sc2.jsonl"
        mock_ds = list(_make_sc2_rows(10))

        with patch("download_open_datasets.load_dataset") as mock_ld:
            mock_ld.return_value = iter(mock_ds)
            samples = list(stream_sc2_instruct(10))

        with open(output_file, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        # Verificar que el JSONL es válido y legible por PAMPAr
        lines = output_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 10
        for line in lines:
            parsed = json.loads(line)
            assert "text" in parsed
            assert "source" in parsed
            assert "license" in parsed
            assert "### Instruction:" in parsed["text"]
            assert "### Response:" in parsed["text"]

    def test_todos_datasets_producen_samples_validos(self):
        """Verifica que todos los streamers producen formato correcto."""
        from download_open_datasets import (
            stream_sc2_instruct,
            stream_commitpackft,
            stream_smollm_python,
            stream_apps,
        )

        mock_data = {
            stream_sc2_instruct: list(_make_sc2_rows(5)),
            stream_commitpackft: list(_make_commit_rows(5)),
            stream_smollm_python: list(_make_smollm_rows(5)),
            stream_apps: list(_make_apps_rows(5)),
        }

        for fn, data in mock_data.items():
            with patch("download_open_datasets.load_dataset") as mock_ld:
                mock_ld.return_value = iter(data)
                results = list(fn(5))

            assert len(results) > 0, f"{fn.__name__} no produjo samples"
            for r in results:
                assert isinstance(r["text"], str)
                assert isinstance(r["source"], str)
                assert isinstance(r["license"], str)
                assert "### Instruction:" in r["text"]
                assert "### Response:" in r["text"]
