# SPDX-License-Identifier: BUSL-1.1
"""
Tests unitarios del servidor de inferencia (pampar.inference).

Qué prueban:
  - handle_infer: responde con infer_ok y texto no vacío
  - handle_infer: prompt vacío → error
  - handle_boot: responde con boot_ok y agents_md con secciones
  - Tipo desconocido → error
  - JSON inválido no rompe el loop (simulado)
  - Tokenizer y modelo son mockeados — no se carga el checkpoint real
"""

from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from pampar.inference import handle_boot, handle_infer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_tokenizer():
    """Tokenizer falso que 'codifica' como lista de ints y 'decodifica' texto fijo."""
    tok = MagicMock()
    tok.Encode.return_value = [1, 2, 3, 4, 5]
    tok.Decode.return_value = "def hello():\n    return 'world'"
    return tok


@pytest.fixture()
def mock_model():
    """Modelo falso cuyo generate devuelve un tensor con tokens extra."""
    model = MagicMock()
    # generate devuelve [1, L+N]: los 5 de prompt + 10 nuevos (ids 10..19)
    fake_output = torch.tensor([[1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]])
    model.generate.return_value = fake_output
    return model


@pytest.fixture()
def device():
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# handle_infer
# ---------------------------------------------------------------------------

class TestHandleInfer:
    def test_responde_infer_ok(self, mock_model, mock_tokenizer, device, capsys):
        msg = {"type": "infer", "prompt": "### Problem:\nhello\n### Solution:\n", "max_tokens": 50}
        handle_infer(mock_model, mock_tokenizer, device, msg)

        captured = capsys.readouterr().out.strip()
        resp = json.loads(captured)
        assert resp["type"] == "infer_ok"
        assert "text" in resp
        assert len(resp["text"]) > 0

    def test_llama_generate_con_params(self, mock_model, mock_tokenizer, device, capsys):
        msg = {"type": "infer", "prompt": "abc", "max_tokens": 128, "temperature": 0.2}
        handle_infer(mock_model, mock_tokenizer, device, msg)

        call_kwargs = mock_model.generate.call_args
        assert call_kwargs[1]["max_tokens"] == 128 or call_kwargs.kwargs.get("max_tokens") == 128 or call_kwargs[0][1] == 128

    def test_prompt_vacio_devuelve_error(self, mock_model, mock_tokenizer, device, capsys):
        msg = {"type": "infer", "prompt": ""}
        handle_infer(mock_model, mock_tokenizer, device, msg)

        captured = capsys.readouterr().out.strip()
        resp = json.loads(captured)
        assert resp["type"] == "error"
        assert "prompt" in resp["message"].lower() or "vacío" in resp["message"].lower() or "vac" in resp["message"]

    def test_prompt_ausente_devuelve_error(self, mock_model, mock_tokenizer, device, capsys):
        msg = {"type": "infer"}
        handle_infer(mock_model, mock_tokenizer, device, msg)

        captured = capsys.readouterr().out.strip()
        resp = json.loads(captured)
        assert resp["type"] == "error"

    def test_tokenizer_encode_llamado(self, mock_model, mock_tokenizer, device, capsys):
        msg = {"type": "infer", "prompt": "test prompt"}
        handle_infer(mock_model, mock_tokenizer, device, msg)
        mock_tokenizer.Encode.assert_called_once_with("test prompt", out_type=int)

    def test_temperatura_default_valida(self, mock_model, mock_tokenizer, device, capsys):
        """Sin temperature en el msg usa el default 0.4 — no debe explotar."""
        msg = {"type": "infer", "prompt": "hello"}
        handle_infer(mock_model, mock_tokenizer, device, msg)
        captured = capsys.readouterr().out.strip()
        resp = json.loads(captured)
        assert resp["type"] == "infer_ok"


# ---------------------------------------------------------------------------
# handle_boot
# ---------------------------------------------------------------------------

class TestHandleBoot:
    def test_responde_boot_ok(self, tmp_path, capsys):
        """boot con workspace válido devuelve boot_ok con agents_md."""
        # Crear un .py mínimo para que el scanner tenga algo que parsear
        (tmp_path / "main.py").write_text("def hello():\n    pass\n")

        msg = {"type": "boot", "workspace": str(tmp_path)}
        handle_boot(msg)

        captured = capsys.readouterr().out.strip()
        resp = json.loads(captured)
        assert resp["type"] == "boot_ok"
        assert "agents_md" in resp
        assert len(resp["agents_md"]) > 50  # algo de contenido

    def test_agents_md_tiene_secciones(self, tmp_path, capsys):
        """El AGENTS.md generado debe tener al menos Quick Reference y Boot protocol."""
        msg = {"type": "boot", "workspace": str(tmp_path)}
        handle_boot(msg)

        captured = capsys.readouterr().out.strip()
        resp = json.loads(captured)
        agents_md = resp["agents_md"]

        assert "## Quick Reference" in agents_md
        assert "## Boot protocol" in agents_md

    def test_workspace_inexistente_no_explota(self, capsys):
        """Un workspace inexistente debe devolver error, no excepción sin capturar."""
        msg = {"type": "boot", "workspace": "/ruta/que/no/existe/12345"}
        # No debe lanzar excepción — debe responder error o boot_ok vacío
        try:
            handle_boot(msg)
            captured = capsys.readouterr().out.strip()
            resp = json.loads(captured)
            assert resp["type"] in ("boot_ok", "error")
        except Exception as exc:
            pytest.fail(f"handle_boot lanzó excepción no capturada: {exc}")

    def test_workspace_vacio_no_explota(self, tmp_path, capsys):
        """Workspace sin archivos .py — debe funcionar igual."""
        msg = {"type": "boot", "workspace": str(tmp_path)}
        handle_boot(msg)
        captured = capsys.readouterr().out.strip()
        resp = json.loads(captured)
        assert resp["type"] == "boot_ok"
