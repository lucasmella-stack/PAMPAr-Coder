# SPDX-License-Identifier: BUSL-1.1
"""
Tests del runtime.Agente de PAMPAr-Coder v3.

El Agente requiere sentencepiece + tokenizer .model para instanciarse,
por eso estos tests evitan usar el constructor real y en su lugar:
  - Testean los métodos puramente lógicos usando mocks
  - Testean componentes integrados (sin modelo real) para el orquestador

Cubren:
  - _parece_codigo() detecta código correctamente
  - _construir_prompt() incluye todos los bloques esperados
  - _procesar_acciones() parsea y ejecuta [LEER:], [EJECUTAR:], [TESTS:]
  - limpiar_historial() limpia el estado
  - stats() retorna dict con claves esperadas
  - aceptar_finetune/rechazar_finetune retornan strings
  - SYSTEM_PROMPT tiene las instrucciones esperadas
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import torch

# Garantizar que el root del proyecto esté en el PATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from pampar.runtime.agente import SYSTEM_PROMPT


# ==============================================================================
# FIXTURE: Agente con dependencias mockeadas
# ==============================================================================

def _make_agente(tmp_path: Path) -> "Agente":
    """
    Crea un Agente real con modelo mini + tokenizer mockeado.

    Evita depender del tokenizer .model y del checkpoint pre-entrenado.
    El modelo se inicializa desde cero (pesos aleatorios).
    """
    from pampar.runtime.agente import Agente
    from pampar.coder.v3.config import ConfigV3

    config_mini = ConfigV3(
        vocab_size=256,
        dim=32,
        n_streams=4,
        n_levels=2,
        n_heads=2,
        n_kv_heads=1,
        ffn_mult=2.0,
        n_zonas=52,            # Hardcodeado en v2 LLAVES — NO cambiar
        n_territorios=4,
        lateral_bottleneck=8,
        ventana_contexto=2,
        max_seq_len=32,
        dropout=0.0,
        use_checkpoint=False,
    )

    # Mock del tokenizer SentencePiece
    mock_tok = MagicMock()
    mock_tok.Encode.side_effect = lambda text: [1, 2, 3, 4, 5]  # siempre 5 tokens
    mock_tok.Decode.side_effect = lambda ids: "respuesta mockeada"
    mock_tok.vocab_size.return_value = 256
    mock_tok.GetPieceSize.return_value = 256            # necesario para registrar_tokenizer
    mock_tok.IdToPiece.side_effect = lambda i: str(i)  # retorna string para clasificar_token

    # Parchar sentencepiece.SentencePieceProcessor para que retorne nuestro mock
    with patch("pampar.runtime.agente.spm") as mock_spm:
        mock_spm.SentencePieceProcessor.return_value = mock_tok

        agente = Agente(
            checkpoint="no_existe_checkpoint.pt",  # Inicia con pesos aleatorios
            tokenizer_path="no_existe_tokenizer.model",
            config=config_mini,
            workspace_root=str(tmp_path),
            memoria_dir=str(tmp_path / "memoria"),
            device="cpu",
            max_historial=5,
        )

    return agente


@pytest.fixture
def agente(tmp_path: Path):
    """Agente con modelo pequeño y tokenizer mockeado."""
    return _make_agente(tmp_path)


# ==============================================================================
# TESTS: SYSTEM_PROMPT
# ==============================================================================

class TestSystemPrompt:
    def test_contiene_instruccion_leer(self):
        """El SYSTEM_PROMPT debe documentar la acción [LEER:]."""
        assert "[LEER:" in SYSTEM_PROMPT

    def test_contiene_instruccion_ejecutar(self):
        """El SYSTEM_PROMPT debe documentar la acción [EJECUTAR:]."""
        assert "[EJECUTAR:" in SYSTEM_PROMPT

    def test_contiene_instruccion_tests(self):
        """El SYSTEM_PROMPT debe documentar la acción [TESTS:]."""
        assert "[TESTS:" in SYSTEM_PROMPT

    def test_en_espanol(self):
        """El SYSTEM_PROMPT debe contener al menos una palabra española."""
        palabras_esp = ["archivos", "código", "Sos", "siempre", "acceso"]
        assert any(p in SYSTEM_PROMPT for p in palabras_esp)


# ==============================================================================
# TESTS: Métodos lógicos sin llamada al modelo
# ==============================================================================

class TestLogicaSinModelo:
    def test_parece_codigo_con_def(self, agente):
        """Un texto con 'def' debe detectarse como código."""
        assert agente._parece_codigo("def calcular(x):\n    return x * 2") is True

    def test_parece_codigo_con_class(self, agente):
        """Un texto con 'class' debe detectarse como código."""
        assert agente._parece_codigo("class Trainer:\n    pass") is True

    def test_parece_codigo_con_import(self, agente):
        """Un texto con 'import' debe detectarse como código."""
        assert agente._parece_codigo("import torch\nimport numpy as np") is True

    def test_no_parece_codigo_texto_simple(self, agente):
        """Un texto conversacional corriente no debe detectarse como código."""
        assert agente._parece_codigo("¿Cómo estás? Cuéntame sobre Python.") is False

    def test_construir_prompt_incluye_system(self, agente):
        """El prompt construido debe incluir el SYSTEM_PROMPT."""
        prompt = agente._construir_prompt("hola", "")
        assert "PAMPAr" in prompt

    def test_construir_prompt_incluye_mensaje_usuario(self, agente):
        """El mensaje del usuario debe aparecer en el prompt."""
        prompt = agente._construir_prompt("¿qué es un decorador?", "")
        assert "¿qué es un decorador?" in prompt

    def test_construir_prompt_incluye_ctx_rag(self, agente):
        """El contexto RAG debe estar presente si se pasa."""
        ctx = "[MEMORIA RELEVANTE]\nEjemplo de código\n[/MEMORIA RELEVANTE]"
        prompt = agente._construir_prompt("explica esto", ctx)
        assert "[MEMORIA RELEVANTE]" in prompt

    def test_construir_prompt_incluye_historial(self, agente):
        """El historial previo debe aparecer en el prompt."""
        agente._historial = [
            {"role": "user", "text": "primer turno"},
            {"role": "assistant", "text": "primera respuesta"},
        ]
        prompt = agente._construir_prompt("segundo turno", "")
        assert "primer turno" in prompt
        assert "primera respuesta" in prompt

    def test_limpiar_historial(self, agente):
        """limpiar_historial() debe dejar el historial vacío."""
        agente._historial = [
            {"role": "user", "text": "algo"},
            {"role": "assistant", "text": "respuesta"},
        ]
        agente.limpiar_historial()
        assert agente._historial == []

    def test_stats_estructura(self, agente):
        """stats() debe retornar un dict con las claves esperadas."""
        s = agente.stats()
        for clave in ("modelo", "rag", "cola_finetune", "historial_turnos", "device"):
            assert clave in s, f"Falta clave '{clave}' en stats()"

    def test_stats_historial_turnos(self, agente):
        """stats()['historial_turnos'] debe contar pares user/assistant."""
        agente._historial = [
            {"role": "user", "text": "t1"},
            {"role": "assistant", "text": "r1"},
            {"role": "user", "text": "t2"},
            {"role": "assistant", "text": "r2"},
        ]
        assert agente.stats()["historial_turnos"] == 2

    def test_rechazar_finetune_retorna_string(self, agente):
        """rechazar_finetune() debe retornar un string no vacío."""
        msg = agente.rechazar_finetune()
        assert isinstance(msg, str)
        assert len(msg) > 10

    def test_aceptar_finetune_retorna_string(self, agente):
        """aceptar_finetune() debe retornar un string (éxito o error)."""
        # No importa si el training falla (no hay script real)
        msg = agente.aceptar_finetune()
        assert isinstance(msg, str)
        assert len(msg) > 10

    def test_describe_retorna_string(self, agente):
        """describe() debe delegar al modelo y retornar string."""
        desc = agente.describe()
        assert isinstance(desc, str)
        assert len(desc) > 0


# ==============================================================================
# TESTS: _procesar_acciones (parsing de actions)
# ==============================================================================

class TestProcesarAcciones:
    def test_sin_acciones_devuelve_original(self, agente):
        """Sin marcadores de acción, la respuesta se devuelve sin cambios."""
        respuesta = "Esto es una respuesta normal sin acciones."
        resultado = agente._procesar_acciones(respuesta)
        assert resultado == respuesta

    def test_accion_ejecutar(self, agente, tmp_path: Path):
        """[EJECUTAR: codigo] debe ejecutar el código e insertar el output."""
        respuesta = "[EJECUTAR:\nprint('desde accion')\n]"
        resultado = agente._procesar_acciones(respuesta)
        # El marcador debe haberse reemplazado por algo
        assert "[EJECUTAR:" not in resultado
        # El output del código debe estar presente
        assert "desde accion" in resultado or "STDOUT" in resultado

    def test_accion_leer_archivo_valido(self, agente, tmp_path: Path):
        """[LEER: ruta] debe leer el archivo e insertar su contenido."""
        # Crear un archivo en el workspace del agente
        archivo = Path(agente.lector.root) / "leeme.py"
        archivo.write_text("# archivo de prueba\nx = 42\n", encoding="utf-8")

        respuesta = "[LEER: leeme.py]"
        resultado = agente._procesar_acciones(respuesta)
        assert "[LEER:" not in resultado
        # El contenido del archivo debe estar en la respuesta
        assert "leeme.py" in resultado or "x = 42" in resultado

    def test_accion_leer_archivo_inexistente(self, agente):
        """[LEER: archivo_que_no_existe.py] debe insertar un mensaje de error."""
        respuesta = "[LEER: archivo_fantasma.py]"
        resultado = agente._procesar_acciones(respuesta)
        assert "[LEER:" not in resultado
        # El marcador fue reemplazado (por un error del lector)
        assert len(resultado) > 0

    def test_multiples_acciones(self, agente):
        """Múltiples acciones [EJECUTAR:] en la misma respuesta se procesan todas."""
        respuesta = (
            "Primero: [EJECUTAR:\nprint(1)\n]\n"
            "Luego: [EJECUTAR:\nprint(2)\n]"
        )
        resultado = agente._procesar_acciones(respuesta)
        assert "[EJECUTAR:" not in resultado
