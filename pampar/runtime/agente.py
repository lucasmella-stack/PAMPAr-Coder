# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Agente PAMPAr — Orquestador de modelo + memoria + skills.

El Agente es el punto de entrada para interactuar con PAMPAr.
Coordina:
  - PamparV3: el modelo de lenguaje (el cerebro)
  - RAGResidual: la memoria externa (el hipocampo)
  - ClasificadorPareto: qué guardar en memoria
  - ColaFinetune: cuándo proponer aprender de las interacciones
  - Skills: lector de archivos, ejecutor de código, etc.

Loop de razonamiento:
  1. Usuario envía mensaje/código
  2. Clasificador analiza el input → agrega a RAG si es L1+
  3. RAG recupera contexto relevante de interacciones previas
  4. Se construye el prompt con: [RAG ctx] + [historial] + [input]
  5. Modelo genera respuesta con Early Exit
  6. Si la respuesta contiene una acción ([LEER:...], [EJECUTAR:...])
     → skill correspondiente se invoca
     → resultado se vuelve a insertar como contexto
     → modelo genera respuesta final
  7. Respuesta se agrega al historial
  8. Se verifica si la cola de fine-tune está lista → proponer al usuario
"""

import sentencepiece as spm
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pampar.coder.v3 import PamparV3, PRESET_V3
from pampar.coder.v3.config import ConfigV3
from pampar.memoria.clasificador import ClasificadorPareto
from pampar.memoria.rag import RAGResidual
from pampar.memoria.cola_finetune import ColaFinetune
from pampar.skills.lector_archivos import LectorArchivos
from pampar.skills.ejecutar_codigo import EjecutorCodigo
from pampar.runtime.boot import BootProtocol


# =============================================================================
# PROMPT BUILDER
# =============================================================================

SYSTEM_PROMPT = """Sos PAMPAr, un asistente de programación local y offline especializado en Python.
Tenés acceso a la memoria de interacciones previas y podés ejecutar código cuando sea necesario.

Para leer un archivo: [LEER: ruta/al/archivo.py]
Para ejecutar código: [EJECUTAR:
codigo_python_aqui
]
Para ejecutar tests: [TESTS: ruta/tests/]

Respondé siempre en español. El código va siempre en inglés."""


# =============================================================================
# AGENTE
# =============================================================================

class Agente:
    """
    Orquestador principal del sistema PAMPAr.

    Args:
        checkpoint:       Path al checkpoint del modelo (.pt)
        tokenizer_path:   Path al tokenizer (.model de SentencePiece)
        config:           Configuración del modelo
        workspace_root:   Directorio raíz del proyecto del usuario
        memoria_dir:      Directorio para persistir la memoria
        device:           "cuda", "cpu" o "auto"
        max_historial:    Máximo de turnos de historial en el contexto
    """

    def __init__(
        self,
        checkpoint: str = "checkpoints/pampar_v3_best.pt",
        tokenizer_path: str = "data/tokenizer/pampar_48k.model",
        config: ConfigV3 = PRESET_V3,
        workspace_root: str = ".",
        memoria_dir: str = "memoria/data",
        device: str = "auto",
        max_historial: int = 10,
    ):
        self.config = config
        self.max_historial = max_historial

        # ── Dispositivo ──────────────────────────────────────────────────────
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # ── Tokenizer ────────────────────────────────────────────────────────
        self.tok = spm.SentencePieceProcessor()
        self.tok.Load(tokenizer_path)

        # ── Modelo ───────────────────────────────────────────────────────────
        self.modelo = PamparV3(config)
        self.modelo.registrar_tokenizer(self.tok)

        ckpt_path = Path(checkpoint)
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            # Compatibilidad: el checkpoint puede tener wrapper 'model'
            state_dict = state.get("model", state)
            self.modelo.load_state_dict(state_dict, strict=False)
            print(f"[Agente] Modelo cargado desde {checkpoint}")
        else:
            print(f"[Agente] Checkpoint no encontrado, usando pesos iniciales: {checkpoint}")

        self.modelo = self.modelo.to(self.device)
        self.modelo.eval()

        # ── Memoria ──────────────────────────────────────────────────────────
        self.clasificador = ClasificadorPareto()
        self.rag = RAGResidual(directorio=memoria_dir)
        self.cola_ft = ColaFinetune(
            directorio=memoria_dir,
            callback_proponer=self._on_cola_lista,
        )

        # ── Skills ───────────────────────────────────────────────────────────
        self.lector = LectorArchivos(workspace_root=workspace_root)
        self.ejecutor = EjecutorCodigo(cwd=workspace_root)
        # ── Boot Protocol ────────────────────────────────────────────────────
        self.boot = BootProtocol(workspace_root=workspace_root)
        self._scan_resultado = self.boot.ejecutar(self.rag)
        self._system_prompt = self.boot.generar_system_prompt()
        # ── Estado ───────────────────────────────────────────────────────────
        self._historial: List[Dict[str, str]] = []  # [{"role": "user/assistant", "text": "..."}]


        print(f"[Agente] Listo en {self.device} | RAG: {self.rag.stats()['total_entradas']} entradas")

    # ── Inferencia ────────────────────────────────────────────────────────────

    def responder(
        self,
        mensaje: str,
        max_tokens: int = 512,
        temperatura: float = 0.8,
    ) -> str:
        """
        Procesa un mensaje del usuario y genera una respuesta.

        Args:
            mensaje:    Input del usuario (código, pregunta, etc.)
            max_tokens: Máximo de tokens a generar
            temperatura: Control de creatividad (0.1=determinista, 1.0=creativo)
        Returns:
            Respuesta del modelo como string
        """
        # 1. Clasificar el input y agregarlo al RAG si es importante
        textos_existentes = [e.texto for e in self.rag._entradas]
        entrada = self.clasificador.clasificar(
            texto=mensaje,
            tipo="codigo" if self._parece_codigo(mensaje) else "dialogo",
            fragmentos_existentes=textos_existentes,
        )
        if entrada.nivel >= 1:
            self.rag.agregar(entrada)
            self.cola_ft.agregar(entrada)

        # 2. Recuperar contexto del RAG
        resultados_rag = self.rag.recuperar(mensaje, nivel_minimo=1)
        ctx_rag = self.rag.formatear_contexto(resultados_rag)

        # 3. Construir prompt completo
        prompt = self._construir_prompt(mensaje, ctx_rag)

        # 4. Tokenizar y generar
        ids = self.tok.Encode(prompt)
        if len(ids) > self.config.max_seq_len - max_tokens - 50:
            # Truncar prompt para dejar espacio a la respuesta
            ids = ids[-(self.config.max_seq_len - max_tokens - 50):]

        input_tensor = torch.tensor([ids], device=self.device)

        with torch.no_grad():
            output = self.modelo.generate(
                input_tensor,
                max_tokens=max_tokens,
                temperature=temperatura,
                top_k=50,
                top_p=0.95,
            )

        # 5. Decodificar solo los tokens nuevos
        nuevos_ids = output[0, len(ids):].tolist()
        respuesta = self.tok.Decode(nuevos_ids).strip()

        # 6. Procesar acciones si las hay
        respuesta = self._procesar_acciones(respuesta)

        # 7. Actualizar historial
        self._historial.append({"role": "user", "text": mensaje})
        self._historial.append({"role": "assistant", "text": respuesta})
        if len(self._historial) > self.max_historial * 2:
            self._historial = self._historial[-self.max_historial * 2:]

        # 8. Verificar cola de fine-tune
        propuesta = self._verificar_cola()
        if propuesta:
            respuesta += f"\n\n{propuesta}"

        return respuesta

    # ── Internos ──────────────────────────────────────────────────────────────

    def _construir_prompt(self, mensaje: str, ctx_rag: str) -> str:
        """Construye el prompt completo con system, RAG, historial y mensaje."""
        partes = [self._system_prompt]

        if ctx_rag:
            partes.append(ctx_rag)

        # Historial (últimos N turnos)
        for turno in self._historial[-(self.max_historial * 2):]:
            prefijo = "Usuario" if turno["role"] == "user" else "PAMPAr"
            partes.append(f"{prefijo}: {turno['text']}")

        partes.append(f"Usuario: {mensaje}")
        partes.append("PAMPAr:")

        return "\n\n".join(partes)

    def _procesar_acciones(self, respuesta: str) -> str:
        """
        Detecta y ejecuta acciones en la respuesta del modelo.

        Formatos reconocidos:
          [LEER: ruta]
          [EJECUTAR: ... ]
          [TESTS: ruta]
        """
        import re

        # [LEER: ruta]
        for match in re.finditer(r"\[LEER:\s*(.+?)\]", respuesta):
            ruta = match.group(1).strip()
            resultado = self.lector.execute(ruta=ruta)
            reemplazo = resultado.contenido if resultado.exito else f"[ERROR al leer: {resultado.error}]"
            respuesta = respuesta.replace(match.group(0), reemplazo)

        # [EJECUTAR: codigo ]
        for match in re.finditer(r"\[EJECUTAR:\s*\n?(.*?)\n?\]", respuesta, re.DOTALL):
            codigo = match.group(1).strip()
            resultado = self.ejecutor.execute(codigo=codigo)
            reemplazo = resultado.contenido if resultado.contenido else f"[ERROR al ejecutar: {resultado.error}]"
            respuesta = respuesta.replace(match.group(0), reemplazo)

        # [TESTS: ruta]
        for match in re.finditer(r"\[TESTS:\s*(.+?)\]", respuesta):
            ruta = match.group(1).strip()
            resultado = self.ejecutor.ejecutar_tests(ruta_test=ruta)
            reemplazo = resultado.contenido if resultado.contenido else f"[ERROR al correr tests: {resultado.error}]"
            respuesta = respuesta.replace(match.group(0), reemplazo)

        return respuesta

    def _parece_codigo(self, texto: str) -> bool:
        """Heurística rápida para detectar si el input es código."""
        import re
        indicadores = [
            r"\bdef\b", r"\bclass\b", r"\bimport\b", r"\bfor\b.*\bin\b",
            r":\s*$", r"^\s{4}", r"```", r"\(\)", r"=\s*\[",
        ]
        return any(re.search(p, texto, re.MULTILINE) for p in indicadores)

    def _on_cola_lista(self, n_ejemplos: int, stats: dict) -> bool:
        """
        Callback cuando la cola de fine-tune tiene suficientes ejemplos.

        Por ahora solo notifica — el usuario decide si aceptar.
        En futuro: integrar con UI/API para diálogo.
        """
        print(f"\n[Cola Fine-tune] {n_ejemplos} ejemplos listos. Stats: {stats}")
        return False  # No lanzar automáticamente — requiere confirmación del usuario

    def _verificar_cola(self) -> Optional[str]:
        """Retorna mensaje de propuesta si la cola está lista."""
        if len(self.cola_ft) >= self.cola_ft.min_ejemplos:
            stats = self.cola_ft.stats()
            if stats.get("listos") and stats["total"] % 50 == 0:
                # Proponer cada 50 ejemplos nuevos
                return self.cola_ft.proponer_usuario()
        return None

    # ── API de control ────────────────────────────────────────────────────────

    def aceptar_finetune(self) -> str:
        """El usuario acepta la propuesta de fine-tune."""
        exito = self.cola_ft.lanzar_finetune()
        if exito:
            return (
                "Fine-tune iniciado en background. "
                "Seguís pudiendo usarme mientras se entrena. "
                "Cuando termine, la memoria residual se liberará."
            )
        return "No pude iniciar el fine-tune. Verificá que el script de training esté disponible."

    def rechazar_finetune(self) -> str:
        """El usuario rechaza la propuesta — los datos se conservan en RAG."""
        return (
            "Entendido. Los patrones quedan en mi memoria como RAG. "
            "Podés aceptar el entrenamiento más adelante cuando quieras."
        )

    def stats(self) -> dict:
        """Estado actual del sistema."""
        return {
            "modelo": self.modelo.count_params(),
            "rag": self.rag.stats(),
            "cola_finetune": self.cola_ft.stats(),
            "historial_turnos": len(self._historial) // 2,
            "device": str(self.device),
        }

    def limpiar_historial(self) -> None:
        """Limpia el historial de conversación actual."""
        self._historial = []

    def describe(self) -> str:
        """Descripción completa del sistema."""
        return self.modelo.describe()
