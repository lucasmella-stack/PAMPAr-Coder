# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Fase 2: Self-Play — "Experimentación"

Como un programador que prueba su código y aprende de los errores.

Ciclo:
  1. El modelo GENERA código dado un prompt
  2. Se EJECUTA el código en un sandbox seguro
  3. Se evalúa el RESULTADO (compila, ejecuta, correcto)
  4. Se entrena con DPO: preferir soluciones correctas vs incorrectas

Esto es EXTREMADAMENTE eficiente en datos:
  - Cada intento genera un par (bueno, malo) para DPO
  - No necesita datasets masivos — genera sus propios datos
  - El reward es OBJETIVO: el código funciona o no
  - Puede correr en la PC local del usuario (no necesita GPU cloud)

Inspirado en:
  - AlphaCode (DeepMind): self-play para código
  - CodeRL (Salesforce): RL con ejecución
  - phi-1 (Microsoft): calidad sobre cantidad
"""

import ast
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# =============================================================================
# RESULTADO DE EJECUCIÓN
# =============================================================================

class TipoResultado(Enum):
    """Tipos de resultado de ejecución de código."""
    CORRECTO = "correcto"           # Ejecuta y da resultado esperado
    EJECUTA = "ejecuta"             # Ejecuta sin error pero sin verificar resultado
    ERROR_SINTAXIS = "error_sintaxis"     # No parsea
    ERROR_RUNTIME = "error_runtime"       # Error en ejecución
    TIMEOUT = "timeout"             # Excede tiempo límite
    ERROR_SEGURIDAD = "error_seguridad"   # Intento de código peligroso


@dataclass
class ResultadoEjecucion:
    """Resultado completo de ejecutar un fragmento de código."""
    tipo: TipoResultado
    codigo: str
    stdout: str = ""
    stderr: str = ""
    error_msg: str = ""
    tiempo_ms: float = 0.0
    resultado_esperado: Optional[str] = None
    resultado_obtenido: Optional[str] = None
    
    @property
    def exito(self) -> bool:
        return self.tipo in (TipoResultado.CORRECTO, TipoResultado.EJECUTA)
    
    @property
    def reward(self) -> float:
        """Reward escalar para entrenamiento RL/DPO."""
        rewards = {
            TipoResultado.CORRECTO: 1.0,
            TipoResultado.EJECUTA: 0.5,
            TipoResultado.ERROR_RUNTIME: -0.3,
            TipoResultado.ERROR_SINTAXIS: -0.5,
            TipoResultado.TIMEOUT: -0.1,
            TipoResultado.ERROR_SEGURIDAD: -1.0,
        }
        return rewards.get(self.tipo, 0.0)


# =============================================================================
# SANDBOX DE EJECUCIÓN
# =============================================================================

# Módulos prohibidos (seguridad)
MODULOS_PROHIBIDOS = {
    "os", "sys", "subprocess", "shutil", "pathlib",
    "socket", "http", "urllib", "requests",
    "ctypes", "pickle", "shelve",
    "importlib", "runpy", "__import__",
    "eval", "exec", "compile",
    "open",  # Solo se usa como built-in
}

# Funciones prohibidas en builtins
BUILTINS_PROHIBIDOS = {
    "eval", "exec", "compile", "__import__",
    "open", "input", "breakpoint",
    "exit", "quit",
}


def verificar_seguridad(codigo: str) -> Optional[str]:
    """
    Verifica que el código no contenga operaciones peligrosas.
    
    Returns:
        None si es seguro, mensaje de error si es peligroso
    """
    # Intentar parsear AST
    try:
        tree = ast.parse(codigo)
    except SyntaxError:
        return None  # Lo manejará el ejecutor
    
    for node in ast.walk(tree):
        # Importaciones peligrosas
        if isinstance(node, ast.Import):
            for alias in node.names:
                modulo = alias.name.split(".")[0]
                if modulo in MODULOS_PROHIBIDOS:
                    return f"módulo prohibido: {modulo}"
        
        if isinstance(node, ast.ImportFrom):
            if node.module:
                modulo = node.module.split(".")[0]
                if modulo in MODULOS_PROHIBIDOS:
                    return f"módulo prohibido: {modulo}"
        
        # Funciones peligrosas
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in BUILTINS_PROHIBIDOS:
                    return f"función prohibida: {node.func.id}"
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in BUILTINS_PROHIBIDOS:
                    return f"método prohibido: {node.func.attr}"
    
    return None


def _ejecutar_en_proceso(codigo: str, timeout: float) -> Dict:
    """
    Ejecuta código en un subproceso aislado.
    Retorna dict con resultado.
    """
    # Crear archivo temporal
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(codigo)
        temp_path = f.name
    
    try:
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={
                "PATH": os.environ.get("PATH", ""),
                "PYTHONPATH": "",
                "HOME": tempfile.gettempdir(),
            },
        )
        
        return {
            "returncode": result.returncode,
            "stdout": result.stdout[:4096],  # Limitar salida
            "stderr": result.stderr[:4096],
        }
    except subprocess.TimeoutExpired:
        return {"returncode": -1, "stdout": "", "stderr": "TIMEOUT"}
    except Exception as e:
        return {"returncode": -2, "stdout": "", "stderr": str(e)}
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


def ejecutar_codigo_seguro(
    codigo: str,
    timeout: float = 5.0,
    resultado_esperado: Optional[str] = None,
) -> ResultadoEjecucion:
    """
    Ejecuta código Python de forma segura en un sandbox.
    
    El código se ejecuta en un subproceso separado con:
    - Timeout de N segundos
    - Variables de entorno limpias
    - Sin acceso a módulos peligrosos (verificación AST)
    
    Args:
        codigo: Código Python a ejecutar
        timeout: Tiempo máximo en segundos
        resultado_esperado: Output esperado para verificar corrección
    
    Returns:
        ResultadoEjecucion con tipo, stdout, stderr, reward, etc.
    """
    import time
    
    # 1. Verificar seguridad
    error_seg = verificar_seguridad(codigo)
    if error_seg:
        return ResultadoEjecucion(
            tipo=TipoResultado.ERROR_SEGURIDAD,
            codigo=codigo,
            error_msg=error_seg,
        )
    
    # 2. Verificar sintaxis
    try:
        ast.parse(codigo)
    except SyntaxError as e:
        return ResultadoEjecucion(
            tipo=TipoResultado.ERROR_SINTAXIS,
            codigo=codigo,
            error_msg=f"SyntaxError: {e.msg} (línea {e.lineno})",
        )
    
    # 3. Ejecutar en subproceso
    t_start = time.perf_counter()
    result = _ejecutar_en_proceso(codigo, timeout)
    t_elapsed = (time.perf_counter() - t_start) * 1000  # ms
    
    # 4. Interpretar resultado
    if result["stderr"] == "TIMEOUT":
        return ResultadoEjecucion(
            tipo=TipoResultado.TIMEOUT,
            codigo=codigo,
            tiempo_ms=t_elapsed,
        )
    
    if result["returncode"] != 0:
        return ResultadoEjecucion(
            tipo=TipoResultado.ERROR_RUNTIME,
            codigo=codigo,
            stdout=result["stdout"],
            stderr=result["stderr"],
            error_msg=result["stderr"].strip().split("\n")[-1] if result["stderr"] else "",
            tiempo_ms=t_elapsed,
        )
    
    # 5. Verificar resultado si hay expected
    stdout = result["stdout"].strip()
    if resultado_esperado is not None:
        esperado = resultado_esperado.strip()
        if stdout == esperado:
            tipo = TipoResultado.CORRECTO
        else:
            tipo = TipoResultado.EJECUTA  # Ejecuta pero resultado incorrecto
        
        return ResultadoEjecucion(
            tipo=tipo,
            codigo=codigo,
            stdout=stdout,
            tiempo_ms=t_elapsed,
            resultado_esperado=esperado,
            resultado_obtenido=stdout,
        )
    
    return ResultadoEjecucion(
        tipo=TipoResultado.EJECUTA,
        codigo=codigo,
        stdout=stdout,
        tiempo_ms=t_elapsed,
    )


# =============================================================================
# PROMPTS PARA SELF-PLAY
# =============================================================================

PROMPTS_POR_NIVEL = {
    1: [  # BASICO
        "# Programa que imprime 'Hola Mundo'\n",
        "# Calcula la suma de dos números\na = 5\nb = 3\n",
        "# Convierte temperatura de Celsius a Fahrenheit\ncelsius = 25\n",
        "# Crea una lista de números del 1 al 5\n",
        "# Concatena dos strings\nnombre = 'PAMPAr'\nversion = 'v2'\n",
    ],
    2: [  # CONTROL
        "# Determina si un número es par o impar\nn = 7\n",
        "# Encuentra el mayor de tres números\na, b, c = 3, 7, 5\n",
        "# Imprime los números del 1 al 10\n",
        "# Cuenta cuántos números pares hay del 1 al 20\n",
        "# Calcula el factorial de 5 usando un loop\nn = 5\n",
    ],
    3: [  # FUNCIONES
        "# Define una función que calcula el factorial\ndef factorial(n):\n",
        "# Función que verifica si un número es primo\ndef es_primo(n):\n",
        "# Función que invierte un string\ndef invertir(texto):\n",
        "# Función que calcula el máximo común divisor\ndef mcd(a, b):\n",
        "# Función que genera números de Fibonacci\ndef fibonacci(n):\n",
    ],
    4: [  # CLASES
        "# Clase que representa un punto en 2D\nclass Punto:\n",
        "# Clase para una lista enlazada\nclass Nodo:\n",
        "# Clase para una calculadora básica\nclass Calculadora:\n",
        "# Clase para un stack (pila)\nclass Stack:\n",
        "# Clase para representar un estudiante\nclass Estudiante:\n",
    ],
    5: [  # ALGORITMOS
        "# Implementa bubble sort\ndef bubble_sort(arr):\n",
        "# Implementa búsqueda binaria\ndef busqueda_binaria(arr, target):\n",
        "# Implementa merge sort\ndef merge_sort(arr):\n",
        "# Cuenta las ocurrencias de cada carácter\ndef contar_chars(texto):\n",
        "# Verifica si un string es palíndromo\ndef es_palindromo(texto):\n",
    ],
    6: [  # PATRONES
        "# Implementa el patrón Singleton\nclass Singleton:\n",
        "# Implementa un decorador de cache\ndef cache(func):\n",
        "# Implementa un context manager\nclass Timer:\n",
        "# Implementa un iterador personalizado\nclass Rango:\n",
        "# Implementa el patrón Observer\nclass EventEmitter:\n",
    ],
}

# =============================================================================
# MOTOR DE SELF-PLAY
# =============================================================================

class SelfPlayEngine:
    """
    Motor de self-play: el modelo genera, ejecuta y aprende.
    
    Como un programador junior que practica:
    1. Lee el prompt (problema)
    2. Escribe código (genera)
    3. Lo ejecuta (prueba)
    4. Aprende del resultado (DPO/REINFORCE)
    
    Genera pares (preferred, rejected) para DPO:
    - preferred = código que ejecuta correctamente
    - rejected = código con errores
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        nivel: int = 1,
        n_intentos: int = 4,       # Intentos por prompt
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        timeout_ejecucion: float = 5.0,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.nivel = nivel
        self.n_intentos = n_intentos
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.timeout = timeout_ejecucion
        self.device = device
        
        # Estadísticas
        self.stats = {
            "total_generados": 0,
            "correctos": 0,
            "ejecutan": 0,
            "errores_sintaxis": 0,
            "errores_runtime": 0,
            "timeouts": 0,
        }
        
        # Buffer de pares DPO
        self.pares_dpo: List[Dict] = []
    
    def generar_codigo(self, prompt: str) -> str:
        """Genera código usando el modelo."""
        tokens = self.tokenizer.Encode(prompt)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
            )
        
        # Decodificar solo la parte generada
        generated = output[0, len(tokens):].tolist()
        texto = self.tokenizer.Decode(generated)
        
        return prompt + texto
    
    def jugar_ronda(self) -> List[ResultadoEjecucion]:
        """
        Ejecuta una ronda de self-play:
        1. Elige un prompt del nivel actual
        2. Genera N intentos
        3. Ejecuta cada uno
        4. Crea pares DPO de los resultados
        
        Returns:
            Lista de resultados de ejecución
        """
        import random
        
        prompts = PROMPTS_POR_NIVEL.get(self.nivel, PROMPTS_POR_NIVEL[1])
        prompt = random.choice(prompts)
        
        resultados = []
        
        for _ in range(self.n_intentos):
            # Generar
            codigo = self.generar_codigo(prompt)
            self.stats["total_generados"] += 1
            
            # Ejecutar
            resultado = ejecutar_codigo_seguro(
                codigo,
                timeout=self.timeout,
            )
            resultados.append(resultado)
            
            # Stats
            if resultado.tipo == TipoResultado.CORRECTO:
                self.stats["correctos"] += 1
            elif resultado.tipo == TipoResultado.EJECUTA:
                self.stats["ejecutan"] += 1
            elif resultado.tipo == TipoResultado.ERROR_SINTAXIS:
                self.stats["errores_sintaxis"] += 1
            elif resultado.tipo == TipoResultado.ERROR_RUNTIME:
                self.stats["errores_runtime"] += 1
            elif resultado.tipo == TipoResultado.TIMEOUT:
                self.stats["timeouts"] += 1
        
        # Crear pares DPO
        self._crear_pares_dpo(prompt, resultados)
        
        return resultados
    
    def _crear_pares_dpo(self, prompt: str, resultados: List[ResultadoEjecucion]):
        """
        Crea pares (preferred, rejected) para DPO.
        
        preferred = mejor resultado (ejecuta correcto)
        rejected = peor resultado (error)
        """
        # Separar buenos y malos
        buenos = [r for r in resultados if r.exito]
        malos = [r for r in resultados if not r.exito]
        
        if not buenos or not malos:
            return  # Necesitamos al menos uno de cada
        
        # Ordenar por reward
        buenos.sort(key=lambda r: r.reward, reverse=True)
        malos.sort(key=lambda r: r.reward)
        
        # Crear par
        self.pares_dpo.append({
            "prompt": prompt,
            "preferred": buenos[0].codigo,
            "rejected": malos[0].codigo,
            "preferred_reward": buenos[0].reward,
            "rejected_reward": malos[0].reward,
        })
    
    def obtener_lote_dpo(
        self,
        batch_size: int = 4,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Obtiene un batch de pares DPO tokenizados para entrenamiento.
        
        Returns:
            Dict con 'preferred_ids', 'rejected_ids', 'prompt_len'
            o None si no hay suficientes pares
        """
        if len(self.pares_dpo) < batch_size:
            return None
        
        # Tomar batch
        batch = self.pares_dpo[:batch_size]
        self.pares_dpo = self.pares_dpo[batch_size:]
        
        preferred_ids = []
        rejected_ids = []
        prompt_lens = []
        
        for par in batch:
            # Tokenizar
            pref_tokens = self.tokenizer.Encode(par["preferred"])
            rej_tokens = self.tokenizer.Encode(par["rejected"])
            prompt_tokens = self.tokenizer.Encode(par["prompt"])
            
            preferred_ids.append(pref_tokens)
            rejected_ids.append(rej_tokens)
            prompt_lens.append(len(prompt_tokens))
        
        # Pad to same length
        max_len = max(
            max(len(p) for p in preferred_ids),
            max(len(r) for r in rejected_ids),
        )
        
        def pad_sequence(seqs, max_l):
            return torch.tensor([
                s + [0] * (max_l - len(s)) for s in seqs
            ], dtype=torch.long, device=self.device)
        
        return {
            "preferred_ids": pad_sequence(preferred_ids, max_len),
            "rejected_ids": pad_sequence(rejected_ids, max_len),
            "prompt_lens": torch.tensor(prompt_lens, dtype=torch.long),
        }
    
    def get_stats_str(self) -> str:
        """Estadísticas formateadas."""
        total = max(self.stats["total_generados"], 1)
        return (
            f"Self-Play Stats:\n"
            f"  Total: {total}\n"
            f"  Correctos: {self.stats['correctos']} ({100*self.stats['correctos']/total:.1f}%)\n"
            f"  Ejecutan: {self.stats['ejecutan']} ({100*self.stats['ejecutan']/total:.1f}%)\n"
            f"  Errores sintaxis: {self.stats['errores_sintaxis']}\n"
            f"  Errores runtime: {self.stats['errores_runtime']}\n"
            f"  Timeouts: {self.stats['timeouts']}\n"
            f"  Pares DPO acumulados: {len(self.pares_dpo)}\n"
        )


# =============================================================================
# DPO LOSS
# =============================================================================

def dpo_loss(
    model,
    preferred_ids: torch.Tensor,   # [B, L]
    rejected_ids: torch.Tensor,    # [B, L]
    prompt_lens: torch.Tensor,     # [B] length of prompt part
    beta: float = 0.1,            # DPO temperature
    ref_model=None,                # Reference model (None = implicit)
) -> torch.Tensor:
    """
    Direct Preference Optimization loss.
    
    Enseña al modelo a preferir código que ejecuta correctamente
    sobre código con errores.
    
    L_DPO = -log σ(β * (log π(y_w|x) - log π(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x)))
    
    Sin ref_model (implicit DPO):
    L_DPO = -log σ(β * (log π(y_w|x) - log π(y_l|x)))
    
    Args:
        model: The policy model
        preferred_ids: Token IDs of preferred (correct) code
        rejected_ids: Token IDs of rejected (incorrect) code
        prompt_lens: Length of the prompt portion
        beta: Temperature parameter
        ref_model: Reference model for KL constraint (None = implicit)
    
    Returns:
        Scalar loss
    """
    B = preferred_ids.shape[0]
    
    # Forward pass for preferred
    pref_logits, _, _ = model(preferred_ids)
    pref_logprobs = F.log_softmax(pref_logits, dim=-1)
    
    # Forward pass for rejected
    rej_logits, _, _ = model(rejected_ids)
    rej_logprobs = F.log_softmax(rej_logits, dim=-1)
    
    # Gather log probs for actual tokens (shifted by 1)
    def gather_logprobs(logprobs, ids):
        """Get log prob of each token given previous tokens."""
        # logprobs: [B, L, V], ids: [B, L]
        # Shift: predict token t from position t-1
        shifted_logprobs = logprobs[:, :-1, :]  # [B, L-1, V]
        target_ids = ids[:, 1:]                  # [B, L-1]
        
        # Gather
        gathered = shifted_logprobs.gather(
            2, target_ids.unsqueeze(-1)
        ).squeeze(-1)  # [B, L-1]
        
        return gathered
    
    pref_token_logprobs = gather_logprobs(pref_logprobs, preferred_ids)
    rej_token_logprobs = gather_logprobs(rej_logprobs, rejected_ids)
    
    # Mask out prompt tokens AND padding (only score generation, not pad)
    def mask_prompt(logprobs, prompt_lens, ids):
        total_len = ids.shape[1]
        pos = torch.arange(total_len - 1, device=logprobs.device).unsqueeze(0)
        prompt_mask = pos >= prompt_lens.unsqueeze(1)  # [B, L-1]
        # Also mask padding tokens (token 0)
        pad_mask = ids[:, 1:] != 0  # [B, L-1]
        mask = prompt_mask & pad_mask
        return (logprobs * mask.float()).sum(dim=1)  # [B]
    
    pref_sum = mask_prompt(pref_token_logprobs, prompt_lens, preferred_ids)
    rej_sum = mask_prompt(rej_token_logprobs, prompt_lens, rejected_ids)
    
    # Reference model (if provided)
    if ref_model is not None:
        with torch.no_grad():
            ref_pref_logits, _, _ = ref_model(preferred_ids)
            ref_rej_logits, _, _ = ref_model(rejected_ids)
            
            ref_pref_logprobs = F.log_softmax(ref_pref_logits, dim=-1)
            ref_rej_logprobs = F.log_softmax(ref_rej_logits, dim=-1)
            
            ref_pref_sum = mask_prompt(
                gather_logprobs(ref_pref_logprobs, preferred_ids),
                prompt_lens, preferred_ids
            )
            ref_rej_sum = mask_prompt(
                gather_logprobs(ref_rej_logprobs, rejected_ids),
                prompt_lens, rejected_ids
            )
        
        # Full DPO with reference
        logits_diff = beta * (
            (pref_sum - ref_pref_sum) - (rej_sum - ref_rej_sum)
        )
    else:
        # Implicit DPO (no reference model)
        logits_diff = beta * (pref_sum - rej_sum)
    
    # DPO loss: -log sigmoid(logits_diff)
    loss = -F.logsigmoid(logits_diff).mean()
    
    return loss
