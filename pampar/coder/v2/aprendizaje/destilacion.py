# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Fase 3: Destilación de Conocimiento — "Filosofar con un Maestro"

Knowledge Distillation: un modelo grande (profesor) enseña a PAMPAr (alumno).

Como un estudiante que aprende de un profesor experto:
  1. El profesor (GPT-4, Claude, Qwen-72B) resuelve problemas
  2. PAMPAr aprende no solo la RESPUESTA sino el RAZONAMIENTO
  3. Soft targets: aprende la distribución de probabilidad del profesor
  4. Duro + blando: combina CE clásico con KL divergence del profesor

3 modos de destilación:

  A) OFFLINE: Generar dataset con API del profesor → entrenar después
     - Más barato (una sola llamada por ejemplo)
     - Se puede hacer por lotes
     - Funciona con cualquier API (OpenAI, Anthropic, local)

  B) ONLINE: Profesor y alumno corren en paralelo
     - Soft targets en tiempo real
     - Más caro pero mejor calidad
     - Necesita GPU para el profesor local

  C) CHAIN-OF-THOUGHT: El profesor genera razonamiento paso a paso
     - PAMPAr aprende a RAZONAR, no solo a responder
     - El razonamiento se mapea a territorios cerebrales
     - Cada paso del CoT activa diferentes zonas de Brodmann

Innovación PAMPAr: La destilación se hace consciente de los territorios.
El profesor genera, PAMPAr analiza QUÉ TERRITORIO se usó para cada parte
del razonamiento. Esto alimenta el Tálamo con señales de routing correctas.

Inspirado en:
  - Hinton et al. (2015): "Distilling the Knowledge in a Neural Network"
  - phi-1 (Microsoft): "Textbooks Are All You Need"
  - Orca (Microsoft): destilación con reasoning traces
"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# CONFIGURACIÓN DE PROFESORES
# =============================================================================

@dataclass
class ConfigProfesor:
    """Configuración de un modelo profesor para destilación."""
    nombre: str                     # Nombre identificativo
    tipo: str                       # "openai", "anthropic", "local", "ollama"
    modelo: str                     # "gpt-4o-mini", "claude-3-haiku", "qwen2.5-coder:7b"
    api_key: str = ""               # API key (vacío = variable de entorno)
    base_url: str = ""              # URL base del API (para local/ollama)
    max_tokens: int = 2048          # Máximo tokens por respuesta
    temperature: float = 0.3        # Baja para respuestas consistentes
    costo_por_1k_input: float = 0.0  # Costo estimado USD por 1K tokens input
    costo_por_1k_output: float = 0.0 # Costo estimado USD por 1K tokens output


# Profesores pre-configurados
PROFESORES = {
    # APIs de pago (calidad alta, costo bajo con mini/haiku)
    "gpt4o-mini": ConfigProfesor(
        nombre="GPT-4o Mini",
        tipo="openai",
        modelo="gpt-4o-mini",
        costo_por_1k_input=0.00015,
        costo_por_1k_output=0.0006,
    ),
    "claude-haiku": ConfigProfesor(
        nombre="Claude 3.5 Haiku",
        tipo="anthropic",
        modelo="claude-3-5-haiku-20241022",
        costo_por_1k_input=0.0008,
        costo_por_1k_output=0.004,
    ),
    # Modelos locales (gratis, tu PC)
    "qwen-7b": ConfigProfesor(
        nombre="Qwen2.5-Coder 7B",
        tipo="ollama",
        modelo="qwen2.5-coder:7b",
        base_url="http://localhost:11434",
    ),
    "codellama-7b": ConfigProfesor(
        nombre="CodeLlama 7B",
        tipo="ollama",
        modelo="codellama:7b",
        base_url="http://localhost:11434",
    ),
    "deepseek-7b": ConfigProfesor(
        nombre="DeepSeek Coder V2 Lite",
        tipo="ollama",
        modelo="deepseek-coder-v2:16b",
        base_url="http://localhost:11434",
    ),
}


# =============================================================================
# CLIENTE UNIVERSAL DE PROFESOR
# =============================================================================

class ClienteProfesor:
    """
    Cliente universal para llamar a modelos profesores.
    Soporta OpenAI, Anthropic, Ollama y cualquier API compatible.
    """

    def __init__(self, config: ConfigProfesor):
        self.config = config
        self.costo_acumulado = 0.0
        self.llamadas = 0
        self.tokens_input = 0
        self.tokens_output = 0

    def generar(
        self,
        prompt: str,
        system: str = "Eres un experto programador. Responde con código Python limpio y bien documentado.",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Genera respuesta del profesor.

        Args:
            prompt: El problema/pregunta
            system: System prompt
            temperature: Override de temperatura
            max_tokens: Override de max tokens

        Returns:
            Respuesta del profesor como string
        """
        temp = temperature or self.config.temperature
        max_tok = max_tokens or self.config.max_tokens

        if self.config.tipo == "openai":
            return self._llamar_openai(prompt, system, temp, max_tok)
        elif self.config.tipo == "anthropic":
            return self._llamar_anthropic(prompt, system, temp, max_tok)
        elif self.config.tipo == "ollama":
            return self._llamar_ollama(prompt, system, temp, max_tok)
        elif self.config.tipo == "local":
            return self._llamar_local(prompt, system, temp, max_tok)
        else:
            raise ValueError(f"Tipo de profesor no soportado: {self.config.tipo}")

    def _llamar_openai(self, prompt, system, temp, max_tok) -> str:
        """Llama a API OpenAI-compatible."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")

        api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY", "")
        base_url = self.config.base_url or None

        client = OpenAI(api_key=api_key, base_url=base_url)

        response = client.chat.completions.create(
            model=self.config.modelo,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=temp,
            max_tokens=max_tok,
        )

        # Tracking
        self.llamadas += 1
        usage = response.usage
        if usage:
            self.tokens_input += usage.prompt_tokens
            self.tokens_output += usage.completion_tokens
            self.costo_acumulado += (
                usage.prompt_tokens / 1000 * self.config.costo_por_1k_input +
                usage.completion_tokens / 1000 * self.config.costo_por_1k_output
            )

        return response.choices[0].message.content

    def _llamar_anthropic(self, prompt, system, temp, max_tok) -> str:
        """Llama a API Anthropic."""
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("pip install anthropic")

        api_key = self.config.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        client = Anthropic(api_key=api_key)

        response = client.messages.create(
            model=self.config.modelo,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=max_tok,
        )

        self.llamadas += 1
        self.tokens_input += response.usage.input_tokens
        self.tokens_output += response.usage.output_tokens
        self.costo_acumulado += (
            response.usage.input_tokens / 1000 * self.config.costo_por_1k_input +
            response.usage.output_tokens / 1000 * self.config.costo_por_1k_output
        )

        return response.content[0].text

    def _llamar_ollama(self, prompt, system, temp, max_tok) -> str:
        """Llama a Ollama (modelo local)."""
        import urllib.request
        import json as _json

        url = f"{self.config.base_url}/api/chat"
        payload = _json.dumps({
            "model": self.config.modelo,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {
                "temperature": temp,
                "num_predict": max_tok,
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=120) as resp:
            data = _json.loads(resp.read().decode("utf-8"))

        self.llamadas += 1
        return data["message"]["content"]

    def _llamar_local(self, prompt, system, temp, max_tok) -> str:
        """Llama a un servidor local OpenAI-compatible."""
        return self._llamar_openai(prompt, system, temp, max_tok)

    def get_stats(self) -> Dict:
        """Estadísticas de uso."""
        return {
            "profesor": self.config.nombre,
            "llamadas": self.llamadas,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "costo_usd": round(self.costo_acumulado, 4),
        }


# =============================================================================
# TEMPLATES DE PROMPTS PARA DESTILACIÓN
# =============================================================================

SYSTEM_PROMPT_CODIGO = """Eres un profesor experto de programación Python.
Responde SOLO con código Python limpio y bien documentado.
Incluye docstrings en español, type hints, y manejo de errores.
No expliques fuera del código — usa comentarios dentro del código."""

SYSTEM_PROMPT_COT = """Eres un profesor experto de programación.
Antes de escribir código, RAZONA paso a paso:
1. Analiza el problema
2. Identifica los conceptos necesarios (variables, control, funciones, clases, algoritmos)
3. Planifica la solución
4. Escribe el código con explicaciones

Formatea así:
## Razonamiento
[tu razonamiento paso a paso]

## Código
```python
[tu código]
```"""

PROMPTS_DESTILACION = {
    # Nivel 1: Básico
    1: [
        "Escribe un programa que calcule el área de un círculo dado el radio.",
        "Crea un programa que convierta grados Celsius a Fahrenheit.",
        "Escribe un programa que intercambie los valores de dos variables.",
        "Crea un programa que calcule el IMC (índice de masa corporal).",
        "Escribe un programa que determine si un año es bisiesto.",
    ],
    # Nivel 2: Control
    2: [
        "Escribe un programa que imprima los números primos del 1 al 100.",
        "Crea un programa que genere la secuencia de Fibonacci hasta N términos.",
        "Escribe un programa que cuente vocales y consonantes en un texto.",
        "Crea un programa que simule una calculadora básica (+, -, *, /).",
        "Escribe un programa que determine si un número es palíndromo.",
    ],
    # Nivel 3: Funciones
    3: [
        "Implementa una función recursiva para calcular la potencia de un número.",
        "Crea funciones para cifrar y descifrar texto con cifrado César.",
        "Implementa una función decoradora que mida el tiempo de ejecución.",
        "Crea una función generadora que produzca números primos infinitamente.",
        "Implementa funciones para validar emails, URLs y contraseñas.",
    ],
    # Nivel 4: Clases
    4: [
        "Implementa una clase BankAccount con depósito, retiro e historial.",
        "Crea un sistema de herencia: Shape -> Circle, Rectangle, Triangle con áreas.",
        "Implementa una clase LinkedList con append, prepend, delete, search, y __str__.",
        "Crea un sistema de inventario con clases Product, Inventory y Cart.",
        "Implementa una clase Matrix con suma, multiplicación y transposición.",
    ],
    # Nivel 5: Algoritmos
    5: [
        "Implementa quicksort con partición de Lomuto y analiza su complejidad.",
        "Crea un árbol binario de búsqueda con inserción, búsqueda y recorridos.",
        "Implementa el algoritmo de Dijkstra para encontrar el camino más corto.",
        "Crea un sistema de caché LRU (Least Recently Used) con OrderedDict.",
        "Implementa un trie para autocompletar palabras con prefijos.",
    ],
    # Nivel 6: Patrones avanzados
    6: [
        "Implementa un ORM simple para SQLite con decoradores y metaclases.",
        "Crea un framework de testing minimalista con fixtures y assertions.",
        "Implementa un sistema de plugins con carga dinámica y registro.",
        "Crea un event loop asíncrono simplificado (mini asyncio).",
        "Implementa un compilador de expresiones matemáticas (tokenizer + parser + eval).",
    ],
}


# =============================================================================
# GENERADOR DE DATASET DE DESTILACIÓN
# =============================================================================

class GeneradorDestilacion:
    """
    Genera datasets de destilación usando un modelo profesor.

    Modos:
    1. Solo código: el profesor genera código limpio
    2. Chain-of-Thought: el profesor razona paso a paso
    3. Corrección: el profesor corrige código incorrecto generado por PAMPAr

    El output se guarda como JSONL para entrenar después.
    """

    def __init__(
        self,
        profesor: ClienteProfesor,
        output_dir: str = "data/distillation/teacher",
        modo: str = "cot",  # "codigo", "cot", "correccion"
    ):
        self.profesor = profesor
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.modo = modo

        self.generados = 0
        self.errores = 0

    def generar(self, nivel: int = 1) -> Optional[Dict]:
        """
        Genera un solo ejemplo de destilación para el nivel dado.

        Returns:
            Dict con instruction, output, nivel, etc. o None si falla.
        """
        import random
        system = SYSTEM_PROMPT_COT if self.modo == "cot" else SYSTEM_PROMPT_CODIGO
        prompts = PROMPTS_DESTILACION.get(nivel, PROMPTS_DESTILACION[1])
        prompt = random.choice(prompts)

        try:
            respuesta = self.profesor.generar(prompt, system=system)
            ejemplo = self._parsear_respuesta(prompt, respuesta, nivel)
            self.generados += 1
            return ejemplo
        except Exception as e:
            self.errores += 1
            return None

    def generar_dataset(
        self,
        niveles: Optional[List[int]] = None,
        max_por_nivel: int = 100,
        prompts_extra: Optional[List[str]] = None,
        delay: float = 0.5,  # Delay entre llamadas (rate limit)
    ) -> Path:
        """
        Genera dataset completo de destilación.

        Args:
            niveles: Niveles a generar (None = todos)
            max_por_nivel: Máximo ejemplos por nivel
            prompts_extra: Prompts adicionales
            delay: Segundos entre llamadas API

        Returns:
            Path al archivo JSONL generado
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"destilacion_{self.modo}_{timestamp}.jsonl"

        niveles = niveles or list(range(1, 7))
        system = SYSTEM_PROMPT_COT if self.modo == "cot" else SYSTEM_PROMPT_CODIGO

        print(f"  Generando dataset de destilación ({self.modo})...")
        print(f"  Profesor: {self.profesor.config.nombre}")
        print(f"  Niveles: {niveles}")
        print(f"  Output: {output_file}")

        with open(output_file, "w", encoding="utf-8") as f:
            for nivel in niveles:
                prompts = PROMPTS_DESTILACION.get(nivel, [])[:max_por_nivel]

                if prompts_extra and nivel == niveles[-1]:
                    prompts.extend(prompts_extra[:max_por_nivel - len(prompts)])

                print(f"\n  Nivel {nivel}: {len(prompts)} prompts")

                for i, prompt in enumerate(prompts):
                    try:
                        respuesta = self.profesor.generar(prompt, system=system)

                        # Parsear respuesta
                        ejemplo = self._parsear_respuesta(prompt, respuesta, nivel)
                        f.write(json.dumps(ejemplo, ensure_ascii=False) + "\n")
                        self.generados += 1

                        if (i + 1) % 5 == 0:
                            stats = self.profesor.get_stats()
                            print(
                                f"    [{i+1}/{len(prompts)}] "
                                f"costo=${stats['costo_usd']:.4f}"
                            )

                        time.sleep(delay)

                    except Exception as e:
                        self.errores += 1
                        print(f"    ERROR en prompt {i}: {e}")

        stats = self.profesor.get_stats()
        print(f"\n  Generados: {self.generados}, Errores: {self.errores}")
        print(f"  Costo total: ${stats['costo_usd']:.4f}")
        print(f"  Archivo: {output_file}")

        return output_file

    def _parsear_respuesta(
        self, prompt: str, respuesta: str, nivel: int,
    ) -> Dict:
        """Parsea la respuesta del profesor en formato de entrenamiento."""
        ejemplo = {
            "instruction": prompt,
            "output": respuesta,
            "nivel": nivel,
            "profesor": self.profesor.config.nombre,
            "modo": self.modo,
        }

        if self.modo == "cot":
            # Extraer razonamiento y código por separado
            razonamiento = ""
            codigo = ""

            # Buscar sección de razonamiento
            match_razon = re.search(
                r"##\s*Razonamiento\s*\n(.*?)(?=##\s*Código|```)",
                respuesta, re.DOTALL,
            )
            if match_razon:
                razonamiento = match_razon.group(1).strip()

            # Buscar bloques de código
            code_blocks = re.findall(r"```python\s*\n(.*?)```", respuesta, re.DOTALL)
            if code_blocks:
                codigo = "\n\n".join(code_blocks)

            ejemplo["razonamiento"] = razonamiento
            ejemplo["codigo"] = codigo

            # Crear texto completo para training
            ejemplo["text"] = (
                f"# Instrucción: {prompt}\n"
                f"# Razonamiento:\n"
                + "\n".join(f"# {line}" for line in razonamiento.split("\n") if line.strip())
                + f"\n\n{codigo}"
            )
        else:
            # Solo código
            code_blocks = re.findall(r"```python\s*\n(.*?)```", respuesta, re.DOTALL)
            if code_blocks:
                ejemplo["text"] = "\n\n".join(code_blocks)
            else:
                ejemplo["text"] = respuesta

        return ejemplo

    def generar_correccion(
        self,
        modelo_alumno,
        tokenizer,
        device: str = "cuda",
        n_ejemplos: int = 100,
        delay: float = 0.5,
    ) -> Path:
        """
        Modo corrección: PAMPAr genera código → profesor lo corrige.

        Esto es particularmente potente porque:
        1. El alumno intenta resolver el problema
        2. El profesor corrige y explica los errores
        3. Se crean pares (incorrecto, correcto) → DPO
        4. El alumno aprende exactamente donde falla

        Returns:
            Path al archivo JSONL con pares corrección
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"correcciones_{timestamp}.jsonl"

        print(f"  Generando correcciones (alumno genera → profesor corrige)...")

        with open(output_file, "w", encoding="utf-8") as f:
            n = 0
            for nivel, prompts in PROMPTS_DESTILACION.items():
                for prompt in prompts:
                    if n >= n_ejemplos:
                        break

                    try:
                        # 1. Alumno genera
                        tokens = tokenizer.Encode(f"# {prompt}\n")
                        input_ids = torch.tensor([tokens], device=device)

                        modelo_alumno.eval()
                        with torch.no_grad():
                            output = modelo_alumno.generate(
                                input_ids, max_tokens=512, temperature=0.8,
                            )
                        modelo_alumno.train()

                        codigo_alumno = tokenizer.Decode(output[0].tolist())

                        # 2. Profesor corrige
                        prompt_correccion = (
                            f"Un estudiante intentó resolver este problema:\n"
                            f"PROBLEMA: {prompt}\n\n"
                            f"CÓDIGO DEL ESTUDIANTE:\n```python\n{codigo_alumno}\n```\n\n"
                            f"Corrige el código. Si está bien, mejóralo. "
                            f"Explica brevemente cada corrección como comentario."
                        )
                        correccion = self.profesor.generar(
                            prompt_correccion,
                            system="Eres un profesor que corrige código de estudiantes. "
                                   "Responde con el código corregido y comentarios explicativos.",
                        )

                        ejemplo = {
                            "instruction": prompt,
                            "alumno": codigo_alumno,
                            "profesor": correccion,
                            "nivel": nivel,
                            "tipo": "correccion",
                        }

                        # Texto para entrenamiento (versión del profesor)
                        code_blocks = re.findall(
                            r"```python\s*\n(.*?)```", correccion, re.DOTALL,
                        )
                        ejemplo["text"] = (
                            code_blocks[0] if code_blocks else correccion
                        )

                        f.write(json.dumps(ejemplo, ensure_ascii=False) + "\n")
                        n += 1
                        self.generados += 1

                        if n % 10 == 0:
                            stats = self.profesor.get_stats()
                            print(f"    [{n}/{n_ejemplos}] costo=${stats['costo_usd']:.4f}")

                        time.sleep(delay)

                    except Exception as e:
                        self.errores += 1
                        print(f"    ERROR: {e}")

                if n >= n_ejemplos:
                    break

        print(f"  Correcciones generadas: {n}")
        return output_file


# =============================================================================
# DISTILLATION LOSS
# =============================================================================

def distillation_loss(
    student_logits: torch.Tensor,    # [B, L, V] logits del alumno
    teacher_targets: torch.Tensor,   # [B, L] tokens hard del profesor
    hard_targets: torch.Tensor,      # [B, L] targets originales (-100 = ignore)
    temperature: float = 3.0,        # Temperatura de destilación
    alpha: float = 0.5,              # Balance: 0=solo hard, 1=solo soft
    teacher_logits: Optional[torch.Tensor] = None,  # [B, L, V] logits del profesor (online)
) -> Tuple[torch.Tensor, Dict]:
    """
    Pérdida de destilación combinada.

    L = α * L_soft + (1-α) * L_hard

    Donde:
    - L_hard = CrossEntropy(student_logits, hard_targets)  [aprender respuesta]
    - L_soft = KL(student_soft || teacher_soft) * T²       [aprender distribución]

    Si teacher_logits está disponible (modo online), usa soft targets.
    Si no (modo offline), solo usa hard targets del profesor.

    Args:
        student_logits: Logits del alumno (PAMPAr)
        teacher_targets: Token IDs generados por el profesor
        hard_targets: Targets originales para CE loss
        temperature: T para softening de distribuciones
        alpha: Balance entre soft y hard loss
        teacher_logits: Logits del profesor (solo modo online)

    Returns:
        (loss, info_dict)
    """
    B, L, V = student_logits.shape
    info = {}

    # Hard loss: CE estándar con targets originales
    loss_hard = F.cross_entropy(
        student_logits.view(-1, V),
        hard_targets.view(-1),
        ignore_index=-100,
    )
    info["loss_hard"] = loss_hard.item()

    if teacher_logits is not None:
        # Modo ONLINE: tenemos logits del profesor → soft targets
        # Suavizar distribuciones con temperatura
        student_soft = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)

        # KL divergence (scaled by T²)
        loss_soft = F.kl_div(
            student_soft.view(-1, V),
            teacher_soft.view(-1, V),
            reduction="batchmean",
        ) * (temperature ** 2)

        info["loss_soft"] = loss_soft.item()
        loss = alpha * loss_soft + (1 - alpha) * loss_hard
    else:
        # Modo OFFLINE: solo hard targets del profesor
        # CE con tokens generados por el profesor
        loss_teacher = F.cross_entropy(
            student_logits.view(-1, V),
            teacher_targets.view(-1),
            ignore_index=-100,
        )
        info["loss_teacher"] = loss_teacher.item()
        loss = alpha * loss_teacher + (1 - alpha) * loss_hard

    info["loss_total"] = loss.item()
    return loss, info


# =============================================================================
# TERRITORY-AWARE DISTILLATION
# =============================================================================

def territory_aware_distillation(
    model,
    tokenizer,
    teacher_text: str,
    nivel: int = 3,
) -> torch.Tensor:
    """
    Destilación consciente de territorios.

    Analiza qué partes del código del profesor corresponden a qué
    territorio y refuerza la señal de routing del Tálamo.

    Acepta el modelo y tokenizer directamente, realiza un forward pass
    para obtener las activaciones territoriales, y las compara con el
    patrón esperado para ese nivel de dificultad.

    Args:
        model: PampaRCoderV2 (debe tener .talamo y .tok_emb)
        tokenizer: SentencePiece tokenizer
        teacher_text: Texto generado por el profesor
        nivel: Nivel de dificultad (1-6)

    Returns:
        Territory alignment loss (scalar tensor)
    """
    from pampar.coder.v2.aprendizaje.curriculum import clasificar_dificultad, TERRITORIOS_POR_NIVEL

    # Tokenizar el texto del profesor
    tokens = tokenizer.Encode(teacher_text)[:model.config.max_seq_len]
    if len(tokens) < 2:
        return torch.tensor(0.0, requires_grad=True)

    device = next(model.parameters()).device
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

    # Forward pass para obtener activaciones territoriales
    # No usar torch.no_grad() para que el gradiente fluya
    x = model.emb_drop(model.tok_emb(input_ids))
    terr_acts, _ = model.talamo(x, input_ids)

    # Clasificar nivel (usar el provisto o auto-detectar)
    nivel_enum, _ = clasificar_dificultad(teacher_text)
    target = torch.tensor(
        TERRITORIOS_POR_NIVEL[nivel_enum],
        dtype=torch.float32,
        device=device,
    )

    # MSE entre activaciones reales y target
    mean_acts = terr_acts.mean(dim=(0, 1))  # [4]
    loss = F.mse_loss(mean_acts, target)

    return loss * 0.05  # Weight bajo, es señal auxiliar
