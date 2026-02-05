# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Knowledge Distillation para PAMPAr-Coder.

Extrae conocimiento de modelos grandes (DeepSeek, GPT-4, Claude, etc.)
para entrenar el modelo pequeño.

Estrategias:
1. Response Distillation: Generar respuestas con teacher, entrenar student
2. Feature Distillation: Alinear representaciones internas
3. Logit Distillation: Transferir distribuciones de probabilidad

"Aprende de los gigantes para correr en tu laptop"
"""

import os
import json
import torch
import asyncio
import aiohttp
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime
from tqdm import tqdm


@dataclass
class DistillationConfig:
    """Configuración de Knowledge Distillation."""
    
    # Teacher API
    teacher_provider: str = "openai"  # "openai", "anthropic", "deepseek", "local"
    teacher_model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    api_base: Optional[str] = None
    
    # Generación
    max_tokens: int = 512
    temperature: float = 0.3
    num_samples_per_prompt: int = 1
    
    # Rate limiting
    requests_per_minute: int = 60
    concurrent_requests: int = 5
    
    # Output
    output_dir: str = "data/distillation/generated"
    save_every: int = 100


class TeacherAPI:
    """Cliente para APIs de modelos teacher."""
    
    PROVIDERS = {
        "openai": {
            "base": "https://api.openai.com/v1",
            "chat_endpoint": "/chat/completions",
        },
        "anthropic": {
            "base": "https://api.anthropic.com/v1",
            "chat_endpoint": "/messages",
        },
        "deepseek": {
            "base": "https://api.deepseek.com/v1",
            "chat_endpoint": "/chat/completions",
        },
        "openrouter": {
            "base": "https://openrouter.ai/api/v1",
            "chat_endpoint": "/chat/completions",
        },
        "local": {
            "base": "http://localhost:11434/v1",
            "chat_endpoint": "/chat/completions",
        },
    }
    
    def __init__(self, config: DistillationConfig):
        self.config = config
        
        # API key
        self.api_key = os.environ.get(config.api_key_env, "")
        if not self.api_key and config.teacher_provider != "local":
            raise ValueError(f"Set {config.api_key_env} environment variable")
        
        # Provider config
        provider = self.PROVIDERS.get(config.teacher_provider)
        if not provider:
            raise ValueError(f"Unknown provider: {config.teacher_provider}")
        
        self.base_url = config.api_base or provider["base"]
        self.chat_endpoint = provider["chat_endpoint"]
        
        # Rate limiting
        self._semaphore = asyncio.Semaphore(config.concurrent_requests)
        self._last_request_time = 0
        self._min_interval = 60.0 / config.requests_per_minute
    
    def _get_headers(self) -> Dict[str, str]:
        """Headers para cada provider."""
        headers = {"Content-Type": "application/json"}
        
        if self.config.teacher_provider == "anthropic":
            headers["x-api-key"] = self.api_key
            headers["anthropic-version"] = "2023-06-01"
        elif self.config.teacher_provider == "openrouter":
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["HTTP-Referer"] = "https://github.com/pampar-coder"
        elif self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        return headers
    
    def _format_request(self, prompt: str, system: str = None) -> Dict:
        """Formatea request según provider."""
        messages = []
        
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        if self.config.teacher_provider == "anthropic":
            return {
                "model": self.config.teacher_model,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "messages": messages,
            }
        else:
            return {
                "model": self.config.teacher_model,
                "messages": messages,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
            }
    
    def _parse_response(self, response: Dict) -> str:
        """Extrae texto de respuesta según provider."""
        if self.config.teacher_provider == "anthropic":
            return response["content"][0]["text"]
        else:
            return response["choices"][0]["message"]["content"]
    
    async def _rate_limit(self):
        """Espera para cumplir rate limit."""
        import time
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            await asyncio.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()
    
    async def generate(
        self,
        prompt: str,
        system: str = None,
        session: aiohttp.ClientSession = None,
    ) -> Tuple[str, Dict]:
        """
        Genera respuesta del teacher.
        
        Returns:
            response: Texto generado
            metadata: Info adicional (tokens, model, etc.)
        """
        async with self._semaphore:
            await self._rate_limit()
            
            url = f"{self.base_url}{self.chat_endpoint}"
            data = self._format_request(prompt, system)
            headers = self._get_headers()
            
            own_session = session is None
            if own_session:
                session = aiohttp.ClientSession()
            
            try:
                async with session.post(url, json=data, headers=headers) as resp:
                    if resp.status != 200:
                        error = await resp.text()
                        raise Exception(f"API error {resp.status}: {error}")
                    
                    result = await resp.json()
                    text = self._parse_response(result)
                    
                    metadata = {
                        "model": self.config.teacher_model,
                        "provider": self.config.teacher_provider,
                        "prompt_tokens": result.get("usage", {}).get("prompt_tokens"),
                        "completion_tokens": result.get("usage", {}).get("completion_tokens"),
                    }
                    
                    return text, metadata
            finally:
                if own_session:
                    await session.close()
    
    def generate_sync(self, prompt: str, system: str = None) -> Tuple[str, Dict]:
        """Versión síncrona de generate."""
        return asyncio.run(self.generate(prompt, system))


@dataclass
class CodePrompt:
    """Prompt para generación de código."""
    instruction: str
    context: str = ""
    expected_language: str = "python"
    difficulty: str = "medium"
    category: str = "general"


@dataclass
class CodeResponse:
    """Respuesta generada por teacher."""
    prompt: CodePrompt
    response: str
    metadata: Dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class DistillationDataCollector:
    """Recolecta datos de distillation desde teacher models."""
    
    SYSTEM_PROMPT = """You are an expert programmer. Generate clean, well-documented code.
Follow these guidelines:
- Write clear, readable code with meaningful variable names
- Include docstrings and comments where helpful
- Handle edge cases appropriately
- Use idiomatic patterns for the language
- Keep solutions concise but complete"""
    
    PROMPT_TEMPLATES = {
        "function": "Write a {language} function that {instruction}",
        "class": "Create a {language} class that {instruction}",
        "algorithm": "Implement in {language}: {instruction}",
        "debug": "Fix the following {language} code:\n```\n{context}\n```\nProblem: {instruction}",
        "explain": "Explain this {language} code and improve it:\n```\n{context}\n```",
        "convert": "Convert this code to {language}:\n```\n{context}\n```",
    }
    
    def __init__(self, config: DistillationConfig):
        self.config = config
        self.teacher = TeacherAPI(config)
        self.responses: List[CodeResponse] = []
        
        # Crear directorio output
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def format_prompt(self, prompt: CodePrompt, template: str = "function") -> str:
        """Formatea prompt según template."""
        template_str = self.PROMPT_TEMPLATES.get(template, "{instruction}")
        return template_str.format(
            language=prompt.expected_language,
            instruction=prompt.instruction,
            context=prompt.context,
        )
    
    async def collect_batch(
        self,
        prompts: List[CodePrompt],
        template: str = "function",
        show_progress: bool = True,
    ) -> List[CodeResponse]:
        """
        Recolecta respuestas para un batch de prompts.
        """
        responses = []
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for prompt in prompts:
                formatted = self.format_prompt(prompt, template)
                task = self._collect_one(prompt, formatted, session)
                tasks.append(task)
            
            # Con progreso
            if show_progress:
                for coro in tqdm(
                    asyncio.as_completed(tasks),
                    total=len(tasks),
                    desc="Collecting responses"
                ):
                    result = await coro
                    if result:
                        responses.append(result)
                        self.responses.append(result)
                        
                        # Auto-save
                        if len(self.responses) % self.config.save_every == 0:
                            self.save_responses()
            else:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for r in results:
                    if isinstance(r, CodeResponse):
                        responses.append(r)
                        self.responses.append(r)
        
        return responses
    
    async def _collect_one(
        self,
        prompt: CodePrompt,
        formatted_prompt: str,
        session: aiohttp.ClientSession,
    ) -> Optional[CodeResponse]:
        """Recolecta una respuesta."""
        try:
            response, metadata = await self.teacher.generate(
                formatted_prompt,
                self.SYSTEM_PROMPT,
                session
            )
            return CodeResponse(
                prompt=prompt,
                response=response,
                metadata=metadata,
            )
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def save_responses(self, filename: str = None) -> str:
        """Guarda respuestas a archivo."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"distillation_{timestamp}.jsonl"
        
        path = Path(self.config.output_dir) / filename
        
        with open(path, 'w', encoding='utf-8') as f:
            for response in self.responses:
                record = {
                    "instruction": response.prompt.instruction,
                    "context": response.prompt.context,
                    "language": response.prompt.expected_language,
                    "category": response.prompt.category,
                    "difficulty": response.prompt.difficulty,
                    "response": response.response,
                    "teacher_model": response.metadata.get("model"),
                    "timestamp": response.timestamp,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        print(f"💾 Saved {len(self.responses)} responses to {path}")
        return str(path)
    
    def load_responses(self, path: str) -> List[CodeResponse]:
        """Carga respuestas desde archivo."""
        responses = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                prompt = CodePrompt(
                    instruction=data["instruction"],
                    context=data.get("context", ""),
                    expected_language=data.get("language", "python"),
                    category=data.get("category", "general"),
                    difficulty=data.get("difficulty", "medium"),
                )
                response = CodeResponse(
                    prompt=prompt,
                    response=data["response"],
                    metadata={"model": data.get("teacher_model")},
                    timestamp=data.get("timestamp", ""),
                )
                responses.append(response)
        
        self.responses.extend(responses)
        return responses


class DistillationTrainer:
    """
    Entrena modelo student con datos de distillation.
    
    Modos:
    1. Response distillation: Aprende de texto generado
    2. Logit distillation: Aprende distribuciones (requiere logits del teacher)
    """
    
    def __init__(
        self,
        student_model: torch.nn.Module,
        tokenizer,
        temperatura: float = 2.0,
        peso_distill: float = 0.5,
    ):
        self.student = student_model
        self.tokenizer = tokenizer
        self.temperatura = temperatura
        self.peso_distill = peso_distill
    
    def prepare_distillation_batch(
        self,
        responses: List[CodeResponse],
        max_length: int = 512,
    ) -> Dict[str, torch.Tensor]:
        """
        Prepara batch de datos para distillation training.
        
        Formato: [INST]{instruction}[/INST]{response}
        """
        texts = []
        for r in responses:
            # Formato instruction-response
            text = f"[INST]{r.prompt.instruction}[/INST]{r.response}"
            texts.append(text)
        
        # Tokenizar
        encoded = [
            self.tokenizer.encode(t)[:max_length]
            for t in texts
        ]
        
        # Padding
        max_len = max(len(e) for e in encoded)
        input_ids = torch.zeros(len(encoded), max_len, dtype=torch.long)
        attention_mask = torch.zeros(len(encoded), max_len, dtype=torch.long)
        
        for i, enc in enumerate(encoded):
            input_ids[i, :len(enc)] = torch.tensor(enc)
            attention_mask[i, :len(enc)] = 1
        
        # Labels (shift right)
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calcula loss combinado para distillation.
        
        L = (1-α) * CE(student, labels) + α * KL(student || teacher)
        """
        # Hard label loss (cross-entropy)
        vocab_size = student_logits.size(-1)
        ce_loss = torch.nn.functional.cross_entropy(
            student_logits.view(-1, vocab_size),
            labels.view(-1),
            ignore_index=-100
        )
        
        # Soft label loss (KL divergence)
        T = self.temperatura
        soft_teacher = torch.nn.functional.softmax(teacher_logits / T, dim=-1)
        soft_student = torch.nn.functional.log_softmax(student_logits / T, dim=-1)
        
        kl_loss = torch.nn.functional.kl_div(
            soft_student,
            soft_teacher,
            reduction='batchmean'
        ) * (T ** 2)
        
        # Combinar
        total_loss = (1 - self.peso_distill) * ce_loss + self.peso_distill * kl_loss
        
        metrics = {
            "ce_loss": ce_loss.item(),
            "kl_loss": kl_loss.item(),
            "total_loss": total_loss.item(),
        }
        
        return total_loss, metrics


# =============================================================================
# FUNCIONES DE UTILIDAD
# =============================================================================

def load_existing_prompts(data_dir: str = "data/distillation") -> List[CodePrompt]:
    """
    Carga prompts existentes de archivos de distillation.
    """
    prompts = []
    data_path = Path(data_dir)
    
    # Cargar de varios archivos
    for jsonl_file in data_path.glob("*.jsonl"):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    
                    # Extraer instruction
                    instruction = data.get("instruction") or data.get("input") or ""
                    if not instruction:
                        continue
                    
                    prompt = CodePrompt(
                        instruction=instruction,
                        context=data.get("context", "") or data.get("input", ""),
                        expected_language="python",
                        category=data.get("category", "general"),
                    )
                    prompts.append(prompt)
                except:
                    continue
    
    return prompts


def create_prompts_from_exercises() -> List[CodePrompt]:
    """
    Crea prompts variados para entrenar coding ability.
    """
    exercises = [
        # Algoritmos básicos
        ("Write a function to find the nth Fibonacci number", "algorithm", "easy"),
        ("Implement binary search on a sorted array", "algorithm", "easy"),
        ("Write a function to check if a string is a palindrome", "algorithm", "easy"),
        
        # Data structures
        ("Implement a stack with push, pop, and peek operations", "algorithm", "medium"),
        ("Create a linked list with insert and delete methods", "algorithm", "medium"),
        ("Implement a hash table with collision handling", "algorithm", "medium"),
        
        # String manipulation
        ("Write a function to reverse words in a sentence", "algorithm", "easy"),
        ("Implement string compression (aabbbcc -> a2b3c2)", "algorithm", "medium"),
        ("Find the longest common subsequence of two strings", "algorithm", "hard"),
        
        # Trees and graphs
        ("Implement BFS traversal of a binary tree", "algorithm", "medium"),
        ("Write a function to find if a path exists between two nodes", "algorithm", "medium"),
        ("Implement Dijkstra's shortest path algorithm", "algorithm", "hard"),
        
        # Dynamic programming
        ("Solve the coin change problem with minimum coins", "algorithm", "medium"),
        ("Find the maximum sum subarray (Kadane's algorithm)", "algorithm", "medium"),
        ("Solve the 0/1 knapsack problem", "algorithm", "hard"),
        
        # API and web
        ("Create a simple REST API endpoint with FastAPI", "function", "medium"),
        ("Write a function to make HTTP requests with retries", "function", "medium"),
        ("Implement rate limiting decorator", "function", "hard"),
        
        # File handling
        ("Write a function to read and parse a CSV file", "function", "easy"),
        ("Create a file watcher that detects changes", "function", "medium"),
        ("Implement a simple logging system", "function", "medium"),
        
        # Classes and OOP
        ("Create a class for managing a shopping cart", "class", "medium"),
        ("Implement the observer design pattern", "class", "medium"),
        ("Create a factory pattern for creating different document types", "class", "hard"),
    ]
    
    return [
        CodePrompt(
            instruction=instr,
            expected_language="python",
            category=cat,
            difficulty=diff,
        )
        for instr, cat, diff in exercises
    ]


# =============================================================================
# DEMO / CLI
# =============================================================================

async def demo_distillation():
    """Demo del sistema de distillation."""
    print("=" * 70)
    print("🎓 PAMPAr-Coder: Knowledge Distillation Demo")
    print("   'Aprende de los gigantes para correr en tu laptop'")
    print("=" * 70)
    
    # Verificar API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\n⚠️  Para usar distillation, configura OPENAI_API_KEY")
        print("   O usa otro provider (deepseek, anthropic, local)")
        print("\n💡 Ejemplo con OpenRouter (muchos modelos):")
        print("   export OPENROUTER_API_KEY=sk-...")
        print("   config.teacher_provider = 'openrouter'")
        print("   config.api_key_env = 'OPENROUTER_API_KEY'")
        print("   config.teacher_model = 'deepseek/deepseek-coder'")
        return
    
    # Configuración
    config = DistillationConfig(
        teacher_provider="openai",
        teacher_model="gpt-4o-mini",
        requests_per_minute=20,
        concurrent_requests=3,
    )
    
    # Crear collector
    collector = DistillationDataCollector(config)
    
    # Crear algunos prompts de prueba
    test_prompts = [
        CodePrompt(
            instruction="calculate the factorial of a number recursively",
            expected_language="python",
            category="algorithm",
            difficulty="easy",
        ),
        CodePrompt(
            instruction="implement a simple LRU cache",
            expected_language="python",
            category="algorithm",
            difficulty="medium",
        ),
    ]
    
    print(f"\n📝 Collecting {len(test_prompts)} responses from {config.teacher_model}...")
    
    # Recolectar
    responses = await collector.collect_batch(test_prompts, show_progress=True)
    
    # Mostrar resultados
    print(f"\n✅ Collected {len(responses)} responses:")
    for i, r in enumerate(responses):
        print(f"\n--- Response {i+1} ---")
        print(f"Instruction: {r.prompt.instruction}")
        print(f"Code preview: {r.response[:200]}...")
    
    # Guardar
    path = collector.save_responses()
    print(f"\n💾 Saved to: {path}")


if __name__ == "__main__":
    asyncio.run(demo_distillation())
