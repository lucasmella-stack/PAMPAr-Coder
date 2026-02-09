# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Fase 1: Curriculum Learning — "Infancia"

Aprende como un niño: de lo simple a lo complejo.
Cada nivel de dificultad activa progresivamente más Territorios.

Niveles:
  1. Variables y asignaciones     → SINTAXIS domina
  2. Control de flujo             → LOGICO + SINTAXIS
  3. Funciones                    → SEMANTICA + ESTRUCTURAL
  4. Clases y OOP                 → Todos los territorios
  5. Algoritmos complejos         → LOGICO + ESTRUCTURAL intensos
  6. Patrones de diseño           → Integración total + razonamiento

Métricas de complejidad (para clasificar automáticamente):
  - Profundidad de anidamiento (nesting)
  - Número de conceptos distintos (keywords únicas)
  - Longitud en tokens
  - Presencia de constructos avanzados
"""

import json
import re
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, IterableDataset


# =============================================================================
# NIVELES DE DIFICULTAD
# =============================================================================

class NivelDificultad(IntEnum):
    """Niveles de dificultad del curriculum (como grados escolares)."""
    BASICO = 1         # Variables, print, asignaciones
    CONTROL = 2        # if/else, for, while
    FUNCIONES = 3      # def, return, parámetros
    CLASES = 4         # class, herencia, métodos
    ALGORITMOS = 5     # Sorting, búsqueda, recursión
    PATRONES = 6       # Design patterns, código complejo


# Patrones para detectar nivel de dificultad
PATRONES_NIVEL = {
    NivelDificultad.BASICO: {
        "keywords": {"print", "=", "int", "str", "float", "input"},
        "max_nesting": 0,
        "max_tokens": 128,
    },
    NivelDificultad.CONTROL: {
        "keywords": {"if", "else", "elif", "for", "while", "break", "continue"},
        "max_nesting": 2,
        "max_tokens": 256,
    },
    NivelDificultad.FUNCIONES: {
        "keywords": {"def", "return", "lambda", "yield", "args", "kwargs"},
        "max_nesting": 3,
        "max_tokens": 512,
    },
    NivelDificultad.CLASES: {
        "keywords": {"class", "self", "__init__", "super", "property", "staticmethod"},
        "max_nesting": 4,
        "max_tokens": 1024,
    },
    NivelDificultad.ALGORITMOS: {
        "keywords": {"sorted", "recursive", "binary", "search", "tree", "graph", "stack", "queue"},
        "max_nesting": 5,
        "max_tokens": 2048,
    },
    NivelDificultad.PATRONES: {
        "keywords": {"abstract", "factory", "singleton", "observer", "decorator", "iterator"},
        "max_nesting": 6,
        "max_tokens": 4096,
    },
}

# Territorios dominantes por nivel (qué territorio debe activarse más)
TERRITORIOS_POR_NIVEL = {
    NivelDificultad.BASICO:     [1.0, 0.3, 0.1, 0.2],  # SINTAXIS domina
    NivelDificultad.CONTROL:    [0.6, 0.3, 0.7, 0.3],  # LOGICO sube
    NivelDificultad.FUNCIONES:  [0.5, 0.7, 0.4, 0.6],  # SEMANTICA + ESTRUCTURAL
    NivelDificultad.CLASES:     [0.6, 0.8, 0.5, 0.7],  # Todos activos
    NivelDificultad.ALGORITMOS: [0.4, 0.5, 0.9, 0.8],  # LOGICO + ESTRUCTURAL
    NivelDificultad.PATRONES:   [0.7, 0.8, 0.8, 0.9],  # Integración total
}


# =============================================================================
# CLASIFICADOR DE DIFICULTAD
# =============================================================================

def calcular_nesting(codigo: str) -> int:
    """Calcula la profundidad máxima de anidamiento."""
    max_depth = 0
    current_depth = 0
    for line in codigo.split("\n"):
        stripped = line.lstrip()
        if not stripped:
            continue
        indent = len(line) - len(stripped)
        # Asumimos 4 espacios por nivel
        depth = indent // 4
        current_depth = depth
        max_depth = max(max_depth, current_depth)
    return max_depth


def contar_conceptos(codigo: str) -> Dict[str, int]:
    """Cuenta conceptos de programación presentes en el código."""
    conceptos = {}
    
    # Keywords de Python
    kw_patterns = {
        "def": r"\bdef\b",
        "class": r"\bclass\b",
        "if": r"\bif\b",
        "for": r"\bfor\b",
        "while": r"\bwhile\b",
        "try": r"\btry\b",
        "import": r"\bimport\b",
        "return": r"\breturn\b",
        "lambda": r"\blambda\b",
        "yield": r"\byield\b",
        "async": r"\basync\b",
        "with": r"\bwith\b",
        "raise": r"\braise\b",
        "assert": r"\bassert\b",
        "list_comp": r"\[.+\bfor\b.+\bin\b.+\]",
        "dict_comp": r"\{.+\bfor\b.+\bin\b.+\}",
        "decorator": r"@\w+",
        "type_hint": r":\s*(int|str|float|bool|List|Dict|Optional|Tuple)",
    }
    
    for nombre, patron in kw_patterns.items():
        count = len(re.findall(patron, codigo))
        if count > 0:
            conceptos[nombre] = count
    
    return conceptos


def clasificar_dificultad(codigo: str) -> Tuple[NivelDificultad, float]:
    """
    Clasifica el nivel de dificultad de un fragmento de código.
    
    Returns:
        (nivel, confianza) donde confianza ∈ [0, 1]
    """
    nesting = calcular_nesting(codigo)
    conceptos = contar_conceptos(codigo)
    n_conceptos = len(conceptos)
    n_lineas = len([l for l in codigo.split("\n") if l.strip()])
    
    # Scoring por nivel
    scores = {}
    
    # Nivel 1: Básico - sin funciones, sin clases, poco anidamiento
    if n_conceptos <= 3 and nesting <= 1 and n_lineas <= 15:
        scores[NivelDificultad.BASICO] = 0.9
    else:
        scores[NivelDificultad.BASICO] = max(0, 0.5 - n_conceptos * 0.1)
    
    # Nivel 2: Control
    has_control = any(k in conceptos for k in ["if", "for", "while"])
    has_func = "def" in conceptos
    if has_control and not has_func and nesting <= 2:
        scores[NivelDificultad.CONTROL] = 0.85
    elif has_control:
        scores[NivelDificultad.CONTROL] = 0.4
    else:
        scores[NivelDificultad.CONTROL] = 0.1
    
    # Nivel 3: Funciones
    if has_func and "class" not in conceptos:
        scores[NivelDificultad.FUNCIONES] = 0.8
    elif has_func:
        scores[NivelDificultad.FUNCIONES] = 0.4
    else:
        scores[NivelDificultad.FUNCIONES] = 0.1
    
    # Nivel 4: Clases
    has_class = "class" in conceptos
    if has_class:
        scores[NivelDificultad.CLASES] = 0.85
    else:
        scores[NivelDificultad.CLASES] = 0.05
    
    # Nivel 5: Algoritmos
    is_complex = nesting >= 3 and n_conceptos >= 5 and n_lineas >= 20
    if is_complex and has_func:
        scores[NivelDificultad.ALGORITMOS] = 0.8
    else:
        scores[NivelDificultad.ALGORITMOS] = max(0, nesting * 0.15 + n_conceptos * 0.05)
    
    # Nivel 6: Patrones
    has_patterns = (
        has_class and has_func and nesting >= 3 and 
        n_conceptos >= 7 and 
        any(k in conceptos for k in ["decorator", "lambda", "type_hint"])
    )
    if has_patterns:
        scores[NivelDificultad.PATRONES] = 0.85
    else:
        scores[NivelDificultad.PATRONES] = max(0, n_conceptos * 0.05 - 0.2)
    
    # Seleccionar el nivel con mayor score
    mejor_nivel = max(scores, key=scores.get)
    confianza = scores[mejor_nivel]
    
    return mejor_nivel, confianza


# =============================================================================
# DATASET CON CURRICULUM
# =============================================================================

@dataclass
class EjemploCurriculum:
    """Un ejemplo de entrenamiento con metadatos de dificultad."""
    texto: str
    nivel: NivelDificultad
    confianza: float
    territorios_target: List[float]  # [SINT, SEM, LOG, EST]
    n_tokens: int = 0


class CurriculumDataset(IterableDataset):
    """
    Dataset que sirve ejemplos en orden de dificultad.
    
    Like a school textbook: chapter 1 before chapter 6.
    Opcionalmente mezcla un % de niveles anteriores (revisión).
    """
    
    def __init__(
        self,
        archivos: List[Path],
        tokenizer,
        nivel_actual: NivelDificultad = NivelDificultad.BASICO,
        max_seq_len: int = 4096,
        revision_ratio: float = 0.1,  # 10% de niveles anteriores
        seed: int = 42,
    ):
        self.archivos = archivos
        self.tokenizer = tokenizer
        self.nivel_actual = nivel_actual
        self.max_seq_len = max_seq_len
        self.revision_ratio = revision_ratio
        self.seed = seed
        
        # Buffers por nivel
        self._buffers: Dict[NivelDificultad, List[str]] = {
            n: [] for n in NivelDificultad
        }
        
    def _extraer_codigo(self, texto: str) -> str:
        """Extrae código de un ejemplo (puede tener instrucción + código)."""
        # Buscar bloques de código
        code_blocks = re.findall(r"```(?:python)?\n?(.*?)```", texto, re.DOTALL)
        if code_blocks:
            return "\n".join(code_blocks)
        
        # Si no hay bloques, buscar líneas que parezcan código
        lineas = texto.split("\n")
        codigo_lineas = [
            l for l in lineas 
            if re.match(r"^[\s]*(def |class |if |for |while |import |from |return |print|#)", l)
            or re.match(r"^[\s]*\w+\s*[=\(]", l)
        ]
        return "\n".join(codigo_lineas) if codigo_lineas else texto
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Itera ejemplos en orden curricular."""
        import random
        rng = random.Random(self.seed)
        
        for archivo in self.archivos:
            with open(archivo, "r", encoding="utf-8") as f:
                for linea in f:
                    try:
                        data = json.loads(linea.strip())
                    except json.JSONDecodeError:
                        continue
                    
                    # Extraer texto
                    texto = data.get("text", data.get("output", data.get("code", "")))
                    if not texto or len(texto) < 10:
                        continue
                    
                    # Clasificar dificultad
                    codigo = self._extraer_codigo(texto)
                    nivel, conf = clasificar_dificultad(codigo)
                    
                    # Filtrar por nivel actual
                    if nivel.value > self.nivel_actual.value:
                        continue  # Muy avanzado para el nivel actual
                    
                    # Incluir niveles anteriores con probabilidad revision_ratio
                    if nivel.value < self.nivel_actual.value:
                        if rng.random() > self.revision_ratio:
                            continue  # Skip revisión la mayoría del tiempo
                    
                    # Tokenizar
                    tokens = self.tokenizer.Encode(texto)
                    if len(tokens) > self.max_seq_len:
                        tokens = tokens[:self.max_seq_len]
                    if len(tokens) < 4:
                        continue
                    
                    # Crear tensores
                    input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
                    targets = torch.tensor(tokens[1:], dtype=torch.long)
                    
                    # Territory targets para este nivel
                    terr_target = torch.tensor(
                        TERRITORIOS_POR_NIVEL[nivel],
                        dtype=torch.float32,
                    )
                    
                    yield {
                        "input_ids": input_ids,
                        "targets": targets,
                        "nivel": nivel.value,
                        "terr_target": terr_target,
                    }


# =============================================================================
# CURRICULUM MANAGER
# =============================================================================

class CurriculumManager:
    """
    Gestiona la progresión del curriculum.
    
    Como un profesor que decide cuándo pasar al siguiente tema:
    - Si el alumno domina el nivel actual (loss baja) → avanzar
    - Si el alumno struggle → quedarse más tiempo
    - Siempre revisar lo anterior (spaced repetition)
    """
    
    def __init__(
        self,
        criterio_avance: float = 2.0,    # Loss máxima para avanzar
        paciencia: int = 3,               # Epochs sin mejora antes de avanzar
        min_epochs_nivel: int = 1,        # Mínimo epochs por nivel
        max_epochs_nivel: int = 5,        # Máximo epochs por nivel
    ):
        self.criterio_avance = criterio_avance
        self.paciencia = paciencia
        self.min_epochs_nivel = min_epochs_nivel
        self.max_epochs_nivel = max_epochs_nivel
        
        self.nivel_actual = NivelDificultad.BASICO
        self.epochs_en_nivel = 0
        self.mejor_loss_nivel = float("inf")
        self.sin_mejora = 0
        
        # Historial
        self.historial: List[Dict] = []
    
    def reportar_epoch(self, loss: float, accuracy: float = 0.0) -> Dict:
        """
        Reporta resultado de un epoch y decide si avanzar.
        
        Returns:
            Dict con 'accion' ('continuar' o 'avanzar'), 'nivel_nuevo', etc.
        """
        self.epochs_en_nivel += 1
        
        # Tracking
        if loss < self.mejor_loss_nivel:
            self.mejor_loss_nivel = loss
            self.sin_mejora = 0
        else:
            self.sin_mejora += 1
        
        self.historial.append({
            "nivel": self.nivel_actual.name,
            "epoch": self.epochs_en_nivel,
            "loss": loss,
            "accuracy": accuracy,
        })
        
        # Decisión de avance
        resultado = {
            "accion": "continuar",
            "nivel_actual": self.nivel_actual,
            "nivel_nuevo": self.nivel_actual,
            "epochs_en_nivel": self.epochs_en_nivel,
            "mejor_loss": self.mejor_loss_nivel,
            "razon": "",
        }
        
        # ¿Ya es el último nivel?
        if self.nivel_actual == NivelDificultad.PATRONES:
            resultado["razon"] = "nivel máximo alcanzado"
            return resultado
        
        # ¿Mínimo de epochs cumplido?
        if self.epochs_en_nivel < self.min_epochs_nivel:
            resultado["razon"] = f"mínimo {self.min_epochs_nivel} epochs por nivel"
            return resultado
        
        # Criterios de avance
        should_advance = False
        
        if loss <= self.criterio_avance:
            should_advance = True
            resultado["razon"] = f"loss {loss:.4f} <= criterio {self.criterio_avance}"
        elif self.sin_mejora >= self.paciencia:
            should_advance = True
            resultado["razon"] = f"sin mejora por {self.sin_mejora} epochs"
        elif self.epochs_en_nivel >= self.max_epochs_nivel:
            should_advance = True
            resultado["razon"] = f"máximo {self.max_epochs_nivel} epochs alcanzado"
        
        if should_advance:
            # Avanzar al siguiente nivel
            nuevo = NivelDificultad(self.nivel_actual.value + 1)
            resultado["accion"] = "avanzar"
            resultado["nivel_nuevo"] = nuevo
            
            # Reset para nuevo nivel
            self.nivel_actual = nuevo
            self.epochs_en_nivel = 0
            self.mejor_loss_nivel = float("inf")
            self.sin_mejora = 0
        
        return resultado
    
    def get_estado(self) -> Dict:
        """Estado actual del curriculum para checkpoint."""
        return {
            "nivel_actual": self.nivel_actual.value,
            "epochs_en_nivel": self.epochs_en_nivel,
            "mejor_loss_nivel": self.mejor_loss_nivel,
            "sin_mejora": self.sin_mejora,
            "historial": self.historial,
        }
    
    def cargar_estado(self, estado: Dict):
        """Restaura estado desde checkpoint."""
        self.nivel_actual = NivelDificultad(estado["nivel_actual"])
        self.epochs_en_nivel = estado["epochs_en_nivel"]
        self.mejor_loss_nivel = estado["mejor_loss_nivel"]
        self.sin_mejora = estado["sin_mejora"]
        self.historial = estado.get("historial", [])


# =============================================================================
# TERRITORY ALIGNMENT LOSS
# =============================================================================

def territory_alignment_loss(
    terr_acts: torch.Tensor,     # [B, L, 4] actual activations
    terr_target: torch.Tensor,   # [B, 4] or [4] target per level
    weight: float = 0.1,
) -> torch.Tensor:
    """
    Pérdida que incentiva que los territorios correctos se activen
    para cada nivel de dificultad.
    
    En Nivel 1 (BASICO), queremos SINTAXIS alto, LOGICO bajo.
    En Nivel 5 (ALGORITMOS), queremos LOGICO + ESTRUCTURAL altos.
    
    Esto "enseña" al Tálamo a enrutar correctamente sin necesidad
    de datasets masivos — las reglas del curriculum guían.
    """
    # Promediar activaciones sobre la secuencia
    mean_acts = terr_acts.mean(dim=1)  # [B, 4]
    
    # Expandir target si necesario
    if terr_target.dim() == 1:
        terr_target = terr_target.unsqueeze(0).expand_as(mean_acts)
    
    # MSE entre activaciones reales y target
    loss = torch.nn.functional.mse_loss(mean_acts, terr_target)
    
    return weight * loss


# =============================================================================
# UTILIDADES
# =============================================================================

def crear_curriculum_desde_jsonl(
    archivos: List[str],
    output_dir: str,
    max_por_nivel: int = 50000,
) -> Dict[NivelDificultad, int]:
    """
    Pre-procesa archivos JSONL y los separa por nivel de dificultad.
    
    Crea archivos: nivel_1_basico.jsonl, nivel_2_control.jsonl, etc.
    
    Returns:
        Conteo de ejemplos por nivel
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    conteos: Dict[NivelDificultad, int] = {n: 0 for n in NivelDificultad}
    handles = {}
    
    try:
        # Abrir archivos de salida
        for nivel in NivelDificultad:
            nombre = f"nivel_{nivel.value}_{nivel.name.lower()}.jsonl"
            handles[nivel] = open(output_path / nombre, "w", encoding="utf-8")
        
        # Procesar archivos de entrada
        for archivo in archivos:
            print(f"Procesando {archivo}...")
            with open(archivo, "r", encoding="utf-8") as f:
                for i, linea in enumerate(f):
                    if i % 10000 == 0:
                        print(f"  línea {i}...")
                    
                    try:
                        data = json.loads(linea.strip())
                    except json.JSONDecodeError:
                        continue
                    
                    texto = data.get("text", data.get("output", data.get("code", "")))
                    if not texto or len(texto) < 10:
                        continue
                    
                    # Extraer y clasificar
                    codigo = texto
                    code_blocks = re.findall(r"```(?:python)?\n?(.*?)```", texto, re.DOTALL)
                    if code_blocks:
                        codigo = "\n".join(code_blocks)
                    
                    nivel, conf = clasificar_dificultad(codigo)
                    
                    if conteos[nivel] >= max_por_nivel:
                        continue
                    
                    # Guardar con metadatos
                    data["_nivel"] = nivel.value
                    data["_confianza"] = round(conf, 3)
                    handles[nivel].write(json.dumps(data, ensure_ascii=False) + "\n")
                    conteos[nivel] += 1
        
    finally:
        for h in handles.values():
            h.close()
    
    # Resumen
    print("\n=== Curriculum generado ===")
    for nivel in NivelDificultad:
        print(f"  Nivel {nivel.value} ({nivel.name}): {conteos[nivel]} ejemplos")
    print(f"  Total: {sum(conteos.values())}")
    
    return conteos
