# PAMPAr-Coder Repository Guidelines

> For AI agents: Claude Code, Codex (OpenAI), Gemini CLI, etc.
> For GitHub Copilot: see `.github/copilot-instructions.md`

## Quick Reference

| Area       | Convention                                                             |
| ---------- | ---------------------------------------------------------------------- |
| Language   | Python 3.13+                                                           |
| Framework  | PyTorch 2.x                                                            |
| Tokenizer  | SentencePiece BPE — **48K** (`pampar_48k.model`) para modelo activo v3 |
| Testing    | pytest — **109 tests, todos deben pasar**                              |
| Type hints | Always required                                                        |
| Docstrings | Google style                                                           |
| Training   | Local — todo corre en la GPU del desarrollador (GTX 1650, 4 GB VRAM)   |
| NO cloud   | No hay RunPod, no hay AWS, no hay nada remoto                          |

## Estado actual del proyecto (Feb 2026)

- **Modelo activo**: `PamparV3` — **108.3M params, vocab 48K**, arquitectura 2D (4 streams × 5 niveles)
- **Checkpoint**: ninguno entrenado aún — pesos aleatorios listos para entrenamiento
- **Tokenizer activo**: `data/tokenizer/pampar_48k.model` (48K vocab)
- **Tests**: **109/109 passing** (4.2s)
- **Siguiente paso**: escribir script de entrenamiento autónomo para v3

## Project Structure

```
PAMPAr-Coder/
├── pampar/
│   ├── __init__.py
│   ├── coder/
│   │   ├── v3/                     # ← ARQUITECTURA ACTIVA
│   │   │   ├── modelo.py           # PamparV3 — 108.3M params
│   │   │   ├── config.py           # ConfigV3 + PRESET_V3 / PRESET_V3_SMALL / PRESET_V3_LARGE
│   │   │   ├── talamo.py           # TalamoInicial — LLAVES 80% + attn 20%
│   │   │   └── bloques.py          # NivelProfundo: GQA + StreamFFN (SwiGLU) + LateralGates
│   │   └── deprecated/             # PampaRCoderV2 (42M, 16K vocab) — solo referencia
│   ├── memoria/
│   │   ├── clasificador.py         # ClasificadorPareto — niveles 0-3 de importancia
│   │   ├── rag.py                  # RAGResidual — recuperación por similitud léxica
│   │   └── cola_finetune.py        # ColaFinetune — cola de candidatos a fine-tuning
│   ├── skills/
│   │   ├── base.py                 # Skill, ResultadoSkill
│   │   ├── lector_archivos.py      # LectorArchivos — lee .py/.json/.md (sandboxed)
│   │   └── ejecutar_codigo.py      # EjecutorCodigo — subprocess con timeout
│   └── runtime/
│       └── agente.py               # Agente — loop de inferencia + acciones [LEER:][EJECUTAR:]
├── biblioteca/
│   ├── indice.json                 # 39 temas Python, 5 categorías, niveles 1-6
│   └── *.jsonl                     # ~140 MB de ejemplos de código
├── data/
│   ├── tokenizer/
│   │   ├── pampar_48k.model        # ACTIVO — 48K vocab (PamparV3)
│   │   └── code_tokenizer.model    # LEGACY — 16K vocab (deprecated v2 únicamente)
│   ├── code/                       # GitHub code dumps
│   └── distillation/               # Datos de destilación
├── checkpoints/                    # Futuro home de pampar_v3_best.pt
├── scripts/
│   ├── aprender_solo.py            # Loop de entrenamiento autónomo (era v2 — necesita v3 port)
│   ├── smoke_test_viaje.py         # Smoke test pre-entrenamiento
│   ├── probar_modelo.py            # Prueba interactiva / automática
│   └── ...                         # Utilidades: curado de datos, eval, benchmark
└── tests/
    ├── conftest.py                 # Fixtures compartidas (modelo mini, tokenizer mock)
    ├── test_v3_arquitectura.py     # ConfigV3, forward pass, loss, generate, early exit
    ├── test_memoria.py             # ClasificadorPareto, RAGResidual, ColaFinetune
    ├── test_skills.py              # LectorArchivos, EjecutorCodigo
    └── test_runtime.py             # SYSTEM_PROMPT, Agente (lógica sin GPU)
```

## Architecture Overview — PamparV3 (v3)

PamparV3 es una arquitectura cortical 2D inspirada en el cerebro humano:

### Grilla 2D: 4 streams × 5 niveles

```
tok_emb [48K × 640]
  → TalamoInicial → terr_acts [B, L, 4]  /  zona_acts [B, L, 52]
  → 4 streams paralelos (dim=640)

Cada NivelProfundo (×5):
  1. GQA Atención compartida (8 Q heads / 2 KV heads, head_dim=80)
  2. Re-routing ligero del Tálamo (Linear dim→52, sin bias)
  3. 4 × StreamFFN SwiGLU independientes (uno por stream)
  4. Lateral gates por stream (bottleneck=128, como fibras blancas)

→ norm_f (RMSNorm) → lm_head (weight-tied, vocab=48K)
```

### Componentes clave

1. **TalamoInicial** — orquestador: LLAVES (80% reglas INT8) + attn_proj (20%) + context_conv
2. **52 Zonas de Brodmann** — representación interna de especialización de código
3. **4 Territorios/Streams** — SINTAXIS (1-15), SEMÁNTICA (16-30), LÓGICO (31-42), ESTRUCTURAL (43-52)
4. **Lateral Gates** — cada stream recibe del resto: comunicación horizontal por nivel
5. **Early Exit** — si confianza > 90%, salta niveles restantes (mínimo 2)

### Parámetros PRESET_V3

| Parámetro         | Valor       |
| ----------------- | ----------- |
| `dim`             | 640         |
| `n_streams`       | 4           |
| `n_levels`        | 5           |
| `n_heads`         | 8           |
| `n_kv_heads`      | 2 (GQA 4:1) |
| `ffn_mult`        | 4.0         |
| `vocab_size`      | 48 000      |
| `max_seq_len`     | 4096        |
| **Total params**  | **108.3M**  |
| **VRAM fp16**     | 200 MB      |
| **VRAM training** | 1.4 GB      |

## Módulos de Memoria, Skills y Runtime

### `pampar.memoria`

- `ClasificadorPareto` — clasifica texto en nivel 0 (ruido) a 3 (valioso) según densidad, loss, novedad
- `RAGResidual` — recuperación: guarda entradas L1+, busca por tokens comunes
- `ColaFinetune` — acumula ejemplos L3; exporta JSONL formato Alpaca cuando hay ≥ umbral

### `pampar.skills`

- `LectorArchivos` — lee archivos `.py/.json/.md/.txt` dentro del workspace, sandboxed
- `EjecutorCodigo` — corre código Python en subprocess con timeout (5s default)

### `pampar.runtime.Agente`

Loop de chat con herramientas:

- Detecta acciones `[LEER: path]`, `[EJECUTAR: code]`, `[TESTS: code]`
- Mantiene historial de N turnos
- Integra RAG y cola de fine-tuning
- `aceptar_finetune()` / `rechazar_finetune()` — feedback del usuario para datos

## Critical Rules

- **vocab_size DEBE coincidir con el tokenizer**: PamparV3 = 48K → usar `pampar_48k.model`
- LLAVES son reglas INT8 pre-computadas, **nunca** en el grafo de gradientes
- Los 4 streams procesan en paralelo — no hay secuencialidad entre streams en un nivel
- `targets.reshape(-1)` siempre, nunca `.view(-1)` (puede fallar con tensores no-contiguos)
- `terr_acts` tiene shape `[B, L, 4]` — clamp al indexar fuera del batch
- `generate()` usa `max_tokens`, NO `max_new_tokens`
- Los imports externos usan `pampar.memoria.*`, `pampar.skills.*`, `pampar.runtime.*`
- Al hacer `patch()` en tests: usar `"pampar.runtime.agente.spm"`, no `"runtime.agente.spm"`
- NO hay cloud, NO hay RunPod — todo corre local en la GPU del desarrollador

## Tokenizers disponibles

| Archivo                | Vocab   | Usar con                     |
| ---------------------- | ------- | ---------------------------- |
| `pampar_48k.model`     | **48K** | **PamparV3 — modelo activo** |
| `code_tokenizer.model` | 16K     | `deprecated/` v2 únicamente  |

## Naming Conventions

- **Español** para conceptos del dominio: `Talamo`, `Territorio`, `Zona`, `LLAVES`, `Agente`
- **Inglés** para ML estándar: `forward`, `embedding`, `hidden_states`, `loss`, `generate`
- **Config v3**: `ConfigV3` con presets `PRESET_V3`, `PRESET_V3_SMALL`, `PRESET_V3_LARGE`
- **Config v2 (deprecated)**: `ConfigV2` con `PRESET_4GB`, `PRESET_8GB`, `PRESET_24GB`, `PRESET_1_5B`

## Uso desde Python

```python
from pampar.coder.v3 import PamparV3, PRESET_V3
import torch

model = PamparV3(PRESET_V3)
model.eval()

# Forward
ids = torch.randint(0, 48_000, (1, 64))
logits, loss, info = model(ids)
# logits: [1, 64, 48000]  |  info: {"exit_nivel": int, "terr_acts": Tensor}

# Con loss
targets = ids.clone()
_, loss, _ = model(ids, targets=targets)

# Generación
gen = model.generate(ids, max_tokens=100, temperature=0.8, top_k=50, top_p=0.95)
```

## Workflow actual

```powershell
# 1. Siempre primero — tests completos
python -m pytest tests/ -v             # 109/109

# 2. Verificar inferencia end-to-end
python _test_inferencia.py

# 3. TODO: entrenamiento v3 (próxima fase)
# Requiere portar/crear scripts/train_v3.py con PamparV3 + pampar_48k.model
```

## Instructions Files

Detailed instructions in `.github/instructions/`:

- `global-profile.instructions.md` — perfil del desarrollador y preferencias generales
- `testing.instructions.md` — reglas de testing (pytest, fixtures, edge cases)
- `git-workflow.instructions.md` — commits convencionales, branching
- `docker-devops.instructions.md` — Docker multi-stage, CI/CD

- **Checkpoint**: `checkpoints/pampar_v2_best.pt` (PRESET_4GB — vocab 16K, dim 384)
- **Tokenizer activo**: `data/tokenizer/code_tokenizer.model` (16K vocab) — debe coincidir con el modelo
- **Entrenamiento**: `scripts/aprender_solo.py` — loop autónomo local con MotorCuriosidad
- **Biblioteca**: `biblioteca/` — 39 temas de Python (~140 MB), lista para entrenamiento
- **Tests**: 134/134 passing

## Project Structure

```
PAMPAr-Coder/
├── pampar/
│   └── coder/
│       └── v2/
│           ├── modelo.py          # PampaRCoderV2 — modelo principal
│           ├── config.py          # ConfigV2 + presets (PRESET_4GB, PRESET_8GB, PRESET_24GB, PRESET_1_5B)
│           ├── talamo.py          # TálamoBrodmann — orquestador central
│           ├── llaves.py          # LLAVES — routing basado en reglas (80%)
│           ├── zonas.py           # 52 zonas de Brodmann
│           ├── bloques.py         # BloqueTerritorial + FFN simbiótico
│           └── aprendizaje/
│               ├── curiosidad.py          # MotorCuriosidad — ZDP de Vygotsky
│               ├── memoria_jerarquica.py  # MemoriaJerarquica — L0/L1/L2 con Pareto
│               └── __init__.py
├── biblioteca/
│   ├── indice.json            # 40 temas en 5 categorías, niveles 1-6
│   └── *.jsonl                # Datos por tema (~140 MB total)
├── data/
│   ├── tokenizer/
│   │   ├── code_tokenizer.model  # ACTIVO — 16K vocab (modelo actual)
│   │   └── pampar_48k.model      # 48K vocab (para futuros modelos más grandes)
│   ├── code/                  # Código de GitHub (~143 MB)
│   └── distillation/          # Datos destilados (~2.8 GB)
├── checkpoints/
│   └── pampar_v2_best.pt      # ACTIVO — 42M params, vocab 16K
├── scripts/
│   ├── aprender_solo.py       # Loop de entrenamiento autónomo (PRINCIPAL)
│   ├── smoke_test_viaje.py    # Smoke test — correr antes de entrenar (12 checks)
│   ├── probar_modelo.py       # Prueba interactiva y automática del modelo
│   └── poblar_biblioteca.py   # Clasifica datos en temas para biblioteca/
└── tests/                     # 134 tests pytest
```

## Architecture Overview

PAMPAr-Coder usa una arquitectura inspirada en el cerebro humano:

1. **52 Zonas de Brodmann** — áreas especializadas de procesamiento de código
2. **4 Territorios** — SINTAXIS (1-15), SEMÁNTICA (16-30), LÓGICO (31-42), ESTRUCTURAL (43-52)
3. **LLAVES** — routing basado en reglas (INT8, 80% del peso de decisión)
4. **Tálamo** — orquestador central con ventana causal Conv1D (32 tokens)
5. **Fronteras Simbióticas** — 6 conexiones entre territorios con gates aprendidos
6. **Early Exit** — si confianza > 90%, salta capas restantes
7. **MotorCuriosidad** — Zona de Desarrollo Próximo de Vygotsky: elige qué estudiar
8. **MemoriaJerarquica** — L0 (reciente) → L1 (importante) → L2 (dominado), Pareto

## Aprendizaje Autónomo (Viaje Intelectual)

El modelo aprende solo, sin supervisión humana, usando:

- `MotorCuriosidad`: elige el siguiente tema maximizando `zona_proximal × novedad × spacing_effect`
  - Loss óptima de aprendizaje: ~1.5 (campana gaussiana)
  - Tema dominado cuando loss < 0.7 por 5 sesiones consecutivas
  - Sube de nivel cuando 70% de temas están dominados
- `LectorBiblioteca`: lee JSONL de `biblioteca/`, tokeniza, devuelve batches
- `MemoriaJerarquica`: guarda patrones difíciles para replay posterior
- Replay cada 50 pasos, consolidación L2→pesos cada 300 pasos

## Critical Rules

- **vocab_size del modelo DEBE coincidir con el tokenizer**: modelo actual = 16K → usar `code_tokenizer.model`
- LLAVES son patrones regex, NO se entrenan — nunca incluirlos en el grafo de gradientes
- Los territorios procesan en paralelo, luego se combinan via fronteras simbióticas
- `targets.reshape(-1)` siempre, nunca `.view(-1)` (puede fallar con tensores no-contiguos)
- `terr_acts` tiene shape `[B, L-1, 4]` (se computa sobre `input_ids[:, :-1]`) — clamp al acceder
- Al iterar `indice.json`, filtrar `if not isinstance(temas, list): continue` (tiene meta-keys)
- `MemoriaJerarquica.guardar(path)` / `.cargar(path)` — no `guardar_estado`/`cargar_estado`
- NO hay cloud, NO hay RunPod — todo corre local en la GPU del desarrollador

## Tokenizers disponibles

| Archivo                | Vocab | Usar con                                |
| ---------------------- | ----- | --------------------------------------- |
| `code_tokenizer.model` | 16K   | Modelos actuales (PRESET_4GB entrenado) |
| `pampar_48k.model`     | 48K   | Modelos futuros más grandes             |

## Naming Conventions

- **Español** para conceptos del dominio: `Talamo`, `Territorio`, `Zona`, `LLAVES`, `MotorCuriosidad`
- **Inglés** para ML estándar: `forward`, `embedding`, `hidden_states`, `loss`
- **Config**: `ConfigV2` con presets `PRESET_4GB`, `PRESET_8GB`, `PRESET_24GB`, `PRESET_1_5B`
- NO existe `PRESET_3B` — borrado del plan original por restricciones de hardware

## Workflow de entrenamiento

```powershell
# 1. Antes de entrenar — smoke test (12 checks, ~15 segundos)
python scripts/smoke_test_viaje.py --checkpoint checkpoints/pampar_v2_best.pt

# 2. Lanzar entrenamiento
python scripts/aprender_solo.py `
  --checkpoint checkpoints/pampar_v2_best.pt `
  --tokenizer data/tokenizer/code_tokenizer.model `
  --batch-size 2 --seq-len 512 --lr 5e-5

# 3. Probar el modelo (sin detener el entrenamiento)
python scripts/probar_modelo.py --auto
python scripts/probar_modelo.py --prompt "def fibonacci("
```

## Instructions Files

Detailed instructions in `.github/instructions/`:

- `pampar-architecture.instructions.md` - Brodmann zones, LLAVES, territories
