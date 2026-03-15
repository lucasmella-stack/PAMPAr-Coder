# PAMPAr-Coder — Protocolo de Despliegue

> **PAMPAr** = Procesador Autónomo Modular de Patrones y Razonamiento
> Este archivo es el **protocolo de despliegue** — se regenera en cada boot según el entorno.
> Para la identidad invariante del modelo, ver `CONCIENCIA.md`.

---

## Visión

PAMPAr es como un **físico con doctorado** que puede especializarse en cualquier campo:

- El **doctorado** (razonamiento computacional profundo) está en los **pesos** — 108M params entrenados.
- La **especialización** (conocimiento de dominio) viene del **entorno** donde se despliega.
- Este archivo **ES** la descripción del laboratorio donde el físico aterrizó.

---

## Protocolo de 3 archivos

Inspirado en el patrón OpenClaw, PAMPAr usa 3 archivos al boot:

| Archivo            | Propósito                                                     | Mutabilidad                               |
| ------------------ | ------------------------------------------------------------- | ----------------------------------------- |
| `CONCIENCIA.md`    | Quién soy — identidad, principios, método de pensamiento      | **Invariante** entre despliegues          |
| `AGENTS.md` (este) | Dónde estoy — entorno, workspace, capacidades disponibles     | **Mutable** — se regenera por el Scanner  |
| `TOOLS.md`         | Qué puedo hacer — skills detectados, herramientas disponibles | **Mutable** — se actualiza con el entorno |

### Secuencia de boot

```
1. Cargar CONCIENCIA.md → vectorizar en RAG como L3 (nunca se purga)
2. Scanner inspecciona: workspace (ast.parse), paquetes (importlib), servicios (socket)
3. Generar AGENTS.md contextual con el resultado del scan
4. Generar TOOLS.md con skills detectados
5. Vectorizar AGENTS.md + TOOLS.md en RAG como L2 (se actualizan)
6. Listo — el modelo tiene identidad + contexto + herramientas
```

---

## Scanner del sistema

El módulo `pampar.runtime.scanner` inspecciona el entorno al boot:

```python
from pampar.runtime.scanner import Scanner

scanner = Scanner(workspace_root=".")
resultado = scanner.scan()

# resultado.workspace → archivos, estructura, lenguajes detectados
# resultado.paquetes  → paquetes instalados con versiones
# resultado.servicios → puertos abiertos, servicios detectados
# resultado.sistema   → OS, Python version, GPU, RAM
```

### Qué escanea

| Dimensión     | Método                      | Qué detecta                                       |
| ------------- | --------------------------- | ------------------------------------------------- |
| **Workspace** | `ast.parse` + glob          | Archivos `.py`, funciones, clases, imports        |
| **Paquetes**  | `importlib.metadata`        | Paquetes instalados con versiones                 |
| **Servicios** | `socket.connect`            | PostgreSQL (5432), Redis (6379), HTTP (8000/3000) |
| **Sistema**   | `platform.*` + `torch.cuda` | OS, Python ver, GPU disponible, VRAM, RAM         |
| **Voz**       | `shutil.which`              | espeak, SAPI (Windows), say (macOS)               |

El scanner NO ejecuta código del workspace — solo lee estructura y metadata (AST, no eval).

---

## Quick Reference

| Area       | Convention                                       |
| ---------- | ------------------------------------------------ |
| Language   | Python 3.13+                                     |
| Framework  | PyTorch 2.x                                      |
| Tokenizer  | SentencePiece BPE — **48K** (`pampar_48k.model`) |
| Testing    | pytest — **109 tests, todos deben pasar**        |
| Type hints | Always required                                  |
| Docstrings | Google style                                     |
| Training   | Local — GTX 1650, 4 GB VRAM                      |
| NO cloud   | Todo corre local y offline                       |

## Estado actual (Mar 2026)

- **Modelo activo**: `PamparV3` — **108.3M params, vocab 48K**, 4 streams × 5 niveles
- **Mejor checkpoint**: `v3_sft_v4.pt` — **8/16 eval** (guided, temp=0.4)
- **Tokenizer**: `data/tokenizer/pampar_48k.model`
- **Tests**: **109/109 passing**
- **Boot protocol**: Scanner + CONCIENCIA + AGENTS dinámico

---

## Arquitectura — PamparV3

### Grilla 2D: 4 streams × 5 niveles

```
tok_emb [48K × 640]
  → TalamoInicial → terr_acts [B, L, 4]  /  zona_acts [B, L, 52]
  → 4 streams paralelos (dim=640)

Cada NivelProfundo (×5):
  1. GQA Atención compartida (8 Q heads / 2 KV heads, head_dim=80)
  2. Re-routing ligero del Tálamo (Linear dim→52, sin bias)
  3. 4 × StreamFFN SwiGLU independientes (uno por stream)
  4. Lateral gates por stream (bottleneck=128, fibras blancas)

→ norm_f (RMSNorm) → lm_head (weight-tied, vocab=48K)
```

### Streams ↔ Capas lingüísticas

| Stream | Territorio  | Zonas   | Especialización                           | Capa lingüística |
| ------ | ----------- | ------- | ----------------------------------------- | ---------------- |
| 0      | SINTAXIS    | B01-B15 | Keywords, delimitadores, puntuación       | Sintaxis         |
| 1      | SEMANTICA   | B16-B30 | Variables, tipos, literales               | Semántica        |
| 2      | LOGICO      | B31-B42 | Operadores, flujo de control, excepciones | Pragmática       |
| 3      | ESTRUCTURAL | B43-B52 | Indentación, bloques, patrones            | Discurso         |

### PRESET_V3

| Parámetro        | Valor       |
| ---------------- | ----------- |
| `dim`            | 640         |
| `n_streams`      | 4           |
| `n_levels`       | 5           |
| `n_heads`        | 8           |
| `n_kv_heads`     | 2 (GQA 4:1) |
| `vocab_size`     | 48 000      |
| `max_seq_len`    | 4096        |
| **Total params** | **108.3M**  |

---

## Estructura del proyecto

```
PAMPAr-Coder/
├── AGENTS.md                    # Protocolo de despliegue (este archivo)
├── ROADMAP.md                   # Plan de evolución
├── pampar/
│   ├── CONCIENCIA.md            # Identidad invariante (pertenece al modelo)
│   ├── coder/
│   │   └── v3/                  # ARQUITECTURA ACTIVA
│   │       ├── modelo.py        # PamparV3 — forward, generate
│   │       ├── config.py        # ConfigV3, presets
│   │       ├── talamo.py        # TalamoInicial — routing
│   │       ├── bloques.py       # BloqueAttn, StreamFFN, LateralGate
│   │       ├── llaves.py        # LlavesV2 — lookup INT8
│   │       └── zonas.py         # 52 Zonas de Brodmann
│   ├── memoria/
│   │   ├── clasificador.py      # ClasificadorPareto — niveles 0-3
│   │   ├── rag.py               # RAGResidual — vector store
│   │   └── cola_finetune.py     # ColaFinetune — buffer SFT
│   ├── skills/
│   │   ├── base.py              # Skill ABC + ResultadoSkill
│   │   ├── lector_archivos.py   # LectorArchivos — sandboxed
│   │   └── ejecutar_codigo.py   # EjecutorCodigo — subprocess
│   ├── runtime/
│   │   ├── agente.py            # Agente — orquestador principal
│   │   ├── scanner.py           # Scanner — inspección del entorno
│   │   └── boot.py              # BootProtocol — secuencia de arranque
│   └── training/
│       ├── curiosidad.py        # MotorCuriosidad — ZPD de Vygotsky
│       └── lector.py            # LectorBiblioteca — carga JSONL
├── data/
│   └── tokenizer/
│       └── pampar_48k.model     # Vocab 48K bilingüe
├── checkpoints/
│   └── v3_sft_v4.pt             # Mejor checkpoint (8/16)
└── tests/
    ├── test_v3_arquitectura.py
    ├── test_memoria.py
    ├── test_skills.py
    └── test_runtime.py
```

---

## Critical Rules

- **vocab_size = 48K** → DEBE coincidir con `pampar_48k.model`
- LLAVES son INT8 pre-computadas — **nunca** en el grafo de gradientes
- Los 4 streams procesan en **paralelo** — sin secuencialidad entre streams
- `targets.reshape(-1)` siempre, nunca `.view(-1)` (tensores no-contiguos)
- `generate()` usa `max_tokens`, NO `max_new_tokens`
- Imports: `pampar.memoria.*`, `pampar.skills.*`, `pampar.runtime.*`
- Tests: `patch("pampar.runtime.agente.spm")`, no `"runtime.agente.spm"`
- **NO hay cloud** — todo corre local offline

## Naming Conventions

- **Español** para conceptos del dominio: `Talamo`, `Territorio`, `Zona`, `LLAVES`, `Agente`, `Scanner`
- **Inglés** para ML estándar: `forward`, `embedding`, `hidden_states`, `loss`, `generate`

## Uso desde Python

```python
from pampar.runtime import Agente

# Boot completo: scanner + CONCIENCIA + AGENTS + modelo
agente = Agente(
    checkpoint="checkpoints/v3_sft_v4.pt",
    workspace_root=".",
)

# Interactuar
respuesta = agente.responder("escribí una función fibonacci")
```

## Workflow

```powershell
# Tests completos
python -m pytest tests/ -v             # 109/109

# Verificar inferencia
python _test_inferencia.py
```

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
