# PAMPAr — Repository Guidelines

> **PAMPAr** = Procesador Autónomo Modular de Patrones y Razonamiento
> Para AI agents: Claude Code, Codex, Gemini CLI, GitHub Copilot.

---

## Visión

PAMPAr es un **motor de razonamiento puro** de 108M parámetros. No memoriza respuestas — aprende a **pensar con información de referencia**.

La analogía: un físico que entiende termodinámica puede resolver problemas de química, ingeniería o biología. No memorizó cada campo — tiene los axiomas correctos.

- Los **pesos** (108M params) contienen la capacidad de **razonar**: leer documentación, entender un problema, derivar una solución step-by-step.
- El **dispositivo** (PC, móvil, servidor) provee el **conocimiento**: docs de Python, MDN, man pages, archivos del usuario — vía RAG local.
- El modelo no necesita "saber Python". Necesita saber **usar la referencia que tiene disponible** para resolver cualquier problema.

**Objetivo**: un modelo local que razona con la misma metodología que los mejores modelos, usando la información del dispositivo como RAG.

---

## Estado actual

- **Modelo activo**: `PamparV3` — **108.3M params**, vocab 48K, 4 streams × 5 niveles
- **Mejor checkpoint**: `v3_ghidra_v9.pt` — Routing Score 89, eval 6/16 (38%)
- **Tokenizer**: `data/tokenizer/pampar_48k.model` (48K, bilingüe ES+código)
- **Runtime**: Agente + RAGResidual + Scanner + BootProtocol — funcional
- **Classroom**: Mentor conversacional con Qwen-plus — lecciones dinámicas, 21 conceptos adaptativos, absorción + práctica + corrección
- **Bio-Mechanisms**: Neuromodulación, LTP, Sleep Consolidation, Neurogenesis, Synaptic Pruning — `bio_mechanisms.py`
- **Teacher API**: Qwen-plus via DashScope (principal), GitHub Models gpt-4o-mini (alternativa)
- **Training data**: `master_sft.jsonl` — 1,253 ejemplos (en expansión vía Classroom)

---

## Quick Reference

| Area       | Convention                                             |
| ---------- | ------------------------------------------------------ |
| Language   | Python 3.13+                                           |
| Framework  | PyTorch 2.6+                                           |
| Tokenizer  | SentencePiece BPE — **48K** (`pampar_48k.model`)       |
| Type hints | Always required                                        |
| Docstrings | Google style                                           |
| Training   | Local GTX 1650 (4 GB) + RunPod A100 para fases pesadas |
| Budget     | $300-500 USD total                                     |

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

## Subsistemas

### 1. Modelo (`pampar/coder/v3/`)

| Archivo             | Líneas | Propósito                                                                       |
| ------------------- | ------ | ------------------------------------------------------------------------------- |
| `modelo.py`         | 310    | PamparV3: forward, generate (nucleus sampling)                                  |
| `config.py`         | 226    | ConfigV3, 3 presets (V3/SMALL/LARGE)                                            |
| `bloques.py`        | 395    | RMSNorm, RoPE, BloqueAttn (GQA), StreamFFN (SwiGLU), LateralGate, NivelProfundo |
| `talamo.py`         | 133    | TalamoInicial: LLAVES 80% + attn_proj 20% + context_conv                        |
| `llaves.py`         | 266    | LlavesV2: clasificar_token(), tabla INT8, agregar_zonas_a_territorios           |
| `zonas.py`          | 265    | Territorio(IntEnum), Zona(IntEnum), ZONAS dict, ZONA_TERRITORIO                 |
| `ghidra_probe.py`   | 343    | GhidraProbe: 36 forward hooks, diagnosis/debugging                              |
| `engrama_stream.py` | 359    | BancoEngrama: O(1) activation memory, cosine-gated injection                    |

### 2. Memoria (`pampar/memoria/`)

| Archivo            | Propósito                                                                     |
| ------------------ | ----------------------------------------------------------------------------- |
| `clasificador.py`  | ClasificadorPareto: scoring L0-L3 por densidad, novedad, loss, frecuencia     |
| `rag.py`           | RAGResidual: FAISS + sentence-transformers (fallback TF-IDF), 5K entradas max |
| `cola_finetune.py` | ColaFinetune: acumula L3, exporta JSONL, propone mini-SFT                     |

### 3. Runtime (`pampar/runtime/`)

| Archivo             | Propósito                                                          |
| ------------------- | ------------------------------------------------------------------ |
| `agente.py`         | Orquestador: prompt→RAG→generar→skills→retry→auto-SFT              |
| `scanner.py`        | Inspección del dispositivo: OS, GPU, paquetes, servicios, archivos |
| `boot.py`           | BootProtocol: CONCIENCIA.md (L3) → Scanner (L2) → Workspace (L1)   |
| `generar_agents.py` | Genera AGENTS.md contextual desde ResultadoScan                    |

### 4. Skills (`pampar/skills/`)

| Archivo              | Propósito                                                 |
| -------------------- | --------------------------------------------------------- |
| `lector_archivos.py` | Lee archivos del dispositivo (30+ extensiones, sandboxed) |
| `ejecutar_codigo.py` | Ejecuta código en subprocess con timeout y blocklist      |

### 5. Inference (`pampar/inference.py`)

Servidor JSON-lines stdin/stdout para extensión VS Code. Commands: `infer`, `boot`.

### 6. Classroom — Mentor Conversacional + Bio-Mechanisms

Sistema donde Qwen-plus actúa como mentor conversacional — genera explicaciones, ejemplos y ejercicios dinámicos. PamparV3 absorbe el conocimiento via gradient descent en 3 phases por lección.

**Flujo**: StudentProfile → Mentor genera lección → Phase A (absorber explicación+ejemplo) → Phase B (alumno intenta ejercicio) → Phase C (mentor corrige, entrenar en solución+replay) → actualizar perfil.

| Módulo                     | Líneas | Responsabilidad                                                                |
| -------------------------- | ------ | ------------------------------------------------------------------------------ |
| `classroom.py`             | ~608   | ClassroomEngine — motor conversacional (orquestador)                           |
| `classroom_curriculum.py`  | ~433   | ClassroomConfig + CONCEPT_TREE (21 conceptos) + StudentProfile + concept_level |
| `classroom_teacher.py`     | ~252   | Mentor API (Qwen/GitHub/OpenRouter) + parse de lecciones                       |
| `classroom_training.py`    | ~211   | Tokenización + LR diferencial + train_step                                     |
| `classroom_memory.py`      | ~187   | EWC + ReplayBuffer + LessonResult + compute_ewc_baseline                       |
| `classroom_events.py`      | ~104   | Formateo dict-based de eventos para consola                                    |
| `classroom_persistence.py` | ~123   | Guardado de checkpoints, sesiones JSONL, grabaciones HTML                      |
| `classroom_server.py`      | ~255   | HTTP SSE server + CLI entry point                                              |
| `bio_mechanisms.py`        | ~497   | 5 bio-mechanisms coordinados por BioOrchestrator                               |

**CONCEPT_TREE**: 21 conceptos en 5 niveles con prerequisitos (arithmetic → algorithms).
**StudentProfile**: mastery tracking adaptativo — prioriza refuerzo, luego nuevos, luego repaso.

| Mecanismo          | Propósito                                                         |
| ------------------ | ----------------------------------------------------------------- |
| **EWC**            | Elastic Weight Consolidation — penaliza cambios en pesos críticos |
| **Replay Buffer**  | Mezcla ejemplos nuevos con anteriores (consolidación tipo sueño)  |
| **LR Diferencial** | LLAVES 0.01x, atención 0.1x, embed 0.1x, FFN 1.0x                 |
| **Curriculum**     | 5 niveles progresivos: básico → avanzado                          |
| **Grabación**      | Genera HTML con replay interactivo de cada sesión                 |

**Bio-Mechanisms** (5 mecanismos de neurociencia en `bio_mechanisms.py`):

| Mecanismo               | Implementación                                                       |
| ----------------------- | -------------------------------------------------------------------- |
| **Neuromodulación**     | Dopamina/Norepinefrina modulan LR dinámicamente (×0.3 a ×3.0)        |
| **LTP**                 | Fortalece `LateralGate.scale` de streams activos (Hebb rule, cada 5) |
| **Sleep Consolidation** | REM (aleatorio) + SWS (ordenado por dificultad), cada 15 lecciones   |
| **Neurogenesis**        | LoRA adapters (rank=8) en StreamFFN cuando loss > 4.0, max 8         |
| **Synaptic Pruning**    | Poda `LateralGate.scale < 0.03` cada 30 lecciones (decay ×0.5)       |

Coordinados por `BioOrchestrator.after_lesson()`. Desactivables con `--no-bio`.

**Resultados piloto mentor conversacional (5 lecciones)**: Loss absorción ~7-8, loss ejercicios 5.89→3.94 (mejora), brain score 88.24% estable.

**APIs soportadas**: `qwen` (Qwen-plus via DashScope, principal), `github` (gpt-4o-mini), `openrouter` (requiere créditos).

---

## Estructura del proyecto

```
PAMPAr-Coder/
├── AGENTS.md                    # Este archivo — guía para AI agents
├── README.md                    # Documentación pública
├── PLAN.md                      # Plan de training y evolución
├── pampar/
│   ├── CONCIENCIA.md            # Identidad invariante del modelo
│   ├── coder/
│   │   └── v3/                  # ARQUITECTURA ACTIVA (108M)
│   │       ├── modelo.py        # PamparV3 — forward, generate
│   │       ├── config.py        # ConfigV3, presets
│   │       ├── talamo.py        # TalamoInicial — routing
│   │       ├── bloques.py       # GQA, SwiGLU, LateralGate, NivelProfundo
│   │       ├── llaves.py        # LlavesV2 — lookup INT8
│   │       ├── zonas.py         # 52 Zonas de Brodmann
│   │       ├── ghidra_probe.py  # Instrumentación read-only
│   │       └── engrama_stream.py# Memoria de activaciones
│   ├── memoria/
│   │   ├── clasificador.py      # ClasificadorPareto — niveles L0-L3
│   │   ├── rag.py               # RAGResidual — vector store local
│   │   └── cola_finetune.py     # ColaFinetune — buffer auto-SFT
│   ├── skills/
│   │   ├── lector_archivos.py   # Lee archivos (sandboxed)
│   │   └── ejecutar_codigo.py   # Ejecuta código (subprocess)
│   ├── runtime/
│   │   ├── agente.py            # Orquestador principal
│   │   ├── scanner.py           # Inspección del dispositivo
│   │   ├── boot.py              # Secuencia de arranque
│   │   └── generar_agents.py    # Generador de AGENTS.md
│   └── inference.py             # Servidor JSON-lines para VS Code
├── scripts/
│   ├── classroom.py             # ClassroomEngine — motor conversacional (~608 líneas)
│   ├── classroom_curriculum.py  # ClassroomConfig + CONCEPT_TREE + StudentProfile + concept_level
│   ├── classroom_teacher.py     # Mentor API — Qwen/GitHub/OpenRouter + parse de lecciones
│   ├── classroom_training.py    # Tokenización + LR diferencial + train_step
│   ├── classroom_events.py      # Formateo dict-based de eventos para consola
│   ├── classroom_memory.py      # EWC + ReplayBuffer + LessonResult + compute_ewc_baseline
│   ├── classroom_persistence.py # Guardado de checkpoints, sesiones, grabaciones HTML
│   ├── classroom_server.py      # HTTP SSE server + CLI entry point
│   ├── bio_mechanisms.py        # 5 bio-mechanisms (Neuromod, LTP, Sleep, Neurogenesis, Pruning)
│   └── classroom_replay.html    # Player HTML para replays
├── sessions/                    # Grabaciones de sesiones classroom
├── data/
│   ├── tokenizer/
│   │   └── pampar_48k.model     # Vocab 48K bilingüe
│   └── *.jsonl                  # Datasets de training
├── checkpoints/
│   └── v3_ghidra_v9.pt          # Mejor checkpoint actual
├── _archive/                    # Backups de archivos antes de refactorizar
└── tests/
```

---

## Critical Rules

- **vocab_size = 48K** → DEBE coincidir con `pampar_48k.model`
- **Tokenizer path**: usar `PRESET_V3.tokenizer_path` o constante compartida — no hardcodear
- LLAVES son INT8 pre-computadas — **nunca** en el grafo de gradientes
- Los 4 streams procesan en **paralelo** — sin secuencialidad entre streams
- `targets.reshape(-1)` siempre, nunca `.view(-1)` (tensores no-contiguos)
- `generate()` usa `max_tokens`, NO `max_new_tokens`
- Imports: `pampar.memoria.*`, `pampar.skills.*`, `pampar.runtime.*`
- **Backups**: antes de borrar/refactorizar, mover el original a `_archive/`

## Naming Conventions

- **Español** para conceptos del dominio: `Talamo`, `Territorio`, `Zona`, `LLAVES`, `Agente`, `Scanner`
- **Inglés** para ML estándar: `forward`, `embedding`, `hidden_states`, `loss`, `generate`

## Paradigma de inferencia

```
1. Usuario hace una pregunta/pedido
2. Scanner provee contexto del dispositivo (OS, paquetes, archivos)
3. RAGResidual busca referencia relevante (docs, código, memoria)
4. Prompt se arma: [SYSTEM] + [REFERENCIA RAG] + [CONTEXTO DISPOSITIVO] + [PREGUNTA]
5. Modelo RAZONA sobre la referencia y genera solución step-by-step
6. Skills ejecutan la solución si aplica (código, lectura, tests)
7. Si falla → retry con error como contexto → ColaFinetune acumula patrones
```

---

## Instructions Files

Detailed instructions in `.github/instructions/`:

- `global-profile.instructions.md` — perfil del desarrollador
- `testing.instructions.md` — reglas de testing (pytest)
- `git-workflow.instructions.md` — commits convencionales
- `docker-devops.instructions.md` — Docker, CI/CD
