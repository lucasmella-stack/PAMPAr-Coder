# PAMPAr-Coder

> **Motor de razonamiento puro** — 108M params, local-first, RAG desde el dispositivo.

## Qué es PAMPAr-Coder

PAMPAr-Coder es un modelo de lenguaje de 108M parámetros que **razona sobre información de referencia** en lugar de memorizar respuestas. Funciona como un físico: entiende los axiomas fundamentales y puede derivar soluciones para cualquier dominio usando documentación disponible en el dispositivo.

- **Pesos**: capacidad de razonar (leer docs, entender problemas, derivar soluciones step-by-step)
- **Dispositivo**: conocimiento vía RAG local (docs de Python, MDN, man pages, archivos del usuario)
- **Hardware**: diseñado para correr en consumer hardware (GTX 1650, 4 GB VRAM)

**Estado actual**: `v3_ghidra_v9.pt` — Routing Score 89, eval 6/16 (38%). Sistema Classroom bio-inspirado operativo con 5 mecanismos de neurociencia real (Neuromodulación, LTP, Sleep Consolidation, Neurogenesis, Synaptic Pruning).

---

## Arquitectura 2D (PamparV3)

```
tok_emb [48K x 640]
  -> TalamoInicial (LLAVES 80% + attn_proj 20% + context_conv)
      -> terr_acts [B, L, 4]  /  zona_acts [B, L, 52]
  -> 4 streams paralelos (dim=640)

  NivelProfundo x5:
    1. GQA Atencion compartida (8 Q heads / 2 KV heads, head_dim=80)
    2. Re-routing ligero del Talamo
    3. 4 x StreamFFN SwiGLU independientes
    4. Lateral gates por stream (bottleneck=128)

  -> norm_f (RMSNorm) -> lm_head (weight-tied, vocab=48K)
```

### Los 4 Streams

| Stream          | Zonas Brodmann | Procesa                             |
| --------------- | -------------- | ----------------------------------- |
| **SINTAXIS**    | B01-B15        | Keywords, operadores, puntuacion    |
| **SEMANTICA**   | B16-B30        | Tipos, variables, literales         |
| **LOGICO**      | B31-B42        | Control flow, condicionales, bucles |
| **ESTRUCTURAL** | B43-B52        | Bloques, indentacion, scope         |

### Parametros

| Parametro        | Valor       |
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

## Innovaciones

### Sistema LLAVES (TalamoInicial)

- **80% reglas explicitas**: routing basado en patrones de codigo (INT8, pre-computado)
- **20% atencion aprendida**: ajuste fino para casos ambiguos
- Produce `terr_acts` y `zona_acts` sin overhead en inferencia

### Arquitectura Cortical 2D

- **4 streams x 5 niveles** = grilla donde filas especializan y columnas refinan
- **GQA 4:1**: menor VRAM, misma calidad
- **Lateral gates** (bottleneck 128): comunicacion entre streams como fibras blancas
- **Re-routing** por nivel: el Talamo adapta que stream lidera segun contexto acumulado

### RAG desde el dispositivo

El modelo usa la maquina donde esta instalado como fuente de conocimiento:

- Scanner detecta OS, paquetes, archivos disponibles
- RAGResidual indexa documentacion local (FAISS + sentence-transformers)
- El modelo razona sobre la referencia, no memoriza contenido

---

## Classroom — Entrenamiento Bio-Inspirado

Sistema de aprendizaje interactivo donde un modelo profesor (gpt-4o-mini) enseña a PamparV3 mediante lecciones progresivas, protegiendo el conocimiento previo.

### Mecanismos base

| Mecanismo                              | Proposito                                                                  |
| -------------------------------------- | -------------------------------------------------------------------------- |
| **EWC** (Elastic Weight Consolidation) | Protege pesos importantes — penaliza cambios en params criticos            |
| **Replay Buffer**                      | Mezcla ejemplos nuevos con anteriores (simula consolidacion durante sueño) |
| **LR Diferencial**                     | LLAVES/Talamo 0.01x, atencion 0.1x, embedding 0.1x, FFN 1.0x               |
| **Curriculum progresivo**              | 5 niveles: basico → avanzado, el profesor adapta la dificultad             |

### Bio-Mechanisms (`bio_mechanisms.py`)

5 mecanismos basados en neurociencia real, integrados como hook post-leccion:

| Mecanismo               | Inspiracion biologica      | Implementacion                                                                        |
| ----------------------- | -------------------------- | ------------------------------------------------------------------------------------- |
| **Neuromodulacion**     | Dopamina + Norepinefrina   | Modula LR dinamicamente segun exito/error (×0.3 a ×3.0)                               |
| **LTP**                 | Potenciacion a largo plazo | Fortalece `LateralGate.scale` de streams con alta activacion consistente (Hebb rule)  |
| **Sleep Consolidation** | Fases REM + SWS            | Replay periodico (cada 15 lecciones): aleatorio (REM) + ordenado por dificultad (SWS) |
| **Neurogenesis**        | Neuronas nuevas hipocampo  | Inyecta LoRA adapters (rank=8, ~10K params) en StreamFFN cuando loss > 4.0            |
| **Synaptic Pruning**    | Poda sinaptica (~50%)      | Reduce `LateralGate.scale < 0.03` cada 30 lecciones (decay ×0.5)                      |

Todos coordinados por `BioOrchestrator.after_lesson()`. Desactivables con `--no-bio`.

### Resultados del piloto (12 lecciones con bio-mechanisms)

- Loss bajando: 4.13 → 2.86 (aprendizaje activo)
- Neuromodulacion: LR factor oscila entre ×1.05 y ×1.16 segun performance
- LTP aplicada en L05 y L10 — `LateralGate.scale` diferenciados por nivel
- Sleep en L05 y L10 — consolidacion con loss ~2.8
- 1 LoRA adapter creado en L01 (loss > 4.0)
- Pruning en L10 — stream 3 nivel 0 reducido a scale=0.0114
- Modelo sigue generando correctamente post-bio

### Uso

```bash
# Con GitHub Models API (gpt-4o-mini, 20K req/min)
python scripts/classroom_server.py \
  --checkpoint checkpoints/v3_ghidra_v9.pt \
  --teacher github --model gpt-4o-mini \
  --max-lessons 10 --level 1

# Con interfaz web (SSE + dashboard)
python scripts/classroom_server.py \
  --checkpoint checkpoints/v3_ghidra_v9.pt \
  --teacher github --port 8787

# Replay de sesion grabada
# Abrir sessions/classroom_*.html en el navegador
```

---

## Subsistemas

| Modulo        | Componentes                 | Proposito                                                                 |
| ------------- | --------------------------- | ------------------------------------------------------------------------- |
| **Modelo**    | `pampar/coder/v3/`          | PamparV3: forward, generate, routing, bloques                             |
| **Memoria**   | `pampar/memoria/`           | ClasificadorPareto (L0-L3), RAGResidual (FAISS), ColaFinetune             |
| **Runtime**   | `pampar/runtime/`           | Agente (orquestador), Scanner (device), BootProtocol                      |
| **Skills**    | `pampar/skills/`            | LectorArchivos (30+ ext), EjecutorCodigo (subprocess)                     |
| **Inference** | `pampar/inference.py`       | Servidor JSON-lines stdin/stdout para VS Code                             |
| **Classroom** | `scripts/classroom*.py`     | Aula bio-inspirada modular (5 archivos ≤300 lineas + engine)              |
| **Bio-Mech**  | `scripts/bio_mechanisms.py` | 5 mecanismos de neurociencia: Neuromod, LTP, Sleep, Neurogenesis, Pruning |

---

## Instalacion

```bash
git clone https://github.com/lucasmella-stack/PAMPAr-Coder.git
cd PAMPAr-Coder
pip install -r requirements.txt
```

---

## Uso

### Instanciar el modelo

```python
from pampar.coder.v3 import PamparV3, PRESET_V3
import torch

model = PamparV3(PRESET_V3)
model.eval()

# Forward pass
ids = torch.randint(0, 48_000, (1, 64))
with torch.no_grad():
    logits, loss, info = model(ids)

# Generacion autoregresiva
gen = model.generate(ids, max_tokens=100, temperature=0.8, top_k=50)
```

### Usar el Agente (con RAG + Skills)

```python
from pampar.runtime import Agente

agente = Agente(
    checkpoint="checkpoints/v3_ghidra_v9.pt",
    workspace_root=".",
)
respuesta = agente.responder("como leer un CSV con pandas?")
```

---

## Estructura del Proyecto

```
PAMPAr-Coder/
+-- pampar/
|   +-- coder/v3/           # Arquitectura activa (108M)
|   |   +-- modelo.py       # PamparV3 -- forward, generate
|   |   +-- config.py       # ConfigV3 + presets
|   |   +-- talamo.py       # TalamoInicial -- routing
|   |   +-- bloques.py      # GQA, SwiGLU, LateralGate, NivelProfundo
|   |   +-- llaves.py       # LlavesV2 -- INT8 lookup
|   |   +-- zonas.py        # 52 Zonas de Brodmann
|   |   +-- ghidra_probe.py # Instrumentacion read-only
|   |   +-- engrama_stream.py # Memoria de activaciones
|   +-- memoria/
|   |   +-- clasificador.py # ClasificadorPareto (L0-L3)
|   |   +-- rag.py          # RAGResidual (FAISS + TF-IDF fallback)
|   |   +-- cola_finetune.py# ColaFinetune (buffer auto-SFT)
|   +-- skills/
|   |   +-- lector_archivos.py  # Lee archivos (sandboxed)
|   |   +-- ejecutar_codigo.py  # Ejecuta codigo (subprocess)
|   +-- runtime/
|   |   +-- agente.py       # Orquestador principal
|   |   +-- scanner.py      # Inspeccion del dispositivo
|   |   +-- boot.py         # Secuencia de arranque
|   |   +-- generar_agents.py # Generador de AGENTS.md
|   +-- inference.py        # Servidor JSON-lines para VS Code
+-- scripts/
|   +-- classroom.py              # ClassroomEngine (motor principal)
|   +-- classroom_curriculum.py   # Config + currículo 5 niveles
|   +-- classroom_teacher.py      # Teacher API (GitHub/OpenRouter)
|   +-- classroom_memory.py       # EWC + ReplayBuffer + LessonResult
|   +-- classroom_server.py       # HTTP SSE server + CLI (entry point)
|   +-- bio_mechanisms.py         # 5 mecanismos bio (Neuromod, LTP, Sleep, Neurogenesis, Pruning)
|   +-- classroom_replay.html     # Player HTML para replays de sesiones
+-- sessions/               # Grabaciones de sesiones classroom
+-- data/
|   +-- tokenizer/
|   |   +-- pampar_48k.model # Vocab 48K bilingue (activo)
|   +-- *.jsonl             # Datasets de training
+-- checkpoints/
|   +-- v3_ghidra_v9.pt     # Mejor checkpoint actual
+-- scripts/                # Training, eval, data generation
+-- _archive/               # Backups pre-refactorizacion
+-- tests/                  # Tests pytest
```

---

## Interpretar el Loss

| Loss  | Significado              |
| ----- | ------------------------ |
| ~10.7 | Sin entrenar (log 48000) |
| 7-8   | Pesos aleatorios         |
| 5-7   | Comenzando a aprender    |
| 2-4   | Aprendizaje activo       |
| 1.5-2 | Zona optima              |
| < 1.5 | Tema bien aprendido      |
| < 0.7 | Tema dominado            |

---

## Tests

```powershell
python -m pytest tests/ -v
```

---

## Filosofia

> _"No necesitas 72 billones de parametros. Necesitas la arquitectura correcta y los axiomas correctos."_

1. **Razonamiento > memorización** -- el modelo aprende a usar referencias, no a memorizar
2. **El dispositivo es la base de conocimiento** -- RAG local, no cloud
3. **El codigo es estructurado** -- 4 streams especializados + LLAVES 80% reglas
4. **Hardware consumer** -- 1.4 GB VRAM para training fp16

---

## Roadmap

- [x] Arquitectura territorial (52 zonas de Brodmann, 4 streams x 5 niveles)
- [x] Sistema LLAVES (routing INT8, 80% reglas)
- [x] Tokenizer BPE 48K bilingue (ES + codigo)
- [x] GQA 4:1, SwiGLU, lateral gates
- [x] Modulo memoria (ClasificadorPareto, RAG, ColaFinetune)
- [x] Skills (LectorArchivos, EjecutorCodigo)
- [x] Runtime.Agente (loop con herramientas)
- [x] GhidraProbe (diagnostico read-only)
- [x] EngramaStream (memoria de activaciones)
- [x] Classroom bio-inspirado (EWC, replay buffer, LR diferencial, curriculum)
- [x] Grabacion y replay HTML de sesiones classroom
- [x] Integracion GitHub Models API (gpt-4o-mini como profesor)
- [x] Bio-mechanisms: Neuromodulacion, LTP, Sleep Consolidation, Neurogenesis, Synaptic Pruning
- [ ] Expansion de training data (textbook + SFT multi-language)
- [ ] KV cache en generate()
- [ ] Multi-language execution (JS, Rust, Bash)
- [ ] Benchmarks vs modelos de referencia
- [ ] Integracion VS Code (extension)

---

## Licencia

AGPL-3.0-or-later -- Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
