<p align="center">
  <img src="PAMPAR-coder.png" alt="PAMPAr-Coder" width="200" />
</p>

<h1 align="center">PAMPAr-Coder</h1>

<p align="center">
  <strong>Pure reasoning engine</strong> — 62.6M params, local-first, on-device RAG.
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-BUSL--1.1-blue" alt="License" /></a>
  <img src="https://img.shields.io/badge/params-62.6M-green" alt="Params" />
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python" />
  <img src="https://img.shields.io/badge/pytorch-2.x-orange" alt="PyTorch" />
</p>

---

## What is PAMPAr-Coder

PAMPAr-Coder is a 62.6M parameter language model that **reasons over reference information** rather than memorizing answers. It works like a physicist: it understands the fundamental axioms and can derive solutions for any domain using documentation available on the device.

- **Weights**: reasoning capability (read docs, understand problems, derive solutions step-by-step)
- **Device**: knowledge via local RAG (Python docs, MDN, man pages, user files)
- **Hardware**: designed to run on consumer hardware (GTX 1650, 4 GB VRAM)

**Current state**: `v3_train.pt` — 98K steps, Mixed Selectivity (FiLM). Classroom system with conversational mentor (Qwen-plus) + 5 bio-inspired mechanisms. Tree of 21 concepts with adaptive prerequisites.

---

## 2D Architecture (PamparV3)

```
tok_emb [48K x 640]
  -> TalamoInicial (LLAVES 80% + attn_proj 20% + context_conv)
      -> terr_acts [B, L, 4]  /  zona_acts [B, L, 52]
  -> 4 parallel streams (dim=640)

  NivelProfundo x5:
    1. Shared GQA Attention (8 Q heads / 2 KV heads, head_dim=80)
    2. Lightweight Thalamus re-routing
    3. 4 x independent StreamFFN SwiGLU
    4. Lateral gates per stream (bottleneck=128)

  -> norm_f (RMSNorm) -> lm_head (weight-tied, vocab=48K)
```

### The 4 Streams

| Stream         | Brodmann Zones | Processes                          |
| -------------- | -------------- | ---------------------------------- |
| **SYNTAX**     | B01-B15        | Keywords, operators, punctuation   |
| **SEMANTICS**  | B16-B30        | Types, variables, literals         |
| **LOGIC**      | B31-B42        | Control flow, conditionals, loops  |
| **STRUCTURAL** | B43-B52        | Blocks, indentation, scope         |

### Parameters

| Parameter        | Value       |
| ---------------- | ----------- |
| `dim`            | 640         |
| `n_streams`      | 4           |
| `n_levels`       | 5           |
| `n_heads`        | 8           |
| `n_kv_heads`     | 2 (GQA 4:1) |
| `vocab_size`     | 48,000      |
| `max_seq_len`    | 4096        |
| **Total params** | **62.6M**   |

---

## Key Innovations

### LLAVES System (TalamoInicial)

- **80% explicit rules**: routing based on code patterns (INT8, pre-computed)
- **20% learned attention**: fine-tuning for ambiguous cases
- Produces `terr_acts` and `zona_acts` with zero inference overhead

### 2D Cortical Architecture

- **4 streams × 5 levels** = grid where rows specialize and columns refine
- **GQA 4:1**: lower VRAM, same quality
- **Lateral gates** (bottleneck 128): cross-stream communication like white-matter fibers
- **Re-routing** per level: the Thalamus adapts which stream leads based on accumulated context

### On-Device RAG

The model uses the machine where it's installed as its knowledge source:

- Scanner detects OS, packages, available files
- RAGResidual indexes local documentation (FAISS + sentence-transformers)
- The model reasons over references, it doesn't memorize content

---

## Classroom — Conversational Mentor + Bio-Mechanisms

A learning system where a mentor model (Qwen-plus via DashScope) teaches PamparV3 through dynamic conversations, like a tutor in a chat. The mentor generates unique explanations, examples, and exercises for each lesson — the student absorbs knowledge via gradient descent.

### Lesson Flow

```
1. StudentProfile selects adaptive concept (21 concepts with prerequisites)
2. Mentor generates lesson: explanation + example + exercise + solution
3. Phase A — Absorb: train on explanation + example (all tokens)
4. Phase B — Practice: student attempts the exercise
5. Phase C — Correct: mentor evaluates, train on correct solution + replay
6. Update student profile (mastery per concept)
```

### Concept Tree (CONCEPT_TREE)

21 concepts organized in 5 levels with prerequisites:

| Level | Concepts                                                              |
| ----- | --------------------------------------------------------------------- |
| 1     | arithmetic → variables_types → conditionals, strings, functions_basic |
| 2     | loops_for → loops_while, lists → tuples_sets, dicts                   |
| 3     | recursion, higher_order, generators, error_handling                   |
| 4     | classes_basic → inheritance, dunder_methods                           |
| 5     | decorators, context_managers, algorithms, file_io                     |

`StudentProfile` tracks mastery per concept and selects adaptively:

- Prioritizes concepts with attempts but not yet mastered (reinforcement)
- Then new concepts whose prerequisites are met
- Finally spaced review of mastered concepts

### Core Mechanisms

| Mechanism                              | Purpose                                                                 |
| -------------------------------------- | ----------------------------------------------------------------------- |
| **EWC** (Elastic Weight Consolidation) | Protects important weights — penalizes changes to critical params       |
| **Replay Buffer**                      | Mixes new and previous examples (simulates sleep consolidation)         |
| **Differential LR**                    | LLAVES/Thalamus 0.01×, attention 0.1×, embedding 0.1×, FFN 1.0×        |
| **Conversational Absorption**          | Trains on mentor explanations + examples (knowledge distillation)       |

### Bio-Mechanisms (`bio_mechanisms.py`)

5 mechanisms based on real neuroscience, integrated as post-lesson hooks:

| Mechanism              | Biological Inspiration     | Implementation                                                                       |
| ---------------------- | -------------------------- | ------------------------------------------------------------------------------------ |
| **Neuromodulation**    | Dopamine + Norepinephrine  | Dynamically modulates LR based on success/error (×0.3 to ×3.0)                       |
| **LTP**                | Long-term potentiation     | Strengthens `LateralGate.scale` of streams with consistent high activation (Hebb rule)|
| **Sleep Consolidation**| REM + SWS phases           | Periodic replay (every 15 lessons): random (REM) + sorted by difficulty (SWS)        |
| **Neurogenesis**       | New hippocampal neurons    | Injects LoRA adapters (rank=8, ~10K params) into StreamFFN when loss > 4.0           |
| **Synaptic Pruning**   | Synaptic pruning (~50%)    | Reduces `LateralGate.scale < 0.03` every 30 lessons (decay ×0.5)                     |

All coordinated by `BioOrchestrator.after_lesson()`. Can be disabled with `--no-bio`.

### Mentor Pilot Results (5 lessons)

- Absorption loss: ~7-8 (new content from mentor)
- Exercise loss decreasing: 5.89 → 5.44 → 4.40 → 3.94 → 4.38
- Brain score stable: 88.24% (prior knowledge preservation)
- EWC penalty growing: 0.000002 → 0.000044 (active regularization)
- Each lesson is UNIQUE — mentor generates dynamically, no repetition

### Usage

```bash
# Conversational mentor with Qwen-plus (recommended)
python scripts/classroom_server.py \
  --checkpoint checkpoints/v3_train.pt \
  --checkpoint-out checkpoints/v3_classroom_mentor.pt \
  --teacher qwen --model qwen-plus \
  --max-lessons 200 --lr 1e-5 --ewc-lambda 50 --no-bio --no-ui

# With bio-inspired mechanisms enabled
python scripts/classroom_server.py \
  --checkpoint checkpoints/v3_train.pt \
  --teacher qwen --model qwen-plus \
  --max-lessons 200 --lr 1e-5

# With web interface (SSE + dashboard)
python scripts/classroom_server.py \
  --checkpoint checkpoints/v3_train.pt \
  --teacher qwen --port 8787

# With GitHub Models API (alternative)
python scripts/classroom_server.py \
  --checkpoint checkpoints/v3_train.pt \
  --teacher github --model gpt-4o-mini

# Replay a recorded session
# Open sessions/classroom_*.html in browser
```

---

## Subsystems

| Module        | Components                  | Purpose                                                                                        |
| ------------- | --------------------------- | ---------------------------------------------------------------------------------------------- |
| **Model**     | `pampar/coder/v3/`          | PamparV3: forward, generate, routing, blocks                                                   |
| **Memory**    | `pampar/memoria/`           | ClasificadorPareto (L0-L3), RAGResidual (FAISS), ColaFinetune                                  |
| **Runtime**   | `pampar/runtime/`           | Agent (orchestrator), Scanner (device), BootProtocol                                           |
| **Skills**    | `pampar/skills/`            | LectorArchivos (30+ ext), EjecutorCodigo (subprocess)                                          |
| **Inference** | `pampar/inference.py`       | JSON-lines stdin/stdout server for VS Code                                                     |
| **Classroom** | `scripts/classroom*.py`     | Conversational mentor: engine + teacher + curriculum + training + events + memory + persistence |
| **Bio-Mech**  | `scripts/bio_mechanisms.py` | 5 neuroscience mechanisms: Neuromod, LTP, Sleep, Neurogenesis, Pruning                         |

---

## Installation

```bash
git clone https://github.com/lucasmella-stack/PAMPAr-Coder.git
cd PAMPAr-Coder
pip install -r requirements.txt
```

---

## Usage

### Instantiate the model

```python
from pampar.coder.v3 import PamparV3, PRESET_V3
import torch

model = PamparV3(PRESET_V3)
model.eval()

# Forward pass
ids = torch.randint(0, 48_000, (1, 64))
with torch.no_grad():
    logits, loss, info = model(ids)

# Autoregressive generation
gen = model.generate(ids, max_tokens=100, temperature=0.8, top_k=50)
```

### Use the Agent (with RAG + Skills)

```python
from pampar.runtime import Agente

agent = Agente(
    checkpoint="checkpoints/v3_train.pt",
    workspace_root=".",
)
response = agent.responder("how to read a CSV with pandas?")
```

---

## Project Structure

```
PAMPAr-Coder/
├── pampar/
│   ├── coder/v3/              # Active architecture (62.6M)
│   │   ├── modelo.py          # PamparV3 — forward, generate
│   │   ├── config.py          # ConfigV3 + presets
│   │   ├── talamo.py          # TalamoInicial — routing
│   │   ├── bloques.py         # GQA, SwiGLU, LateralGate, NivelProfundo
│   │   ├── llaves.py          # LlavesV2 — INT8 lookup
│   │   ├── zonas.py           # 52 Brodmann Zones
│   │   ├── ghidra_probe.py    # Read-only instrumentation
│   │   └── engrama_stream.py  # Activation memory
│   ├── memoria/
│   │   ├── clasificador.py    # ClasificadorPareto (L0-L3)
│   │   ├── rag.py             # RAGResidual (FAISS + TF-IDF fallback)
│   │   └── cola_finetune.py   # ColaFinetune (auto-SFT buffer)
│   ├── skills/
│   │   ├── lector_archivos.py # File reader (sandboxed)
│   │   └── ejecutar_codigo.py # Code executor (subprocess)
│   ├── runtime/
│   │   ├── agente.py          # Main orchestrator
│   │   ├── scanner.py         # Device inspection
│   │   └── boot.py            # Boot sequence
│   └── inference.py           # JSON-lines server for VS Code
├── scripts/
│   ├── classroom.py           # ClassroomEngine (~600 lines)
│   ├── classroom_curriculum.py# CONCEPT_TREE (21 concepts) + StudentProfile
│   ├── classroom_teacher.py   # Mentor API (GitHub/OpenRouter/Qwen)
│   ├── classroom_training.py  # Tokenization + differential LR + train_step
│   ├── classroom_events.py    # Console event formatting
│   ├── classroom_memory.py    # EWC + ReplayBuffer + compute_ewc_baseline
│   ├── classroom_persistence.py # Checkpoint + session + HTML recording save
│   ├── classroom_server.py    # HTTP SSE server + CLI (entry point)
│   └── bio_mechanisms.py      # 5 bio mechanisms
├── data/tokenizer/
│   └── pampar_48k.model       # 48K bilingual vocab (active)
├── checkpoints/               # Model checkpoints (gitignored)
├── tests/                     # pytest test suite
└── _archive/                  # Pre-refactoring backups
```

---

## Understanding the Loss

| Loss  | Meaning                        |
| ----- | ------------------------------ |
| ~10.7 | Untrained (log 48000)          |
| 7-8   | Random weights                 |
| 5-7   | Beginning to learn             |
| 2-4   | Active learning                |
| 1.5-2 | Optimal zone                   |
| < 1.5 | Topic well learned             |
| < 0.7 | Topic mastered                 |

---

## Tests

```bash
python -m pytest tests/ -v
```

142 tests, all passing.

---

## Philosophy

> _"You don't need 72 billion parameters. You need the right architecture and the right axioms."_

1. **Reasoning > memorization** — the model learns to use references, not to memorize
2. **The device is the knowledge base** — local RAG, not cloud
3. **Code is structured** — 4 specialized streams + LLAVES 80% rules
4. **Consumer hardware** — 1.4 GB VRAM for fp16 training

---

## Roadmap

- [x] Territorial architecture (52 Brodmann zones, 4 streams × 5 levels)
- [x] LLAVES system (INT8 routing, 80% rules)
- [x] BPE 48K bilingual tokenizer (ES + code)
- [x] GQA 4:1, SwiGLU, lateral gates
- [x] Memory module (ClasificadorPareto, RAG, ColaFinetune)
- [x] Skills (LectorArchivos, EjecutorCodigo)
- [x] Runtime.Agent (tool-use loop)
- [x] GhidraProbe (read-only diagnostics)
- [x] EngramaStream (activation memory)
- [x] Bio-inspired Classroom (EWC, replay buffer, differential LR, curriculum)
- [x] HTML session recording and replay
- [x] GitHub Models API integration (gpt-4o-mini as teacher)
- [x] Bio-mechanisms: Neuromodulation, LTP, Sleep Consolidation, Neurogenesis, Synaptic Pruning
- [x] Conversational mentor: Qwen-plus generates dynamic lessons as tutor
- [x] CONCEPT_TREE: 21 concepts with adaptive prerequisites
- [x] StudentProfile: per-concept mastery tracking
- [x] Loss masking: -100 on prompt tokens (train only on responses)
- [x] Conversational absorption: train on mentor explanations + examples
- [ ] Multimodal: image/diagram input support
- [ ] Training data expansion (textbook + SFT multi-language)
- [ ] KV cache in generate()
- [ ] Multi-language execution (JS, Rust, Bash)
- [ ] Benchmarks against reference models
- [ ] VS Code extension

---

## License

BUSL-1.1 — Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi

Change Date: April 7, 2030 — License converts to Apache-2.0. See [LICENSE](LICENSE) for details.
