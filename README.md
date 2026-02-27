# PAMPAr-Coder 🧠⚡

> **"Una grilla 2D de streams corticales donde cada token es clasificado por el Tálamo y refinado en paralelo por 4 streams especializados a lo largo de 5 niveles de profundidad."**

## ¿Qué es PAMPAr-Coder?

PAMPAr-Coder es un **modelo de lenguaje cerebral especializado en programación**, diseñado para correr en hardware consumer (GTX 1650, 4 GB VRAM) y **aprender de forma completamente autónoma** a través de un sistema de curiosidad inspirado en Vygotsky.

**Estado actual: PamparV3 — 108.3M params, vocab 48K, arquitectura lista para entrenamiento.**

---

## Arquitectura 2D (v3)

```
Input [B, L]
  → tok_emb [48K × 640]
    → TalamoInicial  (LLAVES 80% + attn_proj + context_conv)
        ↓ terr_acts [B, L, 4]   ↓ zona_acts [B, L, 52]
    → 4 streams (dim=640 cada uno, inicializados desde tok_emb)

    ┌────────────── NivelProfundo × 5 ──────────────┐
    │                                               │
    │  GQA Atención compartida (8Q / 2KV heads)     │
    │  Re-routing ligero del Tálamo                 │
    │                                               │
    │  ┌──────────┬──────────┬──────────┬──────────┐│
    │  │ SINTAXIS │SEMÁNTICA │  LÓGICO  │ESTRUCTUR.││
    │  │  SwiGLU  │  SwiGLU  │  SwiGLU  │  SwiGLU  ││
    │  └────┬─────┴────┬─────┴────┬─────┴────┬─────┘│
    │       └──── Lateral Gates (fibras blancas) ───┘│
    └───────────────────────────────────────────────┘

  → norm_f (RMSNorm) → lm_head (weight-tied con tok_emb)
```

### Los 4 Streams / Territorios

| Stream          | Zonas de Brodmann | Procesa                                |
| --------------- | ----------------- | -------------------------------------- |
| **SINTAXIS**    | 1–15              | Keywords, operadores, puntuación       |
| **SEMÁNTICA**   | 16–30             | Tipos, nombres de variables, literales |
| **LÓGICO**      | 31–42             | Control flow, condicionales, bucles    |
| **ESTRUCTURAL** | 43–52             | Bloques, indentación, scope            |

---

## Innovaciones Clave

### 🔑 Sistema LLAVES (TalamoInicial)

- **80% reglas explícitas**: routing basado en patrones de código (INT8, pre-computado)
- **20% atención aprendida**: ajuste fino para casos ambiguos
- Produce `terr_acts` [B, L, 4] y `zona_acts` [B, L, 52] — sin overhead en inferencia

### 🧠 Arquitectura Cortical 2D

- **4 streams** × **5 niveles** = grilla donde filas especializan y columnas refinan
- **GQA 4:1** (8 Q heads, 2 KV heads): menor VRAM, misma calidad
- **Lateral gates** (bottleneck 128): los streams se comunican como fibras blancas
- **Re-routing** por nivel: el Tálamo adapta qué stream lidera según contexto acumulado

### ⚡ Early Exit

Si la confianza de un token > 90%, salta los niveles restantes — el código es estructuralmente predecible.

### 🛠️ Agente con Skills

El módulo `runtime.Agente` puede leer archivos, ejecutar código y correr tests — usando los outputs del modelo para tomar decisiones reales.

---

## Hardware y Presets

| Preset            | Params   | VRAM fp16  | VRAM training | vocab | Uso                          |
| ----------------- | -------- | ---------- | ------------- | ----- | ---------------------------- |
| `PRESET_V3_SMALL` | ~60M     | ~115 MB    | ~810 MB       | 48K   | Experimentación rápida       |
| `PRESET_V3`       | **108M** | **200 MB** | **~1.4 GB**   | 48K   | **Modelo activo (GTX 1650)** |
| `PRESET_V3_LARGE` | ~220M    | ~420 MB    | ~2.9 GB       | 48K   | Cloud / 24 GB VRAM           |

> Entrenamiento completamente **local**. Sin cloud, sin RunPod.

---

## Instalación

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
# logits: [1, 64, 48000]
# info: {"exit_nivel": int, "terr_acts": Tensor}

# Generación autoregresiva
gen = model.generate(ids, max_tokens=100, temperature=0.8, top_k=50)
```

### Verificar el modelo

```powershell
# 109/109 tests pasando (arquitectura, memoria, skills, runtime)
python -m pytest tests/ -v

# Test de inferencia rápido
python _test_inferencia.py
```

---

## Estructura del Proyecto

```
PAMPAr-Coder/
├── pampar/
│   ├── coder/
│   │   ├── v3/                    # ← ARQUITECTURA ACTIVA
│   │   │   ├── modelo.py          # PamparV3 (108.3M params)
│   │   │   ├── config.py          # ConfigV3 + PRESET_V3/SMALL/LARGE
│   │   │   ├── talamo.py          # TalamoInicial (LLAVES + atención)
│   │   │   └── bloques.py         # NivelProfundo: GQA + StreamFFN + LateralGates
│   │   └── deprecated/            # v2 (42M, 16K vocab) — preservado como referencia
│   ├── memoria/
│   │   ├── clasificador.py        # ClasificadorPareto (niveles 0-3 de importancia)
│   │   ├── rag.py                 # RAGResidual (recuperación por similitud)
│   │   └── cola_finetune.py       # ColaFinetune (batches candidatos a fine-tuning)
│   ├── skills/
│   │   ├── lector_archivos.py     # LectorArchivos (lee .py, .json, .md — sandboxed)
│   │   └── ejecutar_codigo.py     # EjecutorCodigo (subprocess con timeout)
│   └── runtime/
│       └── agente.py              # Agente (loop inferencia + herramientas)
├── biblioteca/
│   ├── indice.json                # 39 temas Python, niveles 1-6, 5 categorías
│   └── *.jsonl                    # ~140 MB de ejemplos
├── data/
│   └── tokenizer/
│       ├── pampar_48k.model       # ← TOKENIZER ACTIVO (vocab 48K)
│       └── code_tokenizer.model   # 16K — solo para deprecated v2
├── checkpoints/                   # Futuros checkpoints de v3
├── scripts/                       # Utilidades (datacuration, evaluación, etc.)
└── tests/                         # 109 tests pytest
    ├── test_v3_arquitectura.py
    ├── test_memoria.py
    ├── test_skills.py
    └── test_runtime.py
```

---

## Tokenizer — Regla crítica

| Archivo                | Vocab   | Usar con                     |
| ---------------------- | ------- | ---------------------------- |
| `pampar_48k.model`     | **48K** | **PamparV3 — modelo activo** |
| `code_tokenizer.model` | 16K     | Solo deprecated v2           |

**El vocab del tokenizer DEBE coincidir con `config.vocab_size` del modelo.**

---

## Interpretar el Loss

| Loss  | Significado                    |
| ----- | ------------------------------ |
| ~10.7 | Sin entrenar (log 48000)       |
| ~7–8  | Pesos aleatorios (visto en v3) |
| 5–7   | Comenzando a aprender          |
| 2–4   | Aprendizaje activo             |
| 1.5–2 | Zona óptima de ZPD (Vygotsky)  |
| < 1.5 | Tema bien aprendido            |
| < 0.7 | Tema dominado                  |

---

## Tests

```powershell
python -m pytest tests/ -v
# 109/109 passing (4.2s)
```

Cobertura:

- `test_v3_arquitectura.py` — ConfigV3, forward pass, early exit, generación
- `test_memoria.py` — ClasificadorPareto, RAGResidual, ColaFinetune
- `test_skills.py` — LectorArchivos, EjecutorCodigo
- `test_runtime.py` — SYSTEM_PROMPT, Agente (métodos lógicos), acciones

---

## Filosofía

> _"No necesitas 72 billones de parámetros. Necesitas la arquitectura correcta y la curiosidad correcta."_

1. **El código es estructurado** → 4 streams especializados + LLAVES 80% reglas
2. **El código es predecible** → Early exit agresivo (umbral 90%)
3. **El conocimiento es jerárquico** → 5 niveles de profundidad refinan el token
4. **Los contextos se comunican** → Lateral gates en cada nivel (fibras blancas)
5. **Hardware consumer** → 1.4 GB VRAM total para entrenamiento fp16

---

## Roadmap

- [x] Arquitectura territorial (52 zonas de Brodmann, 4 streams)
- [x] Sistema LLAVES (routing INT8, 80% reglas)
- [x] Tokenizer BPE 48K especializado en código
- [x] Arquitectura 2D (4 streams × 5 niveles, GQA, SwiGLU, lateral gates)
- [x] Early Exit (umbral 90%, mínimo 2 niveles)
- [x] Módulo memoria (ClasificadorPareto, RAG, ColaFinetune)
- [x] Skills (LectorArchivos, EjecutorCodigo)
- [x] Runtime.Agente (loop con herramientas)
- [x] Test suite completo (109/109)
- [ ] Script de entrenamiento autónomo para v3
- [ ] Checkpoint v3 entrenado sobre biblioteca/
- [ ] Benchmarks vs CodeLlama/StarCoder small
- [ ] Integración VS Code (extensión)

---

## Licencia

AGPL-3.0-or-later — Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi

```
Input → Embedding → [BloqueTerrritorial ×6] → LM Head → Output
                          ↓
              Tálamo (LLAVES 80% + Atención 20%)
                          ↓
    ┌─────────────────────┴─────────────────────┐
    │                                           │
    ▼                                           ▼
┌───────────────┐                    ┌───────────────┐
│   SINTAXIS    │◄───── Frontera ───►│   SEMANTICA   │
│ Keywords/Ops  │                    │  Tipos/Vars   │
└───────┬───────┘                    └───────┬───────┘
        │                                    │
        │◄─────── Fronteras Simbi. ─────────►│
        │                                    │
┌───────▼───────┐                    ┌───────▼───────┐
│    LOGICO     │◄───── Frontera ───►│ ESTRUCTURAL   │
│ Control Flow  │                    │ Bloques/Scope │
└───────────────┘                    └───────────────┘
```

### Los 4 Territorios

| Territorio      | Zonas Brodmann | Procesa                                |
| --------------- | -------------- | -------------------------------------- |
| **SINTAXIS**    | 1–15           | Keywords, operadores, puntuación       |
| **SEMÁNTICA**   | 16–30          | Tipos, nombres de variables, literales |
| **LÓGICO**      | 31–42          | Control flow, condicionales, bucles    |
| **ESTRUCTURAL** | 43–52          | Bloques, indentación, scope            |

---

## Innovaciones Clave

### 🔑 Sistema LLAVES

- **80% reglas explícitas**: Routing instantáneo basado en patrones de código (INT8)
- **20% atención aprendida**: Ajuste fino para casos ambiguos
- Pre-computado al registrar el tokenizer — sin overhead en inferencia

### 🧠 Viaje Intelectual Autónomo

El modelo aprende solo, sin intervención humana:

1. **MotorCuriosidad** (Vygotsky ZPD): elige el próximo tema maximizando aprendizaje real
   - Zona óptima: loss entre ~1.2 y ~2.0
   - Evita temas demasiado fáciles (aburridos) o demasiado difíciles (frustración)
   - Spacing effect: más sesiones a temas difíciles
2. **MemoriaJerarquica** L0 → L1 → L2 con política Pareto: conserva patrones más valiosos
3. **Biblioteca**: 39 temas de Python organizados por nivel (1=básico, 6=experto)

### ⚡ Early Exit

Si la confianza de un token > 90%, salta capas restantes — código Python es predecible.

---

## Hardware Soportado

| Config      | Params  | VRAM    | Vocab | Notas                        |
| ----------- | ------- | ------- | ----- | ---------------------------- |
| PRESET_4GB  | **42M** | ~0.5 GB | 16K   | **Modelo actual (GTX 1650)** |
| PRESET_8GB  | ~56M    | ~1 GB   | 16K   | En preparación               |
| PRESET_24GB | ~133M   | ~3 GB   | 48K   | En preparación               |
| PRESET_1_5B | ~1.5B   | ~12 GB  | 48K   | Roadmap                      |

> Todo el entrenamiento es **local**. No hay cloud, no hay RunPod, no hay nada remoto.

---

## Instalación

```bash
git clone https://github.com/lucasmella-stack/PAMPAr-Coder.git
cd PAMPAr-Coder
pip install -r requirements.txt
```

---

## Uso

### Lanzar entrenamiento autónomo

```powershell
# Smoke test primero (12 checks, ~15 segundos)
python scripts/smoke_test_viaje.py --checkpoint checkpoints/pampar_v2_best.pt

# Entrenar
python scripts/aprender_solo.py `
  --checkpoint checkpoints/pampar_v2_best.pt `
  --tokenizer data/tokenizer/code_tokenizer.model `
  --batch-size 2 --seq-len 512 --lr 5e-5 --guardar-cada 500
```

### Probar el modelo

```powershell
# Test automático (6 prompts, devuelve % score)
python scripts/probar_modelo.py --auto

# Prompt puntual
python scripts/probar_modelo.py --prompt "def fibonacci("

# Modo interactivo (REPL)
python scripts/probar_modelo.py
```

### Usar desde Python

```python
from pampar.coder.v2.modelo import PampaRCoderV2
from pampar.coder.v2.config import ConfigV2, PRESET_4GB
import sentencepiece as spm
import torch

tok = spm.SentencePieceProcessor("data/tokenizer/code_tokenizer.model")
ck = torch.load("checkpoints/pampar_v2_best.pt", map_location="cpu")
config = ConfigV2(**PRESET_4GB)
model = PampaRCoderV2(config)
model.load_state_dict(ck["model_state_dict"])
model.eval()

ids = tok.encode("def fibonacci(")
logits, _, _ = model(torch.tensor([ids]))
next_token = logits[0, -1].argmax().item()
print(tok.decode([next_token]))
```

---

## Tests

```powershell
python -m pytest tests/ -v
# 130/130 passing
```

---

## Estructura del Proyecto

```
PAMPAr-Coder/
├── pampar/coder/v2/
│   ├── modelo.py              # PampaRCoderV2 (42M params, PRESET_4GB)
│   ├── config.py              # ConfigV2 + presets
│   ├── talamo.py              # TálamoBrodmann
│   ├── llaves.py              # LLAVES (INT8 routing)
│   ├── zonas.py               # 52 zonas de Brodmann
│   ├── bloques.py             # BloqueTerritorial + FFN simbiótico
│   └── aprendizaje/
│       ├── curiosidad.py          # MotorCuriosidad (Vygotsky ZPD)
│       └── memoria_jerarquica.py  # L0/L1/L2, Pareto
├── biblioteca/
│   ├── indice.json            # 40 temas, 5 categorías, niveles 1–6
│   └── *.jsonl                # Datos (~140 MB)
├── data/tokenizer/
│   ├── code_tokenizer.model   # 16K ← MODELO ACTUAL
│   └── pampar_48k.model       # 48K ← futuros modelos
├── checkpoints/
│   └── pampar_v2_best.pt      # 42M params, vocab 16K ← ACTIVO
├── scripts/
│   ├── aprender_solo.py       # Loop autónomo
│   ├── smoke_test_viaje.py    # Pre-flight (12 checks)
│   └── probar_modelo.py       # Testing
└── tests/                     # 134 tests pytest
```

---

## Tokenizer — Regla crítica

| Archivo              | Vocab   | Usar cuando                       |
| -------------------- | ------- | --------------------------------- |
| code_tokenizer.model | **16K** | Modelo actual (pampar_v2_best.pt) |
| pampar_48k.model     | 48K     | Futuros modelos más grandes       |

**El vocab del tokenizer DEBE coincidir con el vocab del modelo.**

---

## Interpretar el Loss

| Loss  | Significado           |
| ----- | --------------------- |
| ~99   | Sin entrenar          |
| 5–7   | Comenzando a aprender |
| 2–4   | Aprendizaje activo    |
| 1.5–2 | Zona óptima de ZPD    |
| < 1.5 | Tema bien aprendido   |
| < 0.7 | Tema dominado         |

---

## Filosofía

> _"No necesitas 72 billones de parámetros. Necesitas la arquitectura correcta y la curiosidad correcta."_

1. **El código es estructurado** → Más peso a reglas (LLAVES 80%)
2. **El código es predecible** → Early exit agresivo
3. **El aprendizaje es autónomo** → Curiosidad + memoria, sin etiquetas humanas
4. **Hardware consumer** → Optimizado para 4GB VRAM

---

## Roadmap

- [x] Arquitectura territorial (52 zonas de Brodmann)
- [x] Sistema LLAVES (routing INT8)
- [x] Tokenizer BPE 16K especializado en código
- [x] MotorCuriosidad (Vygotsky ZPD)
- [x] MemoriaJerarquica (L0/L1/L2, Pareto)
- [x] Biblioteca de 39 temas Python
- [x] Viaje Intelectual autónomo
- [ ] Benchmarks vs CodeLlama/StarCoder small
- [ ] Escalar a PRESET_8GB con vocab 48K
- [ ] Integración con VS Code (extensión)

---

## Licencia

AGPL-3.0-or-later — Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
