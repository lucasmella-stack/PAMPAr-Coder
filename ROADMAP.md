# PAMPAr-Coder — Arquitectura y Roadmap

> Estado técnico actual, decisiones de diseño, y plan de evolución.
> Última actualización: limpieza de `deprecated/` completada ✅

---

## 1. Arquitectura actual — PamparV3

### 1.1 Visión general

PamparV3 es un LLM de **108M parámetros** diseñado para código Python/español, optimizado para correr en hardware limitado (GTX 1650, 4.3 GiB VRAM).

```
Tokens (int)
    │
    ▼
[Embeddings] 48K vocab, dim=640, weight-tied con lm_head
    │
    ▼
┌─────────────────────────────┐
│       TalamoInicial          │  routing: qué streams procesan cada token
│  80% LLAVES (INT8 + reglas) │
│  20% attn_proj (aprendido)  │
│  + context_conv causal k=32 │
└────────────┬────────────────┘
             │ [B, L, 4, dim] — 4 streams con pesos distintos
             ▼
  ┌──────────────────────────────────────────────┐
  │          5 × NivelProfundo                   │
  │                                              │
  │  ┌─────────────────────────────────────┐     │
  │  │ Para cada nivel:                    │     │
  │  │                                     │     │
  │  │  TalamoNivel (re-routing ligero)    │     │
  │  │         │                           │     │
  │  │  [× 4 streams en paralelo]          │     │
  │  │  BloqueAttn — GQA 4:1              │     │
  │  │    8 Q heads, 2 KV heads           │     │
  │  │    head_dim=80, RoPE, Flash Attn   │     │
  │  │         │                           │     │
  │  │  StreamFFN — SwiGLU                │     │
  │  │    hidden=640×4=2560               │     │
  │  │         │                           │     │
  │  │  LateralGate (bottleneck=128)      │     │
  │  │    cada stream recibe info         │     │
  │  │    de los otros 3 streams          │     │
  │  │                                     │     │
  │  │  Early Exit (umbral=0.90)          │     │
  │  │    si max_prob ≥ 0.90 en ≥90%     │     │
  │  │    de tokens → salir antes         │     │
  │  └─────────────────────────────────────┘     │
  └──────────────────────────────────────────────┘
             │ [B, L, dim] — combinación de 4 streams
             ▼
        RMSNorm + lm_head → logits [B, L, 48000]
```

### 1.2 Los 4 streams (territorios)

| Stream | Territorio  | Zonas   | Especialización                            |
| ------ | ----------- | ------- | ------------------------------------------ |
| 0      | SINTAXIS    | B01-B15 | Keywords, delimitadores, puntuación        |
| 1      | SEMANTICA   | B16-B30 | Variables, tipos, literales, magic methods |
| 2      | LOGICO      | B31-B42 | Operadores, flujo de control, excepciones  |
| 3      | ESTRUCTURAL | B43-B52 | Indentación, bloques, patrones             |

### 1.3 Sistema LLAVES (v2, ahora en v3/)

- **52 Zonas de Brodmann** para código (inspiradas en neurociencia)
- **INT8** lookup tables: 52 bytes/token vs 208 bytes en FP32
- `clasificar_token()`: reglas exactas + regex → asigna zona + confianza
- `LlavesV2.forward()`: lookup en tabla → `[B, L, 52]` activaciones
- `agregar_zonas_a_territorios()`: reduce 52→4 por media por territorio

### 1.4 Archivos del modelo

```
pampar/coder/v3/
├── config.py      # ConfigV3 dataclass, PRESET_V3/SMALL/LARGE
├── modelo.py      # PamparV3 — forward, generate, gradient checkpointing
├── talamo.py      # TalamoInicial — routing entrada
├── bloques.py     # RMSNorm, RoPE, BloqueAttn, StreamFFN, LateralGate,
│                  # TalamoNivel, NivelProfundo
├── llaves.py      # LlavesV2, clasificar_token, normalizar  ← MOVIDO de deprecated/
└── zonas.py       # Zona (52), Territorio (4), ZONAS, ZONA_TERRITORIO ← MOVIDO de deprecated/
```

### 1.5 Sistema de memoria del agente

```
pampar/memoria/
├── clasificador.py   # ClasificadorPareto — niveles 0-3 (importancia del chunk)
├── rag_residual.py   # RAGResidual — vector store + retrieval BM25-like
└── cola_finetune.py  # ColaFinetune — acumula ejemplos para futuro SFT
```

### 1.6 Training

```
pampar/training/
├── curiosidad.py   # MotorCuriosidad — ZPD de Vygotsky para elección de temas
└── lector.py       # LectorBiblioteca — lee JSONL de biblioteca/ (39 temas)
```

---

## 2. La idea: Vectorización del sistema

### 2.1 Qué significa

La idea es que el **system prompt del agente** (actualmente un string estático inyectado en cada prompt) sea **vectorizado y almacenado en RAGResidual** como entradas de nivel 3 (máxima importancia), en lugar de ocupar tokens del contexto.

**Estado actual:**

```python
# agente.py — el system prompt ocupa ~300 tokens SIEMPRE
prompt = f"[SYSTEM]\n{SYSTEM_PROMPT}\n[USER]\n{msg}\n[ASSISTANT]\n"
```

**Visión propuesta:**

```python
# El system prompt se vectoriza una sola vez al iniciar el agente
# y se recupera como contexto semántico comprimido, no como tokens

rag.agregar(EntradaRAG(
    texto=SYSTEM_PROMPT,
    tipo="system",
    nivel=3,            # máxima importancia, nunca se purga
    territorio="LOGICO" # instrucciones = flujo lógico
))

# En cada turno, el contexto RAG enriquece el prompt, no lo reemplaza
ctx = rag.recuperar(query=msg_usuario, top_k=3)
prompt = f"{ctx.formatear()}\n[USER]\n{msg_usuario}\n[ASSISTANT]\n"
```

### 2.2 Por qué importa

| Problema actual                                          | Solución con vectorización                                     |
| -------------------------------------------------------- | -------------------------------------------------------------- |
| System prompt ocupa ~300 tokens fijos                    | Se recuperan solo los fragmentos relevantes                    |
| Contexto disponible = 4096 - 300 = ~3796                 | Contexto útil ≈ 4096 (system comprimido)                       |
| El agente "olvida" instrucciones si el contexto se llena | RAG siempre recupera lo relevante                              |
| Un solo system prompt para todos los casos               | RAG semántico: recupera la regla más aplicable al query actual |

### 2.3 Cómo implementarlo (plan técnico)

**Fase 1** — Fragmentar el system prompt en chunks semánticos:

```python
# En lugar de un SYSTEM_PROMPT monolítico, fragmentar por "regla"
REGLAS = [
    "Leer el archivo antes de modificarlo.",
    "Ejecutar tests después de cada cambio.",
    "Responder en español cuando el usuario habla en español.",
    ...
]
for regla in REGLAS:
    rag.agregar(EntradaRAG(texto=regla, tipo="system", nivel=3))
```

**Fase 2** — Recuperación dinámica:

```python
# El agente recupera las 3 reglas más relevantes para el query actual
ctx = rag.recuperar(query=msg_usuario, top_k=3)
prompt = construir_prompt(msg_usuario, ctx_rag=ctx.formatear())
```

**Fase 3** (avanzada) — Embeddings reales:

- Entrenar un encoder pequeño (128-256 dim) como tower separado de PamparV3
- Vectorizar chunks con el encoder, guardar en FAISS
- Recuperar por similitud coseno, no por overlap de palabras

---

## 3. Estado de checkpoints

| Checkpoint     | Datos                         | Score eval (guided, temp=0.4) | Notas                    |
| -------------- | ----------------------------- | ----------------------------- | ------------------------ |
| `v3_sft.pt`    | 43K Magicoder (inglés)        | 0/16                          | Genera nombres en inglés |
| `v3_sft_v4.pt` | 1,220 ejemplos Python/español | **8/16 (50%)**                | ✅ Mejor actual          |

### Casos del eval que pasan con v3_sft_v4.pt

| #   | Función            | Estado               |
| --- | ------------------ | -------------------- |
| 01  | suma_digitos       | ✅                   |
| 02  | suma_digitos (var) | ✅                   |
| 03  | es_palindromo      | ✅                   |
| 04  | factorial          | ✅                   |
| 06  | numero_random      | ✅ (suerte con rand) |
| 07  | contar_vocales     | ✅                   |
| 10  | fibonacci          | ✅                   |
| 13  | Stack              | ✅                   |

### Casos que fallan (y por qué)

| #   | Función          | Fallo    | Causa raíz                                  |
| --- | ---------------- | -------- | ------------------------------------------- |
| 05  | fizzbuzz         | LÓGICA   | Genera `if n % 15 == 0` en orden incorrecto |
| 08  | cuadrados_pares  | LÓGICA   | Genera `x * 2` en vez de `x ** 2`           |
| 09  | invertir_dict    | LÓGICA   | Variable fantasma `{v: k for v, k in ...}`  |
| 11  | busqueda_binaria | LÓGICA   | Comparador invertido                        |
| 12  | merge_sort       | SINTAXIS | Indentación corrupta en función recursiva   |
| 14  | Punto            | LÓGICA   | `self.x = x; self.y = x` (typo sistemático) |
| 15  | memoize          | LÓGICA   | No implementa closure correctamente         |
| 16  | primos           | TIEMPO   | Genera trial division en O(n) no O(√n)      |

---

## 4. Plan de entrenamiento

### Fase A — Pre-training curricular con MotorCuriosidad (PRÓXIMO PASO)

**Objetivo:** reforzar las bases de lógica Python que el modelo introduce incorrectamente.

```bash
python scripts/train_v3.py \
  --checkpoint checkpoints/v3_sft_v4.pt \
  --biblioteca data/biblioteca/ \
  --lr 3e-5 \
  --epochs 3
```

**Cómo funciona `MotorCuriosidad`:**

- Evalúa `curiosidad(tema) = zona_proximal × novedad × urgencia_temporal × bonus_mejora`
- ZPD de Vygotsky: elige temas donde `loss_media` está en la "zona óptima" (ni muy fácil ni imposible)
- Esto maximiza la tasa de aprendizaje real por batch consumido

**Temas prioritarios basados en fallos del eval:**

1. `bucles_for_while` — (fizzbuzz, cuadrados_pares)
2. `diccionarios` — (invertir_dict)
3. `busqueda_algoritmos` — (búsqueda binaria)
4. `recursion` — (merge_sort)
5. `clases_oop` — (Punto, memoize)
6. `matematica_basica` — (primos, potencias)

### Fase B — SFT v5 (después del curricular)

**Datos a generar** (`scripts/generar_curriculum.py` usando PamparV3):

- ~3,000 ejemplos por topic × 6 tópicos = 18K ejemplos
- Formato Alpaca: `{"instruction": ..., "input": ..., "output": ...}`
- Curado automático: solo ejemplos donde pytest pasa

```bash
python scripts/destilar.py \
  --modelo checkpoints/v3_post_curricular.pt \
  --output data/distillation/sft_v5_curado.jsonl \
  --n 18000 --filtrar-con-pytest
```

### Fase C — Objetivo 12/16

Para pasar de 8→12/16 se necesitan:

- fizzbuzz (5): orden de condicionales — dato de curricular
- cuadrados_pares (8): `**` vs `*` — dato de curricular
- invertir_dict (9): comprensión de dict — dato de curricular
- busqueda_binaria (11): comparadores — dato de curricular

Los 4 casos restantes (merge_sort, Punto, memoize, primos) requieren SFT v5 focalizado.

---

## 5. Roadmap de capacidades

```
ESTADO ACTUAL          CORTO PLAZO           MEDIANO PLAZO         LARGO PLAZO
─────────────          ──────────────        ─────────────         ─────────────
8/16 eval      →       12/16 eval     →      Runtime loop   →      VS Code ext.

v3_sft_v4.pt           v3_post_curricular   Agente ejecuta         Integración
108M params            + SFT v5             tests, lee archivos    IDE completa
                                            y aprende del loop

Vectorización          Fragmentar           Encoder semántico      RAG con FAISS
system prompt          SYSTEM_PROMPT        128-dim embeddings      cosine search
(Fase 1: texto)        (Fase 2: indexar)    (Fase 3: vectores)
```

### Milestone 1 — 12/16 (corto plazo)

- [ ] Correr `train_v3.py` con biblioteca/ + MotorCuriosidad (3 epochs)
- [ ] Generar SFT v5 curado con pytest-filtrado
- [ ] Fine-tune sobre SFT v5
- [ ] Eval guided+temp=0.4 → objetivo ≥ 12/16

### Milestone 2 — Runtime loop (mediano plazo)

- [ ] El agente ejecuta el código que genera y observa el output
- [ ] Si falla, agrega el par (prompt, error) a `ColaFinetune`
- [ ] Cuando la cola supera umbral → trigger de mini-SFT automático
- [ ] Vectorizar SYSTEM_PROMPT en RAGResidual (Fase 1)

### Milestone 3 — VS Code extension (largo plazo)

- [ ] Extension que carga PamparV3 localmente (CPU/GPU)
- [ ] Completado inline de código con el modelo
- [ ] Panel de chat con el agente
- [ ] Memoria persistente entre sesiones (RAG en disco)

---

## 6. Estructura de carpetas (post-limpieza)

```
PAMPAr-Coder/
├── pampar/
│   ├── coder/
│   │   ├── __init__.py         # exporta solo v3
│   │   └── v3/
│   │       ├── __init__.py
│   │       ├── config.py       # ConfigV3, presets
│   │       ├── modelo.py       # PamparV3 — forward, generate
│   │       ├── talamo.py       # TalamoInicial — routing entrada
│   │       ├── bloques.py      # BloqueAttn, StreamFFN, LateralGate, etc.
│   │       ├── llaves.py       # LlavesV2 + clasificar_token (era deprecated/)
│   │       └── zonas.py        # 52 Zonas de Brodmann (era deprecated/)
│   ├── memoria/
│   │   ├── clasificador.py     # ClasificadorPareto — importancia 0-3
│   │   ├── rag_residual.py     # RAGResidual — vector store de sesión
│   │   └── cola_finetune.py    # buffer de ejemplos para SFT online
│   └── training/
│       ├── curiosidad.py       # MotorCuriosidad (ZPD)
│       └── lector.py           # LectorBiblioteca — carga data/biblioteca/
├── scripts/
│   ├── train_v3.py             # training curricular con MotorCuriosidad
│   ├── eval_v3.py              # evaluación (guided mode, normaliz. indent)
│   ├── destilar.py             # generación de SFT data desde modelo base
│   └── generar_curriculum.py   # generación de ejemplos por tema
├── checkpoints/
│   ├── v3_sft_v4.pt            # ✅ mejor checkpoint (8/16)
│   └── history.json
└── data/
    ├── biblioteca/             # 39 temas Python (JSONL) — para curricular
    ├── distillation/           # datos SFT curados
    └── tokenizer/
        └── pampar_48k.model    # vocab 48K bilingüe (Python + español)
```

> **Nota:** `deprecated/` fue eliminado. `LlavesV2` y `zonas.py` viven ahora en `pampar/coder/v3/` donde pertenecen.
> Scripts legacy (`aprender_solo.py`, `evaluate_v2.py`, etc.) siguen apuntando a v2 — ignorar o portar según necesidad.
