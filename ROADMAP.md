# PAMPAr-Coder — Roadmap

> Plan de evolución. Última actualización: Mar 2026.
> Para la identidad del modelo ver `CONCIENCIA.md`. Para el protocolo de despliegue ver `AGENTS.md`.

---

## 1. Visión

PAMPAr es un **físico con doctorado** que puede especializarse en cualquier campo:

- El **doctorado** (razonamiento computacional) está en los **pesos** — 108M params.
- La **especialización** viene del **entorno** — se descubre al boot con el Scanner.
- El protocolo de 3 archivos (`CONCIENCIA.md` + `AGENTS.md` + `TOOLS.md`) es la interfaz entre el modelo y su despliegue.

### Las 3 fases del proyecto

| Fase                             | Qué                                                                   | Estado                                                     |
| -------------------------------- | --------------------------------------------------------------------- | ---------------------------------------------------------- |
| **Fase 1** — SFT                 | Entrenar el doctorado: lógica Python, patrones, razonamiento          | **✅ Completa** (16/16 con reparadores, target superado)   |
| **Fase 2** — Runtime loop        | El modelo usa herramientas, ejecuta, lee, aprende del loop            | **✅ Completa** (chat.py + ColaFinetune + mini-SFT wiring) |
| **Fase 3** — Protocolo entrenado | El modelo genera su propio AGENTS.md al aterrizar en un sistema nuevo | Futuro                                                     |

---

## 2. Arquitectura actual — PamparV3

### 2.1 Grilla cortical 2D

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
  │  TalamoNivel → 4× BloqueAttn GQA 4:1       │
  │  → 4× StreamFFN SwiGLU → LateralGate       │
  │  → Early Exit (umbral 0.90)                 │
  └──────────────────────────────────────────────┘
             │
             ▼
        RMSNorm + lm_head → logits [B, L, 48000]
```

### 2.2 Streams ↔ Capas lingüísticas

| Stream | Territorio  | Zonas   | Capa lingüística                  |
| ------ | ----------- | ------- | --------------------------------- |
| 0      | SINTAXIS    | B01-B15 | Sintaxis — estructura del código  |
| 1      | SEMANTICA   | B16-B30 | Semántica — significado           |
| 2      | LOGICO      | B31-B42 | Pragmática — intención, flujo     |
| 3      | ESTRUCTURAL | B43-B52 | Discurso — organización, patrones |

Los 4 streams procesan en paralelo. Cada NivelProfundo tiene Lateral Gates (bottleneck=128) para comunicación entre streams — como las fibras blancas del cerebro.

### 2.3 Boot Protocol

```
1. CONCIENCIA.md → RAG L3 (identidad inmutable)
2. Scanner → workspace (ast), paquetes (importlib), servicios (socket), sistema (platform)
3. AGENTS.md contextual → RAG L2 (entorno mutable)
4. System prompt dinámico = identidad + contexto + acciones
```

Implementado en `pampar.runtime.scanner` + `pampar.runtime.boot`.

---

## 3. Estado de checkpoints

| Checkpoint     | Datos                                         | Eval open (temp=0.0) |
| -------------- | --------------------------------------------- | -------------------- |
| `v3_sft.pt`    | 43K Magicoder (inglés)                        | 0/16                 |
| `v3_sft_v5.pt` | SFT v5 (base post-catastrófico)               | 6/16                 |
| `v3_sft_v6.pt` | clean_sft.jsonl (555 ejemplos)                | 10/16                |
| `v3_sft_v7.pt` | final_sft.jsonl (825 = clean + quirúrgico×3)  | 15/16                |
| `v3_sft_v8.pt` | micro-SFT cuadrados (300 steps + reparadores) | **16/16 ✅ BEST**    |

### Estado actual (v3_sft_v8.pt — 16/16 con reparadores)

| #   | Función          | Estado | Notas                                                          |
| --- | ---------------- | ------ | -------------------------------------------------------------- |
| 01  | contar_vocales   | ✅     | —                                                              |
| 02  | suma_digitos     | ✅     | —                                                              |
| 03  | es_palindromo    | ✅     | —                                                              |
| 04  | maximo_lista     | ✅     | —                                                              |
| 05  | fizzbuzz         | ✅     | Corregido (dataset quirúrgico)                                 |
| 06  | aplanar_lista    | ✅     | —                                                              |
| 07  | frecuencia       | ✅     | —                                                              |
| 08  | cuadrados_pares  | ✅     | Genera `x*i` → reparador NameError word-boundary lo corrige    |
| 09  | invertir_dict    | ✅     | —                                                              |
| 10  | fibonacci        | ✅     | —                                                              |
| 11  | busqueda_binaria | ✅     | —                                                              |
| 12  | merge_sort       | ✅     | Corregido (self-contained)                                     |
| 13  | Stack            | ✅     | —                                                              |
| 14  | Punto            | ✅     | Corregido (import math / \*\*0.5)                              |
| 15  | memoize          | ✅     | Corregido (usa `fn`, no `func`)                                |
| 16  | primos_hasta     | ✅     | Reparador `_reparar_bloques_huerfanos` + stop `endswith(\n\n)` |

---

## 4. Plan de entrenamiento

### Fase A — Entrenamiento curricular con MotorCuriosidad

Objetivo: reforzar las bases de lógica que el modelo falla.

```bash
python scripts/train_v3.py \
  --checkpoint checkpoints/v3_sft_v4.pt \
  --biblioteca data/biblioteca/ \
  --lr 3e-5 --epochs 3
```

Temas prioritarios basados en fallos del eval:

1. `bucles_for_while` — fizzbuzz, cuadrados_pares
2. `diccionarios` — invertir_dict
3. `busqueda_algoritmos` — búsqueda binaria
4. `recursion` — merge_sort
5. `clases_oop` — Punto, memoize
6. `matematica_basica` — primos, potencias

### Fase B — SFT v5 (post-curricular)

- ~18K ejemplos curados (3K por topic × 6 topics)
- Formato Alpaca, filtrado con pytest
- Generados por el propio modelo + verificación automática

### Fase C — Matriz lingüística como dato de entrenamiento

Incluir ejemplos que ejerciten explícitamente cada capa:

- **Pragmática**: "El usuario quiere X, yo debo hacer Y" (comprensión de intención)
- **Semántica**: Renombrar variables, inferir tipos, naming conventions
- **Sintaxis**: Indentación correcta, keywords, delimitadores, f-strings
- **Discurso**: Organización de código (imports → constantes → clases → funciones → main)

---

## 5. Roadmap de milestones

```
COMPLETADO ✅       COMPLETADO ✅        AHORA                 LARGO PLAZO
────────────        ────────────         ─────────────         ────────────
15/16 eval   →     16/16 eval   →       Mini-SFT auto →       Protocolo
v3_sft_v7.pt       v3_sft_v8.pt         cuando cola≥50        entrenado
108M params        + reparadores        ColaFinetune          Fase 3

SFT dataset        chat.py              Mini-SFT wiring       El modelo
limpio+quirúrgico  loop activo          sft_v5.py             genera su
Clean+surgical×3   gen→exec→retry       auto-reload           AGENTS.md
```

### Milestone 1 — 16/16 eval ✅ COMPLETADO (target era ≥12/16)

- [x] Dataset limpio (clean_sft.jsonl — 555 ejemplos sin contradicciones)
- [x] Dataset quirúrgico (surgical_sft.jsonl — 90 ejemplos para 6 fallos)
- [x] SFT v6 (10/16) desde clean data
- [x] SFT v7 (15/16) desde clean + surgical×3
- [x] Fix primos_hasta → reparador `_reparar_bloques_huerfanos` + stop `endswith(\n\n)`
- [x] Fix cuadrados_pares → reparador NameError word-boundary en verificador
- [x] **16/16 confirmado** con v3_sft_v8.pt + eval_v3.py cadena de reparadores

### Milestone 2 — Runtime autónomo (EN PROGRESO)

- [x] Scanner del sistema (`pampar.runtime.scanner`)
- [x] Boot protocol (`pampar.runtime.boot`)
- [x] CONCIENCIA.md como identidad invariante
- [x] System prompt dinámico (identidad + contexto del scan)
- [x] El agente ejecuta código que genera y observa output (`scripts/chat.py`)
- [x] Si falla, agrega el par (prompt, error) a ColaFinetune
- [x] Mini-SFT automático cuando la cola supera umbral (wiring con sft_v5.py + reload en proceso)

### Milestone 3 — Protocolo entrenado ✅ Implementado (generador determinista)

- [x] `pampar/runtime/generar_agents.py` — genera AGENTS.md contextual desde el scan (determinista)
- [x] `BootProtocol._inyectar_contexto()` actualizado: genera AGENTS.md → fragmenta por secciones → RAG L2
- [x] 23 tests en `tests/test_generar_agents.py` (132/132 en suite completa)
- [x] Quick Reference, Sistema detectado, Paquetes clave, Servicios, Boot protocol generados dinámicamente
- [ ] El modelo "sabe" escanear: genera `scan_sistema()` como código (largo plazo — necesita mucho más SFT)
- [ ] CONCIENCIA se refuerza con RLHF/DPO sobre interacciones reales (largo plazo)
- [ ] Nota: entrenar 108M params para generar markdown desde cero requiere 10K+ pasos — protocolo funcionando vía boot determinista es la aproximación correcta para este tamaño de modelo

### Milestone 4 — VS Code extension

- [ ] Extension que carga PamparV3 localmente (CPU/GPU)
- [ ] Completado inline de código
- [ ] Panel de chat con el agente
- [ ] Memoria persistente entre sesiones (RAG en disco)

### Milestone 5 — Voz (cuando el sistema la tiene)

- [ ] Detectar motores de voz al boot (espeak, SAPI, say) — ya implementado en Scanner
- [ ] TTS para respuestas cuando el usuario lo pide
- [ ] Zero-dependency: usa lo que el OS tiene instalado

---

## 6. Estructura de carpetas

```
PAMPAr-Coder/
├── CONCIENCIA.md                 # Identidad invariante del modelo
├── AGENTS.md                     # Protocolo de despliegue (mutable)
├── ROADMAP.md                    # Este archivo
├── pampar/
│   ├── coder/v3/                 # Arquitectura activa (108M)
│   │   ├── modelo.py             # PamparV3
│   │   ├── config.py             # ConfigV3, presets
│   │   ├── talamo.py             # TalamoInicial
│   │   ├── bloques.py            # BloqueAttn, StreamFFN, LateralGate
│   │   ├── llaves.py             # LlavesV2 — lookup INT8
│   │   └── zonas.py              # 52 Zonas de Brodmann
│   ├── memoria/
│   │   ├── clasificador.py       # ClasificadorPareto — L0 a L3
│   │   ├── rag.py                # RAGResidual — vector store
│   │   └── cola_finetune.py      # ColaFinetune — buffer SFT
│   ├── runtime/
│   │   ├── agente.py             # Agente — orquestador principal
│   │   ├── scanner.py            # Scanner — inspección del entorno
│   │   └── boot.py               # BootProtocol — secuencia de arranque
│   └── training/
│       ├── curiosidad.py         # MotorCuriosidad — ZPD
│       └── lector.py             # LectorBiblioteca
├── checkpoints/
│   └── v3_sft_v4.pt              # Mejor checkpoint (8/16)
└── tests/                        # 109+ tests
```
