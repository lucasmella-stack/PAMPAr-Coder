# PAMPAr — Plan de Evolución

> Plan aprobado: **Option B — "Staged Physics"**
> Budget total: **$300-500 USD**
> Modelo: PamparV3, 108.3M params, vocab 48K

---

## Objetivo

Transformar PamparV3 de un modelo que solo conoce patrones Python (38% eval)
a un **motor de razonamiento multi-lenguaje** que usa documentación de referencia
para resolver problemas en cualquier dominio.

### Métricas target

| Métrica | Actual | Target |
|---------|--------|--------|
| Python eval | 6/16 (38%) | 70%+ |
| Multi-language | 0% | 50%+ |
| Doc consultation (RAG) | 0% | 60%+ |
| Debugging | 0% | 50%+ |

---

## Fase 1 — Continual Pretrain: "Textbook Physics"

**Objetivo**: inyectar los axiomas fundamentales de razonamiento con referencia.

### Datos (~540K-640K tokens)

6 pilares de axiomas, cada uno con ~90K-100K tokens de texto tipo textbook:

| Pilar | Contenido | Fuente |
|-------|-----------|--------|
| **Lógica y razonamiento** | Proposiciones, inferencia, truth tables, deducción | Generado + Wikipedia |
| **Estructuras de datos** | Arrays, trees, graphs, hashmaps — cross-language | Generado + docs oficiales |
| **Patrones de código** | Design patterns, idioms en Python/JS/Rust/C | Generado + libros open |
| **Comprensión de docs** | Cómo leer una API reference, man page, docstring | MDN, Python docs, Rust Book |
| **Debugging** | Stack traces, error messages, bisección, logging | Generado + StackOverflow curado |
| **Multi-language syntax** | Equivalencias Python↔JS↔Rust↔C↔Bash↔SQL | Generado + Rosetta Code |

### Formato

```
<textbook>
## Capítulo: [tema]

[Explicación clara del concepto]

### Ejemplo
[Código con comentarios]

### Ejercicio resuelto
[Problema → razonamiento step-by-step → solución]
</textbook>
```

### Costo estimado: $30-60

- Distilación desde GPT-4o/Claude para generar textbooks
- ~6 scripts de generación, uno por pilar
- Validación manual de quality (sampling 5%)

### Hardware

- Generación de datos: API calls (local)
- Continual pretrain: **RunPod A100 40GB** (~2-4 horas)

---

## Fase 2 — SFT: "Chain-of-Thought con Referencia"

**Objetivo**: enseñar al modelo a usar documentación de referencia para resolver problemas.

### Datos (~20K ejemplos)

| Categoría | Ejemplos | Descripción |
|-----------|----------|-------------|
| Python + ref | 5K | Problemas con snippet de docs como contexto |
| JavaScript + ref | 3K | DOM, Node.js, ES6+ con MDN como referencia |
| Rust + ref | 2K | Ownership, traits, lifetimes con Rust Book |
| SQL + ref | 2K | Queries con schema como referencia |
| Bash/CLI + ref | 1K | Comandos con man pages como referencia |
| Debugging | 3K | Stack traces → diagnóstico → fix |
| Cross-language | 2K | "Traducir" lógica entre lenguajes |
| RAG-grounded | 2K | Preguntas que requieren buscar en docs primero |

### Formato SFT

```json
{
  "instruction": "[PROBLEMA] Implementar un servidor HTTP básico",
  "reference": "[REFERENCIA] Fragmento de docs de http.server de Python...",
  "reasoning": "[RAZONAMIENTO] 1. Necesito importar http.server\n2. Crear handler...\n3. Bind al puerto...",
  "output": "[SOLUCIÓN] import http.server\n..."
}
```

### Costo estimado: $110-150

- Distilación masiva desde GPT-4o/Claude
- 20K ejemplos × ~$0.006/ejemplo promedio
- Quality filter: score > 0.7 de auto-evaluación

### Hardware

- Generación de datos: API calls (local)
- SFT: **RunPod A100 40GB** (~4-8 horas)

---

## Fase 3 — Corrección: "GhidraProbe + NeuroTrainer"

**Objetivo**: corregir routing y pesos usando diagnóstico local.

### Proceso

1. Correr `eval_v3.py` para identificar categorías débiles
2. GhidraProbe analiza activaciones en ejemplos fallidos
3. NeuroTrainer aplica correcciones targeted:
   - LLAVES: ajustar reglas INT8 para tokens multi-language
   - Routing: corregir `terr_acts` donde el Tálamo asigna mal
   - Pesos: mini-SFT de 50-100 steps en categorías fallidas

### Costo: $0

- 100% local en GTX 1650
- ~2 horas por ronda de corrección
- 3-5 rondas estimadas

---

## Timeline estimado

| Fase | Duración | Costo | Output |
|------|----------|-------|--------|
| Fase 1 — Pretrain data | 1-2 semanas | $30-60 | ~600K tokens textbook |
| Fase 1 — Training | 1 día RunPod | incluido | Checkpoint pretrained |
| Fase 2 — SFT data | 2-3 semanas | $110-150 | ~20K SFT examples |
| Fase 2 — Training | 1 día RunPod | incluido | Checkpoint SFT |
| Fase 3 — Correction | 1 semana | $0 | Checkpoint final |
| **Total** | **5-7 semanas** | **$160-235** | **Motor de razonamiento** |

---

## Pre-requisitos (Blocks 2-3)

Antes de empezar el training, necesitamos limpiar y preparar el código:

### Block 2 — Cleanup de código muerto

Scripts que importan módulos v2 eliminados (borrar con backup a `_archive/`):

- `scripts/aprender_solo.py`
- `scripts/train.py`
- `scripts/train_cerebral.py`
- `scripts/destilar.py`
- `scripts/evaluate_v2.py`
- `scripts/generar_curriculum.py`
- `scripts/smoke_test_viaje.py`
- `scripts/test_llaves.py`

Scripts mixtos v2/v3 rotos (borrar con backup):

- `scripts/benchmark.py`
- `scripts/probar_modelo.py`
- `scripts/eval_honesta.py`

Módulos huérfanos:

- `pampar/training/` — no importado por nada

### Block 3 — Refactoring para multi-language

| Archivo | Cambio |
|---------|--------|
| `zonas.py` | Agregar keywords JS/Rust/C/Bash/SQL a ZONAS |
| `llaves.py` | Expandir `clasificar_token()` para multi-language |
| `clasificador.py` | Generalizar `_calcular_densidad()` más allá de Python |
| `ejecutar_codigo.py` | Agregar soporte para Node.js, Bash |
| `config.py` | Extraer `TOKENIZER_PATH` como constante compartida |

---

## Checkpoints esperados

| Nombre | Fase | Descripción |
|--------|------|-------------|
| `v3_ghidra_v9.pt` | Actual | Score 89, 6/16 (38%) — baseline |
| `v3_pretrain_f1.pt` | Fase 1 | Post continual pretrain |
| `v3_sft_f2.pt` | Fase 2 | Post SFT multi-language |
| `v3_corrected_f3.pt` | Fase 3 | Post GhidraProbe correction — target final |

---

## Notas

- **Arquitectura LOCKED**: no tocar la grilla 4×5, GQA, SwiGLU, LLAVES 80/20
- **Backups siempre**: antes de borrar/refactorizar → `_archive/`
- **RunPod**: A100 40GB para fases 1 y 2, el código de `cloud/runpod/` ya existe
- **Evaluación**: `scripts/eval_v3.py` como benchmark consistente entre fases
