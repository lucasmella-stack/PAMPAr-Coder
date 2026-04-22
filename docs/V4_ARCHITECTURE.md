# PAMPAr V4 — Recurrent-Depth Multimodal Transformer

> Estado: **Fase 0 + Fase 1 completas (scaffold + contexto multimodal-ready).**
> v3 sigue siendo producción. v4 evoluciona en paralelo sin tocar v3.

---

## Visión

PAMPAr V4 evoluciona la arquitectura de v3 incorporando ideas de OpenMythos
(Recurrent-Depth Transformer + DeepSeek-MoE) y dejando rieles tendidos para
escalar a multimodal (texto, imagen, audio, video, code-AST, diagrama, tabla).

**Principios guía:**

1. **No romper v3.** Todo v4 vive en `pampar/coder/v4/`. Los checkpoints v3
   siguen cargando con código v3. v3 nunca se modifica.
2. **Multimodal-ready desde el día uno.** Embeddings polimórficos vía
   `ModalityRouter`, contexto del modulator con slot `modality_id[8]`,
   tokens reservados en el vocab.
3. **Cada fase es un branch separado.** Si algo falla, rollback con
   `git checkout main`.
4. **Mixed Selectivity como columna vertebral.** Toda la arquitectura
   nueva se monta sobre el sistema FiLM ya validado en v3.

---

## Fases

| Fase              | Descripción                                                    | Estado |
| ----------------- | -------------------------------------------------------------- | ------ |
| 0                 | Scaffold `v4/`, `modalities/`, `tests/v4/`, doc                | ✅     |
| 1                 | `ContextModulatorV4` con vector de 71 dims (multimodal + loop) | ✅     |
| 2                 | Modulators jerárquicos (backbone compartido entre niveles)     | ✅     |
| 3                 | `ModalityRouter` + `TextEncoder` cableado en el modelo         | ✅     |
| 4                 | Recurrent loop (Prelude → R×T → Coda) + LTI + Loop-RoPE + ACT  | ⏳     |
| 5                 | FFN compartido entre niveles + validación A/B contra v3        | ⏳     |
| 6 (futuro)        | Encoder de imagen (PatchEncoder ViT)                           | —      |
| 7 (futuro lejano) | Encoder de audio (mel-spec → linear)                           | —      |

---

## Diseño del vector de contexto (Fase 1)

`ContextModulatorV4.CONTEXT_DIM = 71`

| Slot          | Dims | Fuente                             | Significado                     |
| ------------- | ---- | ---------------------------------- | ------------------------------- |
| `zona_acts`   | 52   | TálamoNivel / TalamoInicial        | Tipo léxico (zonas de Brodmann) |
| `terr_acts`   | 4    | TálamoNivel                        | Peso por territorio (stream)    |
| `depth`       | 1    | `nivel_idx / (n_levels - 1)`       | Profundidad normalizada         |
| `conf`        | 1    | exit_head (no_grad)                | Confianza actual                |
| `loop_idx`    | 1    | Iteración del recurrent (0 si N/A) | Posición en el loop             |
| `modality_id` | 8    | One-hot de `ModalityId`            | Tipo de modalidad               |
| `stream_oh`   | 4    | One-hot del stream actual          | Identidad del stream a modular  |

**Comparado con v3 (63 dims):** +8 modality + 1 loop − 1 n_levels = +8 netos.
En modo solo-texto con `loop_idx=0`, los slots multimodal/loop quedan en cero
y el comportamiento es equivalente a v3 (los pesos de la proyección sobre
esos canales no aportan).

---

## Modalidades reservadas

`pampar.coder.v4.modalities.ModalityId` (IntEnum, **orden inmutable**):

| ID  | Nombre   | Estado | Encoder previsto                             |
| --- | -------- | ------ | -------------------------------------------- |
| 0   | TEXT     | ✅     | `TextEncoder` (wrapper sobre `nn.Embedding`) |
| 1   | IMAGE    | ⏳     | PatchEncoder ViT-style (16×16)               |
| 2   | AUDIO    | ⏳     | Mel-spectrogram → linear                     |
| 3   | VIDEO    | ⏳     | Frame patches + temporal pooling             |
| 4   | CODE_AST | ⏳     | Embedding por tipo de nodo AST               |
| 5   | DIAGRAM  | ⏳     | Vector graphics tokenizados                  |
| 6   | TABLE    | ⏳     | Filas/columnas como tokens                   |
| 7   | OTHER    | ⏳     | Slot abierto                                 |

**Nunca reordenar.** Los checkpoints dependen del one-hot.

---

## Tokens reservados en el vocabulario

`ConfigV4.vocab_active = 47_000` (vs `vocab_size = 48_000`).

Los IDs `47_000..47_999` quedan reservados para tokens especiales de modalidades
no-texto: `<IMG>`, `</IMG>`, `<PATCH_i_j>`, `<AUDIO>`, `</AUDIO>`, etc. Hoy
no se usan; el embedding se mantiene de tamaño 48000 para que cualquier
checkpoint futuro siga cargando.

---

## Zonas reservadas

`ConfigV4.n_zonas = 52` (activas, código/lenguaje natural)
`ConfigV4.n_zonas_reserved = 38` (techo conceptual para visión, audición, cross-modal)

Hoy solo las 52 activas se materializan. Las reservadas son planificación
arquitectónica — distribución sugerida cuando llegue el momento:

- 53–70: visuales (V1 edges, V2 contours, V4 shapes, IT objects)
- 71–80: auditivas (A1 frequency, Wernicke speech)
- 81–90: cross-modal (corteza parietal — integra modalidades)

---

## Plan de Fase 2 — Modulators jerárquicos

Hoy cada `ContextModulatorV4` es independiente entre niveles. Se propone
un `HierarchicalModulator`:

```
ctx[71] → SharedProj(71 → bottleneck) → SiLU       # compartido entre niveles
                ↓
        per-(nivel, stream) Linear(bottleneck → dim*2)  # head específica
```

**Beneficio:** menos params, consistencia de representación contextual entre
profundidades, mejor inducción de patrones cross-level.

**Activación:** `ConfigV4.use_hierarchical_modulators=True`. Solo después
de validar que la Fase 1 mantiene paridad numérica con v3.

---

## Plan de Fase 4 — Recurrent loop

Reemplazar el stack secuencial de N niveles por:

```
Input → Prelude (1 nivel) → Recurrent (1 nivel × T loops) → Coda (1 nivel) → Output

Recurrent update rule (LTI-stable):
  h_{t+1} = A · h_t + B · e + RecurrentNivel(h_t, e, loop_idx=t)
  donde ρ(A) < 1 garantizado por A := exp(-exp(log_A) · exp(log_dt))
```

**Componentes (port simplificado de OpenMythos):**

- `LTIInjection` — garantiza ρ(A) < 1 por construcción
- `loop_index_embedding` — sinusoidal del índice t inyectado en h
- `ACTHalting` — cada token decide cuándo dejar de pensar
- `RecurrentBlock` — orquesta el loop con early exit

**Conexión con Mixed Selectivity:** el `loop_idx` se inyecta al
`ContextModulatorV4` (slot ya reservado en Fase 1), por lo que cada
iteración del loop modula el FFN de forma distinta aunque comparta pesos.

---

## Cómo cargar un modelo v4 (cuando exista `PamparV4`)

```python
from pampar.coder.v4 import ConfigV4

cfg = ConfigV4(
    dim=640,
    n_streams=4,
    n_levels=5,
    use_recurrent_loop=False,   # Fase 1: igual que v3
    use_hierarchical_modulators=False,
)
# Próximo paso (Fase 3): from pampar.coder.v4.modelo import PamparV4
# model = PamparV4(cfg)
```

---

## Rollback

Cualquier fase se descarta con:

```bash
git checkout main
git branch -D feat/v4-XXX
```

v3 nunca cambia. Los checkpoints v3 nunca se invalidan. Los tests de v3
deben mantenerse 100% verdes en cada commit a v4.
