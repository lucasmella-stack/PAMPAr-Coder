# PAMPAr-Coder Repository Guidelines

> For AI agents: Claude Code, Codex (OpenAI), Gemini CLI, etc.
> For GitHub Copilot: see `.github/copilot-instructions.md`

## Quick Reference

| Area       | Convention                                                            |
| ---------- | --------------------------------------------------------------------- |
| Language   | Python 3.13+                                                          |
| Framework  | PyTorch 2.x                                                           |
| Tokenizer  | SentencePiece BPE — 16K para modelo actual, 48K preparado para futuro |
| Testing    | pytest (134 tests, todos deben pasar)                                 |
| Type hints | Always required                                                       |
| Docstrings | Google style                                                          |
| Training   | Local — todo corre en la GPU del desarrollador (4GB VRAM)             |
| NO cloud   | No hay RunPod, no hay AWS, no hay nada remoto                         |

## Estado actual del proyecto (Feb 2026)

- **Modelo activo**: `PampaRCoderV2` — 42M params, vocab 16K, arquitectura territorial
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
