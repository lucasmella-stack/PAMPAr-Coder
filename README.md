# PAMPAr-Coder 🧠⚡

> **"Un cerebro artificial donde el tálamo orquesta tokens hacia territorios especializados para generar código — y aprende solo."**

## ¿Qué es PAMPAr-Coder?

PAMPAr-Coder es un **modelo de lenguaje cerebral especializado en programación**, diseñado para correr en hardware consumer (GTX 1650, 4GB VRAM) y **aprender de forma completamente autónoma** a través de un sistema de curiosidad inspirado en Vygotsky.

Estado actual: **42M params, vocab 16K, entrenamiento activo con Viaje Intelectual.**

---

## Arquitectura Territorial

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
