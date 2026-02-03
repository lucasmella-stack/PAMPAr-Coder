# PAMPAr-Coder 🧠⚡

> **"Un cerebro artificial donde el tálamo orquesta tokens hacia territorios especializados para generar código."**

## ¿Qué es PAMPAr-Coder?

PAMPAr-Coder es un **modelo de lenguaje cerebral especializado en programación**, diseñado para correr eficientemente en hardware consumer (GTX 1650, 4GB VRAM).

### Arquitectura Territorial

```
Input → Embedding → [BloqueTerrritorial ×N] → LM Head → Output
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
        │◄────── Fronteras Bidirec ─────────►│
        │                                    │
┌───────▼───────┐                    ┌───────▼───────┐
│    LOGICO     │◄───── Frontera ───►│ ESTRUCTURAL   │
│ Control Flow  │                    │ Bloques/Scope │
└───────────────┘                    └───────────────┘
```

### Los 4 Territorios

| Territorio | Procesa | Ejemplos |
|------------|---------|----------|
| **SINTAXIS** | Keywords, operadores | `def`, `function`, `+`, `==` |
| **SEMÁNTICA** | Tipos, nombres, literales | `int`, `myVar`, `"string"` |
| **LÓGICO** | Control flow | `if`, `while`, `try`, `return` |
| **ESTRUCTURAL** | Bloques, indentación | `{`, `}`, `:`, tabs |

## Innovaciones Clave

### 🔑 Sistema LLAVES
**L**inguistic **L**exical **A**nchoring for **V**ectorized **E**ntry **S**election

- **80% reglas explícitas**: Routing instantáneo basado en patrones de código
- **20% atención aprendida**: Ajuste fino para casos ambiguos
- **Pre-computado**: Al registrar el tokenizer, se calculan todas las activaciones

### ⚡ Early Exit
- Si la confianza > 90%, salta capas restantes
- Acelera inferencia sin perder calidad
- Especialmente efectivo en código (muy predecible)

### 🔗 Fronteras Bidireccionales
- 6 conexiones entre los 4 territorios
- Gates aprendidos regulan el flujo
- Permite que SINTAXIS informe a LÓGICO, etc.

## Rendimiento

| Hardware | Parámetros | VRAM | Velocidad |
|----------|------------|------|-----------|
| **GTX 1650** (4GB) | 44M | 0.44 GB | 25+ tok/s |
| **RTX 3060** (8GB) | 56M | ~1 GB | 50+ tok/s |
| **RTX 4090** (24GB) | 133M | ~3 GB | 100+ tok/s |

## Instalación

```bash
git clone https://github.com/lucasmella-stack/PAMPAr-Coder.git
cd PAMPAr-Coder
pip install -r requirements.txt
```

## Uso Rápido

```python
from pampar.coder import PampaRCoder, crear_modelo, CODER_4GB

# Crear modelo para tu GPU
model = crear_modelo("4GB")  # GTX 1650
model = crear_modelo("8GB")  # RTX 3060
model = crear_modelo("24GB") # RTX 4090

# O con configuración custom
from pampar.coder import ConfigPampaRCoder
config = ConfigPampaRCoder(
    vocab_size=8000,
    dim=256,
    n_capas=6,
    peso_llaves=0.75,
    usar_early_exit=True,
)
model = PampaRCoder(config)

# Generación
import torch
prompt = torch.tensor([[1, 2, 3, 4, 5]])  # Token IDs
generated = model.generate(
    prompt,
    max_new_tokens=100,
    temperature=0.8,
    use_early_exit=True  # ⚡ Más rápido
)
```

## Test

```bash
python -m pytest tests/
# O directamente:
python scripts/test_coder.py
```

## Estructura del Proyecto

```
PAMPAr-Coder/
├── pampar/
│   └── coder/
│       ├── __init__.py           # Exports
│       ├── config.py             # Configuraciones (4GB, 8GB, 24GB)
│       ├── llaves_codigo.py      # 🔑 Sistema LLAVES para código
│       ├── territorios_codigo.py # 🏛️ 4 territorios especializados
│       └── model.py              # 🚀 Modelo PampaRCoder
├── scripts/
│   └── test_coder.py             # Tests
├── requirements.txt
└── README.md
```

## Filosofía

Inspirado en la arquitectura PAMPAr-o1, PAMPAr-Coder aplica el concepto de **territorios cerebrales** al dominio específico del código:

1. **El código es estructurado** → Más peso a reglas (LLAVES 80%)
2. **El código es predecible** → Early exit agresivo
3. **El código tiene patrones** → Territorios especializados
4. **Hardware consumer** → Optimizado para GTX 1650

> *"No necesitas 72 billones de parámetros. Necesitas la arquitectura correcta."*

## Licencia

AGPL-3.0-or-later

Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi

## Próximos Pasos

- [ ] Script de entrenamiento
- [ ] Dataset de código (Python, JS, Rust)
- [ ] Tokenizer BPE especializado
- [ ] Benchmarks vs CodeLlama/StarCoder
- [ ] Integración con VS Code

---

**PAMPAr-Coder** - Código generado por territorios cerebrales 🧠⚡
