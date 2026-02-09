# 🧠 Aprendizaje Cerebral: Paradigma de Entrenamiento Bio-Inspirado

## La Tesis Central

> **"Un cerebro humano aprende a programar con ~10,000 horas de práctica, no con 5.5 trillones de tokens.
> La diferencia: los humanos razonan, experimentan y consolidan — no memorizan brute-force."**

PAMPAr-Coder ya tiene la **arquitectura cerebral** (Territorios, Tálamo, Zonas de Brodmann).
Lo que falta es un **paradigma de entrenamiento** que explote esa arquitectura.

## Por qué Funciona: phi-1 como Prueba

Microsoft demostró con **phi-1** (1.3B params) que un modelo pequeño puede competir con
GPT-3.5 en código usando **solo 7B tokens de calidad "textbook"** vs 5.5T tokens de Qwen.

La clave: **CALIDAD > CANTIDAD**.

Nuestra arquitectura cerebral amplifica esto:
- **LLAVES** (75% reglas) → routing gratuito, no necesita aprender qué es `def`, `if`, `for`
- **52 Zonas Brodmann** → especialización natural por tipo de token
- **Early Exit** → tokens simples son baratos, recursos se enfocan en tokens difíciles
- **4 Territorios FFN** → cada territorio se vuelve experto en su dominio

## Las 5 Fases del Aprendizaje Cerebral

### Fase 1: INFANCIA — Curriculum Learning ($5-15)
```
Nivel 1: Variables y asignaciones     (SINTAXIS domina)
Nivel 2: Control de flujo             (LOGICO + SINTAXIS)
Nivel 3: Funciones                    (SEMANTICA + ESTRUCTURAL)
Nivel 4: Clases y OOP                 (todos los territorios)
Nivel 5: Algoritmos complejos         (LOGICO + ESTRUCTURAL)
Nivel 6: Patrones de diseño           (integración total)
```
- Como un niño aprendiendo: simple → complejo
- Cada nivel activa progresivamente más territorios
- LLAVES asegura routing correcto desde el día 0

### Fase 2: EXPERIMENTACIÓN — Self-Play ($10-30)
```
genera código → ejecuta → resultado → aprende
     ↑                                    ↓
     └────────────── feedback ←────────────┘
```
- Como un programador probando código
- No necesita datasets masivos — genera sus propios datos
- Reward: ¿el código ejecuta? ¿da el resultado correcto?
- DPO: aprende de sus propios aciertos vs errores

### Fase 3: FILOSOFAR — Reasoning Chains ($5-15)
```
Problema: "crear función que ordene una lista"
  → LOGICO: necesito comparar elementos (Zona B32_OP_COMP)
  → ESTRUCTURAL: un bucle anidado (Zona B43_BLOCK_FUNC)
  → SINTAXIS: usar for, if, return (Zonas B05, B06, B04)
  → SEMANTICA: nombre descriptivo (Zona B17_ID_FUNC)
```
- El modelo aprende a USAR sus territorios para razonar
- Chain-of-thought: descomponer problemas en sub-problemas
- Cada paso del razonamiento activa diferentes zonas

### Fase 4: SUEÑO — Consolidación Hebbiana ($2-5)
```
"Neuronas que disparan juntas, se conectan juntas"
  → Fortalecer conexiones entre territorios exitosos
  → Debilitar conexiones no usadas
  → Replay de patrones importantes
  → Poda de pesos innecesarios
```
- Como cuando dormimos y el cerebro consolida memorias
- Ajuste fino del Tálamo basado en patrones de éxito
- El modelo se vuelve más eficiente sin datos nuevos

### Fase 5: CURIOSIDAD — Active Learning ($5-10)
```
Confianza Early Exit baja → "No sé esto" → Generar datos de entrenamiento
Confianza alta + error     → "Estoy mal seguro" → Penalización extra
```
- El modelo identifica qué NO sabe usando Early Exit
- Genera o busca datos específicamente para sus debilidades
- Metacognición: aprende a evaluar su propio conocimiento

## Innovaciones Técnicas

### 1. Metacognitive Loss (Pérdida Metacognitiva)
```python
L_meta = α * CE_loss + β * |confidence - accuracy|
# Si confía mucho y falla → penalización alta (sobreconfianza)
# Si no confía y falla → penalización baja (sabe que no sabe)
# Si confía y acierta → recompensa (calibración correcta)
```

### 2. Territory Entropy Regularization
```python
L_entropy = -γ * Σ terr_acts * log(terr_acts)
# Evita que todos los territorios se activen igual (colapso)
# Incentiva especialización: cada territorio es experto en algo
```

### 3. Hebbian Frontier Learning
```python
# Después de predicción exitosa:
frontier_ij += η * activation_i * activation_j  # "fire together, wire together"
# Después de predicción fallida:
frontier_ij -= η * activation_i * activation_j  # "anti-Hebbian"
```

### 4. Code Execution Reward (sin humanos)
```python
reward = {
    'compila': +0.3,        # el código es válido
    'ejecuta': +0.5,        # el código corre sin error
    'correcto': +1.0,       # produce resultado esperado
    'error_sintaxis': -0.5, # error de parsing
    'error_runtime': -0.3,  # error en ejecución
    'timeout': -0.1,        # loop infinito
}
```

## Estimación de Costo Total

| Fase | Tokens | Costo GPU (A40) | Días |
|------|--------|-----------------|------|
| Infancia | 3-5B | $5-15 | 1-2 |
| Experimentación | 1-3B (generados) | $10-30 | 2-4 |
| Filosofar | 0.5-1B | $5-15 | 1-2 |
| Sueño | 0 (replay) | $2-5 | 0.5 |
| Curiosidad | 0.5-1B (targeted) | $5-10 | 1 |
| **TOTAL** | **5-10B** | **$27-75** | **5-10** |

vs Qwen: 5,500B tokens, $50,000-200,000, meses.

## Cómo Entrenarlo en Tu PC

### Requisitos Mínimos
- **8GB VRAM**: LoRA fine-tuning (fases 2-5 después de pre-training cloud)
- **16GB VRAM**: Full fine-tuning con gradient checkpointing
- **24GB VRAM**: Entrenamiento completo todas las fases

### Flujo Recomendado
1. **Cloud A40** ($30-50): Fase 1 (pre-training curriculum) + Fase 3 (reasoning)
2. **Tu PC** (gratis): Fase 2 (self-play) + Fase 4 (consolidation) + Fase 5 (active learning)

### Por qué Tu PC es Suficiente para Self-Play
- Self-play no procesa datasets masivos — genera 1 ejemplo, entrena, repite
- Cada ciclo: generar 10 programas → ejecutar → aprender = ~1 minuto en RTX 3060
- 1000 ciclos/día = modelo mejorando constantemente = 0 costo de GPU cloud

## Implementación

```
pampar/coder/v2/aprendizaje/
├── __init__.py           # Exports
├── curriculum.py         # Fase 1: Niveles de dificultad
├── self_play.py          # Fase 2: Generación + ejecución
├── razonamiento.py       # Fase 3: Chains of thought
├── neuroplasticidad.py   # Fase 4: Hebbian + consolidación
└── metacognicion.py      # Fase 5: Active learning + meta-loss

scripts/
├── train_cerebral.py     # Pipeline completo 5 fases
└── generar_curriculum.py # Preparar datos por nivel
```
