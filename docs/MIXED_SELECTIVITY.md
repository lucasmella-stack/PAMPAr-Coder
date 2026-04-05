# Mixed Selectivity: FFN Compartido + Modulación Contextual (FiLM)

> Última actualización: Abril 2026  
> Autor: Lucas (concepto) + implementación en PamparV3

---

## Resumen

PamparV3 reemplaza las **4 copias independientes de StreamFFN** (una por territorio) con **1 FFN compartido** + **4 ContextModulators** por nivel. El mismo bloque de pesos se lee de 4 formas distintas según un vector contextual de 63 dimensiones.

**Resultado:** 62.6M params (antes ~105M) — **40% de reducción** sin perder capacidad expresiva.

---

## Motivación

### Neurociencia: "Mixed Selectivity"

Una neurona cortical no responde a un solo estímulo. Rigotti et al. (2013) demostraron que las neuronas exhiben **selectividad mixta**: la misma neurona que responde a "ubicación" también codifica "tiempo" y "contexto de tarea". Esta propiedad es _necesaria_ para computación cognitiva compleja.

### La conexión con PamparV3

PamparV3 ya tiene un sistema de routing (Tálamo) que genera:

- `zona_acts [B, L, 52]` — activación de 52 zonas de Brodmann (tipo de token)
- `terr_acts [B, L, 4]` — pesos de los 4 territorios (sintaxis, semántica, lógico, estructural)

La idea de Lucas: _"Si ya sabemos QUÉ tipo de token es y QUÉ territorio domina... ¿por qué no usar esa info para LEER el mismo FFN de forma diferente en vez de tener 4 copias?"_

---

## Diseño técnico

### Vector contextual (63 dimensiones)

```
ctx = [zona_acts(52), terr_acts(4), depth(1), conf(1), n_levels(1), stream_one_hot(4)]
       ─────────── ──────────── ──────── ─────── ──────────── ────────────────
       Tipo token   Dominio     Nivel    Confianza  Meta         Identidad
```

| Indicador   | Dims | Fuente                      | Interpretación                     |
| ----------- | ---- | --------------------------- | ---------------------------------- |
| `zona_acts` | 52   | TálamoInicial               | keyword, variable, string, etc.    |
| `terr_acts` | 4    | TálamoInicial               | peso por territorio                |
| `depth`     | 1    | `nivel_idx / n_levels`      | 0.0=superficial, 1.0=profundo      |
| `conf`      | 1    | `exit_head` (con `no_grad`) | 0-1, ¿el modelo ya entendió?       |
| `n_levels`  | 1    | `config.n_niveles / 10`     | normalización del modelo           |
| `stream_oh` | 4    | one-hot del stream actual   | identidad del stream que se modula |

### ContextModulator (FiLM)

```python
class ContextModulator(nn.Module):
    CONTEXT_DIM = 63

    def __init__(self, dim: int, bottleneck: int = 128):
        self.proj = nn.Sequential(
            nn.Linear(63, bottleneck),    # comprimir
            nn.SiLU(),
            nn.Linear(bottleneck, dim*2), # generar gamma + beta
        )
        # La última capa inicia en zeros → gamma≈0, beta≈0 → identidad

    def forward(self, ffn_out, zona_acts, terr_acts, stream_idx, nivel_idx, n_levels, conf):
        ctx = self._build_context(zona_acts, terr_acts, stream_idx, nivel_idx, n_levels, conf)
        gamma, beta = self.proj(ctx).chunk(2, dim=-1)
        return (1 + gamma) * ffn_out + beta
```

La fórmula FiLM `(1 + γ) · x + β`:

- **γ (gamma)** escala cada dimensión — amplifica features relevantes, suprime irrelevantes
- **β (beta)** desplaza — inyecta información contextual que el FFN base no tiene
- Al iniciar con γ=0, β=0 → pasa el FFN sin modificar → entrenamiento estable

### Flujo en NivelProfundo

```
1. Combinar: x_combined = Σ streams[t] × terr_acts[:,:,t]
2. Atención: x_attn = BloqueAttn(x_combined)
3. Re-route: zona_acts actualizado = TálamoNivel(x_attn)
          → conf_value = exit_head(x_combined + x_attn)  [no_grad]
4. FFN:     h_base = ffn_shared(norm(stream + x_attn))     ← 1 sola FFN
5. Modular: h_mod = modulator_t(h_base, ctx)               ← 4 modulators
6. Weight:  h = h_mod × terr_acts[:,:,t]                   ← territorial gating
7. Lateral: fibras blancas entre streams
8. Exit?:   si conf > 0.90 → salir temprano
```

---

## Conteo de parámetros

| Componente                | Legacy (4 FFN)    | Mixed Selectivity |
| ------------------------- | ----------------- | ----------------- |
| Embeddings (tok_emb/head) | 30.7M             | 30.7M             |
| Atención GQA ×5           | 5.1M              | 5.1M              |
| **StreamFFN**             | **4× ×5 = 65.5M** | **1× ×5 = 16.4M** |
| **ContextModulators**     | —                 | **4× ×5 = 3.4M**  |
| LateralGates ×5           | 3.3M              | 3.3M              |
| Tálamo + routing + norms  | ~6M               | ~6M               |
| **TOTAL**                 | **~105M**         | **~62.6M**        |

**Ahorro neto: 42.4M params (40%)**

---

## Configuración

En `ConfigV3`:

```python
use_mixed_selectivity: bool = True   # True = compartido + modulators
modulator_bottleneck: int = 128      # tamaño intermedio del modulator
```

`use_mixed_selectivity=False` restaura el comportamiento original con 4 FFNs independientes. Los checkpoints del modo legacy **no son compatibles** con el modo mixed (keys diferentes en state_dict).

---

## Archivos modificados

| Archivo                             | Cambio                                        |
| ----------------------------------- | --------------------------------------------- |
| `pampar/coder/v3/bloques.py`        | +ContextModulator, NivelProfundo init/forward |
| `pampar/coder/v3/config.py`         | +use_mixed_selectivity, +modulator_bottleneck |
| `pampar/coder/v3/modelo.py`         | checkpointing con zona_acts, docstring        |
| `scripts/test_mixed_selectivity.py` | Test de compilación + forward pass            |

---

## Posibilidades futuras

1. **Más profundidad:** Con 42M ahorrados, subir de 5 a 8+ niveles manteniendo ~105M.
2. **Más streams:** De 4 a 6-8 especialidades. Costo marginal: solo modulators extra (~170K c/u).
3. **Dimensión mayor:** Subir dim de 640 a ~830 para vectores más expresivos.
4. **Cross-level modulators:** Compartir el FFN entre NIVELES también (no solo streams).
5. **Adaptive bottleneck:** El tamaño del modulator podría crecer con la profundidad.

---

## Referencias

- Rigotti, M. et al. (2013). _The importance of mixed selectivity in complex cognitive tasks._ Nature.
- Perez, E. et al. (2018). _FiLM: Visual Reasoning with a General Conditioning Layer._ AAAI.
- Anthropic (2022). _Superposition in Neural Networks._
