# PAMPAr-Coder - Copilot Instructions

> Instrucciones específicas para este proyecto. Se combinan con tu perfil global.

## Proyecto

**PAMPAr-Coder** es un LLM de código 1.5B con arquitectura cerebral inspirada en las 52 zonas de Brodmann.
"El Linux de la IA" - Hacer más con menos hardware.

## Arquitectura

```
Input → Embedding → [BloqueTerrritorial ×N] → LM Head → Output
                          ↓
              TálamoBrodmann (LLAVES 80% + Atención 20%)
              + Conv1D causal (ventana 32 tokens)
                          ↓
    ┌─────────────────────┴─────────────────────┐
    ▼                                           ▼
┌───────────────┐                    ┌───────────────┐
│   SINTAXIS    │◄── simbiosis ────►│   SEMÁNTICA   │
│  Zonas 1-15   │                    │  Zonas 16-30  │
└───────────────┘                    └───────────────┘
    ▼                                           ▼
┌───────────────┐                    ┌───────────────┐
│    LÓGICO     │◄── simbiosis ────►│ ESTRUCTURAL   │
│  Zonas 31-42  │                    │  Zonas 43-52  │
└───────────────┘                    └───────────────┘
```

## Componentes Clave

| Archivo | Propósito |
|---------|-----------|
| `pampar/coder/v2/modelo.py` | PampaRCoderV2 con 52 zonas |
| `pampar/coder/v2/config.py` | ConfigPampaRCoderV2 + presets |
| `pampar/coder/v2/talamo.py` | Tálamo orquestador con LLAVES + context conv |
| `pampar/coder/v2/llaves.py` | LLAVES lookup tables (INT8, 256 niveles) |
| `pampar/coder/v2/bloques.py` | BloqueTerrritorial + relaciones simbióticas |
| `pampar/coder/v2/zonas.py` | Definición de las 52 zonas de Brodmann |
| `pampar/coder/v2/aprendizaje/` | Metacognición, neuroplasticidad, memoria errores |
| `cloud/runpod/train_cloud.py` | Script de entrenamiento en cloud |

## Convenciones

- **Idioma código**: Inglés
- **Comentarios/docs**: Español o Inglés según contexto
- **Nombres de clases**: Español para conceptos de dominio (`Talamo`, `Territorio`, `Zona`, `MemoriaErrores`)
- **Variables**: Inglés (`input_ids`, `hidden_states`)

## Stack

- PyTorch 2.x
- SentencePiece (tokenizer BPE, 48K vocab)
- Hugging Face datasets
- RunPod/Cloud para entrenamiento

## Comandos frecuentes

```bash
# Entrenar localmente
python scripts/train.py --config 1.5B --epochs 10

# Entrenar en cloud (RunPod)
ssh root@IP -p PORT
cd /workspace/PAMPAr-Coder
screen -S train
python3 cloud/runpod/train_cloud.py --config 1_5B > training.log 2>&1

# Ver progreso
tail -f training.log
```

## Reglas específicas

1. **LLAVES** son patrones regex que clasifican tokens - NUNCA usar ML para esto
2. **Territorios** procesan en paralelo, luego combinan via soporte simbiótico
3. **Cuantización INT8** (256 niveles) para LLAVES lookup tables
4. **Early Exit** usa percentil 10 per-token (no promedio global)
5. **Gradient checkpointing** siempre activo para modelos >500M params
6. **Tests** en `tests/` con pytest
7. **Ventana de contexto** (32 tokens) usa convolución causal - pad izquierdo
