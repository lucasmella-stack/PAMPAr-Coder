# PAMPAr-Coder - Copilot Instructions

> Instrucciones específicas para este proyecto. Se combinan con tu perfil global.

## Proyecto

**PAMPAr-Coder** es un LLM de código con arquitectura cerebral inspirada en las 52 zonas de Brodmann.
"El Linux de la IA" - Hacer más con menos hardware.

## Arquitectura

```
Input → Embedding → [BloqueTerrritorial ×N] → LM Head → Output
                          ↓
              TálamoBrodmann (LLAVES 80% + Atención 20%)
                          ↓
    ┌─────────────────────┴─────────────────────┐
    ▼                                           ▼
┌───────────────┐                    ┌───────────────┐
│   SINTAXIS    │                    │   SEMÁNTICA   │
│  Zonas 1-13   │                    │  Zonas 14-39  │
└───────────────┘                    └───────────────┘
    ▼                                           ▼
┌───────────────┐                    ┌───────────────┐
│    LÓGICO     │                    │ ESTRUCTURAL   │
│  Zonas 40-44  │                    │  Zonas 45-52  │
└───────────────┘                    └───────────────┘
```

## Componentes Clave

| Archivo | Propósito |
|---------|-----------|
| `pampar/coder/model_v2.py` | Modelo principal con 52 zonas |
| `pampar/coder/talamo_v2.py` | Tálamo orquestador con LLAVES |
| `pampar/coder/zonas_brodmann.py` | Definición de las 52 zonas |
| `cloud/runpod/train_cloud.py` | Script de entrenamiento en cloud |
| `cloud/runpod/config_3b.py` | Configuraciones del modelo |

## Convenciones

- **Idioma código**: Inglés
- **Comentarios/docs**: Español o Inglés según contexto
- **Nombres de clases**: Español para conceptos de dominio (`Talamo`, `Territorio`, `Zona`)
- **Variables**: Inglés (`input_ids`, `hidden_states`)

## Stack

- PyTorch 2.x
- SentencePiece (tokenizer BPE)
- Hugging Face datasets
- RunPod/Cloud para entrenamiento

## Comandos frecuentes

```bash
# Entrenar localmente
python scripts/train.py --config 1.5B --epochs 10

# Entrenar en cloud (RunPod)
ssh root@IP -p PORT
cd /workspace/PAMPAr-Coder
nohup python3 cloud/runpod/train_cloud.py --config 3B > training.log 2>&1 &

# Ver progreso
tail -f training.log
```

## Reglas específicas

1. **LLAVES** son patrones regex que clasifican tokens - NUNCA usar ML para esto
2. **Territorios** procesan en paralelo - mantener independencia
3. **Cuantización INT4** solo para LLAVES lookup tables
4. **Gradient checkpointing** siempre activo para modelos >500M params
5. **Tests** en `tests/` con pytest
